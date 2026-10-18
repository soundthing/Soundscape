"""Gated source separation for sound event detection."""

from __future__ import annotations

import copy
from typing import Any, Protocol

import numpy as np

from ..config import Config

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "model": "htdemucs",
    "gate_labels": [
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Conversation",
        "Narration, monologue",
        "Music",
        "Singing",
        "Song",
    ],
    "gate_threshold": 0.3,
    "gate_vad": True,
    "pad_sec": 0.5,
    "crossfade_sec": 0.05,
    "chunk_sec": 10.0,
    "fusion": "max",
    "sep_weight": 0.5,
    "suppress_under_speech": 1.0,
}


def sep_config(cfg: Config) -> dict[str, Any]:
    """Return DEFAULTS overlaid with cfg.pipeline events.separation."""
    res = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    pipeline = cfg.pipeline if cfg and getattr(cfg, "pipeline", None) else {}
    events = pipeline.get("events") if isinstance(pipeline, dict) else {}
    user_sep = (events.get("separation") or {}) if isinstance(events, dict) else {}
    for k, v in user_sep.items():
        if isinstance(v, list):
            res[k] = list(v)
        else:
            res[k] = v
    return res


def is_enabled(cfg: Config) -> bool:
    """Return True if source separation is enabled in config."""
    return bool(sep_config(cfg)["enabled"])


class Separator(Protocol):
    """Protocol for audio source separators."""

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """(N,) float32 mono at sr -> (N,) float32 effects stem at sr."""
        ...


class DemucsSeparator:
    """Separator implementation using Demucs pretrained models."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        if hasattr(self.model, "eval"):
            self.model.eval()

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract effects stem (sum of all sources except vocals)."""
        import torch
        import torchaudio
        import demucs.apply

        orig_len = len(audio)
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        target_sr = int(getattr(self.model, "samplerate", 44100))
        t = torch.as_tensor(audio, dtype=torch.float32)
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)

        if t.ndim == 1:
            x = torch.stack([t, t], dim=0)
        elif t.ndim == 2:
            if t.shape[0] == 1:
                x = torch.cat([t, t], dim=0)
            elif t.shape[0] == 2:
                x = t
            else:
                x = torch.stack([t[0], t[0]], dim=0)
        else:
            raise ValueError(f"Expected 1-D or 2-D audio tensor, got shape {tuple(t.shape)}")

        with torch.no_grad():
            out = demucs.apply.apply_model(self.model, x[None], device=self.device)

        sources = list(self.model.sources)
        effects_indices = [i for i, name in enumerate(sources) if name != "vocals"]
        if effects_indices:
            effects = out[0][effects_indices].sum(dim=0)
        else:
            effects = torch.zeros_like(out[0][0])

        effects = effects.mean(dim=0)

        if sr != target_sr:
            effects = torchaudio.functional.resample(effects, target_sr, sr)

        effects_np = effects.detach().cpu().numpy().astype(np.float32)

        if len(effects_np) < orig_len:
            effects_np = np.pad(effects_np, (0, orig_len - len(effects_np)))
        elif len(effects_np) > orig_len:
            effects_np = effects_np[:orig_len]

        return effects_np


def load_separator(cfg: Config) -> Separator:
    """Load a DemucsSeparator configured by cfg."""
    try:
        import demucs.pretrained
        import torch
    except ImportError as e:
        raise ImportError(
            'demucs is required for source separation. Install with: pip install -e ".[separation]"'
        ) from e

    c = sep_config(cfg)
    name = c.get("model", "htdemucs")
    model = demucs.pretrained.get_model(name)
    if hasattr(model, "eval"):
        model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "to"):
        model.to(device)
    return DemucsSeparator(model, device)


def gate_spans(
    times: np.ndarray,
    tagger_raw: np.ndarray,
    audioset_names: list[str],
    vad_probs: np.ndarray,
    vad_chunk_sec: float,
    duration: float,
    window_sec: float,
    cfg: Config,
) -> list[tuple[float, float]]:
    """Return sorted, padded, merged time spans where separation should run."""
    c = sep_config(cfg)
    gate_labels = set(c.get("gate_labels") or [])
    gate_threshold = float(c.get("gate_threshold", 0.3))
    gate_vad = bool(c.get("gate_vad", True))
    pad_sec = float(c.get("pad_sec", 0.5))

    cols = [i for i, name in enumerate(audioset_names) if name in gate_labels]

    raw_spans: list[tuple[float, float]] = []
    if (
        times is not None
        and len(times) > 0
        and len(cols) > 0
        and tagger_raw is not None
        and tagger_raw.shape[0] > 0
    ):
        sub = np.asarray(tagger_raw[:, cols], dtype=np.float32)
        max_vals = np.max(sub, axis=1)
        gated = max_vals >= np.float32(gate_threshold)
        half_win = float(window_sec) / 2.0
        for idx in np.where(gated)[0]:
            t = float(times[idx])
            raw_spans.append((t - half_win, t + half_win))

    if gate_vad and vad_probs is not None and len(vad_probs) > 0:
        from . import vad

        v_spans = vad.speech_regions_from_probs(vad_probs, vad_chunk_sec, duration, cfg)
        raw_spans.extend(v_spans)

    if not raw_spans:
        return []

    padded: list[tuple[float, float]] = []
    for s, e in raw_spans:
        s_pad = max(0.0, float(s) - pad_sec)
        e_pad = min(float(duration), float(e) + pad_sec)
        if e_pad > s_pad:
            padded.append((s_pad, e_pad))

    if not padded:
        return []

    padded.sort(key=lambda span: (span[0], span[1]))

    merged: list[tuple[float, float]] = []
    cur_s, cur_e = padded[0]
    for next_s, next_e in padded[1:]:
        if next_s <= cur_e:
            cur_e = max(cur_e, next_e)
        else:
            merged.append((float(cur_s), float(cur_e)))
            cur_s, cur_e = next_s, next_e
    merged.append((float(cur_s), float(cur_e)))

    return merged


def window_mask(
    times: np.ndarray,
    window_sec: float,
    spans: list[tuple[float, float]],
) -> np.ndarray:
    """Return (T,) bool: True if [t - window_sec/2, t + window_sec/2] lies fully inside one span."""
    times_arr = np.asarray(times, dtype=np.float64)
    n = len(times_arr)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not spans:
        return np.zeros(n, dtype=bool)

    half = float(window_sec) / 2.0
    w_start = times_arr - half
    w_end = times_arr + half

    mask = np.zeros(n, dtype=bool)
    tol = 1e-6
    for s, e in spans:
        span_s = float(s) - tol
        span_e = float(e) + tol
        mask |= (w_start >= span_s) & (w_end <= span_e)

    return mask


def separate_spans(
    audio: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    separator: Any,
    cfg: Config,
) -> np.ndarray:
    """Return a copy of audio with spans replaced by separated signal crossfaded at edges."""
    out = np.array(audio, dtype=np.float32, copy=True)
    if not spans or len(audio) == 0:
        return out

    c = sep_config(cfg)
    chunk_sec = float(c.get("chunk_sec", 10.0))
    crossfade_sec = float(c.get("crossfade_sec", 0.05))

    chunk_len = max(1, int(round(chunk_sec * sr)))
    cf_len = max(0, int(round(crossfade_sec * sr)))
    n_samples = len(audio)

    for start, end in spans:
        s_idx = max(0, min(n_samples, int(round(start * sr))))
        e_idx = max(0, min(n_samples, int(round(end * sr))))
        if e_idx <= s_idx:
            continue

        span_len = e_idx - s_idx

        # Split [s_idx, e_idx) into consecutive pieces of at most chunk_sec
        pieces: list[tuple[int, int]] = []
        cur = s_idx
        while cur < e_idx:
            nxt = min(cur + chunk_len, e_idx)
            pieces.append((cur, nxt))
            cur = nxt

        # Separate each piece with crossfade extension
        sep_pieces: list[np.ndarray] = []
        for p_start, p_end in pieces:
            ext_start = max(0, p_start - cf_len)
            ext_end = min(n_samples, p_end + cf_len)
            piece_in = np.asarray(audio[ext_start:ext_end], dtype=np.float32)
            piece_out = separator(piece_in, sr)
            if piece_out is None or len(piece_out) != len(piece_in):
                raise ValueError(
                    f"Separator output length mismatch: got {len(piece_out) if piece_out is not None else None}, expected {len(piece_in)}"
                )
            piece_out = np.asarray(piece_out, dtype=np.float32)
            if piece_out.ndim != 1 or piece_out.shape != piece_in.shape:
                raise ValueError(
                    f"Separator output shape mismatch: got {piece_out.shape}, expected {piece_in.shape}"
                )
            sep_pieces.append(piece_out)

        # Stitch pieces together into span_sep of length span_len
        span_sep = np.zeros(span_len, dtype=np.float32)

        if len(pieces) == 1:
            p_start, p_end = pieces[0]
            ext_start = max(0, p_start - cf_len)
            offset = p_start - ext_start
            span_sep[:] = sep_pieces[0][offset : offset + span_len]
        else:
            for k, (p_start, p_end) in enumerate(pieces):
                sp_s = p_start - s_idx
                sp_e = p_end - s_idx
                ext_start = max(0, p_start - cf_len)
                off = p_start - ext_start
                span_sep[sp_s:sp_e] = sep_pieces[k][off : off + (p_end - p_start)]

            if cf_len > 0:
                for k in range(len(pieces) - 1):
                    p_curr_s, p_curr_e = pieces[k]
                    p_next_s, p_next_e = pieces[k + 1]
                    J = p_curr_e
                    J_sp = J - s_idx

                    half = cf_len // 2
                    blend_start = max(0, J_sp - half)
                    blend_end = min(span_len, blend_start + cf_len)
                    actual_blend_len = blend_end - blend_start

                    if actual_blend_len > 0:
                        ext_curr_s = max(0, p_curr_s - cf_len)
                        k_slice_start = (s_idx + blend_start) - ext_curr_s
                        k_slice = sep_pieces[k][k_slice_start : k_slice_start + actual_blend_len]

                        ext_next_s = max(0, p_next_s - cf_len)
                        next_slice_start = (s_idx + blend_start) - ext_next_s
                        next_slice = sep_pieces[k + 1][next_slice_start : next_slice_start + actual_blend_len]

                        alpha = np.linspace(0.0, 1.0, actual_blend_len, endpoint=False, dtype=np.float32)
                        span_sep[blend_start:blend_end] = (1.0 - alpha) * k_slice + alpha * next_slice

        if cf_len == 0:
            out[s_idx:e_idx] = span_sep
        else:
            edge_cf = min(cf_len, span_len // 2)
            if edge_cf == 0:
                out[s_idx:e_idx] = span_sep
            else:
                out[s_idx + edge_cf : e_idx - edge_cf] = span_sep[edge_cf : span_len - edge_cf]

                ramp_in = np.linspace(0.0, 1.0, edge_cf, endpoint=False, dtype=np.float32)
                out[s_idx : s_idx + edge_cf] = (
                    (1.0 - ramp_in) * audio[s_idx : s_idx + edge_cf]
                    + ramp_in * span_sep[:edge_cf]
                )

                ramp_out = np.linspace(0.0, 1.0, edge_cf, endpoint=False, dtype=np.float32)
                out[e_idx - edge_cf : e_idx] = (
                    (1.0 - ramp_out) * span_sep[span_len - edge_cf :]
                    + ramp_out * audio[e_idx - edge_cf : e_idx]
                )

    return out
