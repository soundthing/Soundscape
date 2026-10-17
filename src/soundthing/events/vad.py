"""Speech masking (Silero VAD per-chunk probabilities and speech regions)."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from ..config import Config

VAD_CHUNK_SEC: float = 512 / 16000


@functools.lru_cache(maxsize=1)
def _load_silero() -> Any:
    """Load Silero VAD model via silero_vad package."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _silero_probs(model: Any, audio16k: np.ndarray) -> np.ndarray:
    """Run Silero VAD over audio16k in 512-sample chunks, returning probabilities."""
    import torch

    if hasattr(model, "reset_states"):
        model.reset_states()

    chunk_size = 512
    n_samples = len(audio16k)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    probs = np.zeros(n_chunks, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = audio16k[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            t_chunk = torch.from_numpy(chunk)
            probs[i] = float(model(t_chunk, 16000))

    return probs


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1-D audio array to target_sr using torchaudio."""
    import torch
    import torchaudio.transforms as T

    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(tensor).squeeze(0)
    return resampled.detach().cpu().numpy()


def _postprocess_spans(
    raw_spans: list[tuple[float, float]],
    duration: float,
    pad_sec: float,
    merge_gap_sec: float,
) -> list[tuple[float, float]]:
    """Pad by pad_sec on both sides, clip to [0, duration], merge gaps < merge_gap_sec, round to 3 decimals."""
    if not raw_spans:
        return []

    padded: list[tuple[float, float]] = []
    for s, e in raw_spans:
        s_pad = max(0.0, float(s) - pad_sec)
        e_pad = min(duration, float(e) + pad_sec)
        if e_pad > s_pad:
            padded.append((s_pad, e_pad))

    if not padded:
        return []

    padded.sort(key=lambda span: (span[0], span[1]))

    merged: list[tuple[float, float]] = []
    cur_s, cur_e = padded[0]
    for next_s, next_e in padded[1:]:
        if next_s - cur_e < merge_gap_sec:
            cur_e = max(cur_e, next_e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = next_s, next_e
    merged.append((cur_s, cur_e))

    result: list[tuple[float, float]] = []
    for s, e in merged:
        s_r, e_r = round(s, 3), round(e, 3)
        if e_r > s_r:
            result.append((s_r, e_r))
    return result


def speech_probs(audio: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """(V,) float32 per-chunk speech probability."""
    vad_cfg = cfg.pipeline.get("events", {}).get("vad", {})
    model_name = vad_cfg.get("model", "silero")
    if model_name == "none":
        return np.zeros(0, dtype=np.float32)
    if model_name != "silero":
        raise ValueError(f"Unknown VAD model: {model_name!r}")

    if audio is None or audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    if audio.ndim == 2:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=-1)
    elif audio.ndim > 2:
        audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))

    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    if sr != 16000:
        audio16k = _resample(audio, sr, 16000)
    else:
        audio16k = audio

    model = _load_silero()
    probs = _silero_probs(model, audio16k)
    return probs.astype(np.float32)


def speech_regions_from_probs(
    probs: np.ndarray,
    chunk_sec: float,
    duration: float,
    cfg: Config,
) -> list[tuple[float, float]]:
    """Convert per-chunk speech probabilities to padded, merged speech spans."""
    vad_cfg = cfg.pipeline.get("events", {}).get("vad", {})
    model_name = vad_cfg.get("model", "silero")
    if model_name == "none" or probs is None or len(probs) == 0:
        return []
    if model_name != "silero":
        raise ValueError(f"Unknown VAD model: {model_name!r}")

    threshold = float(vad_cfg.get("threshold", 0.5))
    min_speech_sec = float(vad_cfg.get("min_speech_sec", 0.25))
    pad_sec = float(vad_cfg.get("pad_sec", 0.1))
    merge_gap_sec = float(vad_cfg.get("merge_gap_sec", 0.3))

    off_threshold = max(threshold - 0.15, 0.01)

    raw_spans: list[tuple[float, float]] = []
    in_speech = False
    start_idx = 0

    for i, p in enumerate(probs):
        if not in_speech:
            if p >= threshold:
                in_speech = True
                start_idx = i
        else:
            if p < off_threshold:
                in_speech = False
                raw_spans.append((start_idx * chunk_sec, i * chunk_sec))

    if in_speech:
        raw_spans.append((start_idx * chunk_sec, len(probs) * chunk_sec))

    # Drop raw spans shorter than min_speech_sec
    filtered_spans = [(s, e) for s, e in raw_spans if (e - s) >= min_speech_sec]

    return _postprocess_spans(filtered_spans, duration, pad_sec, merge_gap_sec)


def speech_regions(audio: np.ndarray, sr: int, cfg: Config) -> list[tuple[float, float]]:
    """Return sorted (start, end) speech spans in seconds."""
    if audio is None or audio.size == 0:
        return []

    if audio.ndim == 2:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            n_samples = audio.shape[1]
        else:
            n_samples = audio.shape[0]
    elif audio.ndim > 2:
        n_samples = audio.shape[0]
    else:
        n_samples = len(audio)

    duration = float(n_samples) / float(sr) if sr > 0 else 0.0
    probs = speech_probs(audio, sr, cfg)
    return speech_regions_from_probs(probs, VAD_CHUNK_SEC, duration, cfg)


def speech_mask(times: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    """(T,) bool: True where window centre falls inside speech."""
    mask = np.zeros(len(times), dtype=bool)
    for s, e in regions:
        mask |= (times >= s) & (times < e)
    return mask
