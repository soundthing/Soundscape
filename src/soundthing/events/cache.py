"""Model output cache for replayable sound event detection tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from soundthing.config import Config
from soundthing.events import separation, vad

CACHE_VERSION: int = 1


@dataclass
class ClipCache:
    video_id: str
    duration: float  # seconds of audio
    key: str  # cache_key(cfg) at build time
    window_sec: float
    hop: float
    times: np.ndarray  # (T,) window centre times
    audioset_names: list[str]  # (A,) tagger label names
    tagger_raw: np.ndarray  # (T, A) float16 raw tagger probs (all 527)
    clap_emb: np.ndarray | None  # (T, D) float16 L2-normalised CLAP audio embeddings; None if CLAP disabled
    vad_chunk_sec: float  # seconds per VAD prob (Silero 512 samples @16k = 0.032)
    vad_probs: np.ndarray  # (V,) float32 per-chunk speech probability
    # Source separation (all None when separation is disabled). Rows where sep_mask is
    # False are copies of the raw rows.
    sep_mask: np.ndarray | None = None  # (T,) bool: window was rescored on the effects stem
    tagger_sep: np.ndarray | None = None  # (T, A) float16 tagger probs on the effects stem
    clap_emb_sep: np.ndarray | None = None  # (T, D) float16 CLAP embeddings on the effects stem


def _window_slice(audio: np.ndarray, start: int, win: int) -> np.ndarray:
    chunk = audio[start : start + win]
    if len(chunk) < win:
        chunk = np.pad(chunk, (0, win - len(chunk)))
    return chunk


def cache_key(cfg: Config) -> str:
    """sha1 hex[:12] of canonical JSON of model/window settings + CACHE_VERSION."""
    ecfg = cfg.pipeline.get("events", {})
    tagger_model = ecfg.get("tagger", {}).get("model")
    clap_cfg = ecfg.get("clap", {})
    clap_model = clap_cfg.get("model")
    clap_enabled = bool(clap_cfg.get("enabled", True))
    window_sec = float(ecfg.get("window_sec", 1.0))
    hop_sec = float(ecfg.get("hop_sec", 0.25))
    vad_model = ecfg.get("vad", {}).get("model")

    payload: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "clap_enabled": clap_enabled,
        "clap_model": clap_model,
        "hop_sec": hop_sec,
        "tagger_model": tagger_model,
        "vad_model": vad_model,
        "window_sec": window_sec,
    }
    if separation.is_enabled(cfg):
        sc = separation.sep_config(cfg)
        payload["separation"] = {
            "chunk_sec": sc.get("chunk_sec"),
            "crossfade_sec": sc.get("crossfade_sec"),
            "gate_labels": sc.get("gate_labels"),
            "gate_threshold": sc.get("gate_threshold"),
            "gate_vad": sc.get("gate_vad"),
            "model": sc.get("model"),
            "pad_sec": sc.get("pad_sec"),
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def build_cache(
    audio: np.ndarray,
    sr: int,
    video_id: str,
    cfg: Config,
    models: Any = None,
) -> ClipCache:
    """Build a ClipCache from audio and models."""
    if models is None:
        from soundthing.events import detector

        models = detector.get_models(cfg)

    if models.audioset_names is None:
        raise ValueError("EventModels.audioset_names must not be None to build cache")

    if audio.ndim == 2:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=-1)
    elif audio.ndim > 2:
        audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    ecfg = cfg.pipeline.get("events", {})
    window_sec = float(ecfg.get("window_sec", 1.0))
    hop_sec = float(ecfg.get("hop_sec", 0.25))
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n_samples = len(audio)
    starts = list(range(0, max(1, n_samples - win + 1), hop))

    tagger_rows: list[np.ndarray] = []
    clap_rows: list[np.ndarray] | None = [] if models.clap is not None else None
    times: list[float] = []

    for s in starts:
        chunk = _window_slice(audio, s, win)
        t_row = models.tagger(chunk[None, :], sr)[0]
        tagger_rows.append(t_row)
        if models.clap is not None:
            c_row = models.clap.embed_audio(chunk, sr)
            if clap_rows is not None:
                clap_rows.append(c_row)
        times.append((s + win / 2.0) / sr)

    if tagger_rows:
        tagger_raw = np.array(tagger_rows, dtype=np.float16)
    else:
        tagger_raw = np.zeros((0, len(models.audioset_names)), dtype=np.float16)

    if clap_rows is not None:
        clap_emb = np.array(clap_rows, dtype=np.float16)
    else:
        clap_emb = None

    times_arr = np.array(times, dtype=np.float64)

    vad_probs = vad.speech_probs(audio, sr, cfg).astype(np.float32)
    vad_chunk_sec = float(vad.VAD_CHUNK_SEC)
    duration = float(n_samples) / float(sr) if sr > 0 else 0.0

    sep_mask = None
    tagger_sep = None
    clap_emb_sep = None

    if separation.is_enabled(cfg):
        if getattr(models, "separator", None) is None:
            raise ValueError(
                "separation is enabled in config but models.separator is None; "
                "a cache built under a separation key must contain separation results"
            )
        spans = separation.gate_spans(
            times_arr,
            tagger_raw,
            list(models.audioset_names),
            vad_probs,
            vad_chunk_sec,
            duration,
            window_sec,
            cfg,
        )
        mask = separation.window_mask(times_arr, window_sec, spans)
        tagger_sep = tagger_raw.copy()
        clap_emb_sep = clap_emb.copy() if clap_emb is not None else None
        if mask.any():
            sep_audio = separation.separate_spans(audio, sr, spans, models.separator, cfg)
            masked_indices = np.where(mask)[0]
            for i in masked_indices:
                chunk = _window_slice(sep_audio, starts[i], win)
                tagger_sep[i] = models.tagger(chunk[None, :], sr)[0]
                if models.clap is not None and clap_emb_sep is not None:
                    clap_emb_sep[i] = models.clap.embed_audio(chunk, sr)

        sep_mask = mask.astype(bool)
        tagger_sep = tagger_sep.astype(np.float16)
        if clap_emb_sep is not None:
            clap_emb_sep = clap_emb_sep.astype(np.float16)

    return ClipCache(
        video_id=video_id,
        duration=duration,
        key=cache_key(cfg),
        window_sec=window_sec,
        hop=hop_sec,
        times=times_arr,
        audioset_names=list(models.audioset_names),
        tagger_raw=tagger_raw,
        clap_emb=clap_emb,
        vad_chunk_sec=vad_chunk_sec,
        vad_probs=vad_probs,
        sep_mask=sep_mask,
        tagger_sep=tagger_sep,
        clap_emb_sep=clap_emb_sep,
    )


def save_cache(c: ClipCache, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "video_id": c.video_id,
        "duration": c.duration,
        "key": c.key,
        "window_sec": c.window_sec,
        "hop": c.hop,
        "audioset_names": c.audioset_names,
        "vad_chunk_sec": c.vad_chunk_sec,
        "has_clap": c.clap_emb is not None,
        "cache_version": CACHE_VERSION,
    }
    arrays: dict[str, Any] = {
        "meta": np.array(json.dumps(meta)),
        "times": c.times,
        "tagger_raw": c.tagger_raw.astype(np.float16),
        "vad_probs": c.vad_probs.astype(np.float32),
    }
    if c.clap_emb is not None:
        arrays["clap_emb"] = c.clap_emb.astype(np.float16)
    if c.sep_mask is not None:
        arrays["sep_mask"] = c.sep_mask.astype(bool)
    if c.tagger_sep is not None:
        arrays["tagger_sep"] = c.tagger_sep.astype(np.float16)
    if c.clap_emb_sep is not None:
        arrays["clap_emb_sep"] = c.clap_emb_sep.astype(np.float16)

    np.savez_compressed(path, **arrays)


def load_cache(path: Path) -> ClipCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("cache_version") != CACHE_VERSION:
            raise ValueError(
                f"Cache version mismatch: got {meta.get('cache_version')}, expected {CACHE_VERSION}"
            )
        video_id = str(meta["video_id"])
        duration = float(meta["duration"])
        key = str(meta["key"])
        window_sec = float(meta["window_sec"])
        hop = float(meta["hop"])
        audioset_names = list(meta["audioset_names"])
        vad_chunk_sec = float(meta["vad_chunk_sec"])
        times = data["times"]
        tagger_raw = data["tagger_raw"].astype(np.float16)
        vad_probs = data["vad_probs"].astype(np.float32)
        clap_emb = (
            data["clap_emb"].astype(np.float16)
            if meta.get("has_clap", False) and "clap_emb" in data
            else None
        )
        sep_mask = data["sep_mask"].astype(bool) if "sep_mask" in data else None
        tagger_sep = data["tagger_sep"].astype(np.float16) if "tagger_sep" in data else None
        clap_emb_sep = data["clap_emb_sep"].astype(np.float16) if "clap_emb_sep" in data else None

    return ClipCache(
        video_id=video_id,
        duration=duration,
        key=key,
        window_sec=window_sec,
        hop=hop,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=clap_emb,
        vad_chunk_sec=vad_chunk_sec,
        vad_probs=vad_probs,
        sep_mask=sep_mask,
        tagger_sep=tagger_sep,
        clap_emb_sep=clap_emb_sep,
    )


def cache_path(video_id: str, key: str, root: Path | None = None) -> Path:
    from soundthing.harness import paths

    return (root or paths.cache_root()) / key / f"{video_id}.npz"
