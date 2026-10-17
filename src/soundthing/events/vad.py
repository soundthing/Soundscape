"""Speech masking (Silero VAD or Whisper timestamps)."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from ..config import Config


@functools.lru_cache(maxsize=1)
def _load_silero() -> Any:
    """Load Silero VAD model via silero_vad package."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _silero_timestamps(
    model: Any,
    audio16k: np.ndarray,
    threshold: float,
    min_speech_sec: float,
) -> list[tuple[float, float]]:
    """Run Silero VAD to get speech timestamps in seconds."""
    import torch
    from silero_vad import get_speech_timestamps

    tensor = torch.from_numpy(audio16k)
    min_ms = int(min_speech_sec * 1000)
    stamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_ms,
        return_seconds=True,
    )
    res: list[tuple[float, float]] = []
    for item in stamps:
        if isinstance(item, dict):
            res.append((float(item["start"]), float(item["end"])))
        else:
            res.append((float(item[0]), float(item[1])))
    return res


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


def speech_regions(audio: np.ndarray, sr: int, cfg: Config) -> list[tuple[float, float]]:
    """Return sorted (start, end) speech spans in seconds."""
    vad_cfg = cfg.pipeline.get("events", {}).get("vad", {})
    model_name = vad_cfg.get("model", "silero")
    if model_name == "none":
        return []
    if model_name != "silero":
        raise ValueError(f"Unknown VAD model: {model_name!r}")

    if audio is None or audio.size == 0:
        return []

    if audio.ndim == 2:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=-1)
    elif audio.ndim > 2:
        audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))

    audio = np.ascontiguousarray(audio, dtype=np.float32)

    duration = float(len(audio)) / float(sr)
    if duration <= 0:
        return []

    if sr != 16000:
        audio16k = _resample(audio, sr, 16000)
    else:
        audio16k = audio

    threshold = float(vad_cfg.get("threshold", 0.5))
    min_speech_sec = float(vad_cfg.get("min_speech_sec", 0.25))
    pad_sec = float(vad_cfg.get("pad_sec", 0.1))
    merge_gap_sec = float(vad_cfg.get("merge_gap_sec", 0.3))

    model = _load_silero()
    raw_spans = _silero_timestamps(model, audio16k, threshold, min_speech_sec)

    return _postprocess_spans(raw_spans, duration, pad_sec, merge_gap_sec)


def speech_mask(times: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    """(T,) bool: True where window centre falls inside speech."""
    mask = np.zeros(len(times), dtype=bool)
    for s, e in regions:
        mask |= (times >= s) & (times < e)
    return mask
