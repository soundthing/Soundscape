"""Environment component for the evaluation harness (Person 2).

Provides cache generation, replay prediction from cached logits, fixed objective
scoring (time-weighted accuracy + macro-averaged accuracy over labeled environments),
and search space definition for tuning environment classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import wave

import numpy as np

from soundthing.config import Config
from soundthing.contracts import (
    GENERAL_ENV,
    ClipLabels,
    EnvSegment,
    ExtractedInputs,
    load_labels,
)
from soundthing.environment.audio import audio_env_logits
from soundthing.environment.classifier import segments_from_logits
from soundthing.environment.segmentation import detect_spans
from soundthing.environment.visual import visual_env_logits
from soundthing.harness import paths
from soundthing.harness.components.base import Clip, Component, Evaluation

CACHE_VERSION: int = 1


@dataclass
class EnvCache:
    video_id: str
    duration: float
    key: str
    spans: list[tuple[float, float]]
    env_ids: list[str]  # non-general envs, config order
    audio_logits: np.ndarray | None  # (S, E) raw prompt similarities, pre-temperature
    visual_logits: np.ndarray | None  # (S, E) or None (audio-only clips)


def cache_path(video_id: str, key: str, root: Path | None = None) -> Path:
    base = root or paths.cache_root()
    return base / "environment" / key / f"{video_id}.npz"


def cache_key(cfg: Config) -> str:
    """sha1[:12] of canonical JSON of segmentation, models, and environment prompts."""
    ecfg = cfg.pipeline.get("environment", {})
    seg_cfg = ecfg.get("segmentation", {})
    visual_model = ecfg.get("visual", {}).get("model")
    audio_model = ecfg.get("audio", {}).get("model")

    envs_data = [
        {
            "id": env.id,
            "audio_prompts": list(env.audio_prompts),
            "visual_prompts": list(env.visual_prompts),
        }
        for env in cfg.environments
    ]
    payload: dict[str, Any] = {
        "audio_model": audio_model,
        "cache_version": CACHE_VERSION,
        "environments": envs_data,
        "segmentation": seg_cfg,
        "visual_model": visual_model,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def save_env_cache(cache: EnvCache, path: Path | None = None) -> Path:
    """Save EnvCache to an .npz file with JSON metadata."""
    if path is None:
        path = cache_path(cache.video_id, cache.key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "video_id": cache.video_id,
        "duration": float(cache.duration),
        "key": cache.key,
        "spans": [(float(s), float(e)) for s, e in cache.spans],
        "env_ids": list(cache.env_ids),
        "has_audio": cache.audio_logits is not None,
        "has_visual": cache.visual_logits is not None,
        "cache_version": CACHE_VERSION,
    }
    arrays: dict[str, Any] = {
        "meta": np.array(json.dumps(meta)),
        "spans": np.asarray(cache.spans, dtype=np.float64).reshape(-1, 2),
    }
    if cache.audio_logits is not None:
        arrays["audio_logits"] = np.asarray(cache.audio_logits, dtype=np.float32)
    if cache.visual_logits is not None:
        arrays["visual_logits"] = np.asarray(cache.visual_logits, dtype=np.float32)

    np.savez_compressed(path, **arrays)
    return path


def load_cache_file(path: Path) -> EnvCache:
    """Load an EnvCache from an .npz file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cache file not found at {path}")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("cache_version") != CACHE_VERSION:
            raise ValueError(
                f"Cache version mismatch: got {meta.get('cache_version')}, expected {CACHE_VERSION}"
            )
        video_id = str(meta["video_id"])
        duration = float(meta["duration"])
        key = str(meta["key"])
        env_ids = list(meta["env_ids"])
        if "spans" in data:
            spans_arr = data["spans"].reshape(-1, 2)
            spans = [(float(r[0]), float(r[1])) for r in spans_arr]
        else:
            spans = [(float(s), float(e)) for s, e in meta["spans"]]
        audio_logits = (
            data["audio_logits"].astype(np.float32)
            if meta.get("has_audio", False) and "audio_logits" in data
            else None
        )
        visual_logits = (
            data["visual_logits"].astype(np.float32)
            if meta.get("has_visual", False) and "visual_logits" in data
            else None
        )
    return EnvCache(
        video_id=video_id,
        duration=duration,
        key=key,
        spans=spans,
        env_ids=env_ids,
        audio_logits=audio_logits,
        visual_logits=visual_logits,
    )


def load_env_cache(video_id: str | Path, cfg: Config | None = None) -> EnvCache:
    """Load cached environment logits for video_id under cfg's cache_key."""
    if isinstance(video_id, Path) or cfg is None:
        return load_cache_file(Path(video_id))
    key = cache_key(cfg)
    path = cache_path(str(video_id), key)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing environment cache for clip {video_id!r} with key {key!r} at {path}"
        )
    return load_cache_file(path)


def _audio_info(audio_path: Path) -> tuple[float, int]:
    """Return (duration_sec, sample_rate) for an audio file."""
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.duration), int(info.samplerate)
    except Exception:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = float(frames) / float(rate) if rate > 0 else 0.0
            return duration, int(rate)


def _find_video_file(audio_path: Path, video_id: str) -> Path | None:
    """Look for an associated video file near audio_path."""
    for ext in [".mp4", ".mkv", ".avi", ".webm", ".mov"]:
        candidate = audio_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
        candidate2 = audio_path.parent / f"{video_id}{ext}"
        if candidate2.is_file():
            return candidate2
    return None


def _find_frames_dir(audio_path: Path, video_id: str) -> Path:
    """Look for an extracted frames directory near audio_path."""
    candidates = [
        audio_path.parent / video_id / "frames",
        audio_path.parent / "frames" / video_id,
        audio_path.parent / f"{video_id}_frames",
        audio_path.parent / "frames",
        audio_path.with_suffix(".frames"),
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("frame_*.jpg")):
            return c
    return audio_path.parent / "__nonexistent_frames__"


def build_cache(video_id: str, audio_path: Path, cfg: Config) -> Path:
    """Run slow environment models on a clip and save the resulting EnvCache."""
    audio_path = Path(audio_path)
    duration, sample_rate = _audio_info(audio_path)
    video_file = _find_video_file(audio_path, video_id)

    ecfg = cfg.pipeline.get("environment", {})
    seg_cfg = ecfg.get("segmentation", {})
    max_segment_sec = float(seg_cfg.get("max_segment_sec", 30.0))

    if video_file is not None:
        spans = detect_spans(str(video_file), duration, cfg)
    else:
        if duration <= 0.0:
            spans = []
        elif duration <= max_segment_sec:
            spans = [(0.0, duration)]
        else:
            spans = []
            cur = 0.0
            while cur < duration:
                end = min(duration, cur + max_segment_sec)
                spans.append((cur, end))
                cur = end

    frames_dir = _find_frames_dir(audio_path, video_id)
    inputs = ExtractedInputs(
        video_id=video_id,
        video_path=str(video_file) if video_file is not None else "",
        audio_path=str(audio_path),
        frames_dir=str(frames_dir),
        fps=1.0,
        duration=duration,
        sample_rate=sample_rate,
        audio_stereo_path=str(audio_path),
    )

    envs = [e for e in cfg.environments if e.id != GENERAL_ENV]
    env_ids = [e.id for e in envs]

    audio_logits = audio_env_logits(inputs, spans, envs, cfg)
    visual_logits = (
        visual_env_logits(inputs, spans, envs, cfg) if inputs.frame_paths() else None
    )

    key = cache_key(cfg)
    cache = EnvCache(
        video_id=video_id,
        duration=duration,
        key=key,
        spans=spans,
        env_ids=env_ids,
        audio_logits=audio_logits,
        visual_logits=visual_logits,
    )
    out_path = cache_path(video_id, key)
    save_env_cache(cache, out_path)
    return out_path


def predict_segments(cache: EnvCache, cfg: Config) -> list[EnvSegment]:
    """Replay environment prediction from cached logits using current cfg."""
    return segments_from_logits(
        spans=cache.spans,
        audio_logits=cache.audio_logits,
        visual_logits=cache.visual_logits,
        env_ids=cache.env_ids,
        cfg=cfg,
        duration=cache.duration,
    )


def load_clips(cfg: Config, labels_dir: Path) -> list[Clip]:
    """Load labels and caches for clips with environment labels."""
    labels_dir = Path(labels_dir)
    clips: list[Clip] = []
    if not labels_dir.is_dir():
        return clips

    key = cache_key(cfg)
    for path in sorted(labels_dir.glob("*.json")):
        labels = load_labels(path)
        if not labels.segments:
            continue
        c_path = cache_path(labels.video_id, key)
        if not c_path.is_file():
            raise FileNotFoundError(
                f"Missing environment cache for clip {labels.video_id!r} with key {key!r} at {c_path}"
            )
        cache = load_cache_file(c_path)
        clips.append(Clip(video_id=labels.video_id, labels=labels, cache=cache))

    clips.sort(key=lambda c: c.video_id)
    return clips


def compute_objective_from_stats(
    totals: np.ndarray,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Compute (objective, time_acc, macro_acc) from summed stats."""
    t = np.asarray(totals, dtype=np.float64)
    is_1d = t.ndim == 1
    if is_1d:
        t = t[None, :]

    correct = t[:, 0::2]  # (B, n_envs)
    total = t[:, 1::2]  # (B, n_envs)

    sum_correct = correct.sum(axis=1)  # (B,)
    sum_total = total.sum(axis=1)  # (B,)

    time_acc = np.where(sum_total > 0, sum_correct / np.maximum(sum_total, 1e-12), 0.0)

    # Environments with 0 total seconds are excluded from the macro mean
    valid = total > 0  # (B, n_envs)
    per_env = np.where(valid, correct / np.maximum(total, 1e-12), 0.0)
    valid_counts = valid.sum(axis=1)  # (B,)
    macro_acc = np.where(
        valid_counts > 0, per_env.sum(axis=1) / np.maximum(valid_counts, 1), 0.0
    )

    objective = 0.5 * time_acc + 0.5 * macro_acc

    if is_1d:
        return float(objective[0]), float(time_acc[0]), float(macro_acc[0])
    return objective, time_acc, macro_acc


def stat_objective(totals: np.ndarray) -> np.ndarray | float:
    """Recompute objective values from summed stats."""
    totals_arr = np.asarray(totals, dtype=np.float64)
    obj, _, _ = compute_objective_from_stats(totals_arr)
    return obj


def evaluate(clips: list[Clip], cfg: Config) -> Evaluation:
    """Evaluate predicted segments against ground truth clip labels."""
    all_envs = list(cfg.environments)
    all_env_ids = [e.id for e in all_envs]
    n_envs = len(all_env_ids)
    n_stats = 2 * n_envs

    per_clip: dict[str, dict[str, Any]] = {}
    preds: dict[str, list[EnvSegment]] = {}
    confusion: dict[str, dict[str, float]] = {
        l_env: {p_env: 0.0 for p_env in all_env_ids} for l_env in all_env_ids
    }

    all_clip_stats: list[list[float]] = []

    for clip in clips:
        pred = predict_segments(clip.cache, cfg)
        preds[clip.video_id] = pred

        correct_sec: dict[str, float] = {e: 0.0 for e in all_env_ids}
        total_sec: dict[str, float] = {e: 0.0 for e in all_env_ids}

        for ls in clip.labels.segments:
            l_env = ls.env
            if l_env not in total_sec:
                total_sec[l_env] = 0.0
                correct_sec[l_env] = 0.0
                confusion[l_env] = {p_env: 0.0 for p_env in all_env_ids}
            total_sec[l_env] += ls.end - ls.start

            for ps in pred:
                overlap = max(0.0, min(ls.end, ps.end) - max(ls.start, ps.start))
                if overlap > 0.0:
                    p_env = ps.env
                    if p_env not in confusion[l_env]:
                        confusion[l_env][p_env] = 0.0
                    confusion[l_env][p_env] += overlap
                    if p_env == l_env:
                        correct_sec[l_env] += overlap

        clip_stat_list: list[float] = []
        for e in all_env_ids:
            clip_stat_list.append(float(correct_sec[e]))
            clip_stat_list.append(float(total_sec[e]))

        all_clip_stats.append(clip_stat_list)
        per_clip[clip.video_id] = {
            "stats": clip_stat_list,
        }

    if all_clip_stats:
        summed_stats = np.sum(all_clip_stats, axis=0)
    else:
        summed_stats = np.zeros(n_stats, dtype=np.float64)

    obj, time_acc, macro_acc = compute_objective_from_stats(summed_stats)

    total_pred_sec = sum(
        (ps.end - ps.start) for clip in clips for ps in preds[clip.video_id]
    )
    general_pred_sec = sum(
        (ps.end - ps.start)
        for clip in clips
        for ps in preds[clip.video_id]
        if ps.env == GENERAL_ENV
    )
    general_rate = (
        float(general_pred_sec / total_pred_sec) if total_pred_sec > 0.0 else 0.0
    )

    per_env_accuracy: dict[str, float] = {}
    for i, e in enumerate(all_env_ids):
        c_sec = summed_stats[2 * i]
        t_sec = summed_stats[2 * i + 1]
        per_env_accuracy[e] = float(c_sec / t_sec) if t_sec > 0.0 else 0.0

    diagnostics = {
        "confusion": confusion,
        "per_env_accuracy": per_env_accuracy,
    }

    metrics = {
        "objective": float(obj),
        "time_acc": float(time_acc),
        "macro_acc": float(macro_acc),
        "general_rate": float(general_rate),
    }

    return Evaluation(
        objective=float(obj),
        metrics=metrics,
        per_clip=per_clip,
        diagnostics=diagnostics,
    )


def limits(metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
    """Check candidate metrics against the current best."""
    if best is None:
        return {"time_acc": True, "macro_acc": True}

    def _get_metric(obj: Any, name: str) -> float:
        if isinstance(obj, dict):
            if "metrics" in obj and isinstance(obj["metrics"], dict) and name in obj["metrics"]:
                return float(obj["metrics"][name])
            if name in obj:
                return float(obj[name])
        if hasattr(obj, name):
            return float(getattr(obj, name))
        if hasattr(obj, "metrics") and isinstance(obj.metrics, dict) and name in obj.metrics:
            return float(obj.metrics[name])
        raise KeyError(f"Metric {name!r} not found in best entry")

    best_time_acc = _get_metric(best, "time_acc")
    best_macro_acc = _get_metric(best, "macro_acc")
    return {
        "time_acc": bool(metrics["time_acc"] >= best_time_acc - 1e-9),
        "macro_acc": bool(metrics["macro_acc"] >= best_macro_acc - 1e-9),
    }


def space(cfg: Config, labels_dir: Path | None = None) -> list[dict[str, Any]]:
    """Tunable parameters for the environment component."""
    ecfg = cfg.pipeline.get("environment", {})
    fusion_cfg = ecfg.get("fusion", {})
    seg_cfg = ecfg.get("segmentation", {})

    return [
        {
            "name": "environment.fusion.audio_weight",
            "type": "float",
            "bounds": [0.0, 1.0],
            "current": float(fusion_cfg.get("audio_weight", 0.3)),
            "shared": False,
        },
        {
            "name": "environment.fusion.visual_weight",
            "type": "float",
            "bounds": [0.0, 1.0],
            "current": float(fusion_cfg.get("visual_weight", 0.7)),
            "shared": False,
        },
        {
            "name": "environment.fusion.temperature",
            "type": "float",
            "bounds": [0.001, 1.0],
            "current": float(fusion_cfg.get("temperature", 0.01)),
            "shared": False,
            "scale": "log",
        },
        {
            "name": "environment.fusion.visual_similarity_floor",
            "type": "float",
            "bounds": [-1.0, 1.0],
            "current": float(fusion_cfg.get("visual_similarity_floor", 0.0)),
            "shared": False,
        },
        {
            "name": "environment.confidence_threshold",
            "type": "float",
            "bounds": [0.0, 1.0],
            "current": float(ecfg.get("confidence_threshold", 0.45)),
            "shared": False,
        },
        {
            "name": "environment.segmentation.threshold",
            "type": "float",
            "bounds": [1.0, 100.0],
            "current": float(seg_cfg.get("threshold", 27.0)),
            "shared": False,
            "needs_recache": True,
        },
        {
            "name": "environment.segmentation.min_segment_sec",
            "type": "float",
            "bounds": [1.0, 30.0],
            "current": float(seg_cfg.get("min_segment_sec", 5.0)),
            "shared": False,
            "needs_recache": True,
        },
        {
            "name": "environment.segmentation.max_segment_sec",
            "type": "float",
            "bounds": [5.0, 60.0],
            "current": float(seg_cfg.get("max_segment_sec", 30.0)),
            "shared": False,
            "needs_recache": True,
        },
    ]


def build_env_caches(labels_dir: Path, audio_dir: Path, cfg: Config) -> int:
    """Build environment caches for all clips in labels_dir with audio in audio_dir."""
    labels_dir = Path(labels_dir)
    audio_dir = Path(audio_dir)
    if not labels_dir.is_dir():
        return 0
    count = 0
    for path in sorted(labels_dir.glob("*.json")):
        labels = load_labels(path)
        if not labels.segments:
            continue
        audio_path = audio_dir / f"{labels.video_id}.wav"
        if not audio_path.is_file():
            found = False
            for ext in [".wav", ".flac", ".mp3", ".ogg"]:
                cand = audio_dir / f"{labels.video_id}{ext}"
                if cand.is_file():
                    audio_path = cand
                    found = True
                    break
            if not found:
                continue
        build_cache(labels.video_id, audio_path, cfg)
        count += 1
    return count


class EnvironmentComponent:
    name: str = "environment"
    owner: str = "person2"
    sections: tuple[str, ...] = ("environment.",)
    code_paths: tuple[str, ...] = ("src/soundthing/environment",)
    tuning_log: str = "src/soundthing/environment/tuning-log.md"

    @property
    def n_stats(self) -> int:
        from soundthing.config import load_config

        return 2 * len(load_config().environments)

    def cache_key(self, cfg: Config) -> str:
        return cache_key(cfg)

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path:
        return build_cache(video_id, audio_path, cfg)

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        return load_clips(cfg, labels_dir)

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        return evaluate(clips, cfg)

    def limits(
        self, metrics: dict[str, float], best: dict[str, Any] | None
    ) -> dict[str, bool]:
        return limits(metrics, best)

    def space(
        self, cfg: Config, labels_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        return space(cfg, labels_dir)

    def stat_objective(self, totals: np.ndarray) -> np.ndarray | float:
        return stat_objective(totals)


COMPONENT = EnvironmentComponent()
