"""Score sound event detection against dev or holdout ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import uuid

from ...config import Config, load_config
from ...contracts import (
    GENERAL_ENV,
    ClipLabels,
    EnvSegment,
    load_labels,
    validate_segments,
)
from .. import experiments
from . import ledger, paths
from .cache import ClipCache, cache_key, cache_path, load_cache
from .objective import ClipMatch, diagnostics, f_beta, grid_f1, match_clip
from .overrides import apply_overrides
from .replay import replay, text_embeddings_for


@dataclass
class ScoreResult:
    run_id: str
    objective: float  # F0.5 conditioned (labeled env at 0.7 confidence)
    baseline: float  # F0.5 unconditioned
    wrong_env: float  # F0.5 with experiments.wrong_env_segments
    safety_recall: float
    fp_per_min: float  # conditioned
    grid_f1: float
    per_clip: dict[str, dict]  # video_id -> {"conditioned": counts, "baseline": counts, "wrong_env": counts}
    diagnostics: dict
    limits: dict[str, bool]  # safety_recall, fp_per_min, above_baseline, wrong_env (vs current best)
    passed: bool  # all limits
    overrides: dict
    split: str

    def to_dict(self) -> dict:
        return asdict(self)


def labeled_env_segments(
    labels: ClipLabels,
    cfg: Config,
    confidence: float = 0.7,
) -> list[EnvSegment]:
    """Convert ground truth labeled environment segments into full-coverage EnvSegments."""
    for s in labels.segments:
        if s.env not in cfg.env_ids:
            raise KeyError(
                f"Unknown environment {s.env!r} in labels for clip {labels.video_id!r}"
            )

    if not labels.segments:
        result = [
            EnvSegment(
                start=0.0,
                end=float(labels.duration),
                env=GENERAL_ENV,
                confidence=1.0,
                probs={GENERAL_ENV: 1.0},
            )
        ]
        validate_segments(result, labels.duration)
        return result

    sorted_segs = sorted(labels.segments, key=lambda s: (s.start, s.end))
    result: list[EnvSegment] = []
    cur_time = 0.0

    for s in sorted_segs:
        if s.start - cur_time > 1e-6:
            result.append(
                EnvSegment(
                    start=cur_time,
                    end=float(s.start),
                    env=GENERAL_ENV,
                    confidence=1.0,
                    probs={GENERAL_ENV: 1.0},
                )
            )
            cur_time = float(s.start)

        seg_start = cur_time
        seg_end = float(s.end)
        if s.env == GENERAL_ENV:
            result.append(
                EnvSegment(
                    start=seg_start,
                    end=seg_end,
                    env=GENERAL_ENV,
                    confidence=1.0,
                    probs={GENERAL_ENV: 1.0},
                )
            )
        else:
            conf = float(confidence)
            result.append(
                EnvSegment(
                    start=seg_start,
                    end=seg_end,
                    env=s.env,
                    confidence=conf,
                    probs={s.env: conf, GENERAL_ENV: float(1.0 - conf)},
                )
            )
        cur_time = seg_end

    if float(labels.duration) - cur_time > 1e-6:
        result.append(
            EnvSegment(
                start=cur_time,
                end=float(labels.duration),
                env=GENERAL_ENV,
                confidence=1.0,
                probs={GENERAL_ENV: 1.0},
            )
        )
    elif result:
        result[-1].end = float(labels.duration)

    validate_segments(result, labels.duration)
    return result


def score(
    overrides: dict[str, Any] | None = None,
    split: str = "dev",
    cfg: Config | None = None,
    best: dict[str, Any] | None = None,
    labels_dir: Path | str | None = None,
) -> ScoreResult:
    """Replay every cached clip under a candidate config and return objective, limits, and counts."""
    # 1. Config resolution and overrides
    if cfg is None:
        cfg = load_config(Path.cwd() / "config")
    cfg = apply_overrides(cfg, overrides or {})

    # 2. Labels directory resolution and loading
    if labels_dir is not None:
        ldir = Path(labels_dir)
    elif split == "dev":
        ldir = paths.dev_labels_dir()
    elif split == "holdout":
        ldir = paths.holdout_dir() / "labels"
    else:
        raise ValueError(f"Unknown split: {split!r}")

    label_files = sorted(ldir.glob("*.json"))
    if not label_files:
        raise ValueError(f"No label files found in {ldir}")

    all_labels = [load_labels(p) for p in label_files]
    all_labels.sort(key=lambda l: l.video_id)

    # 3. Cache verification and loading
    key = cache_key(cfg)
    caches: list[ClipCache] = []
    for lbl in all_labels:
        cpath = cache_path(lbl.video_id, key)
        if not cpath.exists():
            raise FileNotFoundError(
                f"Cache missing for clip {lbl.video_id!r} with key {key!r} at {cpath}. "
                f"caches must be built/recached for this window/hop/model."
            )
        caches.append(load_cache(cpath))

    # 4. Text embeddings (only if CLAP enabled and at least one cache has clap_emb)
    clap_cfg = cfg.pipeline.get("events", {}).get("clap", {})
    clap_enabled = bool(clap_cfg.get("enabled", True))
    has_clap_cache = any(c.clap_emb is not None for c in caches)

    if clap_enabled and has_clap_cache:
        text_emb = text_embeddings_for(cfg)
    else:
        text_emb = None

    # 5. Replay and match in three modes
    tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

    cond_matches: list[ClipMatch] = []
    base_matches: list[ClipMatch] = []
    wrong_matches: list[ClipMatch] = []

    cond_preds_by_clip: dict[str, list] = {}
    labels_by_clip: dict[str, ClipLabels] = {}
    per_clip: dict[str, dict[str, dict[str, Any]]] = {}

    for lbl, cache in zip(all_labels, caches):
        if cache.duration != lbl.duration:
            cache = replace(cache, duration=lbl.duration)

        cond_segs = labeled_env_segments(lbl, cfg)
        wrong_segs = experiments.wrong_env_segments(cond_segs, cfg)

        pred_cond = replay(cache, cond_segs, cfg, text_emb=text_emb)
        pred_base = replay(cache, None, cfg, text_emb=text_emb)
        pred_wrong = replay(cache, wrong_segs, cfg, text_emb=text_emb)

        m_cond = match_clip(pred_cond, lbl, cfg, tol)
        m_base = match_clip(pred_base, lbl, cfg, tol)
        m_wrong = match_clip(pred_wrong, lbl, cfg, tol)

        cond_matches.append(m_cond)
        base_matches.append(m_base)
        wrong_matches.append(m_wrong)

        vid = lbl.video_id
        cond_preds_by_clip[vid] = pred_cond
        labels_by_clip[vid] = lbl

        per_clip[vid] = {
            "conditioned": {
                "tp_fine": int(m_cond.tp_fine),
                "tp_coarse": int(m_cond.tp_coarse),
                "fp": int(m_cond.fp),
                "fn": int(m_cond.fn),
                "minutes": float(m_cond.minutes),
                "safety_tp": int(m_cond.safety_tp),
                "safety_total": int(m_cond.safety_total),
            },
            "baseline": {
                "tp_fine": int(m_base.tp_fine),
                "tp_coarse": int(m_base.tp_coarse),
                "fp": int(m_base.fp),
                "fn": int(m_base.fn),
                "minutes": float(m_base.minutes),
                "safety_tp": int(m_base.safety_tp),
                "safety_total": int(m_base.safety_total),
            },
            "wrong_env": {
                "tp_fine": int(m_wrong.tp_fine),
                "tp_coarse": int(m_wrong.tp_coarse),
                "fp": int(m_wrong.fp),
                "fn": int(m_wrong.fn),
                "minutes": float(m_wrong.minutes),
                "safety_tp": int(m_wrong.safety_tp),
                "safety_total": int(m_wrong.safety_total),
            },
        }

    # 6. Objective, limits, grid_f1, and diagnostics
    objective = f_beta(cond_matches, 0.5)
    baseline = f_beta(base_matches, 0.5)
    wrong_env = f_beta(wrong_matches, 0.5)

    total_safety_tp = sum(m.safety_tp for m in cond_matches)
    total_safety_total = sum(m.safety_total for m in cond_matches)
    safety_recall = (
        float(total_safety_tp / total_safety_total)
        if total_safety_total > 0
        else 1.0
    )

    total_fp = sum(m.fp for m in cond_matches)
    total_minutes = sum(m.minutes for m in cond_matches)
    fp_per_min = (
        float(total_fp / total_minutes)
        if total_minutes > 0.0
        else 0.0
    )

    gf1 = grid_f1(cond_preds_by_clip, labels_by_clip)
    diag = diagnostics(cond_matches, cfg)

    # 7. Limit checks vs current best
    if best is None:
        best = ledger.best()

    if best is None:
        limits = {
            "safety_recall": True,
            "fp_per_min": True,
            "above_baseline": bool(objective >= baseline),
            "wrong_env": True,
        }
    else:
        limits = {
            "safety_recall": bool(
                safety_recall >= float(best["safety_recall"]) - 1e-9
            ),
            "fp_per_min": bool(fp_per_min <= float(best["fp_per_min"]) + 1e-9),
            "above_baseline": bool(objective >= baseline - 1e-9),
            "wrong_env": bool(wrong_env >= float(best["wrong_env"]) - 1e-9),
        }

    passed = bool(all(limits.values()))

    # 8. Result assembly (run_id 12-char hex, score does not write to ledger)
    run_id = uuid.uuid4().hex[:12]
    stored_overrides = overrides if overrides is not None else {}

    return ScoreResult(
        run_id=run_id,
        objective=float(objective),
        baseline=float(baseline),
        wrong_env=float(wrong_env),
        safety_recall=float(safety_recall),
        fp_per_min=float(fp_per_min),
        grid_f1=float(gf1),
        per_clip=per_clip,
        diagnostics=diag,
        limits=limits,
        passed=passed,
        overrides=stored_overrides,
        split=split,
    )
