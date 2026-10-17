"""Search space description and overrides validation for events tuning.

Defines module-level parameter bounds, describes the full tunable space from
config and dev event counts, and validates proposed overrides against the space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soundthing.config import Config
from soundthing.contracts import load_labels
from soundthing.events.tuning import overrides, paths


MIN_CLASS_EVENTS: int = 20

THRESHOLD_BOUNDS: tuple[float, float] = (0.05, 0.95)


@dataclass(frozen=True)
class ParamSpec:
    type: str
    lo: float | int
    hi: float | int
    default: float | int | None = None


BOUNDS: dict[str, ParamSpec] = {
    "events.clap.calibration.center": ParamSpec("float", 0.0, 0.6),
    "events.clap.calibration.scale": ParamSpec("float", 2.0, 60.0),
    "events.vad.suppress_under_speech": ParamSpec("float", 0.0, 1.0),
    "events.vad.threshold": ParamSpec("float", 0.2, 0.9),
    "events.vad.min_speech_sec": ParamSpec("float", 0.05, 1.0),
    "events.vad.pad_sec": ParamSpec("float", 0.0, 0.5),
    "events.vad.merge_gap_sec": ParamSpec("float", 0.0, 1.0),
    "events.postprocess.smoothing_sec": ParamSpec("float", 0.0, 2.0),
    "events.postprocess.min_event_sec": ParamSpec("float", 0.05, 1.5),
    "events.postprocess.merge_gap_sec": ParamSpec("float", 0.0, 2.0),
    "events.postprocess.coarse_margin": ParamSpec("float", 0.0, 0.3),
    "events.postprocess.sustained_repeat_sec": ParamSpec("float", 10.0, 120.0),
    "events.postprocess.max_on_screen": ParamSpec("int", 1, 4),
    "events.postprocess.onset_shift_sec": ParamSpec("float", -0.5, 0.5, default=0.0),
    "thresholds.default": ParamSpec("float", 0.05, 0.95),
    "thresholds.safety_default": ParamSpec("float", 0.05, 0.95),
}


def _get_nested(data: Any, path: list[str]) -> Any:
    cur = data
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def describe_space(cfg: Config, dev_event_counts: dict[str, int] | None) -> list[dict[str, Any]]:
    """Describe the tuning search space for the given Config and dev event counts.

    Entries:
    1. BOUNDS entries: non-class-specific tunables from pipeline.yaml and thresholds.yaml.
    2. Per-class thresholds: for classes with at least MIN_CLASS_EVENTS dev events.
    3. Shared entries: vocabulary and priors (proposal/scoring only).
    """
    entries: list[dict[str, Any]] = []

    # 1. Non-class-specific entries from BOUNDS table
    for name, spec in BOUNDS.items():
        val: Any = None
        if name == "thresholds.default":
            val = cfg.thresholds.default
        elif name == "thresholds.safety_default":
            val = cfg.thresholds.safety_default
        elif name.startswith("events."):
            val = _get_nested(cfg.pipeline.get("events") or {}, name[len("events."):].split("."))

        if val is None:
            if spec.default is not None:
                val = spec.default
            else:
                continue

        if spec.type == "int":
            current_val = int(val)
        elif spec.type == "float":
            current_val = float(val)
        else:
            current_val = val

        entries.append({
            "name": name,
            "type": spec.type,
            "bounds": [spec.lo, spec.hi],
            "current": current_val,
            "shared": False,
        })

    # 2. Per-class thresholds
    if dev_event_counts is not None:
        for cid in cfg.class_ids:
            if dev_event_counts.get(cid, 0) >= MIN_CLASS_EVENTS:
                entries.append({
                    "name": f"thresholds.classes.{cid}",
                    "type": "float",
                    "bounds": [THRESHOLD_BOUNDS[0], THRESHOLD_BOUNDS[1]],
                    "current": float(cfg.threshold(cid)),
                    "shared": False,
                })

    # 3. Shared entries
    for cid in cfg.class_ids:
        sc = cfg.sound_class(cid)
        entries.append({
            "name": f"vocabulary.{cid}.clap",
            "type": "list[str]",
            "bounds": None,
            "current": list(sc.clap),
            "shared": True,
        })
        entries.append({
            "name": f"vocabulary.{cid}.audioset",
            "type": "list[str]",
            "bounds": None,
            "current": list(sc.audioset),
            "shared": True,
        })

    entries.append({
        "name": "priors.floor",
        "type": "float",
        "bounds": [0.0, 1.0],
        "current": float(cfg.priors.floor),
        "shared": True,
    })

    for env_name, env_priors in cfg.priors.environments.items():
        entries.append({
            "name": f"priors.{env_name}.default",
            "type": "float",
            "bounds": [0.0, 1.0],
            "current": float(env_priors.default),
            "shared": True,
        })
        for cid, weight in env_priors.classes.items():
            entries.append({
                "name": f"priors.{env_name}.classes.{cid}",
                "type": "float",
                "bounds": [0.0, 1.0],
                "current": float(weight),
                "shared": True,
            })

    return entries


def space_index(space: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map space entry name to entry dictionary."""
    return {entry["name"]: entry for entry in space}


def validate_overrides(overrides: dict[str, Any], space: list[dict[str, Any]]) -> None:
    """Validate override parameters against the search space.

    Raises ValueError if a key is not in the space, if numeric values are out of
    bounds or not numeric, if int values are non-integral, or if list[str] values
    are not lists of strings.
    """
    idx = space_index(space)
    for key, val in overrides.items():
        if key not in idx:
            if key.startswith("thresholds.classes."):
                raise ValueError(
                    f"Parameter {key!r} not in search space. "
                    f"Per-class thresholds require at least MIN_CLASS_EVENTS ({MIN_CLASS_EVENTS}) dev events."
                )
            raise ValueError(f"Unknown parameter {key!r} not in search space.")

        entry = idx[key]
        expected_type = entry["type"]
        bounds = entry["bounds"]

        if isinstance(val, bool):
            raise ValueError(f"Parameter {key!r} received boolean value {val!r}, expected {expected_type}.")

        if expected_type == "float":
            if not isinstance(val, (int, float)):
                raise ValueError(f"Parameter {key!r} expected float, got {type(val).__name__}: {val!r}")
            val_num = float(val)
            if bounds is not None:
                lo, hi = bounds
                if not (lo <= val_num <= hi):
                    raise ValueError(f"Parameter {key!r} value {val} outside bounds [{lo}, {hi}].")

        elif expected_type == "int":
            if not isinstance(val, (int, float)):
                raise ValueError(f"Parameter {key!r} expected int, got {type(val).__name__}: {val!r}")
            if isinstance(val, float) and not val.is_integer():
                raise ValueError(f"Parameter {key!r} expected int, got non-integral value {val!r}")
            val_int = int(val)
            if bounds is not None:
                lo, hi = bounds
                if not (lo <= val_int <= hi):
                    raise ValueError(f"Parameter {key!r} value {val} outside bounds [{lo}, {hi}].")

        elif expected_type == "list[str]":
            if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                raise ValueError(f"Parameter {key!r} expected list[str], got {val!r}")
        else:
            raise ValueError(f"Unknown spec type {expected_type!r} for parameter {key!r}")


def is_shared(key: str) -> bool:
    """True for keys starting with 'vocabulary.' or 'priors.'."""
    return overrides.touches_shared({key: None})



def dev_event_counts(labels_dir: Path | None = None) -> dict[str, int]:
    """Count events per class_id over all *.json in labels_dir.

    Default: paths.dev_labels_dir(). Missing directory returns {}.
    """
    if labels_dir is None:
        labels_dir = paths.dev_labels_dir()
    labels_path = Path(labels_dir)
    if not labels_path.is_dir():
        return {}
    counts: dict[str, int] = {}
    for path in sorted(labels_path.glob("*.json")):
        labels = load_labels(path)
        for ev in labels.events:
            counts[ev.class_id] = counts.get(ev.class_id, 0) + 1
    return counts
