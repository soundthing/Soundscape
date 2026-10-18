"""Config overrides for the tuning harness."""

from __future__ import annotations

import copy
from typing import Any

from soundthing.config import Config


def touches_shared(overrides: dict[str, Any]) -> bool:
    """Return True if any key starts with 'vocabulary.' or 'priors.'."""
    return any(k.startswith("vocabulary.") or k.startswith("priors.") for k in overrides)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted key-value overrides to a deep copy of cfg and validate.

    Dotted keys:
    - events.<a>.<b>... -> cfg.pipeline['events'][a][b]...
    - thresholds.default, thresholds.safety_default -> float
    - thresholds.classes.<id> -> float (<id> in class_ids or coarse_groups)
    - vocabulary.<class_id>.clap / vocabulary.<class_id>.audioset -> list[str]
    - priors.floor -> float
    - priors.<env>.default -> float
    - priors.<env>.classes.<class_id> -> float
    """
    new_cfg = copy.deepcopy(cfg)
    class_ids_set = set(new_cfg.class_ids)
    env_ids_set = set(new_cfg.env_ids)

    for key, val in overrides.items():
        if not isinstance(key, str):
            raise KeyError(f"Override key must be a string: {key!r}")
        parts = key.split(".")

        section = parts[0]
        if section == "events":
            if len(parts) < 2:
                raise KeyError(f"events key path too short: {key!r}")
            if "events" not in new_cfg.pipeline or not isinstance(new_cfg.pipeline["events"], dict):
                raise KeyError(f"pipeline has no 'events' section: {key!r}")
            cur = new_cfg.pipeline["events"]
            for p in parts[1:-1]:
                if not isinstance(cur, dict) or p not in cur:
                    raise KeyError(f"Path does not exist in pipeline['events']: {key!r}")
                cur = cur[p]
            leaf = parts[-1]
            if not isinstance(cur, dict) or leaf not in cur:
                raise KeyError(f"Leaf key does not exist in pipeline['events']: {key!r}")
            cur[leaf] = val

        elif section == "thresholds":
            if key == "thresholds.default":
                new_cfg.thresholds.default = float(val)
            elif key == "thresholds.safety_default":
                new_cfg.thresholds.safety_default = float(val)
            elif len(parts) == 3 and parts[1] == "classes":
                cid = parts[2]
                if cid not in class_ids_set and cid not in new_cfg.coarse_groups:
                    raise KeyError(f"Unknown class or coarse group {cid!r} in key {key!r}")
                new_cfg.thresholds.classes[cid] = float(val)
            else:
                raise KeyError(f"Unknown thresholds key: {key!r}")

        elif section == "vocabulary":
            if len(parts) == 3 and parts[2] in ("clap", "audioset"):
                cid = parts[1]
                field_name = parts[2]
                if cid not in class_ids_set:
                    raise KeyError(f"Unknown class_id {cid!r} in key {key!r}")
                sc = new_cfg.sound_class(cid)
                if not isinstance(val, list):
                    raise TypeError(f"Expected list[str] for {key!r}, got {type(val)}")
                setattr(sc, field_name, list(val))
            else:
                raise KeyError(f"Unknown vocabulary key: {key!r}")

        elif section == "priors":
            if key == "priors.floor":
                new_cfg.priors.floor = float(val)
            elif len(parts) == 3 and parts[2] == "default":
                env = parts[1]
                if env not in env_ids_set or env not in new_cfg.priors.environments:
                    raise KeyError(f"Unknown environment {env!r} in key {key!r}")
                new_cfg.priors.environments[env].default = float(val)
            elif len(parts) == 4 and parts[2] == "classes":
                env = parts[1]
                cid = parts[3]
                if env not in env_ids_set or env not in new_cfg.priors.environments:
                    raise KeyError(f"Unknown environment {env!r} in key {key!r}")
                if cid not in class_ids_set:
                    raise KeyError(f"Unknown class_id {cid!r} in key {key!r}")
                new_cfg.priors.environments[env].classes[cid] = float(val)
            else:
                raise KeyError(f"Unknown priors key: {key!r}")

        elif section in ("environment", "captions", "extract"):
            if len(parts) < 2:
                raise KeyError(f"{section} key path too short: {key!r}")
            if section not in new_cfg.pipeline or not isinstance(new_cfg.pipeline[section], dict):
                new_cfg.pipeline[section] = {}
            cur = new_cfg.pipeline[section]
            for p in parts[1:-1]:
                if not isinstance(cur, dict):
                    raise KeyError(f"Path does not exist in pipeline[{section!r}]: {key!r}")
                if p not in cur:
                    cur[p] = {}
                cur = cur[p]
            leaf = parts[-1]
            cur[leaf] = val

        else:
            raise KeyError(f"Unknown top-level section in key: {key!r}")

    new_cfg.validate()
    return new_cfg
