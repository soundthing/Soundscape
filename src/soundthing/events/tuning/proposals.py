"""Generate human-review proposals for shared configuration changes.

Shared configuration files (config/vocabulary.yaml and config/priors.yaml)
belong to the whole team / other roles and cannot be written automatically
by tuning scripts. This module drafts markdown proposals for review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soundthing.config import load_config
from soundthing.events.tuning import paths


def _format_diff_val(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return repr(round(val, 6))
    if isinstance(val, int):
        return str(val)
    return json.dumps(val)


def _get_field(res: Any, field_name: str, default: Any = None) -> Any:
    if res is None:
        return default
    if isinstance(res, dict):
        return res.get(field_name, default)
    return getattr(res, field_name, default)


def propose_shared_change(
    overrides: dict,
    result: Any,
    note: str,
    *,
    cfg: Any = None,
    out_dir: Path | None = None,
) -> Path:
    """Create a markdown proposal for shared configuration overrides.

    Args:
        overrides: Non-empty dict of overrides with keys starting with
            'vocabulary.' or 'priors.'.
        result: ScoreResult object, dict of score metrics, or None.
        note: Explanation of the change.
        cfg: Optional Config instance; defaults to load_config().
        out_dir: Directory where the proposal is written; defaults to
            paths.proposals_dir().

    Returns:
        Path to the created proposal markdown file.
    """
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("overrides must be a non-empty dict")

    for key in overrides:
        if not (key.startswith("vocabulary.") or key.startswith("priors.")):
            raise ValueError(
                f"Key {key!r} is not a shared configuration key (must start with 'vocabulary.' or 'priors.')"
            )

    if cfg is None:
        cfg = load_config()

    target_files: list[str] = []
    has_vocab = any(k.startswith("vocabulary.") for k in overrides)
    has_priors = any(k.startswith("priors.") for k in overrides)
    if has_vocab:
        target_files.append("config/vocabulary.yaml")
    if has_priors:
        target_files.append("config/priors.yaml")

    diff_lines: list[str] = []
    for key in sorted(overrides.keys()):
        parts = key.split(".")
        if parts[0] == "vocabulary":
            if len(parts) != 3 or parts[2] not in ("clap", "audioset"):
                raise ValueError(f"Invalid vocabulary key: {key!r}")
            cid = parts[1]
            try:
                sc = cfg.sound_class(cid)
            except KeyError:
                raise ValueError(f"Unknown sound class id: {cid!r}")
            curr_val = sc.clap if parts[2] == "clap" else sc.audioset
        elif parts[0] == "priors":
            if len(parts) == 2 and parts[1] == "floor":
                curr_val = cfg.priors.floor
            elif len(parts) == 3 and parts[2] == "default":
                env = parts[1]
                if env not in cfg.priors.environments:
                    raise ValueError(f"Unknown environment: {env!r}")
                curr_val = cfg.priors.environments[env].default
            elif len(parts) == 4 and parts[2] == "classes":
                env = parts[1]
                cid = parts[3]
                if env not in cfg.priors.environments:
                    raise ValueError(f"Unknown environment: {env!r}")
                if cid not in cfg.class_ids:
                    raise ValueError(f"Unknown sound class id: {cid!r}")
                curr_val = cfg.priors.environments[env].classes.get(cid, None)
            else:
                raise ValueError(f"Invalid priors key: {key!r}")
        else:
            raise ValueError(f"Unknown prefix: {key!r}")

        new_val = overrides[key]
        diff_lines.append(f"- {key}: {_format_diff_val(curr_val)}")
        diff_lines.append(f"+ {key}: {_format_diff_val(new_val)}")

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    date_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    run_id_val = _get_field(result, "run_id")
    run_id = str(run_id_val) if run_id_val else "norun"

    out_dir_path = Path(out_dir) if out_dir is not None else paths.proposals_dir()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    proposal_path = out_dir_path / f"{ts}-{run_id}.md"

    if result is None:
        scores_section = "not scored"
    else:
        objective = _get_field(result, "objective")
        baseline = _get_field(result, "baseline")
        wrong_env = _get_field(result, "wrong_env")
        safety_recall = _get_field(result, "safety_recall")
        fp_per_min = _get_field(result, "fp_per_min")
        grid_f1 = _get_field(result, "grid_f1")
        passed = _get_field(result, "passed")
        limits = _get_field(result, "limits")

        def _fmt(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        score_lines = [
            f"- Objective: {_fmt(objective)}",
            f"- Baseline: {_fmt(baseline)}",
            f"- Wrong env: {_fmt(wrong_env)}",
            f"- Safety recall: {_fmt(safety_recall)}",
            f"- FP per min: {_fmt(fp_per_min)}",
            f"- Grid F1: {_fmt(grid_f1)}",
            f"- Passed: {passed}",
        ]
        if limits is not None:
            if isinstance(limits, dict):
                score_lines.append("- Limits:")
                for l_k in sorted(limits.keys()):
                    score_lines.append(f"  - {l_k}: {limits[l_k]}")
            else:
                score_lines.append(f"- Limits: {limits}")
        scores_section = "\n".join(score_lines)

    diff_block = "\n".join(diff_lines)
    targets_str = ", ".join(target_files)

    content = f"""# Shared config proposal

- Date: {date_str}
- Run ID: {run_id}
- Target files: {targets_str}

## Note

{note}

## Change

```diff
{diff_block}
```

## Scores

{scores_section}

## Notice

This proposal modifies shared interface files (AGENTS.md §2). These files cannot be modified automatically by tuning scripts; a human must review and apply this change by hand after team approval.
"""

    with open(proposal_path, "w", encoding="utf-8") as f:
        f.write(content)

    return proposal_path
