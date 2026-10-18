"""Score stage pipelines against dev or holdout ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import uuid

from soundthing.config import Config, load_config
from soundthing.harness import components, ledger, paths
from soundthing.harness.components.events import labeled_env_segments
from soundthing.harness.overrides import apply_overrides


@dataclass
class ScoreResult:
    run_id: str
    component: str
    objective: float
    metrics: dict[str, float]
    per_clip: dict[str, dict]
    diagnostics: dict
    limits: dict[str, bool]
    passed: bool
    overrides: dict
    split: str

    def __getattr__(self, name: str) -> Any:
        if name != "metrics" and hasattr(self, "metrics") and isinstance(self.metrics, dict) and name in self.metrics:
            return self.metrics[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.metrics:
            for k, v in self.metrics.items():
                d[k] = v
        return d


def score(
    overrides: dict[str, Any] | None = None,
    split: str = "dev",
    cfg: Config | None = None,
    best: dict[str, Any] | None = None,
    labels_dir: Path | str | None = None,
    component: str = "events",
) -> ScoreResult:
    """Replay clips under candidate overrides for the requested component."""
    comp = components.get(component)
    stored_overrides = dict(overrides or {})

    # Check override keys start with one of comp.sections
    for k in stored_overrides:
        if not any(k.startswith(sec) for sec in comp.sections):
            raise ValueError(
                f"Override key {k!r} not in component {component!r} sections: {comp.sections}"
            )

    # 1. Config resolution and overrides
    if cfg is None:
        cfg = load_config(Path.cwd() / "config")
    if stored_overrides:
        cfg = apply_overrides(cfg, stored_overrides)

    # 2. Labels directory resolution
    if labels_dir is not None:
        ldir = Path(labels_dir)
    elif split == "dev":
        ldir = paths.dev_labels_dir()
    elif split == "holdout":
        ldir = paths.holdout_dir() / "labels"
    else:
        raise ValueError(f"Unknown split: {split!r}")

    # 3. Component clip loading and evaluation
    clips = comp.load_clips(cfg, ldir)
    evaluation = comp.evaluate(clips, cfg)

    # 4. Limit checks vs current best
    if best is None:
        best = ledger.best(component)

    limits = comp.limits(evaluation.metrics, best)
    passed = bool(all(limits.values()))

    run_id = uuid.uuid4().hex[:12]

    return ScoreResult(
        run_id=run_id,
        component=component,
        objective=float(evaluation.objective),
        metrics=dict(evaluation.metrics),
        per_clip=evaluation.per_clip,
        diagnostics=evaluation.diagnostics,
        limits=limits,
        passed=passed,
        overrides=stored_overrides,
        split=split,
    )


__all__ = ["ScoreResult", "labeled_env_segments", "score"]
