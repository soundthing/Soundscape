"""Component protocol: what the harness core needs from each tunable stage.

A component scores one stage (events, environment, captions, pipeline). The core
(`score`, `sweep`, `bootstrap`, `promote`, `ledger`, `runner`, MCP) is component-agnostic
and talks to stages only through this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from soundthing.config import Config
from soundthing.contracts import ClipLabels

# Owner of each override-key prefix. Promotion writes config only for keys owned by the
# current role (see `harness_role`); other keys become proposals for their owner.
SECTION_OWNERS: dict[str, str] = {
    "events.": "person1",
    "thresholds.": "person1",
    "environment.": "person2",
    "captions.": "person3",
    "extract.": "person3",
}


def key_owner(key: str) -> str | None:
    """Role that owns an override key, or None for shared/unknown keys."""
    for prefix, owner in SECTION_OWNERS.items():
        if key.startswith(prefix):
            return owner
    return None


@dataclass
class Clip:
    """One scored clip: its labels plus whatever the component cached for it."""

    video_id: str
    labels: ClipLabels
    cache: Any


@dataclass
class Evaluation:
    """What `Component.evaluate` returns.

    `metrics` is flat scalars and always contains "objective". `per_clip[video_id]` is
    JSON-safe and always contains "stats": list[float] of length `Component.n_stats`;
    summing stats over clips and passing them to `stat_objective` must reproduce
    `objective` (this is what the paired bootstrap resamples).
    """

    objective: float
    metrics: dict[str, float]
    per_clip: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Component(Protocol):
    name: str  # "events" | "environment" | "captions" | "pipeline"
    owner: str  # "person1" | "person2" | "person3"
    sections: tuple[str, ...]  # override-key prefixes it may tune, e.g. ("events.", "thresholds.")
    code_paths: tuple[str, ...]  # repo-relative code an experiment may change
    tuning_log: str  # repo-relative markdown file promotion appends to
    n_stats: int  # length of per_clip[...]["stats"]

    def cache_key(self, cfg: Config) -> str | None:
        """Hash of the settings the slow cached stage depends on; None if uncached."""

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path | None:
        """Run the slow model stage for one clip and save it; None if uncached."""

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        """Load labels (sorted by video_id) and caches. Raise FileNotFoundError naming
        the clip and cache key when a needed cache is missing. Skip clips the component
        cannot score (e.g. environment: clips without env segments)."""

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        """Replay the fast stage under cfg and score it with the fixed objective."""

    def limits(self, metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
        """Limit checks vs the current best ledger entry (None: no best yet)."""

    def space(self, cfg: Config, labels_dir: Path | None) -> list[dict[str, Any]]:
        """Tunable parameters, same dict shape as harness.space.describe_space."""

    def stat_objective(self, totals: np.ndarray) -> np.ndarray:
        """totals: (B, n_stats) summed per-clip stats -> (B,) objective values."""
