"""Stage interfaces. Each owner implements one of these; the pipeline only calls
through them. `backend` in config/pipeline.yaml selects stub vs real.
"""

from __future__ import annotations

from typing import Protocol

from .config import Config
from .contracts import CaptionEvent, EnvSegment, ExtractedInputs


class EnvironmentClassifier(Protocol):
    """Person 2. Must return segments satisfying contracts.validate_segments."""

    def __call__(self, inputs: ExtractedInputs, cfg: Config) -> list[EnvSegment]: ...


class EventDetector(Protocol):
    """Person 1. `segments=None` means the unconditioned audio-only baseline."""

    def __call__(
        self, inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg: Config
    ) -> list[CaptionEvent]: ...


def get_environment_classifier(cfg: Config) -> EnvironmentClassifier:
    backend = cfg.pipeline["environment"]["backend"]
    if backend == "stub":
        from .environment.stub import classify_stub

        return classify_stub
    if backend == "real":
        from .environment.classifier import classify

        return classify
    raise ValueError(f"unknown environment backend {backend!r}")


def get_event_detector(cfg: Config) -> EventDetector:
    backend = cfg.pipeline["events"]["backend"]
    if backend == "stub":
        from .events.stub import detect_stub

        return detect_stub
    if backend == "real":
        from .events.detector import detect

        return detect
    raise ValueError(f"unknown events backend {backend!r}")
