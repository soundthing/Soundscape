"""Shot/scene segmentation (PySceneDetect)."""

from __future__ import annotations

from ..config import Config


def detect_spans(video_path: str, duration: float, cfg: Config) -> list[tuple[float, float]]:
    """Return sorted, contiguous (start, end) spans covering [0, duration].

    - Detect shot boundaries with PySceneDetect (detector/threshold from
      pipeline.yaml environment.segmentation).
    - Merge spans shorter than min_segment_sec into a neighbour.
    - Split spans longer than max_segment_sec.
    """
    raise NotImplementedError("TODO(Person 2): PySceneDetect segmentation")
