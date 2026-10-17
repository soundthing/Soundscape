"""Shot/scene segmentation (PySceneDetect)."""

from __future__ import annotations

import math

from ..config import Config


def detect_spans(video_path: str, duration: float, cfg: Config) -> list[tuple[float, float]]:
    """Return sorted, contiguous (start, end) spans covering [0, duration].

    - Detect shot boundaries with PySceneDetect (detector/threshold from
      pipeline.yaml environment.segmentation).
    - Merge spans shorter than min_segment_sec into a neighbour.
    - Split spans longer than max_segment_sec.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    settings = cfg.pipeline["environment"]["segmentation"]
    if settings["detector"] != "content":
        raise ValueError(f"unsupported scene detector {settings['detector']!r}")
    minimum = float(settings["min_segment_sec"])
    maximum = float(settings["max_segment_sec"])
    if duration <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid duration or segmentation length limits")

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=float(settings["threshold"]), min_scene_len=1))
    manager.detect_scenes(video)
    cuts = sorted({min(duration, max(0.0, end.get_seconds()))
                   for _, end in manager.get_scene_list()})
    edges = [0.0, *(cut for cut in cuts if 0.0 < cut < duration), float(duration)]

    # Remove the boundary next to a short span. Prefer the longer neighbour.
    while len(edges) > 2:
        short = next((i for i in range(len(edges) - 1)
                      if edges[i + 1] - edges[i] < minimum), None)
        if short is None:
            break
        if short == 0:
            del edges[1]
        elif short == len(edges) - 2:
            del edges[-2]
        elif edges[short] - edges[short - 1] >= edges[short + 2] - edges[short + 1]:
            del edges[short]
        else:
            del edges[short + 1]

    spans = []
    for start, end in zip(edges, edges[1:]):
        pieces = math.ceil((end - start) / maximum)
        for i in range(pieces):
            spans.append((start + (end - start) * i / pieces,
                          start + (end - start) * (i + 1) / pieces))
    return spans
