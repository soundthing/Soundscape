"""Remove redundant ancestor labels only for intervals covered by descendants."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..contracts import CaptionEvent


def suppress_ancestors(events: Sequence[CaptionEvent], ancestors: Mapping[str, frozenset[str]]) -> list[CaptionEvent]:
    """Input events already passed their class thresholds and duration checks.

    Descendants need no qualifying parent. Siblings/unrelated sources survive.
    Unknown ids (e.g. old external classes) have no ancestry. No input is mutated.
    """
    by_ancestor: dict[str, list[tuple[float, float]]] = {}
    for event in events:
        for parent in ancestors.get(event.class_id, ()):
            by_ancestor.setdefault(parent, []).append((event.start, event.end))
    # Union overlapping descendant spans before subtracting them from parents.
    blockers: dict[str, list[tuple[float, float]]] = {}
    for parent, spans in by_ancestor.items():
        merged: list[tuple[float, float]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(merged[-1][1], end)
            else:
                merged.append((start, end))
        blockers[parent] = merged
    output = []
    for event in events:
        start = event.start
        for low, high in blockers.get(event.class_id, ()):
            if high <= start:
                continue
            if low >= event.end:
                break
            if low > start:
                output.append(replace(event, start=start, end=min(low, event.end)))
            start = max(start, high)
            if start >= event.end:
                break
        if start < event.end:
            output.append(replace(event, start=start))
    return sorted(output, key=lambda e: (e.start, e.end, e.class_id))
