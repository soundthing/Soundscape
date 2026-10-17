"""Caption track writers — OWNER: Person 3.  list[CaptionEvent] -> .vtt / .srt"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import CaptionEvent


@dataclass
class Cue:
    start: float
    end: float
    text: str


def events_to_cues(events: list[CaptionEvent], min_cue: float = 1.0, max_cue: float = 6.0) -> list[Cue]:
    """Clamp cue durations to [min_cue, max_cue] for readability."""
    cues = []
    for e in sorted(events, key=lambda e: e.start):
        dur = min(max(e.end - e.start, min_cue), max_cue)
        cues.append(Cue(e.start, e.start + dur, e.caption_text()))
    return cues


def _ts(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_vtt(cues: list[Cue]) -> str:
    lines = ["WEBVTT", ""]
    for c in cues:
        lines += [f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}", c.text, ""]
    return "\n".join(lines)


def to_srt(cues: list[Cue]) -> str:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{_ts(c.start, ',')} --> {_ts(c.end, ',')}", c.text, ""]
    return "\n".join(lines)


def write_captions(events: list[CaptionEvent], path: str | Path, min_cue: float = 1.0, max_cue: float = 6.0) -> None:
    path = Path(path)
    cues = events_to_cues(events, min_cue, max_cue)
    text = to_srt(cues) if path.suffix.lower() == ".srt" else to_vtt(cues)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def merge_with_speech(sound_cues: list[Cue], speech_subtitle_path: str | Path) -> list[Cue]:
    """Parse an existing .srt/.vtt speech track and interleave sound cues with it
    (sound cues on their own line; avoid overlapping text where possible)."""
    raise NotImplementedError("TODO(Person 3): speech subtitle merge")
