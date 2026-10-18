"""Caption track writers — OWNER: Person 3.  list[CaptionEvent] -> .vtt / .srt"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import re
from pathlib import Path

from .contracts import CaptionEvent
from .events.hierarchy import suppress_ancestors


@dataclass
class Cue:
    start: float
    end: float
    text: str


def events_to_cues(events: list[CaptionEvent], min_cue: float = 1.0, max_cue: float = 6.0,
                   hierarchy: Mapping[str, frozenset[str]] | None = None) -> list[Cue]:
    """Clamp durations, then trim ancestor overlap when hierarchy is provided."""
    display = [replace(e, end=e.start + min(max(e.end - e.start, min_cue), max_cue)) for e in events]
    if hierarchy:
        display = suppress_ancestors(display, hierarchy)
    return [Cue(e.start, e.end, e.caption_text()) for e in sorted(display, key=lambda e: e.start)]


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


def write_captions(events: list[CaptionEvent], path: str | Path, min_cue: float = 1.0, max_cue: float = 6.0,
                   speech_subtitle_path: str | Path | None = None,
                   hierarchy: Mapping[str, frozenset[str]] | None = None) -> None:
    path = Path(path)
    cues = events_to_cues(events, min_cue, max_cue, hierarchy)
    if speech_subtitle_path is not None:
        cues = merge_with_speech(cues, speech_subtitle_path)
    text = to_srt(cues) if path.suffix.lower() == ".srt" else to_vtt(cues)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_timestamp(value: str) -> float:
    match = re.fullmatch(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})", value)
    if not match or int(match[2]) > 59 or int(match[3]) > 59:
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    return int(match[1] or 0) * 3600 + int(match[2]) * 60 + int(match[3]) + int(match[4]) / 1000


def read_subtitles(path: str | Path) -> list[Cue]:
    """Read UTF-8 SRT or VTT cue text and times; layout/style metadata is ignored."""
    path = Path(path)
    if path.suffix.lower() not in {".srt", ".vtt"}:
        raise ValueError("Subtitles must be an .srt or .vtt file")
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeError as exc:
        raise ValueError(f"{path.name}: subtitles must use UTF-8 encoding") from exc
    blocks = re.split(r"\n[ \t]*\n", text.strip())
    if path.suffix.lower() == ".vtt":
        if not re.fullmatch(r"WEBVTT(?:[ \t].*)?", blocks[0].split("\n")[0]):
            raise ValueError(f"{path.name}: missing WEBVTT header")
        if "-->" in blocks[0]:
            raise ValueError(f"{path.name}: separate the WEBVTT header and cues with a blank line")
        blocks = blocks[1:]
    cues = []
    for number, block in enumerate(blocks, 1):
        lines = block.strip().split("\n")
        if not block.strip() or (path.suffix.lower() == ".vtt" and re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0])):
            continue
        index = 0 if "-->" in lines[0] else 1
        timing = re.fullmatch(r"(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[index] if index < len(lines) else "")
        if not timing:
            raise ValueError(f"{path.name}: malformed timing in cue {number}")
        try:
            start, end = (_parse_timestamp(timing[i]) for i in (1, 2))
        except ValueError as exc:
            raise ValueError(f"{path.name}: malformed timestamp in cue {number}") from exc
        if end <= start:
            raise ValueError(f"{path.name}: cue {number} must end after it starts")
        caption = "\n".join(lines[index + 1:])
        if caption:
            cues.append(Cue(start, end, caption))
    return sorted(cues, key=lambda cue: (cue.start, cue.end))


def merge_with_speech(sound_cues: list[Cue], speech_subtitle_path: str | Path) -> list[Cue]:
    """Combine speech and sound text on one timeline without moving either.

    Split at every cue boundary so overlapping speech and sounds share one
    multiline cue. Speech comes first. Adjacent identical intervals coalesce.
    Resolution is milliseconds, matching SRT/VTT output. Inputs are not mutated.
    """
    speech = read_subtitles(speech_subtitle_path)
    changes: dict[int, list[tuple[int, str | None]]] = {}
    for index, cue in enumerate([*speech, *sorted(sound_cues, key=lambda cue: (cue.start, cue.end))]):
        if not math.isfinite(cue.start) or not math.isfinite(cue.end) or cue.start < 0 or cue.end <= cue.start:
            raise ValueError("Caption cues require finite, nonnegative starts and positive durations")
        start, end = round(cue.start * 1000), round(cue.end * 1000)
        if end <= start or not cue.text:
            continue
        changes.setdefault(start, []).append((index, cue.text))
        changes.setdefault(end, []).append((index, None))
    boundaries = sorted(changes)
    active: dict[int, str] = {}
    merged = []
    for start, end in zip(boundaries, boundaries[1:]):
        for index, text in changes[start]:
            if text is None:
                active.pop(index, None)
            else:
                active[index] = text
        # Identical simultaneous text appears once rather than on duplicate lines.
        text = "\n".join(dict.fromkeys(active[index] for index in sorted(active)))
        if not text:
            continue
        if merged and merged[-1].end == start / 1000 and merged[-1].text == text:
            merged[-1].end = end / 1000
        else:
            merged.append(Cue(start / 1000, end / 1000, text))
    return merged


def merge_subtitle_files(sound_path: str | Path, speech_path: str | Path, output_path: str | Path) -> None:
    """Merge existing caption files without extracting video or running models."""
    output = Path(output_path)
    if output.suffix.lower() not in {".srt", ".vtt"}:
        raise ValueError("Merged output must be an .srt or .vtt file")
    if output.resolve() in {Path(sound_path).resolve(), Path(speech_path).resolve()}:
        raise ValueError("Choose an output path different from both input subtitle files")
    cues = merge_with_speech(read_subtitles(sound_path), speech_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_srt(cues) if output.suffix.lower() == ".srt" else to_vtt(cues), encoding="utf-8")
