"""Shared data contracts between pipeline stages.

These are the ONLY types that cross ownership boundaries. Change them only with
agreement from everyone (see docs/interfaces.md). Each stage reads/writes these
as JSON files so any stage can be run, tested, or faked in isolation.

    Person 3 (extract)      -> ExtractedInputs   -> everyone        inputs.json
    Person 2 (environment)  -> list[EnvSegment]  -> Person 1        env_segments.json
    Person 1 (events)       -> list[CaptionEvent]-> Person 3        events.json
    everyone (labeling)     -> ClipLabels        -> Person 3 (eval) labels/<video_id>.json
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar

# Version 2 retires the five unmapped event classes and requires PANNs-only runs.
# Optional prior coefficient formats affect configuration, not these JSON payloads.
SCHEMA_VERSION = 2

# Fallback environment used when the classifier is not confident. It must also
# exist in config/environments.yaml.
GENERAL_ENV = "general"

# Allowed caption modifiers. Rendered in this order, e.g. "[door slams, right]".
MODIFIERS = (
    "left",
    "right",
    "distant",
    "nearby",
    "approaching",
    "receding",
    "loud",
    "quiet",
    "continues",
)


class ContractError(ValueError):
    """Raised when data does not satisfy a shared contract."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _check_span(start: float, end: float, what: str) -> None:
    _check(math.isfinite(start) and math.isfinite(end), f"{what}: non-finite time")
    _check(start >= 0, f"{what}: start {start} < 0")
    _check(end > start, f"{what}: end {end} must be > start {start}")


def _check_unit(x: float, what: str) -> None:
    _check(0.0 <= x <= 1.0, f"{what}: {x} not in [0, 1]")


# ---------------------------------------------------------------------------
# Person 3 -> everyone
# ---------------------------------------------------------------------------


@dataclass
class ExtractedInputs:
    """Output of ffmpeg extraction. Paths are relative to the JSON file's directory
    when serialized, absolute after `load()`."""

    video_id: str
    video_path: str
    audio_path: str  # 16 kHz mono PCM WAV — model input
    frames_dir: str  # JPEGs named frame_000000.jpg ...; frame i is at t = i / fps
    fps: float  # frame sampling rate (~1)
    duration: float  # seconds
    sample_rate: int = 16000
    audio_stereo_path: str | None = None  # original-rate stereo WAV, for left/right modifiers

    def validate(self) -> None:
        _check(bool(self.video_id), "video_id is empty")
        _check(self.fps > 0, f"fps {self.fps} must be > 0")
        _check(self.duration > 0, f"duration {self.duration} must be > 0")
        _check(self.sample_rate > 0, "sample_rate must be > 0")

    def frame_time(self, index: int) -> float:
        return index / self.fps

    def frame_paths(self) -> list[Path]:
        return sorted(Path(self.frames_dir).glob("frame_*.jpg"))


# ---------------------------------------------------------------------------
# Person 2 -> Person 1
# ---------------------------------------------------------------------------


@dataclass
class EnvSegment:
    """Environment of one contiguous span of the video.

    Segments returned by the environment classifier must be sorted, non-overlapping
    and cover [0, duration] without gaps (see `validate_segments`).
    """

    start: float
    end: float
    env: str  # argmax of probs, or GENERAL_ENV when confidence is low
    confidence: float  # probs[env] before any fallback; in [0, 1]
    probs: dict[str, float]  # distribution over environment ids; sums to ~1

    def validate(self) -> None:
        _check_span(self.start, self.end, "EnvSegment")
        _check_unit(self.confidence, "EnvSegment.confidence")
        _check(bool(self.probs), "EnvSegment.probs is empty")
        for k, v in self.probs.items():
            _check_unit(v, f"EnvSegment.probs[{k}]")
        total = sum(self.probs.values())
        _check(abs(total - 1.0) < 1e-3, f"EnvSegment.probs sums to {total}, expected 1")


def validate_segments(segments: list[EnvSegment], duration: float, tol: float = 0.05) -> None:
    _check(len(segments) > 0, "no environment segments")
    for s in segments:
        s.validate()
    _check(abs(segments[0].start) <= tol, "segments must start at 0")
    _check(abs(segments[-1].end - duration) <= tol, f"segments must end at duration {duration}")
    for a, b in zip(segments, segments[1:]):
        _check(abs(a.end - b.start) <= tol, f"gap/overlap between segments at {a.end} / {b.start}")


def env_probs_at(segments: list[EnvSegment], t: float) -> dict[str, float]:
    """Environment distribution at time t (nearest segment if t is out of range)."""
    for s in segments:
        if s.start <= t < s.end:
            return s.probs
    return segments[-1].probs if t >= segments[-1].end else segments[0].probs


# ---------------------------------------------------------------------------
# Person 1 -> Person 3
# ---------------------------------------------------------------------------


@dataclass
class CaptionEvent:
    """Final event; ids may be curated or AudioSet-based. Confidence is a score, not calibrated probability."""
    """A final, display-ready non-speech caption."""

    start: float
    end: float
    label: str  # caption text without brackets, e.g. "door slams"
    class_id: str  # vocabulary id (or coarse group id) e.g. "door_slam" / "impact"; used by eval
    confidence: float
    modifiers: list[str] = field(default_factory=list)  # subset of MODIFIERS

    def validate(self) -> None:
        _check_span(self.start, self.end, f"CaptionEvent({self.label})")
        _check_unit(self.confidence, "CaptionEvent.confidence")
        _check(bool(self.label), "CaptionEvent.label is empty")
        _check(bool(self.class_id), "CaptionEvent.class_id is empty")
        for m in self.modifiers:
            _check(m in MODIFIERS, f"unknown modifier {m!r}; allowed: {MODIFIERS}")

    def caption_text(self) -> str:
        mods = [m for m in MODIFIERS if m in self.modifiers]
        return "[" + ", ".join([self.label, *mods]) + "]"


# ---------------------------------------------------------------------------
# Hand labels (everyone -> Person 3)
# ---------------------------------------------------------------------------


@dataclass
class LabeledSegment:
    start: float
    end: float
    env: str


@dataclass
class LabeledEvent:
    start: float
    end: float
    class_id: str  # fine-grained vocabulary id
    note: str = ""


@dataclass
class ClipLabels:
    """Ground truth for one evaluation clip: labels/<video_id>.json"""

    video_id: str
    duration: float
    labeler: str
    segments: list[LabeledSegment]
    events: list[LabeledEvent]

    def validate(self) -> None:
        _check(self.duration > 0, "duration must be > 0")
        for s in self.segments:
            _check_span(s.start, s.end, "LabeledSegment")
        for e in self.events:
            _check_span(e.start, e.end, f"LabeledEvent({e.class_id})")


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

@dataclass
class DirectionFrame:
    """Soundtrack channel strengths at a video time, independent of caption events."""

    time: float
    levels: dict[str, float]

    def validate(self) -> None:
        _check(math.isfinite(self.time) and self.time >= 0, "DirectionFrame: invalid time")
        for channel, level in self.levels.items():
            _check(channel in {"FL", "FR", "FC", "SL", "SR"}, "DirectionFrame: unknown channel")
            _check(math.isfinite(level), "DirectionFrame: non-finite level")
            _check_unit(level, f"DirectionFrame.levels[{channel}]")


@dataclass
class DirectionTrack:
    """Stereo/5.1 speaker-feed activity, not physical source localization.

    LFE is deliberately excluded. Back and side 5.1 surround feeds are named SL/SR.
    Frames use media timestamps; levels map -60..0 dBFS to 0..1 with smoothing.
    """

    duration: float
    mode: str
    channels: list[str]
    frames: list[DirectionFrame]

    def validate(self) -> None:
        _check(math.isfinite(self.duration) and self.duration >= 0, "DirectionTrack: invalid duration")
        layouts = {"stereo": ["FL", "FR"], "surround51": ["FL", "FR", "FC", "SL", "SR"], "unavailable": []}
        _check(self.mode in layouts, "DirectionTrack: invalid mode")
        _check(self.channels == layouts[self.mode], "DirectionTrack: invalid channels for mode")
        if self.mode == "unavailable":
            _check(not self.frames, "DirectionTrack: unavailable track has frames")
            return
        _check(bool(self.frames) and self.frames[0].time == 0, "DirectionTrack: frames must start at zero")
        previous = -1.0
        for frame in self.frames:
            frame.validate()
            _check(previous < frame.time < self.duration, "DirectionTrack: frame outside duration or unordered")
            _check(set(frame.levels) == set(self.channels), "DirectionTrack: channel keys do not match")
            previous = frame.time

T = TypeVar("T")


def _envelope(kind: str, data: Any) -> dict:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "data": data}


def _unwrap(obj: Any, kind: str) -> Any:
    # Accept both the envelope and the bare payload (handy for hand-written fakes).
    if isinstance(obj, dict) and "kind" in obj and "data" in obj:
        _check(obj["kind"] == kind, f"expected kind {kind!r}, got {obj['kind']!r}")
        _check(
            obj.get("schema_version") == SCHEMA_VERSION,
            f"schema_version {obj.get('schema_version')} != {SCHEMA_VERSION}",
        )
        return obj["data"]
    return obj


def _write(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_inputs(inputs: ExtractedInputs, path: Path) -> None:
    inputs.validate()
    base = Path(path).resolve().parent
    d = asdict(inputs)
    for key in ("video_path", "audio_path", "frames_dir", "audio_stereo_path"):
        if d[key] is not None:
            p = Path(d[key]).resolve()
            try:
                d[key] = p.relative_to(base).as_posix()
            except ValueError:
                d[key] = p.as_posix()
    _write(path, _envelope("extracted_inputs", d))


def load_inputs(path: Path) -> ExtractedInputs:
    base = Path(path).resolve().parent
    d = _unwrap(_read(path), "extracted_inputs")
    for key in ("video_path", "audio_path", "frames_dir", "audio_stereo_path"):
        if d.get(key) is not None:
            d[key] = str((base / d[key]).resolve())
    inputs = ExtractedInputs(**d)
    inputs.validate()
    return inputs


def save_segments(segments: list[EnvSegment], path: Path) -> None:
    for s in segments:
        s.validate()
    _write(path, _envelope("env_segments", [asdict(s) for s in segments]))


def load_segments(path: Path) -> list[EnvSegment]:
    segs = [EnvSegment(**d) for d in _unwrap(_read(path), "env_segments")]
    for s in segs:
        s.validate()
    return segs


def save_events(events: Iterable[CaptionEvent], path: Path) -> None:
    events = sorted(events, key=lambda e: (e.start, e.end))
    for e in events:
        e.validate()
    _write(path, _envelope("caption_events", [asdict(e) for e in events]))


def load_events(path: Path) -> list[CaptionEvent]:
    evs = [CaptionEvent(**d) for d in _unwrap(_read(path), "caption_events")]
    for e in evs:
        e.validate()
    return evs


def save_labels(labels: ClipLabels, path: Path) -> None:
    labels.validate()
    _write(path, _envelope("clip_labels", asdict(labels)))


def load_labels(path: Path) -> ClipLabels:
    d = _unwrap(_read(path), "clip_labels")
    labels = ClipLabels(
        video_id=d["video_id"],
        duration=d["duration"],
        labeler=d.get("labeler", ""),
        segments=[LabeledSegment(**s) for s in d.get("segments", [])],
        events=[LabeledEvent(**e) for e in d.get("events", [])],
    )
    labels.validate()
    return labels


def save_direction(track: DirectionTrack, path: Path) -> None:
    track.validate()
    _write(path, _envelope("direction", asdict(track)))


def load_direction(path: Path) -> DirectionTrack:
    data = _unwrap(_read(path), "direction")
    track = DirectionTrack(duration=data["duration"], mode=data["mode"],
                           channels=data["channels"],
                           frames=[DirectionFrame(**frame) for frame in data["frames"]])
    track.validate()
    return track
