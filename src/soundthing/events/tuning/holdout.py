"""Holdout evaluation split stored outside git tracking.

The holdout dataset is stored outside repository tracking so tuning agents
cannot inspect label contents or overfit to it, and is evaluated strictly on promotion.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Iterable

from soundthing.config import Config, load_config
from soundthing.contracts import ClipLabels, load_labels, save_labels
from soundthing.events.tuning import paths

AUDIO_EXTENSIONS: tuple[str, ...] = (
    "wav",
    "flac",
    "mp3",
    "m4a",
    "mp4",
    "mkv",
    "webm",
)


def holdout_labels_dir() -> Path:
    """Directory where holdout label JSON files are stored."""
    return paths.holdout_dir() / "labels"


def holdout_audio_dir() -> Path:
    """Directory where holdout audio files are stored."""
    return paths.holdout_dir() / "audio"


def holdout_ids() -> list[str]:
    """Return sorted stems of *.json in holdout_labels_dir(), or [] if missing."""
    d = holdout_labels_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if p.is_file())


def dev_ids(labels_dir: Path | None = None) -> list[str]:
    """Return sorted stems of *.json in labels_dir or paths.dev_labels_dir()."""
    d = Path(labels_dir) if labels_dir is not None else paths.dev_labels_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if p.is_file())


def assert_disjoint(ids: Iterable[str]) -> None:
    """Raise ValueError listing every given id that is also a holdout id."""
    current_holdout = set(holdout_ids())
    overlap = sorted({i for i in ids if i in current_holdout})
    if overlap:
        raise ValueError(f"IDs overlap with holdout split: {', '.join(overlap)}")


def import_holdout(
    src_labels_dir: Path,
    src_audio_dir: Path | None = None,
    cfg: Config | None = None,
) -> list[str]:
    """Import hand labels and matching audio into the hidden holdout split."""
    src_labels = Path(src_labels_dir)
    json_paths = sorted(src_labels.glob("*.json")) if src_labels.is_dir() else []

    if cfg is None:
        cfg = load_config()

    valid_class_ids = set(cfg.class_ids)
    valid_env_ids = set(cfg.env_ids)

    loaded_labels: list[ClipLabels] = []
    for json_path in json_paths:
        if not json_path.is_file():
            continue
        labels = load_labels(json_path)
        if json_path.stem != labels.video_id:
            raise ValueError(
                f"File stem {json_path.stem!r} does not match video_id "
                f"{labels.video_id!r} in {json_path.name}"
            )
        for ev in labels.events:
            if ev.class_id not in valid_class_ids:
                raise ValueError(
                    f"Unknown class_id {ev.class_id!r} in {json_path.name}"
                )
        for seg in labels.segments:
            if seg.env not in valid_env_ids:
                raise ValueError(
                    f"Unknown environment id {seg.env!r} in {json_path.name}"
                )
        loaded_labels.append(labels)

    current_dev = set(dev_ids())
    overlap = sorted({l.video_id for l in loaded_labels if l.video_id in current_dev})
    if overlap:
        raise ValueError(
            f"Imported IDs overlap with dev split: {', '.join(overlap)}"
        )

    dest_labels = holdout_labels_dir()
    dest_labels.mkdir(parents=True, exist_ok=True)
    for labels in loaded_labels:
        save_labels(labels, dest_labels / f"{labels.video_id}.json")

    missing_audio: list[str] = []
    if src_audio_dir is not None:
        src_audio = Path(src_audio_dir)
        dest_audio = holdout_audio_dir()
        for labels in loaded_labels:
            found = False
            for ext in AUDIO_EXTENSIONS:
                cand = src_audio / f"{labels.video_id}.{ext}"
                if cand.is_file():
                    dest_audio.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, dest_audio / f"{labels.video_id}.{ext}")
                    found = True
                    break
            if not found:
                missing_audio.append(labels.video_id)
    else:
        missing_audio = [l.video_id for l in loaded_labels]

    if missing_audio:
        warnings.warn(
            f"Missing audio for holdout IDs: {', '.join(sorted(missing_audio))}",
            UserWarning,
        )

    return sorted(l.video_id for l in loaded_labels)
