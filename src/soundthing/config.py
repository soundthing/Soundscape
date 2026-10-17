"""Load and cross-validate the shared config files in config/.

Everything environment- or class-specific lives in YAML; code reads it via `Config`.
Run `soundthing check-config` after editing any config file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .contracts import GENERAL_ENV

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigError(ValueError):
    pass


@dataclass
class Environment:
    id: str
    display: str
    visual_prompts: list[str] = field(default_factory=list)
    audio_prompts: list[str] = field(default_factory=list)


@dataclass
class SoundClass:
    id: str
    caption: str
    audioset: list[str] = field(default_factory=list)
    clap: list[str] = field(default_factory=list)
    coarse: str | None = None
    safety: bool = False
    sustained: bool = False


@dataclass
class CoarseGroup:
    id: str
    caption: str


@dataclass
class EnvPriors:
    default: float
    classes: dict[str, float]

    def weight(self, class_id: str) -> float:
        return self.classes.get(class_id, self.default)


@dataclass
class Priors:
    floor: float
    environments: dict[str, EnvPriors]


@dataclass
class Thresholds:
    default: float
    safety_default: float
    classes: dict[str, float]


@dataclass
class Config:
    environments: list[Environment]
    classes: list[SoundClass]
    coarse_groups: dict[str, CoarseGroup]
    priors: Priors
    thresholds: Thresholds
    pipeline: dict[str, Any]
    config_dir: Path

    # -- lookups -------------------------------------------------------------

    @property
    def env_ids(self) -> list[str]:
        return [e.id for e in self.environments]

    @property
    def class_ids(self) -> list[str]:
        return [c.id for c in self.classes]

    def sound_class(self, class_id: str) -> SoundClass:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(class_id)

    def safety_ids(self) -> set[str]:
        return {c.id for c in self.classes if c.safety}

    def threshold(self, class_id: str) -> float:
        if class_id in self.thresholds.classes:
            return self.thresholds.classes[class_id]
        if class_id in self.safety_ids():
            return self.thresholds.safety_default
        return self.thresholds.default

    def caption_for(self, class_id: str) -> str:
        """Caption text for a fine class id or a coarse group id."""
        if class_id in self.coarse_groups:
            return self.coarse_groups[class_id].caption
        return self.sound_class(class_id).caption

    def coarse_of(self, class_id: str) -> str | None:
        return self.sound_class(class_id).coarse

    # -- validation ----------------------------------------------------------

    def validate(self, audioset_labels: set[str] | None = None) -> list[str]:
        """Raise ConfigError on hard errors; return a list of warnings."""
        errors: list[str] = []
        warnings: list[str] = []

        env_ids = self.env_ids
        class_ids = self.class_ids
        if len(set(env_ids)) != len(env_ids):
            errors.append("duplicate environment ids")
        if len(set(class_ids)) != len(class_ids):
            dupes = sorted({c for c in class_ids if class_ids.count(c) > 1})
            errors.append(f"duplicate class ids: {dupes}")
        if GENERAL_ENV not in env_ids:
            errors.append(f"environments.yaml must define '{GENERAL_ENV}'")
        overlap = set(class_ids) & set(self.coarse_groups)
        if overlap:
            errors.append(f"coarse group ids collide with class ids: {sorted(overlap)}")

        for e in self.environments:
            if e.id != GENERAL_ENV and not (e.visual_prompts or e.audio_prompts):
                errors.append(f"environment {e.id}: needs visual_prompts or audio_prompts")

        for c in self.classes:
            if not (c.audioset or c.clap):
                errors.append(f"class {c.id}: needs audioset labels or clap prompts")
            if c.coarse is not None and c.coarse not in self.coarse_groups:
                errors.append(f"class {c.id}: unknown coarse group {c.coarse!r}")
            if audioset_labels is not None:
                for name in c.audioset:
                    if name not in audioset_labels:
                        errors.append(f"class {c.id}: AudioSet label {name!r} not in label list")

        for env_id in env_ids:
            if env_id not in self.priors.environments:
                errors.append(f"priors.yaml: missing block for environment {env_id!r}")
        for env_id, ep in self.priors.environments.items():
            if env_id not in env_ids:
                errors.append(f"priors.yaml: unknown environment {env_id!r}")
            if not 0 <= ep.default <= 1:
                errors.append(f"priors.yaml: {env_id}.default not in [0, 1]")
            for cid, w in ep.classes.items():
                if cid not in class_ids:
                    errors.append(f"priors.yaml: {env_id}: unknown class {cid!r}")
                if not 0 <= w <= 1:
                    errors.append(f"priors.yaml: {env_id}.{cid}={w} not in [0, 1]")
                if cid in self.safety_ids():
                    warnings.append(f"priors.yaml: {env_id}.{cid} is a safety class; prior is ignored")
        if not 0 <= self.priors.floor <= 1:
            errors.append("priors.yaml: floor not in [0, 1]")

        for cid, t in self.thresholds.classes.items():
            if cid not in class_ids and cid not in self.coarse_groups:
                errors.append(f"thresholds.yaml: unknown class {cid!r}")
            if not 0 < t < 1:
                errors.append(f"thresholds.yaml: {cid}={t} not in (0, 1)")

        covered = {cid for ep in self.priors.environments.values() for cid in ep.classes}
        for c in self.classes:
            if not c.safety and c.id not in covered:
                warnings.append(f"class {c.id} is not listed under any environment in priors.yaml")

        if errors:
            raise ConfigError("config errors:\n  " + "\n  ".join(errors))
        return warnings


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path | None = None, validate: bool = True) -> Config:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    env_raw = _load_yaml(d / "environments.yaml")
    vocab_raw = _load_yaml(d / "vocabulary.yaml")
    priors_raw = _load_yaml(d / "priors.yaml")
    thr_raw = _load_yaml(d / "thresholds.yaml")
    pipeline = _load_yaml(d / "pipeline.yaml")

    cfg = Config(
        environments=[Environment(**e) for e in env_raw.get("environments", [])],
        classes=[SoundClass(**c) for c in vocab_raw.get("classes", [])],
        coarse_groups={
            k: CoarseGroup(id=k, **v) for k, v in (vocab_raw.get("coarse_groups") or {}).items()
        },
        priors=Priors(
            floor=float(priors_raw.get("floor", 0.0)),
            environments={
                k: EnvPriors(default=float(v.get("default", 1.0)), classes=dict(v.get("classes") or {}))
                for k, v in (priors_raw.get("environments") or {}).items()
            },
        ),
        thresholds=Thresholds(
            default=float(thr_raw.get("default", 0.5)),
            safety_default=float(thr_raw.get("safety_default", thr_raw.get("default", 0.5))),
            classes=dict(thr_raw.get("classes") or {}),
        ),
        pipeline=pipeline,
        config_dir=d,
    )
    if validate:
        cfg.validate()
    return cfg


def read_audioset_labels(csv_path: str | Path) -> set[str]:
    """Read AudioSet display names from class_labels_indices.csv (index,mid,display_name)."""
    with open(csv_path, encoding="utf-8") as f:
        return {row["display_name"] for row in csv.DictReader(f)}
