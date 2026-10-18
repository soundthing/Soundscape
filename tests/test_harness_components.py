"""Tests for component-agnostic harness core and events component (Track A)."""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soundthing.config import DEFAULT_CONFIG_DIR, Config, load_config
from soundthing.contracts import (
    GENERAL_ENV,
    ClipLabels,
    LabeledEvent,
    LabeledSegment,
    save_labels,
)
from soundthing.events import vad
from soundthing.events.cache import (
    build_cache,
    cache_key,
    cache_path,
    save_cache,
)
from soundthing.events.core import EventModels
from soundthing.events.replay import replay
from soundthing.events.tagger import build_tagger_map
from soundthing.harness import bootstrap, components, ledger, paths, promote
from soundthing.harness.__main__ import main
from soundthing.harness.components.base import Clip, Component, Evaluation
from soundthing.harness.score import score


class FakeTagger:
    def __init__(
        self,
        names: list[str],
        reg_label: str | None = None,
        safety_label: str | None = None,
    ) -> None:
        self.names = names
        self.reg_label = reg_label
        self.safety_label = safety_label

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.full((b, len(self.names)), 0.05, dtype=np.float32)
        peak = float(np.max(np.abs(audio)))
        if 0.3 <= peak < 0.7 and self.reg_label and self.reg_label in self.names:
            probs[:, self.names.index(self.reg_label)] = 0.95
        elif peak >= 0.7 and self.safety_label and self.safety_label in self.names:
            probs[:, self.names.index(self.safety_label)] = 0.95
        return probs


def _setup_clip_and_cache(
    cfg: Config,
    labels_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_id: str = "clip_synth",
    duration: float = 5.0,
) -> ClipLabels:
    labels_dir.mkdir(parents=True, exist_ok=True)
    env_id = next(e for e in cfg.env_ids if e != GENERAL_ENV)

    sc = next(
        c
        for c in cfg.classes
        if c.audioset
        and not c.safety
        and cfg.priors.environments[env_id].weight(c.id) >= 0.7
    )
    safety_sc = next(c for c in cfg.classes if c.audioset and c.safety)

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)

    reg_label = sc.audioset[0]
    safety_label = safety_sc.audioset[0]

    tagger = FakeTagger(
        audioset_names,
        reg_label=reg_label,
        safety_label=safety_label,
    )
    models = EventModels(
        tagger=tagger,
        clap=None,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
    )

    monkeypatch.setattr(
        vad, "speech_probs", lambda a, s, c: np.zeros(10, dtype=np.float32)
    )

    sr = 16000
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    audio[int(0.5 * sr) : int(1.5 * sr)] = 0.5
    audio[int(3.0 * sr) : int(4.0 * sr)] = 0.9

    c = build_cache(audio, sr, video_id, cfg, models=models)
    key = cache_key(cfg)
    save_cache(c, cache_path(video_id, key))

    synth_events = replay(c, None, cfg)
    labeled_events = [
        LabeledEvent(start=e.start, end=e.end, class_id=e.class_id)
        for e in synth_events
    ]

    lbl = ClipLabels(
        video_id=video_id,
        duration=duration,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=duration, env=env_id)],
        events=labeled_events,
    )
    save_labels(lbl, labels_dir / f"{video_id}.json")
    return lbl


@dataclass
class DummyClipCache:
    video_id: str
    data: float = 1.0


class FakeComponent:
    name: str = "fake"
    owner: str = "person2"
    sections: tuple[str, ...] = ("environment.",)
    code_paths: tuple[str, ...] = ("src/soundthing/environment",)
    tuning_log: str = "src/soundthing/environment/references/tuning-log.md"
    n_stats: int = 2

    def cache_key(self, cfg: Config) -> str | None:
        return "fake_key_123"

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path | None:
        return None

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        ldir = Path(labels_dir)
        files = sorted(ldir.glob("*.json"))
        clips: list[Clip] = []
        for p in files:
            lbl = ClipLabels(video_id=p.stem, duration=5.0, labeler="test", segments=[], events=[])
            clips.append(Clip(video_id=p.stem, labels=lbl, cache=DummyClipCache(p.stem)))
        return clips

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        per_clip: dict[str, dict[str, Any]] = {}
        for c in clips:
            per_clip[c.video_id] = {
                "stats": [8.0, 2.0],
                "accuracy": 0.8,
            }
        return Evaluation(
            objective=0.80,
            metrics={"objective": 0.80, "accuracy": 0.80},
            per_clip=per_clip,
            diagnostics={"n_clips": len(clips)},
        )

    def limits(self, metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
        if best is None:
            return {"accuracy": True}
        return {"accuracy": bool(metrics["accuracy"] >= float(best.get("accuracy", 0.0)))}

    def space(self, cfg: Config, labels_dir: Path | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": "environment.threshold",
                "type": "float",
                "bounds": [0.0, 1.0],
                "current": 0.5,
                "shared": False,
            }
        ]

    def stat_objective(self, totals: np.ndarray) -> np.ndarray:
        if totals.ndim == 1:
            totals = totals[None, :]
        num = totals[:, 0]
        denom = totals[:, 0] + totals[:, 1]
        out = np.zeros_like(num, dtype=np.float64)
        valid = denom > 0
        out[valid] = num[valid] / denom[valid]
        return out


FAKE_COMPONENT: FakeComponent = FakeComponent()


@pytest.fixture(autouse=True)
def isolate_harness_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))


def test_registry_get_events_is_component():
    comp = components.get("events")
    assert isinstance(comp, Component)
    assert comp.name == "events"
    assert comp.owner == "person1"
    assert comp.sections == ("events.", "thresholds.")
    assert comp.tuning_log == "src/soundthing/events/references/tuning-log.md"
    assert comp.n_stats == 4


def test_fake_component_end_to_end_score_bootstrap_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "fake_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "clip1.json").write_text("{}", encoding="utf-8")
    (labels_dir / "clip2.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(components.REGISTRY, "fake", "tests.test_harness_components:FAKE_COMPONENT")

    comp = components.get("fake")
    assert isinstance(comp, Component)

    # 1. Score fake component
    cfg = load_config(DEFAULT_CONFIG_DIR)
    res = score(
        overrides={"environment.threshold": 0.7},
        component="fake",
        labels_dir=labels_dir,
        cfg=cfg,
    )
    assert res.component == "fake"
    assert res.objective == 0.80
    assert res.accuracy == 0.80
    assert res.passed is True
    assert "clip1" in res.per_clip
    assert res.per_clip["clip1"]["stats"] == [8.0, 2.0]

    # 2. Paired bootstrap with fake component
    comparator_per_clip = {
        "clip1": {"stats": [6.0, 4.0]},
        "clip2": {"stats": [6.0, 4.0]},
    }
    ci = bootstrap.paired_bootstrap(res.per_clip, comparator_per_clip, component="fake", n_boot=100)
    assert ci["gain"] == pytest.approx(0.20)
    assert ci["lo"] > 0
    assert bootstrap.accepts(ci) is True

    # 3. Ledger best and set_best write best-fake.json
    ledger.set_best(res.to_dict(), component="fake")
    best_file = paths.best_path("fake")
    assert best_file.name == "best-fake.json"
    assert best_file.exists()

    best_read = ledger.best("fake")
    assert best_read is not None
    assert best_read["run_id"] == res.run_id
    assert best_read["component"] == "fake"
    assert best_read["objective"] == 0.80


def test_override_keys_outside_comp_sections_raise_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "clip.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(components.REGISTRY, "fake", "tests.test_harness_components:FAKE_COMPONENT")

    # events component rejects environment. or captions.
    with pytest.raises(ValueError, match="sections"):
        score(overrides={"environment.something": 1.0}, component="events", labels_dir=labels_dir)

    with pytest.raises(ValueError, match="sections"):
        score(overrides={"captions.something": 1.0}, component="events", labels_dir=labels_dir)

    # fake component rejects events. or thresholds.
    with pytest.raises(ValueError, match="sections"):
        score(overrides={"events.postprocess.smoothing_sec": 0.5}, component="fake", labels_dir=labels_dir)


def test_promote_keys_belonging_to_another_role_writes_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "repo"
    config_dir = root / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, config_dir)

    tuning_log = root / "src" / "soundthing" / "environment" / "references" / "tuning-log.md"
    tuning_log.parent.mkdir(parents=True, exist_ok=True)
    tuning_log.write_text("# Environment tuning log\n\n_No runs yet._\n", encoding="utf-8")

    pipe_orig_bytes = (config_dir / "pipeline.yaml").read_bytes()
    thr_orig_bytes = (config_dir / "thresholds.yaml").read_bytes()

    monkeypatch.setitem(components.REGISTRY, "fake", "tests.test_harness_components:FAKE_COMPONENT")
    monkeypatch.setenv("SOUNDTHING_ROLE", "person1")

    # Candidate whose overrides belong to person2 (environment.)
    cand_entry = {
        "run_id": "cand_fake_p2",
        "kind": "score",
        "component": "fake",
        "overrides": {"environment.threshold": 0.7},
        "objective": 0.85,
        "accuracy": 0.85,
        "passed": True,
        "per_clip": {
            "clip1": {"stats": [8.0, 2.0]},
            "clip2": {"stats": [8.0, 2.0]},
        },
    }
    comparator_entry = {
        "run_id": "best_fake",
        "kind": "score",
        "component": "fake",
        "objective": 0.60,
        "accuracy": 0.60,
        "passed": True,
        "per_clip": {
            "clip1": {"stats": [6.0, 4.0]},
            "clip2": {"stats": [6.0, 4.0]},
        },
    }

    fake_ledger = ledger
    fake_ledger.append(comparator_entry)
    fake_ledger.append(cand_entry)
    fake_ledger.set_best(comparator_entry, component="fake")

    res = promote.promote(
        "cand_fake_p2",
        ledger=fake_ledger,
        root=root,
        n_boot=50,
        seed=1,
        component="fake",
    )

    # 1. Promoted True, holdout is n/a for fake component
    assert res["promoted"] is True
    assert res["holdout"] == "n/a"

    # 2. No config file was changed
    assert res["files"] == []
    assert (config_dir / "pipeline.yaml").read_bytes() == pipe_orig_bytes
    assert (config_dir / "thresholds.yaml").read_bytes() == thr_orig_bytes

    # 3. Proposal was written under proposals dir
    assert "proposal" in res
    prop_path = Path(res["proposal"])
    assert prop_path.exists()
    assert prop_path.parent == paths.proposals_dir()
    prop_text = prop_path.read_text(encoding="utf-8")
    assert "environment.threshold" in prop_text
    assert "cand_fake_p2" in prop_text


def test_events_equivalence(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    lbl = _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    res = score(cfg=cfg, labels_dir=labels_dir, component="events")

    assert res.objective > 0.0
    assert res.baseline > 0.0
    assert res.wrong_env > 0.0
    assert res.safety_recall == 1.0
    assert res.fp_per_min == pytest.approx(0.0)
    assert res.passed is True
    assert res.component == "events"

    # Per clip counts and stats
    vid = lbl.video_id
    assert vid in res.per_clip
    modes = res.per_clip[vid]
    assert "stats" in modes
    assert len(modes["stats"]) == 4

    cond = modes["conditioned"]
    assert cond["tp_fine"] > 0
    assert cond["safety_tp"] > 0
    assert cond["safety_total"] > 0
    assert modes["stats"] == [
        float(cond["tp_fine"]),
        float(cond["tp_coarse"]),
        float(cond["fp"]),
        float(cond["fn"]),
    ]


def test_list_components_cli(monkeypatch: pytest.MonkeyPatch):
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)

    rc = main(["list-components"])
    assert rc == 0

    val = out.getvalue()
    parsed = json.loads(val)
    assert isinstance(parsed, list)
    assert len(parsed) == 4

    comp_map = {item["name"]: item for item in parsed}
    assert set(comp_map.keys()) == {"events", "environment", "captions", "pipeline"}

    assert comp_map["events"]["available"] is True
    assert comp_map["events"]["owner"] == "person1"
    assert "events." in comp_map["events"]["sections"]

    owners = {"environment": "person2", "captions": "person3", "pipeline": "person3"}
    for name, owner in owners.items():
        assert comp_map[name]["available"] is True
        assert comp_map[name]["owner"] == owner
