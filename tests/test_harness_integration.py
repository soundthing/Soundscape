"""End-to-end integration test for the events tuning harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from soundthing.config import load_config
from soundthing.contracts import (
    GENERAL_ENV,
    ClipLabels,
    LabeledEvent,
    LabeledSegment,
    save_labels,
)
from soundthing.events.core import EventModels
from soundthing.events.tagger import build_tagger_map
from soundthing.events.cache import (
    build_cache,
    cache_key,
    cache_path,
    save_cache,
)
from soundthing.harness import holdout, ledger, overrides, paths
from soundthing.harness.mcp_server import TuningTools
from soundthing.events.replay import replay
from soundthing.events import vad


class FakeTagger:
    """Deterministic fake tagger for synthetic audio testing."""

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
            probs[:, self.names.index(self.reg_label)] = 0.45
        elif peak >= 0.7 and self.safety_label and self.safety_label in self.names:
            probs[:, self.names.index(self.safety_label)] = 0.95
        return probs


def dir_hashes(directory: Path) -> dict[str, str]:
    """Compute sha256 hashes of all files in directory relative to directory."""
    hashes = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(directory)).replace("\\", "/")
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def test_tuning_integration_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Snapshot real repo hashes before anything touches the system
    repo_root = Path.cwd()
    real_config_hashes = dir_hashes(repo_root / "config")
    real_ref_hashes = dir_hashes(
        repo_root / "src" / "soundthing" / "events" / "references"
    )

    # 1. Isolated TUNING_HOME
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))

    # 2. Repo copy at root = tmp_path / "repo"
    root = tmp_path / "repo"
    shutil.copytree(repo_root / "config", root / "config")
    ref_dir = root / "src" / "soundthing" / "events" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo_root
        / "src"
        / "soundthing"
        / "events"
        / "references"
        / "tuning-log.md",
        ref_dir / "tuning-log.md",
    )

    monkeypatch.chdir(root)

    # Work around git_state in isolated tmp root (not a git repo)
    fake_git = {
        "sha": "1234567890abcdef",
        "dirty": False,
        "branch": "orch/C-integrate",
    }
    monkeypatch.setattr(
        ledger, "git_state", lambda cwd=None: fake_git
    )

    cfg = load_config(root / "config")
    env_id = next(e for e in cfg.env_ids if e != GENERAL_ENV)

    # Pick regular class that has no per-class override and prior 1.0 in env_id
    sc = next(
        c
        for c in cfg.classes
        if c.audioset
        and not c.safety
        and c.id not in cfg.thresholds.classes
        and cfg.priors.environments[env_id].weight(c.id) == 1.0
    )
    # Untier it so it uses thresholds.default (tiers would override the tuned default)
    thr_path = root / "config" / "thresholds.yaml"
    thr_lines = thr_path.read_text(encoding="utf-8").splitlines(keepends=True)
    thr_path.write_text(
        "".join(l for l in thr_lines if not l.startswith(f"  {sc.id}: ")), encoding="utf-8"
    )
    cfg = load_config(root / "config")
    assert sc.id not in cfg.thresholds.class_tiers
    safety_sc = next(c for c in cfg.classes if c.audioset and c.safety)

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    tagger = FakeTagger(
        audioset_names,
        reg_label=sc.audioset[0],
        safety_label=safety_sc.audioset[0],
    )
    models = EventModels(
        tagger=tagger,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        audioset_names=audioset_names,
    )
    monkeypatch.setattr(
        vad, "speech_probs", lambda a, s, c: np.zeros(10, dtype=np.float32)
    )

    tuned_key = "thresholds.default"
    tuned_val = 0.40
    tuned_cfg = overrides.apply_overrides(cfg, {tuned_key: tuned_val})

    # Prepare dev and holdout labels directories
    dev_labels_dir = paths.dev_labels_dir()
    dev_labels_dir.mkdir(parents=True, exist_ok=True)
    holdout_labels_dir = holdout.holdout_labels_dir()
    holdout_labels_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    duration = 6.0
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    # Pulse 1: safety class
    audio[int(1.0 * sr) : int(3.0 * sr)] = 0.9
    # Pulse 2: regular class
    audio[int(4.0 * sr) : int(5.0 * sr)] = 0.5

    key = cache_key(cfg)

    def _setup_clip(video_id: str, dest_dir: Path) -> None:
        c = build_cache(audio, sr, video_id, cfg, models=models)
        save_cache(c, cache_path(video_id, key))

        # Ground-truth labels made under tuned config
        synth_events = replay(c, None, tuned_cfg)
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
        save_labels(lbl, dest_dir / f"{video_id}.json")

    # 3 tiny dev clips and 2 holdout clips (different video ids)
    for i in range(1, 4):
        _setup_clip(f"dev_clip_{i:02d}", dev_labels_dir)
    for i in range(1, 3):
        _setup_clip(f"holdout_clip_{i:02d}", holdout_labels_dir)

    tools = TuningTools(cfg_loader=lambda: load_config(root / "config"))

    # a. tools.score -> passed True, bootstrap None, ledger entries
    res_a = tools.score(
        overrides={tuned_key: tuned_val}, hypothesis="integration"
    )
    assert res_a["passed"] is True
    assert res_a["bootstrap"] is None
    run_id_a = res_a["run_id"]

    ledger_entries = ledger.read()
    score_entries = [
        e
        for e in ledger_entries
        if e.get("kind") == "score" and e.get("hypothesis") == "integration"
    ]
    mcp_call_entries = [
        e
        for e in ledger_entries
        if e.get("kind") == "mcp_call" and e.get("tool") == "score"
    ]
    assert len(score_entries) >= 1
    assert len(mcp_call_entries) >= 1

    # b. tools.sweep -> evaluations == 2, entries with hypothesis "sweep"
    res_b = tools.sweep([tuned_key], "coordinate", 2)
    assert res_b["evaluations"] == 2
    sweep_entries = [
        e
        for e in ledger.read()
        if e.get("kind") == "score" and e.get("hypothesis") == "sweep"
    ]
    assert len(sweep_entries) >= 2

    # c. tools.promote(run_id_from_a) -> promoted True, holdout "pass"
    res_c = tools.promote(run_id_a)
    assert res_c["promoted"] is True
    assert res_c["holdout"] == "pass"

    # thresholds.yaml updated with comments preserved
    thr_text = (root / "config" / "thresholds.yaml").read_text(encoding="utf-8")
    assert "default: 0.4" in thr_text
    assert "# Required scores on the conditioned output" in thr_text

    # best.json updated
    best_data = ledger.best()
    assert best_data is not None
    assert best_data["run_id"] == run_id_a

    # root's tuning-log.md has a promote <run_id> entry
    log_text = (ref_dir / "tuning-log.md").read_text(encoding="utf-8")
    assert f"promote {run_id_a}" in log_text

    # The real repo config/ and references/ are byte-identical
    assert dir_hashes(repo_root / "config") == real_config_hashes
    assert (
        dir_hashes(repo_root / "src" / "soundthing" / "events" / "references")
        == real_ref_hashes
    )

    # No ledger line contains a holdout number
    raw_lines = paths.ledger_path().read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        entry = json.loads(line)
        if entry.get("kind") == "promote":
            assert entry.get("holdout_pass") is True
            for k in entry:
                if "holdout" in k and k != "holdout_pass":
                    pytest.fail(f"Promote entry contains holdout field: {k}={entry[k]}")
        else:
            assert entry.get("split") != "holdout"
            assert not any("holdout" in k for k in entry)

    # d. A second tools.score after promotion returns a bootstrap dict (best exists)
    res_d = tools.score(
        overrides={tuned_key: tuned_val}, hypothesis="after_promote"
    )
    assert res_d["bootstrap"] is not None
    assert isinstance(res_d["bootstrap"], dict)
    assert "gain" in res_d["bootstrap"]
    assert "lo" in res_d["bootstrap"]
    assert "hi" in res_d["bootstrap"]
    assert "accept" in res_d["bootstrap"]
