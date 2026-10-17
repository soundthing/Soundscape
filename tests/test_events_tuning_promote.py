"""Tests for events tuning YAML config writer (promote.py)."""

from __future__ import annotations

import re
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from soundthing.config import DEFAULT_CONFIG_DIR, load_config
from soundthing.events.tuning.promote import (
    PromoteError,
    append_tuning_log,
    format_scalar,
    holdout_check,
    promote,
    set_yaml_value,
    split_target,
    write_overrides,
)


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg_dir = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, cfg_dir)
    return cfg_dir


def test_split_target_valid():
    assert split_target("events.postprocess.smoothing_sec") == (
        "pipeline.yaml",
        ["events", "postprocess", "smoothing_sec"],
    )
    assert split_target("thresholds.default") == ("thresholds.yaml", ["default"])
    assert split_target("thresholds.safety_default") == (
        "thresholds.yaml",
        ["safety_default"],
    )
    assert split_target("thresholds.classes.gunshot") == (
        "thresholds.yaml",
        ["classes", "gunshot"],
    )


def test_split_target_refusals():
    with pytest.raises(PromoteError):
        split_target("events.backend")
    with pytest.raises(PromoteError):
        split_target("events.backend.foo")
    with pytest.raises(PromoteError):
        split_target("thresholds.classes")
    with pytest.raises(PromoteError):
        split_target("thresholds.classes.a.b")
    with pytest.raises(PromoteError):
        split_target("thresholds.other")
    with pytest.raises(PromoteError):
        split_target("vocabulary.gunshot.clap")
    with pytest.raises(PromoteError):
        split_target("priors.floor")
    with pytest.raises(PromoteError):
        split_target("priors.street.default")


def test_format_scalar():
    # Acceptance check 6: format_scalar cases: True, 3, 20.0, 0.35, "max"
    assert format_scalar(True) == "true"
    assert format_scalar(False) == "false"
    assert format_scalar(3) == "3"
    assert format_scalar(20.0) == "20.0"
    assert format_scalar(0.35) == "0.35"
    assert format_scalar("max") == "max"
    assert format_scalar("panns_cnn14") == "panns_cnn14"
    # Special strings that need JSON quoting so they don't deserialize as bool/int
    assert format_scalar("true") == '"true"'
    assert format_scalar("123") == '"123"'

    with pytest.raises(PromoteError):
        format_scalar([1, 2])
    with pytest.raises(PromoteError):
        format_scalar({"a": 1})
    with pytest.raises(PromoteError):
        format_scalar(None)


def test_edit_smoothing_sec_minimal_diff(tmp_config_dir: Path):
    # Acceptance check 1: Changing events.postprocess.smoothing_sec to 0.75:
    # only that one line differs (compare line lists), its trailing comment is kept,
    # load_config(tmp) gives 0.75.
    pipe_path = tmp_config_dir / "pipeline.yaml"
    orig_lines = pipe_path.read_text(encoding="utf-8").splitlines()

    changed = write_overrides({"events.postprocess.smoothing_sec": 0.75}, tmp_config_dir)
    assert changed == [pipe_path]

    new_lines = pipe_path.read_text(encoding="utf-8").splitlines()
    assert len(orig_lines) == len(new_lines)

    diff_indices = [i for i, (l1, l2) in enumerate(zip(orig_lines, new_lines)) if l1 != l2]
    assert len(diff_indices) == 1
    diff_idx = diff_indices[0]

    assert "smoothing_sec: 0.75" in new_lines[diff_idx]
    assert "# two-sided median filter width" in new_lines[diff_idx]

    cfg = load_config(tmp_config_dir)
    assert cfg.pipeline["events"]["postprocess"]["smoothing_sec"] == 0.75


def test_edit_thresholds(tmp_config_dir: Path):
    # Acceptance check 2:
    # thresholds.classes.<first id under classes: in thresholds.yaml> updated (read the id from the yaml);
    # thresholds.classes.<a cfg class id NOT in thresholds.yaml> added under classes: with right indent;
    # thresholds.default changed; comments in thresholds.yaml unchanged.
    thr_path = tmp_config_dir / "thresholds.yaml"
    orig_text = thr_path.read_text(encoding="utf-8")
    thr_yaml = yaml.safe_load(orig_text)

    first_class_id = next(iter(thr_yaml["classes"].keys()))
    cfg = load_config(tmp_config_dir)
    missing_class_id = [cid for cid in cfg.class_ids if cid not in thr_yaml["classes"]][0]

    overrides = {
        f"thresholds.classes.{first_class_id}": 0.82,
        f"thresholds.classes.{missing_class_id}": 0.68,
        "thresholds.default": 0.42,
    }

    changed = write_overrides(overrides, tmp_config_dir)
    assert changed == [thr_path]

    new_text = thr_path.read_text(encoding="utf-8")

    # Comments in thresholds.yaml unchanged
    assert "# Per-class detection thresholds on the (conditioned) score. OWNER: Person 1." in new_text
    assert "# Precision over recall: when in doubt, raise a threshold." in new_text
    assert "# Lookup: classes[id] -> safety_default (if class is safety) -> default." in new_text
    assert "# safety classes bypass priors, so they need a higher bar against false alarms" in new_text

    # Newly added class has right indent (2 spaces)
    assert f"  {missing_class_id}: 0.68" in new_text

    # load_config verification
    cfg2 = load_config(tmp_config_dir)
    assert cfg2.threshold(first_class_id) == 0.82
    assert cfg2.threshold(missing_class_id) == 0.68
    assert cfg2.thresholds.default == 0.42


def test_missing_key_insertion(tmp_config_dir: Path):
    # Acceptance check 3:
    # Missing key events.postprocess.onset_shift_sec = -0.1 is inserted inside the postprocess: block
    # (before modifiers:), and a missing intermediate mapping (e.g. events.clap.newgroup.x) is created
    # correctly; yaml.safe_load gives the values; the captions:/eval: sections are unchanged.
    pipe_path = tmp_config_dir / "pipeline.yaml"
    orig_text = pipe_path.read_text(encoding="utf-8")

    overrides = {
        "events.postprocess.onset_shift_sec": -0.1,
        "events.clap.newgroup.x": 42,
    }

    changed = write_overrides(overrides, tmp_config_dir)
    assert changed == [pipe_path]

    new_text = pipe_path.read_text(encoding="utf-8")
    lines = new_text.splitlines()

    postprocess_idx = next(i for i, l in enumerate(lines) if re.match(r"^\s*postprocess:", l))
    onset_idx = next(i for i, l in enumerate(lines) if "onset_shift_sec:" in l)
    modifiers_idx = next(i for i, l in enumerate(lines) if re.match(r"^\s*modifiers:", l))

    assert postprocess_idx < onset_idx < modifiers_idx

    # Intermediate mapping
    clap_idx = next(i for i, l in enumerate(lines) if re.match(r"^\s*clap:", l))
    newgroup_idx = next(i for i, l in enumerate(lines) if "newgroup:" in l)
    x_idx = next(i for i, l in enumerate(lines) if re.match(r"^\s*x:\s*42", l))
    vad_idx = next(i for i, l in enumerate(lines) if re.match(r"^\s*vad:", l))

    assert clap_idx < newgroup_idx < x_idx < vad_idx

    loaded = yaml.safe_load(new_text)
    assert loaded["events"]["postprocess"]["onset_shift_sec"] == -0.1
    assert loaded["events"]["clap"]["newgroup"]["x"] == 42

    # captions: and eval: unchanged
    captions_marker = "captions:           # Person 3"
    orig_captions_part = orig_text[orig_text.index(captions_marker):]
    new_captions_part = new_text[new_text.index(captions_marker):]
    assert orig_captions_part == new_captions_part


def test_refusals_nothing_written(tmp_config_dir: Path):
    # Acceptance check 4:
    # Refusals with nothing written (files byte-identical afterwards):
    # vocabulary.x.clap, priors.floor, events.backend, a mapping key (events.postprocess),
    # a mix of one good + one bad key.
    bad_overrides_list = [
        {"vocabulary.x.clap": ["prompt"]},
        {"priors.floor": 0.05},
        {"events.backend": "real"},
        {"events.postprocess": 0.5},
        {"events.postprocess.smoothing_sec": 0.75, "events.backend": "real"},
        {"events.postprocess.smoothing_sec": 0.75, "events.postprocess": 0.5},
        {"events.postprocess.smoothing_sec": 0.75, "vocabulary.x.clap": ["a"]},
    ]

    for bad in bad_overrides_list:
        before_bytes = {p.name: p.read_bytes() for p in tmp_config_dir.iterdir()}
        with pytest.raises(PromoteError):
            write_overrides(bad, tmp_config_dir)
        after_bytes = {p.name: p.read_bytes() for p in tmp_config_dir.iterdir()}
        assert before_bytes == after_bytes, f"Files were modified for refused override: {bad}"


def test_crlf_preserved(tmp_config_dir: Path):
    # Acceptance check 5:
    # CRLF: a file converted to CRLF stays CRLF after an edit.
    pipe_path = tmp_config_dir / "pipeline.yaml"
    content = pipe_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    pipe_path.write_bytes(content)

    write_overrides({"events.postprocess.smoothing_sec": 0.75}, tmp_config_dir)

    new_bytes = pipe_path.read_bytes()
    assert b"\r\n" in new_bytes
    assert new_bytes.endswith(b"\r\n")
    # Verify no bare \n
    lines = new_bytes.split(b"\r\n")
    assert all(b"\n" not in line for line in lines)


def test_load_config_validation_failure_restores_file(tmp_config_dir: Path):
    # Acceptance check 7:
    # A value that makes load_config fail validation (e.g. thresholds.classes.<unknown_id>
    # — note thresholds.yaml unknown class ids are a validation error, see Config.validate)
    # -> PromoteError and the file is restored.
    before_bytes = {p.name: p.read_bytes() for p in tmp_config_dir.iterdir()}

    with pytest.raises(PromoteError):
        write_overrides({"thresholds.classes.totally_unknown_class_xyz": 0.5}, tmp_config_dir)

    after_bytes = {p.name: p.read_bytes() for p in tmp_config_dir.iterdir()}
    assert before_bytes == after_bytes


@dataclass
class FakeResult:
    run_id: str = "run_fake"
    objective: float = 0.85
    baseline: float = 0.60
    wrong_env: float = 0.50
    safety_recall: float = 0.95
    fp_per_min: float = 0.20
    grid_f1: float = 0.70
    per_clip: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    passed: bool = True
    overrides: dict = field(default_factory=dict)
    split: str = "dev"


class FakeLedger:
    def __init__(self, entries=None, best_entry=None):
        self.entries = list(entries or [])
        self._best = best_entry
        self.set_best_called = None

    def append(self, entry: dict) -> None:
        self.entries.append(dict(entry))

    def read(self) -> list[dict]:
        return list(self.entries)

    def best(self) -> dict | None:
        return self._best

    def set_best(self, entry: dict) -> None:
        self._best = dict(entry)
        self.set_best_called = dict(entry)

    def git_state(self, cwd=None) -> dict:
        return {"sha": "c0ffee1234567890", "dirty": False, "branch": "orch/C-promote"}


def make_per_clip_data(n_clips: int = 30, fp_cand: int = 0, fp_best: int = 5) -> tuple[dict, dict]:
    cand_per_clip = {}
    best_per_clip = {}
    for i in range(n_clips):
        vid = f"clip_{i:02d}"
        cand_per_clip[vid] = {
            "conditioned": {
                "tp_fine": 10,
                "tp_coarse": 0,
                "fp": fp_cand,
                "fn": 1,
                "minutes": 1.0,
                "safety_tp": 2,
                "safety_total": 2,
            },
            "baseline": {
                "tp_fine": 8,
                "tp_coarse": 0,
                "fp": fp_cand + 1,
                "fn": 2,
                "minutes": 1.0,
                "safety_tp": 2,
                "safety_total": 2,
            },
            "wrong_env": {
                "tp_fine": 6,
                "tp_coarse": 0,
                "fp": fp_cand + 2,
                "fn": 3,
                "minutes": 1.0,
                "safety_tp": 1,
                "safety_total": 2,
            },
        }
        best_per_clip[vid] = {
            "conditioned": {
                "tp_fine": 10,
                "tp_coarse": 0,
                "fp": fp_best,
                "fn": 1,
                "minutes": 1.0,
                "safety_tp": 2,
                "safety_total": 2,
            },
            "baseline": {
                "tp_fine": 8,
                "tp_coarse": 0,
                "fp": fp_best + 1,
                "fn": 2,
                "minutes": 1.0,
                "safety_tp": 2,
                "safety_total": 2,
            },
            "wrong_env": {
                "tp_fine": 6,
                "tp_coarse": 0,
                "fp": fp_best + 2,
                "fn": 3,
                "minutes": 1.0,
                "safety_tp": 1,
                "safety_total": 2,
            },
        }
    return cand_per_clip, best_per_clip


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path / "tuning_home"))
    root = tmp_path / "repo"
    config_dir = root / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, config_dir)

    real_tuning_log = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "soundthing"
        / "events"
        / "references"
        / "tuning-log.md"
    )
    target_tuning_log = root / "src" / "soundthing" / "events" / "references" / "tuning-log.md"
    target_tuning_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(real_tuning_log, target_tuning_log)

    return root


def test_promote_happy_path(repo_root: Path):
    cand_per_clip, best_per_clip = make_per_clip_data(30, fp_cand=0, fp_best=5)

    tuning_log_path = repo_root / "src" / "soundthing" / "events" / "references" / "tuning-log.md"
    # Pre-seed one older ## entry to check ordering
    pre_seed_text = (
        tuning_log_path.read_text(encoding="utf-8").replace(
            "_No runs yet._",
            "## 2026-10-15 — promote old_run\n"
            "Git: 1111222233    Split: dev (30 clips)\n"
            "Params: events.postprocess.smoothing_sec: 0.5000 -> 0.6000\n"
            "Dev F0.5: 0.6000 -> 0.7200   (baseline 0.5500, wrong-env 0.5000)\n"
            "Bootstrap gain: 0.1200 [0.0500, 0.1800]\n"
            "Holdout: pass\n"
            "Outcome: promoted\n",
        )
    )
    tuning_log_path.write_text(pre_seed_text, encoding="utf-8")

    run_id = "cand_001"
    cand_entry = {
        "run_id": run_id,
        "kind": "score",
        "overrides": {"events.postprocess.smoothing_sec": 0.75},
        "objective": 0.88,
        "baseline": 0.70,
        "wrong_env": 0.65,
        "safety_recall": 0.95,
        "fp_per_min": 0.10,
        "passed": True,
        "per_clip": cand_per_clip,
    }
    best_entry = {
        "run_id": "best_000",
        "kind": "score",
        "objective": 0.72,
        "baseline": 0.65,
        "wrong_env": 0.60,
        "safety_recall": 0.95,
        "fp_per_min": 0.30,
        "passed": True,
        "per_clip": best_per_clip,
    }

    ledger = FakeLedger(entries=[best_entry, cand_entry], best_entry=best_entry)

    holdout_called_with = []

    def fake_holdout(ov):
        holdout_called_with.append(ov)
        return True

    res = promote(
        run_id,
        ledger=ledger,
        holdout_fn=fake_holdout,
        root=repo_root,
        n_boot=100,
        seed=42,
    )

    # 1. promoted True
    assert res["promoted"] is True
    assert res["run_id"] == run_id
    assert res["holdout"] == "pass"
    assert res["objective"] == 0.88
    assert res["bootstrap"]["lo"] > 0
    assert "config/pipeline.yaml" in res["files"]

    # 2. pipeline.yaml in root has the new value with comment kept
    pipe_text = (repo_root / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    assert "smoothing_sec: 0.75" in pipe_text
    assert "# two-sided median filter width" in pipe_text

    # 3. ledger.set_best called with git_sha
    assert ledger.set_best_called is not None
    assert ledger.set_best_called["run_id"] == run_id
    assert ledger.set_best_called["git_sha"] == "c0ffee1234567890"

    # 4. tuning-log has new entry right after header and before older entries
    log_text = tuning_log_path.read_text(encoding="utf-8")
    assert f"promote {run_id}" in log_text
    new_idx = log_text.index(f"promote {run_id}")
    old_idx = log_text.index("promote old_run")
    assert new_idx < old_idx
    header_end = log_text.index("```\n\n## ")
    assert new_idx > header_end

    # 5. Ledger got one "promote" entry with holdout_pass True
    promote_entries = [e for e in ledger.entries if e.get("kind") == "promote"]
    assert len(promote_entries) == 1
    p_entry = promote_entries[0]
    assert p_entry["run_id"] == run_id
    assert p_entry["promoted"] is True
    assert p_entry["holdout_pass"] is True
    assert p_entry["reason"] is None
    assert p_entry["git"]["sha"] == "c0ffee1234567890"

    # 6. Return value and ledger entries contain NO holdout numbers (only bool / "pass")
    assert "holdout_pass" in p_entry
    assert isinstance(p_entry["holdout_pass"], bool)
    assert not any("holdout" in k and k != "holdout_pass" for k in p_entry)
    assert res["holdout"] == "pass"
    for k, v in res.items():
        if "holdout" in k:
            assert v == "pass"


def test_promote_refusals(repo_root: Path):
    cand_per_clip, best_per_clip = make_per_clip_data(30, fp_cand=0, fp_best=5)

    base_cand = {
        "run_id": "cand_refuse",
        "kind": "score",
        "overrides": {"events.postprocess.smoothing_sec": 0.75},
        "objective": 0.88,
        "baseline": 0.70,
        "wrong_env": 0.65,
        "safety_recall": 0.95,
        "fp_per_min": 0.10,
        "passed": True,
        "per_clip": cand_per_clip,
    }
    best_entry = {
        "run_id": "best_000",
        "kind": "score",
        "objective": 0.72,
        "passed": True,
        "per_clip": best_per_clip,
    }

    def raise_holdout(ov):
        raise PromoteError("holdout set not available (test error)")

    cases = [
        # (name, cand_mods, holdout_fn, expected_reason_sub, holdout_pass_val)
        ("shared_key", {"overrides": {"vocabulary.gunshot.clap": ["woof"]}}, lambda ov: True, "propose_shared_change", None),
        ("limits_not_passed", {"passed": False}, lambda ov: True, "limits not passed", None),
        ("bootstrap_fail", {"per_clip": best_per_clip}, lambda ov: True, "bootstrap lower bound <= 0", None),
        (
            "bootstrap_no_common_clips",
            {"per_clip": {"disjoint_clip_1": {"conditioned": {"tp_fine": 10, "tp_coarse": 0, "fp": 0, "fn": 1}}}},
            lambda ov: True,
            "bootstrap: no common clips with comparator",
            None,
        ),
        ("holdout_failed", {}, lambda ov: False, "holdout failed", False),
        ("holdout_error", {}, raise_holdout, "holdout set not available", None),
        ("no_overrides", {"overrides": {}}, lambda ov: True, "no overrides", None),
        ("invalid_target", {"overrides": {"events.backend": "real"}}, lambda ov: True, "Refusing to override events.backend", None),
    ]

    for name, cand_mods, holdout_fn, expected_reason, expected_hp in cases:
        cand = {**base_cand, **cand_mods}
        ledger = FakeLedger(entries=[best_entry, cand], best_entry=best_entry)

        config_dir = repo_root / "config"
        before_bytes = {p.name: p.read_bytes() for p in config_dir.iterdir()}

        res = promote(
            "cand_refuse",
            ledger=ledger,
            holdout_fn=holdout_fn,
            root=repo_root,
            n_boot=100,
            seed=42,
        )

        assert res["promoted"] is False, f"Failed for case {name}"
        assert expected_reason in res["reason"], f"Reason mismatch for {name}: {res['reason']}"

        after_bytes = {p.name: p.read_bytes() for p in config_dir.iterdir()}
        assert before_bytes == after_bytes, f"Config modified for refusal case {name}"

        promote_entries = [e for e in ledger.entries if e.get("kind") == "promote"]
        assert len(promote_entries) == 1, f"Expected 1 promote entry for {name}, got {len(promote_entries)}"
        pe = promote_entries[0]
        assert pe["promoted"] is False
        assert pe["reason"] == res["reason"]
        assert pe["holdout_pass"] == expected_hp


def test_promote_no_best_in_ledger(repo_root: Path):
    cand_per_clip, comparator_per_clip = make_per_clip_data(30, fp_cand=0, fp_best=5)

    cand_entry = {
        "run_id": "cand_nobest",
        "kind": "score",
        "overrides": {"events.postprocess.smoothing_sec": 0.75},
        "objective": 0.88,
        "baseline": 0.70,
        "wrong_env": 0.65,
        "passed": True,
        "per_clip": cand_per_clip,
    }

    ledger = FakeLedger(entries=[cand_entry], best_entry=None)

    fake_score_calls = []

    def fake_score(**kwargs):
        fake_score_calls.append(kwargs)
        return FakeResult(
            run_id="cur_baseline",
            objective=0.70,
            per_clip=comparator_per_clip,
        )

    res = promote(
        "cand_nobest",
        ledger=ledger,
        score_fn=fake_score,
        holdout_fn=lambda ov: True,
        root=repo_root,
        n_boot=100,
        seed=42,
    )

    assert res["promoted"] is True
    assert len(fake_score_calls) == 1
    assert fake_score_calls[0]["overrides"] == {}
    assert fake_score_calls[0]["split"] == "dev"
    assert fake_score_calls[0]["cfg"] is not None


def test_promote_unknown_run_id(repo_root: Path):
    ledger = FakeLedger(entries=[])
    with pytest.raises(PromoteError, match="Candidate run_id 'unknown_run' not found in ledger"):
        promote("unknown_run", ledger=ledger, root=repo_root)


def test_holdout_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import soundthing.events.tuning.promote as promote_mod

    tmp_hdir = tmp_path / "holdout_labels"
    tmp_hdir.mkdir(parents=True, exist_ok=True)
    (tmp_hdir / "clip_01.json").write_text("{}", encoding="utf-8")

    mod = types.ModuleType("soundthing.events.tuning.holdout")
    mod.holdout_labels_dir = lambda: tmp_hdir
    monkeypatch.setitem(sys.modules, "soundthing.events.tuning.holdout", mod)
    if "soundthing.events.tuning" in sys.modules:
        monkeypatch.setattr(sys.modules["soundthing.events.tuning"], "holdout", mod, raising=False)
    monkeypatch.setattr(promote_mod, "holdout", mod, raising=False)

    score_calls = []

    def make_score_fn(cand_res: FakeResult, cur_res: FakeResult):
        def _score(overrides=None, split="dev", cfg=None, best=None, labels_dir=None):
            score_calls.append({
                "overrides": overrides,
                "split": split,
                "cfg": cfg,
                "best": best,
                "labels_dir": labels_dir,
            })
            if overrides:
                return cand_res
            return cur_res
        return _score

    base_cur = FakeResult(run_id="cur", objective=0.70, safety_recall=0.90, fp_per_min=0.25)

    # 1. All pass: cand >= cur objective, cand >= cur safety, cand <= cur fp
    cand_pass = FakeResult(run_id="cand", objective=0.75, safety_recall=0.92, fp_per_min=0.20)
    score_calls.clear()
    assert holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_pass, base_cur)) is True
    assert len(score_calls) == 2
    assert all(c["split"] == "holdout" for c in score_calls)
    assert all(c["labels_dir"] == tmp_hdir for c in score_calls)

    # 2. Objective fail
    cand_obj_fail = FakeResult(run_id="cand", objective=0.69, safety_recall=0.92, fp_per_min=0.20)
    assert holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_obj_fail, base_cur)) is False

    # 3. Safety fail
    cand_safety_fail = FakeResult(run_id="cand", objective=0.75, safety_recall=0.88, fp_per_min=0.20)
    assert holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_safety_fail, base_cur)) is False

    # 4. FP fail
    cand_fp_fail = FakeResult(run_id="cand", objective=0.75, safety_recall=0.92, fp_per_min=0.26)
    assert holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_fp_fail, base_cur)) is False

    # 5. Missing dir -> PromoteError
    tmp_missing = tmp_path / "missing_holdout"
    mod_missing = types.ModuleType("soundthing.events.tuning.holdout")
    mod_missing.holdout_labels_dir = lambda: tmp_missing
    monkeypatch.setitem(sys.modules, "soundthing.events.tuning.holdout", mod_missing)
    if "soundthing.events.tuning" in sys.modules:
        monkeypatch.setattr(sys.modules["soundthing.events.tuning"], "holdout", mod_missing, raising=False)
    monkeypatch.setattr(promote_mod, "holdout", mod_missing, raising=False)
    with pytest.raises(PromoteError, match="holdout set not available"):
        holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_pass, base_cur))

    # 6. Empty dir -> PromoteError
    tmp_empty = tmp_path / "empty_holdout"
    tmp_empty.mkdir(parents=True, exist_ok=True)
    mod_empty = types.ModuleType("soundthing.events.tuning.holdout")
    mod_empty.holdout_labels_dir = lambda: tmp_empty
    monkeypatch.setitem(sys.modules, "soundthing.events.tuning.holdout", mod_empty)
    if "soundthing.events.tuning" in sys.modules:
        monkeypatch.setattr(sys.modules["soundthing.events.tuning"], "holdout", mod_empty, raising=False)
    monkeypatch.setattr(promote_mod, "holdout", mod_empty, raising=False)
    with pytest.raises(PromoteError, match="holdout set not available"):
        holdout_check({"events.postprocess.smoothing_sec": 0.75}, score_fn=make_score_fn(cand_pass, base_cur))



def test_append_tuning_log_formatting(tmp_path: Path):
    log_file = tmp_path / "tuning-log.md"
    log_file.write_text(
        "# Tuning log\n\nFormat:\n\n```\n## YYYY-MM-DD — <what changed>\n```\n\n_No runs yet._\n",
        encoding="utf-8",
    )

    entry = {
        "run_id": "run_test_1",
        "overrides": {"events.postprocess.smoothing_sec": 0.75},
        "objective": 0.825,
        "baseline": 0.65,
        "wrong_env": 0.55,
        "per_clip": {f"clip_{i}": {} for i in range(25)},
    }
    old = {"events.postprocess.smoothing_sec": 0.5}
    ci = {"gain": 0.075, "lo": 0.02, "hi": 0.13}

    append_tuning_log(log_file, entry, old, ci, before=0.75, git_sha="1234567890abcdef")

    content = log_file.read_text(encoding="utf-8")
    assert "_No runs yet._" not in content
    assert "## " in content
    assert "promote run_test_1" in content
    assert "Git: 1234567890    Split: dev (25 clips)" in content
    assert "Params: events.postprocess.smoothing_sec: 0.5000 -> 0.7500" in content
    assert "Dev F0.5: 0.7500 -> 0.8250   (baseline 0.6500, wrong-env 0.5500)" in content
    assert "Bootstrap gain: 0.0750 [0.0200, 0.1300]" in content
    assert "Holdout: pass" in content
    assert "Outcome: promoted" in content

    # Add a second entry, verify it is inserted before the first entry
    entry2 = {
        "run_id": "run_test_2",
        "overrides": {"events.postprocess.smoothing_sec": 0.85},
        "objective": 0.85,
        "baseline": 0.65,
        "wrong_env": 0.55,
        "per_clip": {f"clip_{i}": {} for i in range(25)},
    }
    old2 = {"events.postprocess.smoothing_sec": 0.75}
    ci2 = {"gain": 0.025, "lo": 0.01, "hi": 0.04}

    append_tuning_log(log_file, entry2, old2, ci2, before=0.825, git_sha="abcdef1234567890")

    content2 = log_file.read_text(encoding="utf-8")
    idx1 = content2.index("promote run_test_1")
    idx2 = content2.index("promote run_test_2")
    assert idx2 < idx1


def test_promote_write_failure_refusal(repo_root: Path, monkeypatch: pytest.MonkeyPatch):
    cand_per_clip, best_per_clip = make_per_clip_data(30, fp_cand=0, fp_best=5)

    cand_entry = {
        "run_id": "cand_write_fail",
        "kind": "score",
        "overrides": {"events.postprocess.smoothing_sec": 0.75},
        "objective": 0.88,
        "baseline": 0.70,
        "wrong_env": 0.65,
        "passed": True,
        "per_clip": cand_per_clip,
    }
    best_entry = {
        "run_id": "best_000",
        "kind": "score",
        "objective": 0.72,
        "passed": True,
        "per_clip": best_per_clip,
    }

    ledger = FakeLedger(entries=[best_entry, cand_entry], best_entry=best_entry)

    import soundthing.events.tuning.promote as promote_mod

    def mock_write_overrides(overrides, config_dir):
        raise PromoteError("Config validation failed after editing: simulated test error")

    monkeypatch.setattr(promote_mod, "write_overrides", mock_write_overrides)

    tuning_log_path = repo_root / "src" / "soundthing" / "events" / "references" / "tuning-log.md"
    orig_log_text = tuning_log_path.read_text(encoding="utf-8")

    res = promote(
        "cand_write_fail",
        ledger=ledger,
        holdout_fn=lambda ov: True,
        root=repo_root,
        n_boot=100,
        seed=42,
    )

    # Returns refusal
    assert res["promoted"] is False
    assert res["reason"] == "write failed: Config validation failed after editing: simulated test error"

    # Exactly one promote entry in ledger
    promote_entries = [e for e in ledger.entries if e.get("kind") == "promote"]
    assert len(promote_entries) == 1
    pe = promote_entries[0]
    assert pe["promoted"] is False
    assert pe["reason"] == res["reason"]
    assert pe["holdout_pass"] is True
    assert pe["bootstrap"] is not None

    # set_best was NOT called
    assert ledger.set_best_called is None

    # tuning-log was NOT written
    assert tuning_log_path.read_text(encoding="utf-8") == orig_log_text
