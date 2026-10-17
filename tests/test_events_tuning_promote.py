"""Tests for events tuning YAML config writer (promote.py)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from soundthing.config import DEFAULT_CONFIG_DIR, load_config
from soundthing.events.tuning.promote import (
    PromoteError,
    format_scalar,
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
