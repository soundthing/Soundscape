"""Tests for shared-change proposal generator (proposals.py)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from soundthing.config import load_config
from soundthing.events.tuning.proposals import propose_shared_change


@dataclass
class FakeResult:
    run_id: str = "run-1234"
    objective: float = 0.8123
    baseline: float = 0.7500
    wrong_env: float = 0.7200
    safety_recall: float = 0.9500
    fp_per_min: float = 1.2345
    grid_f1: float = 0.8800
    passed: bool = True
    limits: dict[str, bool] = field(
        default_factory=lambda: {
            "safety_recall": True,
            "fp_per_min": True,
            "above_baseline": True,
            "wrong_env": True,
        }
    )


def test_propose_shared_change_fake_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Acceptance check 8:
    # Writes a file under a tmp proposals dir (monkeypatch SOUNDTHING_TUNING_HOME=tmp_path;
    # default out_dir resolves under it) with the diff lines, the note and scores
    # from a fake result object and from a dict; result None -> "not scored".
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()
    first_cid = cfg.class_ids[0]

    overrides = {
        f"vocabulary.{first_cid}.clap": ["custom prompt a", "custom prompt b"],
        "priors.floor": 0.05,
    }

    result = FakeResult()
    path = propose_shared_change(overrides, result, "Tuned CLAP prompt and floor", cfg=cfg)

    assert path.exists()
    assert path.parent == tmp_path / "proposals"
    assert path.name.endswith("-run-1234.md")

    text = path.read_text(encoding="utf-8")
    assert "# Shared config proposal" in text
    assert "Run ID: run-1234" in text
    assert "Target files: config/vocabulary.yaml, config/priors.yaml" in text
    assert "## Note" in text
    assert "Tuned CLAP prompt and floor" in text
    assert "## Change" in text
    assert f"- vocabulary.{first_cid}.clap:" in text
    assert f'+ vocabulary.{first_cid}.clap: ["custom prompt a", "custom prompt b"]' in text
    assert "- priors.floor:" in text
    assert "+ priors.floor: 0.05" in text
    assert "## Scores" in text
    assert "0.8123" in text
    assert "0.7500" in text
    assert "0.7200" in text
    assert "0.9500" in text
    assert "1.2345" in text
    assert "0.8800" in text
    assert "Passed: True" in text
    assert "safety_recall: True" in text
    assert "AGENTS.md §2" in text


def test_propose_shared_change_dict_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()
    first_cid = cfg.class_ids[0]

    overrides = {f"vocabulary.{first_cid}.audioset": ["Label 1"]}
    res_dict = asdict(FakeResult(run_id="run-5678"))

    path = propose_shared_change(overrides, res_dict, "Dict score note", cfg=cfg)
    assert path.exists()
    assert path.name.endswith("-run-5678.md")

    text = path.read_text(encoding="utf-8")
    assert "Run ID: run-5678" in text
    assert "Dict score note" in text
    assert "0.8123" in text


def test_propose_shared_change_result_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()
    first_cid = cfg.class_ids[0]

    overrides = {f"vocabulary.{first_cid}.clap": ["prompt"]}
    path = propose_shared_change(overrides, None, "No scoring performed", cfg=cfg)

    assert path.exists()
    assert path.name.endswith("-norun.md")

    text = path.read_text(encoding="utf-8")
    assert "Run ID: norun" in text
    assert "not scored" in text


def test_propose_shared_change_invalid_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Acceptance check 9:
    # Non-shared key or empty overrides -> ValueError; unknown class id -> ValueError.
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    # Empty overrides
    with pytest.raises(ValueError):
        propose_shared_change({}, None, "note", cfg=cfg)

    # Non-shared keys
    with pytest.raises(ValueError):
        propose_shared_change({"events.postprocess.smoothing_sec": 0.5}, None, "note", cfg=cfg)

    with pytest.raises(ValueError):
        propose_shared_change({"thresholds.default": 0.5}, None, "note", cfg=cfg)

    # Unknown class id in vocabulary
    with pytest.raises(ValueError):
        propose_shared_change({"vocabulary.unknown_class_xyz.clap": ["a"]}, None, "note", cfg=cfg)

    # Unknown class id in priors
    first_env = cfg.env_ids[0]
    with pytest.raises(ValueError):
        propose_shared_change(
            {f"priors.{first_env}.classes.unknown_class_xyz": 0.5}, None, "note", cfg=cfg
        )

    # Unknown environment in priors
    with pytest.raises(ValueError):
        propose_shared_change({"priors.unknown_env_xyz.default": 0.5}, None, "note", cfg=cfg)
