"""Tests for events tuning CLI (__main__.py)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from soundthing.harness.__main__ import main
from soundthing.harness.promote import PromoteError


class FakeTools:
    """Fake TuningTools that records method calls and returns predictable dicts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.promote_result: dict[str, Any] = {"promoted": False, "run_id": "run_001"}
        self.should_raise: Exception | None = None

    def score(self, overrides: dict[str, Any] | None = None, hypothesis: str = "", component: str = "events", **kwargs: Any) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("score", {"overrides": overrides, "hypothesis": hypothesis}))
        return {"run_id": "run_test", "objective": 0.8, "passed": True}

    def sweep(self, params: list[str], strategy: str = "coordinate", budget: int = 20, component: str = "events", **kwargs: Any) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("sweep", {"params": params, "strategy": strategy, "budget": budget}))
        return {"best_params": params, "best_objective": 0.85}

    def promote(self, run_id: str, component: str = "events", **kwargs: Any) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("promote", {"run_id": run_id}))
        return dict(self.promote_result)

    def propose_shared_change(self, overrides: dict[str, Any], note: str) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("propose_shared_change", {"overrides": overrides, "note": note}))
        return {"path": "proposals/test.md", "run_id": "run_prop"}

    def describe_space(self, component: str = "events", **kwargs: Any) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("describe_space", {}))
        return {"params": ["events.postprocess.smoothing_sec"]}

    def get_best(self, component: str = "events", **kwargs: Any) -> dict[str, Any] | None:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("get_best", {}))
        return {"run_id": "run_best", "objective": 0.9}

    def get_ledger(self, limit: int = 20, component: str = "events", **kwargs: Any) -> list[dict[str, Any]]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("get_ledger", {"limit": limit}))
        return [{"run_id": f"run_{i}"} for i in range(min(limit, 3))]

    def request_recache(self, window: float, hop: float, model: str) -> dict[str, Any]:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(("request_recache", {"window": window, "hop": hop, "model": model}))
        return {"status": "submitted", "kernel": "test"}


def test_cli_score_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()
    code = main(
        [
            "score",
            "--set",
            "events.postprocess.smoothing_sec=0.75",
            "--set",
            "thresholds.default=0.6",
            "--hypothesis",
            "h",
        ],
        tools=fake,
    )
    assert code == 0
    assert len(fake.calls) == 1
    call_name, args = fake.calls[0]
    assert call_name == "score"
    assert args["overrides"] == {
        "events.postprocess.smoothing_sec": 0.75,
        "thresholds.default": 0.6,
    }
    assert args["hypothesis"] == "h"
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["run_id"] == "run_test"


def test_cli_overrides_merged_with_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()
    code = main(
        ["score", "--overrides", '{"a": 1, "b": 10}', "--set", "a=2"],
        tools=fake,
    )
    assert code == 0
    assert fake.calls[0][1]["overrides"] == {"a": 2, "b": 10}
    out = capsys.readouterr().out
    assert json.loads(out) == {"run_id": "run_test", "objective": 0.8, "passed": True}


def test_cli_sweep_params_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()
    code = main(
        ["sweep", "--params", "a,b", "--strategy", "optuna", "--budget", "50"],
        tools=fake,
    )
    assert code == 0
    call_name, args = fake.calls[0]
    assert call_name == "sweep"
    assert args["params"] == ["a", "b"]
    assert args["strategy"] == "optuna"
    assert args["budget"] == 50
    out = capsys.readouterr().out
    assert json.loads(out)["best_params"] == ["a", "b"]


def test_cli_promote_return_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()

    # 1. Not promoted -> returns 1, prints dict
    fake.promote_result = {"promoted": False, "run_id": "r1"}
    code = main(["promote", "r1"], tools=fake)
    assert code == 1
    out = capsys.readouterr().out
    assert json.loads(out) == {"promoted": False, "run_id": "r1"}

    # 2. Promoted -> returns 0, prints dict
    fake.promote_result = {"promoted": True, "run_id": "r2"}
    code = main(["promote", "r2"], tools=fake)
    assert code == 0
    out = capsys.readouterr().out
    assert json.loads(out) == {"promoted": True, "run_id": "r2"}


def test_cli_tool_exception_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()

    fake.should_raise = ValueError("invalid value test")
    code = main(["score", "--hypothesis", "test"], tools=fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "error: invalid value test" in err

    fake.should_raise = KeyError("missing_key")
    code = main(["space"], tools=fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "error: 'missing_key'" in err

    fake.should_raise = RuntimeError("runtime failure")
    code = main(["space"], tools=fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "error: runtime failure" in err


    fake.should_raise = PromoteError("promotion failed reason")
    code = main(["promote", "r_fail"], tools=fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "error: promotion failed reason" in err


def test_cli_build_dev_missing_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    import soundthing.harness

    monkeypatch.setitem(sys.modules, "soundthing.harness.datasets", None)
    monkeypatch.delattr(soundthing.harness, "datasets", raising=False)

    fake = FakeTools()
    code = main(["build-dev", "--esc50", str(tmp_path / "esc50")], tools=fake)
    assert code == 2
    err = capsys.readouterr().err
    assert "error: build-dev needs soundthing.harness.datasets (Track B), not available in this tree" in err


def test_cli_build_dev_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake_datasets = types.ModuleType("soundthing.harness.datasets")
    calls: list[tuple[str, Any]] = []

    def fake_esc50_mixes(esc50_dir: Path, cfg: Any, n_per_env: int, seed: int, out_dir: Path) -> list[Path]:
        calls.append(("esc50_mixes", (esc50_dir, n_per_env, seed, out_dir)))
        return [out_dir / "labels" / "clip1.json", out_dir / "labels" / "clip2.json"]

    def fake_build_dev_caches(audio_dir: Path, labels_dir: Path, cfg: Any, **kwargs: Any) -> list[Path]:
        calls.append(("build_dev_caches", (audio_dir, labels_dir)))
        return [Path("c1.npz"), Path("c2.npz")]

    fake_datasets.esc50_mixes = fake_esc50_mixes  # type: ignore[attr-defined]
    fake_datasets.build_dev_caches = fake_build_dev_caches  # type: ignore[attr-defined]

    import soundthing.harness

    monkeypatch.setitem(sys.modules, "soundthing.harness.datasets", fake_datasets)
    monkeypatch.setattr(soundthing.harness, "datasets", fake_datasets, raising=False)

    esc50_dir = tmp_path / "esc50"
    esc50_dir.mkdir()
    out_dir = tmp_path / "custom_dev"

    code = main(
        [
            "build-dev",
            "--esc50",
            str(esc50_dir),
            "--n-per-env",
            "10",
            "--seed",
            "42",
            "--out",
            str(out_dir),
        ],
    )
    assert code == 0
    assert len(calls) == 2
    assert calls[0][0] == "esc50_mixes"
    assert calls[0][1] == (esc50_dir, 10, 42, out_dir)
    assert calls[1][0] == "build_dev_caches"
    assert calls[1][1] == (out_dir / "audio", out_dir / "labels")
    out = capsys.readouterr().out
    counts = json.loads(out)
    assert counts == {"mixes": 2, "caches": 2}


def test_cli_routing_space_best_ledger_propose_recache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    fake = FakeTools()

    # space
    code = main(["space"], tools=fake)
    assert code == 0
    assert fake.calls[-1] == ("describe_space", {})
    assert json.loads(capsys.readouterr().out)["params"] == ["events.postprocess.smoothing_sec"]

    # best
    code = main(["best"], tools=fake)
    assert code == 0
    assert fake.calls[-1] == ("get_best", {})
    assert json.loads(capsys.readouterr().out)["run_id"] == "run_best"

    # ledger --limit 5
    code = main(["ledger", "--limit", "5"], tools=fake)
    assert code == 0
    assert fake.calls[-1] == ("get_ledger", {"limit": 5})
    assert len(json.loads(capsys.readouterr().out)) == 3

    # propose --note
    code = main(
        ["propose", "--set", "vocabulary.siren.clap=['alarm']", "--note", "clarify siren"],
        tools=fake,
    )
    assert code == 0
    assert fake.calls[-1] == (
        "propose_shared_change",
        {"overrides": {"vocabulary.siren.clap": ["alarm"]}, "note": "clarify siren"},
    )
    assert json.loads(capsys.readouterr().out)["run_id"] == "run_prop"

    # recache
    code = main(
        ["recache", "--window", "1.0", "--hop", "0.5", "--model", "panns"],
        tools=fake,
    )
    assert code == 0
    assert fake.calls[-1] == (
        "request_recache",
        {"window": 1.0, "hop": 0.5, "model": "panns"},
    )
    assert json.loads(capsys.readouterr().out)["status"] == "submitted"


def test_claude_md_content() -> None:
    claude_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "soundthing"
        / "harness"
        / "CLAUDE.md"
    )
    assert claude_path.is_file(), f"CLAUDE.md not found at {claude_path}"
    content = claude_path.read_text(encoding="utf-8")
    assert content == "@AGENTS.md\n"
