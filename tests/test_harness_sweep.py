"""Tests for automated parameter search and ledger logging in events tuning."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import pytest

from soundthing.harness import paths
from soundthing.harness.sweep import log_result, result_as_dict, sweep


requires_optuna = pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="needs optuna")


@dataclass
class FakeResult:
    """Fake ScoreResult matching TUNING_DESIGN.md and C_CONVENTIONS.md."""

    run_id: str
    objective: float
    baseline: float = 0.5
    wrong_env: float = 0.5
    safety_recall: float = 1.0
    fp_per_min: float = 0.1
    grid_f1: float = 0.8
    per_clip: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, bool] = field(default_factory=dict)
    passed: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)
    split: str = "dev"


class FakeLedger:
    """In-memory fake ledger matching C_CONVENTIONS.md."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def read(self) -> list[dict[str, Any]]:
        return list(self.entries)

    def best(self) -> dict[str, Any] | None:
        return self._best

    def set_best(self, entry: dict[str, Any]) -> None:
        self._best = dict(entry)

    def git_state(self, cwd: Any = None) -> dict[str, Any]:
        return {"sha": "abc1234", "dirty": False, "branch": "test"}


class FakeScoreFn:
    """Deterministic scoring function tracking calls and run_ids."""

    def __init__(
        self,
        center_smoothing: float = 0.76,
        center_min_event: float = 0.3,
        fail_predicate: Any = None,
    ) -> None:
        self.center_smoothing = center_smoothing
        self.center_min_event = center_min_event
        self.fail_predicate = fail_predicate
        self.calls: list[dict[str, Any]] = []
        self._counter = 0

    def __call__(
        self,
        overrides: dict[str, Any] | None = None,
        split: str = "dev",
        cfg: Any = None,
    ) -> FakeResult:
        self._counter += 1
        run_id = f"run_{self._counter:04d}"
        ovr = dict(overrides or {})

        smoothing = float(
            ovr.get(
                "events.postprocess.smoothing_sec",
                cfg.pipeline["events"]["postprocess"]["smoothing_sec"] if cfg else 0.5,
            )
        )
        min_event = float(
            ovr.get(
                "events.postprocess.min_event_sec",
                cfg.pipeline["events"]["postprocess"]["min_event_sec"] if cfg else 0.3,
            )
        )

        obj = float(
            0.5
            - (smoothing - self.center_smoothing) ** 2
            - (min_event - self.center_min_event) ** 2
        )

        passed = True
        if self.fail_predicate is not None:
            passed = not bool(self.fail_predicate(ovr, obj))

        res = FakeResult(
            run_id=run_id,
            objective=obj,
            passed=passed,
            overrides=ovr,
            split=split,
        )
        self.calls.append(ovr)
        return res


def test_result_as_dict():
    """Verify result_as_dict converts dataclasses, objects with __dict__, and dicts."""
    # 1. Dataclass instance
    r = FakeResult(run_id="r1", objective=0.7)
    d1 = result_as_dict(r)
    assert isinstance(d1, dict)
    assert d1["run_id"] == "r1"
    assert d1["objective"] == 0.7

    # 2. Object with __dict__
    class CustomObj:
        def __init__(self):
            self.run_id = "r2"
            self.objective = 0.8

    d2 = result_as_dict(CustomObj())
    assert isinstance(d2, dict)
    assert d2["run_id"] == "r2"
    assert d2["objective"] == 0.8

    # 3. Plain dict
    d3_input = {"run_id": "r3", "objective": 0.9}
    d3 = result_as_dict(d3_input)
    assert d3 == d3_input
    assert d3 is not d3_input  # returns a copy


def test_log_result():
    """Verify log_result creates correct entry, calls ledger.append, and returns it."""
    ledger = FakeLedger()
    res = FakeResult(run_id="run_100", objective=0.65)
    entry = log_result(res, "sweep", ledger)

    assert entry["kind"] == "score"
    assert entry["hypothesis"] == "sweep"
    assert entry["git"] == {"sha": "abc1234", "dirty": False, "branch": "test"}
    assert entry["run_id"] == "run_100"
    assert entry["objective"] == 0.65

    assert len(ledger.entries) == 1
    assert ledger.entries[0] == entry


def test_acceptance_1_coordinate_single_param(cfg, tmp_path, monkeypatch):
    """Check 1: coordinate on smoothing_sec with budget 6: <= 6 calls, logged, best adopted, improved flag."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()
    # Peak at 0.76 ensures 1.0 strictly improves upon incumbent 0.5:
    # 0.5: obj = 0.5 - (0.5 - 0.76)^2 = 0.4324
    # 1.0: obj = 0.5 - (1.0 - 0.76)^2 = 0.4424 (best among [0.0, 1.0, 1.5, 2.0])
    score_fn = FakeScoreFn(center_smoothing=0.76, center_min_event=0.3)

    res = sweep(
        params=["events.postprocess.smoothing_sec"],
        strategy="coordinate",
        score_fn=score_fn,
        budget=6,
        cfg=cfg,
        ledger=ledger,
        n_values=5,
    )

    assert res["strategy"] == "coordinate"
    assert res["evaluations"] <= 6
    assert res["evaluations"] == 6
    assert len(ledger.entries) == 6

    # Every call logged with hypothesis 'sweep', kind 'score', git sha
    for entry in ledger.entries:
        assert entry["kind"] == "score"
        assert entry["hypothesis"] == "sweep"
        assert entry["git"]["sha"] == "abc1234"

    # Best smoothing is 1.0 (the linspace point in [0, 2] closest to 0.75, skipping incumbent 0.5)
    assert res["best_overrides"]["events.postprocess.smoothing_sec"] == 1.0
    assert res["improved"] is True
    assert res["best_objective"] == pytest.approx(0.5 - (1.0 - 0.76) ** 2)


def test_acceptance_2_coordinate_two_params_stops_when_no_improvement(cfg, tmp_path, monkeypatch):
    """Check 2: two params, coordinate: makes second pass while improving; stops when full pass makes no improvement."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()

    # Linspace for smoothing [0.0, 2.0] n_values=5: [0.0, 0.5, 1.0, 1.5, 2.0]. Peak 1.0 is on the grid.
    # Linspace for min_event [0.05, 1.5] n_values=5: [0.05, 0.4125, 0.775, 1.1375, 1.5]. Peak 0.775 is on the grid.
    score_fn = FakeScoreFn(center_smoothing=1.0, center_min_event=0.775)

    res = sweep(
        params=["events.postprocess.smoothing_sec", "events.postprocess.min_event_sec"],
        strategy="coordinate",
        score_fn=score_fn,
        budget=30,
        cfg=cfg,
        ledger=ledger,
        n_values=5,
    )

    # Call breakdown:
    # 1 base eval (smoothing=0.5, min_event=0.3)
    # Pass 1:
    #   smoothing candidates (0.5 skipped): [0.0, 1.0, 1.5, 2.0] -> 4 calls. Adopts 1.0.
    #   min_event candidates (0.3 not in grid): [0.05, 0.4125, 0.775, 1.1375, 1.5] -> 5 calls. Adopts 0.775.
    # Pass 2:
    #   smoothing candidates (1.0 skipped): [0.0, 0.5, 1.5, 2.0] -> 4 calls. None improves.
    #   min_event candidates (0.775 skipped): [0.05, 0.4125, 1.1375, 1.5] -> 4 calls. None improves.
    # Total calls: 1 + 4 + 5 + 4 + 4 = 18 calls.
    assert res["evaluations"] == 18
    assert len(ledger.entries) == 18
    assert res["improved"] is True
    assert res["best_overrides"] == {
        "events.postprocess.smoothing_sec": 1.0,
        "events.postprocess.min_event_sec": 0.775,
    }
    assert res["best_objective"] == pytest.approx(0.5)


@requires_optuna
def test_acceptance_3_unpassed_candidates_never_adopted(cfg, tmp_path, monkeypatch):
    """Check 3: candidates that do not pass limits are never adopted even if objective is higher."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()

    # Fail any candidate with smoothing == 1.0
    def fail_smoothing_1(ovr, obj):
        return ovr.get("events.postprocess.smoothing_sec") == 1.0

    score_fn = FakeScoreFn(
        center_smoothing=1.0, center_min_event=0.3, fail_predicate=fail_smoothing_1
    )

    res = sweep(
        params=["events.postprocess.smoothing_sec"],
        strategy="coordinate",
        score_fn=score_fn,
        budget=10,
        cfg=cfg,
        ledger=ledger,
        n_values=5,
    )

    # Candidate 1.0 had highest objective (0.5), but passed=False, so must not be adopted.
    assert res["best_overrides"].get("events.postprocess.smoothing_sec") != 1.0

    # Also test with optuna: all trials fail
    def all_fail(ovr, obj):
        return True

    fail_score_fn = FakeScoreFn(fail_predicate=all_fail)
    res_optuna = sweep(
        params=["events.postprocess.smoothing_sec"],
        strategy="optuna",
        score_fn=fail_score_fn,
        budget=5,
        cfg=cfg,
        ledger=ledger,
        seed=42,
    )
    assert res_optuna["improved"] is False
    assert res_optuna["best_overrides"] == {}


@requires_optuna
def test_acceptance_4_int_param_max_on_screen(cfg, tmp_path, monkeypatch):
    """Check 4: int param events.postprocess.max_on_screen suggested/linspace values are ints, deduped."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()
    score_fn = FakeScoreFn()

    res = sweep(
        params=["events.postprocess.max_on_screen"],
        strategy="coordinate",
        score_fn=score_fn,
        budget=10,
        cfg=cfg,
        ledger=ledger,
        n_values=5,
    )

    for item in res["history"]:
        if "events.postprocess.max_on_screen" in item["overrides"]:
            val = item["overrides"]["events.postprocess.max_on_screen"]
            assert isinstance(val, int)
            assert not isinstance(val, bool)
            assert type(val) is int

    # Optuna int suggestion
    res_opt = sweep(
        params=["events.postprocess.max_on_screen"],
        strategy="optuna",
        score_fn=score_fn,
        budget=5,
        cfg=cfg,
        ledger=ledger,
        seed=1,
    )

    for item in res_opt["history"]:
        if "events.postprocess.max_on_screen" in item["overrides"]:
            val = item["overrides"]["events.postprocess.max_on_screen"]
            assert isinstance(val, int)
            assert not isinstance(val, bool)
            assert type(val) is int


@requires_optuna
def test_acceptance_5_optuna_budget_and_determinism(cfg, tmp_path, monkeypatch):
    """Check 5: optuna with budget 8 and seed 0: exactly 8 score calls, values within bounds, deterministic."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))

    ledger1 = FakeLedger()
    score_fn1 = FakeScoreFn(center_smoothing=0.8, center_min_event=0.4)
    res1 = sweep(
        params=["events.postprocess.smoothing_sec", "events.postprocess.min_event_sec"],
        strategy="optuna",
        score_fn=score_fn1,
        budget=8,
        cfg=cfg,
        ledger=ledger1,
        seed=0,
    )

    assert res1["evaluations"] == 8
    assert len(ledger1.entries) == 8
    assert len(res1["history"]) == 8

    # Check bounds
    for item in res1["history"]:
        ovr = item["overrides"]
        if "events.postprocess.smoothing_sec" in ovr:
            assert 0.0 <= ovr["events.postprocess.smoothing_sec"] <= 2.0
        if "events.postprocess.min_event_sec" in ovr:
            assert 0.05 <= ovr["events.postprocess.min_event_sec"] <= 1.5

    # Run 2 with identical seed
    ledger2 = FakeLedger()
    score_fn2 = FakeScoreFn(center_smoothing=0.8, center_min_event=0.4)
    res2 = sweep(
        params=["events.postprocess.smoothing_sec", "events.postprocess.min_event_sec"],
        strategy="optuna",
        score_fn=score_fn2,
        budget=8,
        cfg=cfg,
        ledger=ledger2,
        seed=0,
    )

    assert res1["best_overrides"] == res2["best_overrides"]
    assert res1["best_objective"] == pytest.approx(res2["best_objective"])

    for h1, h2 in zip(res1["history"], res2["history"]):
        assert h1["overrides"] == h2["overrides"]
        assert h1["objective"] == pytest.approx(h2["objective"])


def test_acceptance_6_error_cases(cfg, tmp_path, monkeypatch):
    """Check 6: unknown param, shared param, strategy 'grid', budget 0, base_overrides with shared key -> ValueError."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()
    score_fn = FakeScoreFn()

    # Unknown param
    with pytest.raises(ValueError, match="Unknown parameter|not in search space"):
        sweep(["events.postprocess.nonexistent_param"], score_fn=score_fn, cfg=cfg, ledger=ledger)

    # Shared param: priors.floor
    with pytest.raises(ValueError, match="shared"):
        sweep(["priors.floor"], score_fn=score_fn, cfg=cfg, ledger=ledger)

    # Shared param: vocabulary
    cid0 = cfg.class_ids[0]
    with pytest.raises(ValueError, match="shared"):
        sweep([f"vocabulary.{cid0}.clap"], score_fn=score_fn, cfg=cfg, ledger=ledger)

    # Unknown strategy 'grid'
    with pytest.raises(ValueError, match="Unknown strategy 'grid'"):
        sweep(
            ["events.postprocess.smoothing_sec"],
            strategy="grid",
            score_fn=score_fn,
            cfg=cfg,
            ledger=ledger,
        )

    # Budget 0
    with pytest.raises(ValueError, match="budget must be >= 1"):
        sweep(
            ["events.postprocess.smoothing_sec"],
            budget=0,
            score_fn=score_fn,
            cfg=cfg,
            ledger=ledger,
        )

    # Empty params
    with pytest.raises(ValueError, match="params must not be empty"):
        sweep([], score_fn=score_fn, cfg=cfg, ledger=ledger)

    # Base_overrides with shared key
    with pytest.raises(ValueError, match="base_overrides contains shared key"):
        sweep(
            ["events.postprocess.smoothing_sec"],
            base_overrides={"priors.floor": 0.2},
            score_fn=score_fn,
            cfg=cfg,
            ledger=ledger,
        )


@requires_optuna
def test_acceptance_7_budget_1(cfg, tmp_path, monkeypatch):
    """Check 7: budget 1 -> only the base evaluation, improved False."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    ledger = FakeLedger()
    score_fn = FakeScoreFn()

    res = sweep(
        params=["events.postprocess.smoothing_sec"],
        strategy="coordinate",
        score_fn=score_fn,
        budget=1,
        base_overrides={"events.postprocess.min_event_sec": 0.5},
        cfg=cfg,
        ledger=ledger,
    )

    assert res["evaluations"] == 1
    assert res["improved"] is False
    assert len(res["history"]) == 1
    assert res["best_overrides"] == {"events.postprocess.min_event_sec": 0.5}
    assert res["best_run_id"] == res["history"][0]["run_id"]
    assert res["best_objective"] == res["history"][0]["objective"]
    assert len(ledger.entries) == 1

    # Also for optuna
    ledger_opt = FakeLedger()
    res_opt = sweep(
        params=["events.postprocess.smoothing_sec"],
        strategy="optuna",
        score_fn=score_fn,
        budget=1,
        cfg=cfg,
        ledger=ledger_opt,
    )
    assert res_opt["evaluations"] == 1
    assert res_opt["improved"] is False
    assert len(res_opt["history"]) == 1
