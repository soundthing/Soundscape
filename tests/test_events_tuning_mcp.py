"""Tests for events tuning MCP server."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from soundthing.events.tuning.mcp_server import TuningTools, build_server


def make_fake_per_clip(n: int = 10, tp: int = 5) -> dict[str, Any]:
    return {
        f"clip_{i:02d}": {
            "conditioned": {
                "tp_fine": tp,
                "tp_coarse": 1,
                "fp": 1,
                "fn": 1,
                "minutes": 1.0,
                "safety_tp": 1,
                "safety_total": 1,
            },
            "baseline": {
                "tp_fine": 3,
                "tp_coarse": 1,
                "fp": 2,
                "fn": 2,
                "minutes": 1.0,
                "safety_tp": 1,
                "safety_total": 1,
            },
            "wrong_env": {
                "tp_fine": 3,
                "tp_coarse": 1,
                "fp": 2,
                "fn": 2,
                "minutes": 1.0,
                "safety_tp": 1,
                "safety_total": 1,
            },
        }
        for i in range(n)
    }


@dataclass
class FakeResult:
    run_id: str = "run_0001"
    objective: float = 0.8
    baseline: float = 0.5
    wrong_env: float = 0.5
    safety_recall: float = 1.0
    fp_per_min: float = 0.1
    grid_f1: float = 0.8
    per_clip: dict[str, Any] = field(default_factory=lambda: make_fake_per_clip(10, tp=5))
    diagnostics: dict[str, Any] = field(default_factory=lambda: {
        "per_class": {"siren": {"tp": 2, "fp": 0, "fn": 0}},
        "top_false_positives": [{"video_id": "clip_01", "start": 1.0, "class_id": "siren", "confidence": 0.8}],
    })
    limits: dict[str, bool] = field(default_factory=lambda: {
        "safety_recall": True,
        "fp_per_min": True,
        "above_baseline": True,
        "wrong_env": True,
    })
    passed: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)
    split: str = "dev"


class FakeLedger:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None

    def append(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def read(self) -> list[dict[str, Any]]:
        return list(self.entries)

    def best(self) -> dict[str, Any] | None:
        return self._best

    def set_best(self, entry: dict[str, Any] | None) -> None:
        self._best = dict(entry) if entry is not None else None

    def git_state(self, cwd: Any = None) -> dict[str, Any]:
        return {"sha": "abc1234", "dirty": False, "branch": "test"}


class FakeScoreFn:
    def __init__(self, result: FakeResult | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._counter = 0
        self.result = result

    def __call__(
        self,
        overrides: dict[str, Any] | None = None,
        split: str = "dev",
        cfg: Any = None,
        best: Any = None,
        labels_dir: Any = None,
    ) -> FakeResult:
        self._counter += 1
        self.calls.append({
            "overrides": overrides,
            "split": split,
            "cfg": cfg,
            "best": best,
            "labels_dir": labels_dir,
        })
        if self.result is not None:
            return self.result
        return FakeResult(
            run_id=f"run_{self._counter:04d}",
            objective=0.75,
            per_clip=make_fake_per_clip(10, tp=5),
            diagnostics={
                "per_class": {"siren": {"tp": 2, "fp": 0, "fn": 0}},
                "top_false_positives": [{"video_id": "clip_01", "start": 1.0, "class_id": "siren", "confidence": 0.8}],
            },
            overrides=dict(overrides or {}),
            split=split,
        )


def make_fake_sweep_fn():
    calls: list[dict[str, Any]] = []

    def fake_sweep(
        params: list[str],
        strategy: str = "coordinate",
        score_fn: Any = None,
        budget: int = 20,
        *,
        cfg: Any = None,
        ledger: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append({
            "params": params,
            "strategy": strategy,
            "score_fn": score_fn,
            "budget": budget,
            "cfg": cfg,
            "ledger": ledger,
            "kwargs": kwargs,
        })
        return {
            "strategy": strategy,
            "evaluations": 1,
            "best_overrides": {},
            "best_objective": 0.8,
            "best_run_id": "run_0001",
            "improved": False,
            "history": [],
        }

    fake_sweep.calls = calls  # type: ignore[attr-defined]
    return fake_sweep


def make_fake_promote_fn(return_val: dict[str, Any] | None = None):
    calls: list[dict[str, Any]] = []

    def fake_promote(run_id: str, *, ledger: Any = None, score_fn: Any = None, cfg: Any = None, **kwargs: Any) -> dict[str, Any]:
        calls.append({
            "run_id": run_id,
            "ledger": ledger,
            "score_fn": score_fn,
            "cfg": cfg,
            "kwargs": kwargs,
        })
        if return_val is not None:
            return dict(return_val)
        return {"promoted": True, "holdout": "pass"}

    fake_promote.calls = calls  # type: ignore[attr-defined]
    return fake_promote


def make_fake_propose_fn(tmp_path: Path):
    calls: list[dict[str, Any]] = []

    def fake_propose(overrides: dict[str, Any], result: Any, note: str, *, cfg: Any = None, out_dir: Any = None) -> Path:
        calls.append({"overrides": overrides, "result": result, "note": note, "cfg": cfg})
        prop_file = tmp_path / "proposals" / "test_proposal.md"
        prop_file.parent.mkdir(parents=True, exist_ok=True)
        prop_file.write_text("test proposal", encoding="utf-8")
        return prop_file

    fake_propose.calls = calls  # type: ignore[attr-defined]
    return fake_propose


def make_fake_recache_fn():
    calls: list[dict[str, Any]] = []

    def fake_recache(window: float, hop: float, model: str) -> dict[str, Any]:
        calls.append({"window": window, "hop": hop, "model": model})
        return {"kernel_id": "soundthing/cache-builder", "status": "queued"}

    fake_recache.calls = calls  # type: ignore[attr-defined]
    return fake_recache


@pytest.fixture
def tuning_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg: Any):
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    ledger = FakeLedger()
    score_fn = FakeScoreFn()
    sweep_fn = make_fake_sweep_fn()
    promote_fn = make_fake_promote_fn()
    propose_fn = make_fake_propose_fn(tmp_path)
    recache_fn = make_fake_recache_fn()

    tools = TuningTools(
        score_fn=score_fn,
        ledger=ledger,
        cfg_loader=lambda: cfg,
        sweep_fn=sweep_fn,
        promote_fn=promote_fn,
        propose_fn=propose_fn,
        recache_fn=recache_fn,
    )
    return {
        "tools": tools,
        "ledger": ledger,
        "score_fn": score_fn,
        "sweep_fn": sweep_fn,
        "promote_fn": promote_fn,
        "propose_fn": propose_fn,
        "recache_fn": recache_fn,
        "cfg": cfg,
        "tmp_path": tmp_path,
    }


def test_1_list_tools_names(tuning_env: dict[str, Any]):
    """1. asyncio.run(build_server(tools).list_tools()) names == the 8 tool names exactly."""
    tools = tuning_env["tools"]
    server = build_server(tools)
    tool_list = asyncio.run(server.list_tools())
    names = [t.name for t in tool_list]
    expected = [
        "describe_space",
        "get_best",
        "get_ledger",
        "score",
        "sweep",
        "promote",
        "propose_shared_change",
        "request_recache",
    ]
    assert sorted(names) == sorted(expected)
    assert len(names) == 8


def test_2_every_tool_call_logs_mcp_call(tuning_env: dict[str, Any]):
    """2. Every tool call appends one mcp_call entry (including failing call)."""
    tools = tuning_env["tools"]
    ledger = tuning_env["ledger"]
    cfg = tuning_env["cfg"]
    cid = cfg.class_ids[0]

    # 1. describe_space
    assert len(ledger.entries) == 0
    tools.describe_space()
    assert len([e for e in ledger.entries if e.get("kind") == "mcp_call"]) == 1
    assert ledger.entries[-1]["tool"] == "describe_space"
    assert ledger.entries[-1]["ok"] is True
    assert ledger.entries[-1]["error"] is None

    # 2. get_best
    tools.get_best()
    assert len([e for e in ledger.entries if e.get("kind") == "mcp_call"]) == 2
    assert ledger.entries[-1]["tool"] == "get_best"
    assert ledger.entries[-1]["ok"] is True

    # 3. get_ledger
    tools.get_ledger(limit=10)
    assert len([e for e in ledger.entries if e.get("kind") == "mcp_call"]) == 3
    assert ledger.entries[-1]["tool"] == "get_ledger"
    assert ledger.entries[-1]["args"] == {"limit": 10}
    assert ledger.entries[-1]["ok"] is True

    # 4. score (successful)
    tools.score(overrides={"events.postprocess.smoothing_sec": 0.5}, hypothesis="test")
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 4
    assert mcp_calls[-1]["tool"] == "score"
    assert mcp_calls[-1]["ok"] is True
    assert mcp_calls[-1]["args"]["overrides"] == {"events.postprocess.smoothing_sec": 0.5}

    # 4b. score (failing: unknown key)
    with pytest.raises(ValueError) as exc_info:
        tools.score(overrides={"unknown_param": 1.0})
    assert "unknown_param" in str(exc_info.value)
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 5
    assert mcp_calls[-1]["tool"] == "score"
    assert mcp_calls[-1]["ok"] is False
    assert mcp_calls[-1]["error"] is not None
    assert "unknown_param" in mcp_calls[-1]["error"]

    # 5. sweep
    tools.sweep(params=["events.postprocess.smoothing_sec"], budget=5)
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 6
    assert mcp_calls[-1]["tool"] == "sweep"
    assert mcp_calls[-1]["ok"] is True

    # 6. promote
    tools.promote("run_0001")
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 7
    assert mcp_calls[-1]["tool"] == "promote"
    assert mcp_calls[-1]["args"] == {"run_id": "run_0001"}
    assert mcp_calls[-1]["ok"] is True

    # 7. propose_shared_change
    tools.propose_shared_change(overrides={f"vocabulary.{cid}.clap": ["alert sound"]}, note="proposal note")
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 8
    assert mcp_calls[-1]["tool"] == "propose_shared_change"
    assert mcp_calls[-1]["ok"] is True

    # 8. request_recache
    tools.request_recache(window=1.0, hop=0.5, model="panns")
    mcp_calls = [e for e in ledger.entries if e.get("kind") == "mcp_call"]
    assert len(mcp_calls) == 9
    assert mcp_calls[-1]["tool"] == "request_recache"
    assert mcp_calls[-1]["ok"] is True


def test_3_score_rules_and_rejections(tuning_env: dict[str, Any]):
    """3. score: rejects eval.*, events.backend, window_sec; split='dev'; reply no per_clip; bootstrap None/dict; shared True; logged kind 'score'."""
    tools = tuning_env["tools"]
    ledger = tuning_env["ledger"]
    score_fn = tuning_env["score_fn"]
    cfg = tuning_env["cfg"]
    cid = cfg.class_ids[0]

    # Rejects eval.onset_tolerance_sec
    with pytest.raises(ValueError):
        tools.score(overrides={"eval.onset_tolerance_sec": 0.5})

    # Rejects events.backend
    with pytest.raises(ValueError):
        tools.score(overrides={"events.backend": "real"})

    # Rejects events.window_sec
    with pytest.raises(ValueError):
        tools.score(overrides={"events.window_sec": 1.0})

    # Calls score_fn with split="dev"
    assert len(score_fn.calls) == 0
    reply = tools.score(overrides={"events.postprocess.smoothing_sec": 0.4}, hypothesis="tune smoothing")
    assert len(score_fn.calls) == 1
    assert score_fn.calls[0]["split"] == "dev"

    # Reply has no per_clip
    assert "per_clip" not in reply

    # Bootstrap None without best
    assert reply["bootstrap"] is None

    # Candidate logged with kind "score"
    score_entries = [e for e in ledger.entries if e.get("kind") == "score"]
    assert len(score_entries) == 1
    assert score_entries[0]["hypothesis"] == "tune smoothing"

    # Dict with accept when a best exists
    ledger.set_best({
        "run_id": "run_best",
        "objective": 0.70,
        "per_clip": make_fake_per_clip(10, tp=4),
    })
    reply_with_best = tools.score(overrides={"events.postprocess.smoothing_sec": 0.4})
    assert reply_with_best["bootstrap"] is not None
    assert isinstance(reply_with_best["bootstrap"], dict)
    assert "accept" in reply_with_best["bootstrap"]
    assert isinstance(reply_with_best["bootstrap"]["accept"], bool)
    assert "gain" in reply_with_best["bootstrap"]

    # Shared overrides flagged shared True
    reply_shared = tools.score(overrides={f"vocabulary.{cid}.clap": ["alert sound"]})
    assert reply_shared["shared"] is True
    assert "use propose_shared_change; promote will refuse" in reply_shared.get("note", "")


def test_4_get_ledger_and_get_best(tuning_env: dict[str, Any]):
    """4. get_ledger limit bounds and stripping of per_clip/diagnostics; get_best strips per_clip."""
    tools = tuning_env["tools"]
    ledger = tuning_env["ledger"]

    # Limit bounds for get_ledger
    with pytest.raises(ValueError):
        tools.get_ledger(limit=0)
    with pytest.raises(ValueError):
        tools.get_ledger(limit=201)

    # Populate ledger with entries having per_clip and diagnostics
    ledger.append({
        "kind": "score",
        "run_id": "run_01",
        "objective": 0.8,
        "per_clip": {"clip_01": {}},
        "diagnostics": {"per_class": {}},
    })
    ledger.append({
        "kind": "score",
        "run_id": "run_02",
        "objective": 0.82,
        "per_clip": {"clip_02": {}},
        "diagnostics": {"per_class": {}},
    })

    entries = tools.get_ledger(limit=5)
    assert len(entries) >= 2
    for entry in entries:
        assert "per_clip" not in entry
        assert "diagnostics" not in entry

    # get_best strips per_clip but preserves diagnostics
    assert tools.get_best() is None
    ledger.set_best({
        "run_id": "run_best",
        "objective": 0.85,
        "per_clip": {"clip_01": {}},
        "diagnostics": {"per_class": {"siren": {"tp": 5}}},
    })
    best = tools.get_best()
    assert best is not None
    assert "per_clip" not in best
    assert "diagnostics" in best
    assert best["run_id"] == "run_best"


def test_5_sweep_budget_and_passthrough(tuning_env: dict[str, Any]):
    """5. sweep budget cap -> ValueError above max_sweep_budget; passes ledger/score_fn through to sweep_fn."""
    tools = tuning_env["tools"]
    sweep_fn = tuning_env["sweep_fn"]

    # Budget cap
    with pytest.raises(ValueError):
        tools.sweep(params=["events.postprocess.smoothing_sec"], budget=201)
    with pytest.raises(ValueError):
        tools.sweep(params=["events.postprocess.smoothing_sec"], budget=0)

    # Valid sweep call passes ledger and score_fn
    res = tools.sweep(params=["events.postprocess.smoothing_sec"], strategy="coordinate", budget=15)
    assert isinstance(res, dict)
    assert len(sweep_fn.calls) == 1
    assert sweep_fn.calls[0]["budget"] == 15
    assert sweep_fn.calls[0]["strategy"] == "coordinate"
    assert sweep_fn.calls[0]["score_fn"] == tools.score_fn
    assert sweep_fn.calls[0]["ledger"] == tools.ledger


def test_6_promote_holdout_filtering(tuning_env: dict[str, Any]):
    """6. promote: a fake promote_fn returning {"promoted": True, "holdout": "pass", "holdout_score": 0.9} -> reply has no 'holdout_score'."""
    tools = tuning_env["tools"]
    fake_promote = make_fake_promote_fn(return_val={
        "promoted": True,
        "holdout": "pass",
        "holdout_score": 0.9,
    })
    tools.promote_fn = fake_promote

    reply = tools.promote("run_0001")
    assert "holdout_score" not in reply
    assert reply["promoted"] is True
    assert reply["holdout"] == "pass"


def test_7_propose_shared_change(tuning_env: dict[str, Any]):
    """7. propose_shared_change: path returned, propose_fn got a result and the note, proposal entry logged."""
    tools = tuning_env["tools"]
    ledger = tuning_env["ledger"]
    propose_fn = tuning_env["propose_fn"]
    cfg = tuning_env["cfg"]
    cid = cfg.class_ids[0]

    ovr = {f"vocabulary.{cid}.clap": ["alert sound"]}
    note = f"Testing clap prompt refinement for {cid}"
    reply = tools.propose_shared_change(overrides=ovr, note=note)

    assert "path" in reply
    assert "run_id" in reply
    assert isinstance(reply["path"], str)

    # Check propose_fn received arguments
    assert len(propose_fn.calls) == 1
    assert propose_fn.calls[0]["note"] == note
    assert propose_fn.calls[0]["result"] is not None
    assert propose_fn.calls[0]["overrides"] == ovr

    # Proposal entry logged in ledger
    prop_entries = [e for e in ledger.entries if e.get("kind") == "proposal"]
    assert len(prop_entries) == 1
    assert prop_entries[0]["path"] == reply["path"]
    assert prop_entries[0]["run_id"] == reply["run_id"]


def test_8_request_recache(tuning_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, cfg: Any):
    """8. request_recache: fake recache_fn called; bad hop -> ValueError; with no recache_fn and kaggle_io is None -> RuntimeError."""
    tools = tuning_env["tools"]
    recache_fn = tuning_env["recache_fn"]

    # Fake recache_fn called
    reply = tools.request_recache(window=1.0, hop=0.5, model="panns")
    assert reply.get("status") == "queued"
    assert len(recache_fn.calls) == 1
    assert recache_fn.calls[0] == {"window": 1.0, "hop": 0.5, "model": "panns"}

    # Validation errors
    with pytest.raises(ValueError):
        tools.request_recache(window=1.0, hop=2.0, model="panns")
    with pytest.raises(ValueError):
        tools.request_recache(window=1.0, hop=0.0, model="panns")
    with pytest.raises(ValueError):
        tools.request_recache(window=-0.5, hop=0.2, model="panns")
    with pytest.raises(ValueError):
        tools.request_recache(window=1.0, hop=0.5, model="")

    # With no recache_fn and kaggle_io set to None in sys.modules -> RuntimeError mentioning kaggle_io
    tools_no_recache = TuningTools(ledger=FakeLedger(), cfg_loader=lambda: cfg)
    monkeypatch.setitem(sys.modules, "soundthing.events.tuning.kaggle_io", None)
    with pytest.raises(RuntimeError) as exc_info:
        tools_no_recache.request_recache(window=1.0, hop=0.5, model="panns")
    assert "kaggle_io" in str(exc_info.value)
