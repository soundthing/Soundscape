from __future__ import annotations

from pathlib import Path
import pytest

from soundthing.harness import ledger, paths


@pytest.fixture(autouse=True)
def set_tuning_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))


def test_append_read_round_trip():
    assert ledger.read() == []

    entry1 = {"run_id": "run_01", "objective": 0.85}
    ledger.append(entry1)

    entries = ledger.read()
    assert len(entries) == 1
    read1 = entries[0]
    assert read1["run_id"] == "run_01"
    assert read1["objective"] == 0.85
    assert "ts" in read1
    assert isinstance(read1["ts"], str)


def test_ts_added_if_missing():
    # If ts already provided, it should be kept
    explicit_ts = "2026-01-01T00:00:00+00:00"
    ledger.append({"run_id": "r1", "ts": explicit_ts})

    # If missing, it should be auto-populated
    ledger.append({"run_id": "r2"})

    entries = ledger.read()
    assert len(entries) == 2
    assert entries[0]["ts"] == explicit_ts
    assert "ts" in entries[1]
    assert entries[1]["ts"] != explicit_ts


def test_multiple_appends_keep_order():
    for i in range(10):
        ledger.append({"step": i})

    entries = ledger.read()
    assert len(entries) == 10
    for i, item in enumerate(entries):
        assert item["step"] == i


def test_corrupt_line_raises_value_error():
    ledger.append({"step": 1})
    p = paths.ledger_path()
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n")  # Blank line should be skipped
        f.write("invalid json line\n")
        f.write('{"step": 2}\n')

    with pytest.raises(ValueError, match="line 3"):
        ledger.read()


def test_best_round_trip():
    assert ledger.best() is None

    best_data = {"run_id": "best_run", "objective": 0.92, "git_sha": "abc1234"}
    ledger.set_best(best_data)

    retrieved = ledger.best()
    assert retrieved == best_data

    # Overwrite best
    best_data2 = {"run_id": "better_run", "objective": 0.95}
    ledger.set_best(best_data2)
    assert ledger.best() == best_data2


def test_git_state_worktree():
    state = ledger.git_state()
    assert state["sha"] is not None
    assert len(state["sha"]) == 40
    assert isinstance(state["dirty"], bool)
    assert state["branch"] is None or isinstance(state["branch"], str)


def test_git_state_non_repo(tmp_path: Path):
    non_repo_dir = tmp_path / "not_a_git_repo"
    non_repo_dir.mkdir(parents=True, exist_ok=True)
    state = ledger.git_state(cwd=non_repo_dir)
    assert state == {"sha": None, "dirty": None, "branch": None}
