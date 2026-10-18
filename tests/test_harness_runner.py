"""Tests for experiment runner, worktree isolation, bootstrap gating, and code promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from soundthing.harness import paths
from soundthing.harness import runner as runner_module
from soundthing.harness.runner import (
    cli_handle,
    cli_register,
    mcp_promote_code,
    mcp_run_experiment,
    promote_code,
    run_experiment,
)


class FakeLedger:
    def __init__(self, entries=None, best_entry=None):
        self.entries = list(entries or [])
        self._best = best_entry

    def append(self, entry: dict) -> None:
        self.entries.append(dict(entry))

    def read(self) -> list[dict]:
        return list(self.entries)

    def best(self, component: str = "events") -> dict | None:
        return self._best


@pytest.fixture
def temp_git_repo(tmp_path: Path):
    """Create a temporary git repository for runner testing."""
    repo_dir = tmp_path / "main_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=repo_dir, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "runner@test.local"],
        cwd=repo_dir, capture_output=True, check=True,
    )

    detector = repo_dir / "src" / "soundthing" / "events" / "detector.py"
    detector.parent.mkdir(parents=True, exist_ok=True)
    detector.write_text("def detect(): return []\n", encoding="utf-8")

    config_pipe = repo_dir / "config" / "pipeline.yaml"
    config_pipe.parent.mkdir(parents=True, exist_ok=True)
    config_pipe.write_text("events:\n  window: 1.0\n", encoding="utf-8")

    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True, check=True)

    return repo_dir


def test_accepted_run_keeps_branch_and_writes_patch(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Accepted experiment saves patch, keeps git branch, records ledger entry, and removes worktree."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))
    monkeypatch.setenv("SOUNDTHING_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))

    patch_content = """diff --git a/src/soundthing/events/detector.py b/src/soundthing/events/detector.py
--- a/src/soundthing/events/detector.py
+++ b/src/soundthing/events/detector.py
@@ -1 +1 @@
-def detect(): return []
+def detect(): return ['event']
"""
    ledger = FakeLedger()

    def fake_trusted_score(worktree, component, overrides, pinned):
        # Verify patch was applied in worktree
        modified = Path(worktree) / "src" / "soundthing" / "events" / "detector.py"
        assert "def detect(): return ['event']" in modified.read_text(encoding="utf-8")
        return {
            "objective": 0.88,
            "passed": True,
            "metrics": {"objective": 0.88},
            "limits": {"safety_recall": True},
            "per_clip": {"clip1": {"stats": [1.0, 0.0, 0.0, 0.0]}},
        }

    res = run_experiment(
        component="events",
        hypothesis="return detected event",
        patch=patch_content,
        overrides={"events.window": 1.5},
        main_repo=temp_git_repo,
        ledger=ledger,
        trusted_score_fn=fake_trusted_score,
    )

    assert res["accept"] is True
    assert res["passed"] is True
    run_id = res["run_id"]
    branch_name = f"harness/exp-{run_id}"

    # Verify patch was saved with matching sha256
    patch_file = paths.patches_dir() / f"{run_id}.patch"
    assert patch_file.is_file()
    assert patch_file.read_text(encoding="utf-8") == patch_content
    expected_sha = hashlib.sha256(patch_content.encode("utf-8")).hexdigest()
    assert res["patch_sha256"] == expected_sha

    # Verify branch was kept
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert branch_name in branches

    # Verify worktree was cleaned up by default
    wt_path = paths.experiments_root() / f"soundthing-exp-{run_id}"
    assert not wt_path.exists()

    # Verify ledger entry
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry["kind"] == "experiment"
    assert entry["run_id"] == run_id
    assert entry["hypothesis"] == "return detected event"
    assert entry["accept"] is True
    assert entry["branch"] == branch_name


def test_rejected_run_removes_worktree_and_branch(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rejected experiment removes both the temporary worktree and the git branch."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))
    monkeypatch.setenv("SOUNDTHING_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))

    ledger = FakeLedger()

    def fake_trusted_score(worktree, component, overrides, pinned):
        return {
            "objective": 0.40,
            "passed": False,  # limit check failed
            "metrics": {"objective": 0.40},
            "limits": {"safety_recall": False},
        }

    res = run_experiment(
        component="events",
        hypothesis="failing experiment",
        main_repo=temp_git_repo,
        ledger=ledger,
        trusted_score_fn=fake_trusted_score,
    )

    assert res["accept"] is False
    assert res["passed"] is False
    run_id = res["run_id"]
    branch_name = f"harness/exp-{run_id}"

    # Verify worktree is removed
    wt_path = paths.experiments_root() / f"soundthing-exp-{run_id}"
    assert not wt_path.exists()

    # Verify branch was deleted
    branches = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert branch_name not in branches

    # Ledger still records the attempt
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["accept"] is False


def test_patch_outside_code_paths_raises_before_worktree(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch touching files outside code_paths raises ValueError before any worktree is created."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))
    monkeypatch.setenv("SOUNDTHING_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))

    forbidden_patch = """diff --git a/src/soundthing/contracts.py b/src/soundthing/contracts.py
--- a/src/soundthing/contracts.py
+++ b/src/soundthing/contracts.py
@@ -1 +1 @@
-old
+new
"""
    ledger = FakeLedger()

    with pytest.raises(ValueError, match="outside component code_paths"):
        run_experiment(
            component="events",
            hypothesis="touch contracts",
            patch=forbidden_patch,
            main_repo=temp_git_repo,
            ledger=ledger,
        )

    # Verify no worktree was created
    exp_root = paths.experiments_root()
    if exp_root.exists():
        assert list(exp_root.iterdir()) == []

    # Verify no ledger entry
    assert len(ledger.entries) == 0


def test_promote_code_creates_review_branch_and_validates_patch(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """promote_code creates harness/review/<run_id> and refuses non-accepted runs or tampered patches."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))
    monkeypatch.setenv("SOUNDTHING_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))

    run_id = "exp_accept_01"
    branch_name = f"harness/exp-{run_id}"

    # Create the experiment branch in git
    subprocess.run(["git", "branch", branch_name, "HEAD"], cwd=temp_git_repo, capture_output=True, check=True)

    patch_text = "fake patch content"
    patch_file = paths.patches_dir() / f"{run_id}.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(patch_text, encoding="utf-8")
    patch_sha = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()

    accepted_entry = {
        "kind": "experiment",
        "component": "events",
        "run_id": run_id,
        "accept": True,
        "branch": branch_name,
        "patch_sha256": patch_sha,
        "overrides": {},
    }
    ledger = FakeLedger(entries=[accepted_entry])

    # 1. Successful promotion with holdout passing
    res = promote_code(
        run_id=run_id,
        main_repo=temp_git_repo,
        ledger=ledger,
        holdout_fn=lambda ov: True,
    )
    assert res["promoted"] is True
    assert res["branch"] == f"harness/review/{run_id}"

    # Verify review branch was created in git
    branches = subprocess.run(
        ["git", "branch", "--list", f"harness/review/{run_id}"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert f"harness/review/{run_id}" in branches

    # 2. Refuses non-accepted run
    unaccepted_id = "exp_rej_02"
    ledger.append({
        "kind": "experiment",
        "component": "events",
        "run_id": unaccepted_id,
        "accept": False,
        "branch": f"harness/exp-{unaccepted_id}",
    })
    res_unacc = promote_code(run_id=unaccepted_id, main_repo=temp_git_repo, ledger=ledger)
    assert res_unacc["promoted"] is False
    assert "experiment not accepted" in res_unacc["reason"]

    # 3. Refuses tampered patch
    tampered_id = "exp_tampered_03"
    t_branch = f"harness/exp-{tampered_id}"
    subprocess.run(["git", "branch", t_branch, "HEAD"], cwd=temp_git_repo, capture_output=True, check=True)
    t_patch_file = paths.patches_dir() / f"{tampered_id}.patch"
    t_patch_file.write_text("original patch", encoding="utf-8")
    t_sha = hashlib.sha256(b"original patch").hexdigest()

    ledger.append({
        "kind": "experiment",
        "component": "events",
        "run_id": tampered_id,
        "accept": True,
        "branch": t_branch,
        "patch_sha256": t_sha,
    })

    # Tamper with the patch on disk
    t_patch_file.write_text("tampered patch content", encoding="utf-8")

    res_tampered = promote_code(run_id=tampered_id, main_repo=temp_git_repo, ledger=ledger)
    assert res_tampered["promoted"] is False
    assert "tampered patch" in res_tampered["reason"]


def test_cli_and_mcp_wrappers(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """cli_register adds arguments and cli_handle invokes runner and prints JSON."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "harness_home"))

    # Test parser registration
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    cli_register(subparsers)

    args = parser.parse_args([
        "run-experiment",
        "--component", "events",
        "--hypothesis", "test cli",
        "--set", "events.threshold=0.6",
    ])
    assert args.component == "events"
    assert args.hypothesis == "test cli"
    assert args.set_items == ["events.threshold=0.6"]

    # Test mcp_run_experiment / mcp_promote_code
    ledger = FakeLedger()
    res_mcp = mcp_run_experiment(
        component="events",
        hypothesis="mcp test",
        main_repo=temp_git_repo,
        ledger=ledger,
        trusted_score_fn=lambda *a, **k: {"objective": 0.8, "passed": True},
    )
    assert res_mcp["accept"] is True

    res_promote_mcp = mcp_promote_code(res_mcp["run_id"])
    # Not found in global ledger because we injected a fake, but wrapper runs without error
    assert "promoted" in res_promote_mcp

    # Test cli_handle execution
    monkeypatch.setattr(
        "soundthing.harness.runner.run_experiment",
        lambda **kwargs: {"accept": True, "run_id": "cli_exp_1"},
    )
    code = cli_handle(args)
    assert code == 0

    args_promote = parser.parse_args(["promote-code", "cli_exp_1"])
    monkeypatch.setattr(
        "soundthing.harness.runner.promote_code",
        lambda run_id, **kwargs: {"promoted": True, "branch": "harness/review/cli_exp_1", "run_id": run_id},
    )
    code_promote = cli_handle(args_promote)
    assert code_promote == 0


def test_code_holdout_scores_experiment_branch_code(temp_git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Code-promotion holdout scores the experiment branch's code, not the main checkout's."""
    monkeypatch.setenv("SOUNDTHING_EXPERIMENTS_ROOT", str(tmp_path / "experiments"))
    subprocess.run(["git", "branch", "harness/exp-abc"], cwd=temp_git_repo, capture_output=True, check=True)

    calls = []

    def fake_trusted(worktree, component, overrides, *, split="dev", **kwargs):
        detector = Path(worktree) / "src" / "soundthing" / "events" / "detector.py"
        calls.append((Path(worktree), split, detector.is_file()))
        cand = Path(worktree) != temp_git_repo
        return {"objective": 0.6 if cand else 0.5, "safety_recall": 1.0, "fp_per_min": 1.0}

    monkeypatch.setattr(runner_module, "trusted_score", fake_trusted)
    candidate = {"run_id": "abc", "branch": "harness/exp-abc", "overrides": {}}

    assert runner_module._code_holdout_check(temp_git_repo, candidate) is True
    assert [c[1] for c in calls] == ["holdout", "holdout"]
    assert calls[0][0] != temp_git_repo and calls[0][2]
    assert calls[1][0] == temp_git_repo
    assert not calls[0][0].exists()
