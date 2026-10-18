"""Tests for trusted scorer in an isolated overlay tree."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from soundthing.harness.patches import parse_patch_paths, validate_patch
from soundthing.harness.trusted import trusted_score


@pytest.fixture
def temp_git_repo(tmp_path: Path):
    """Create a temporary git repository with pinned files and a modified worktree."""
    repo_dir = tmp_path / "main_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=repo_dir, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "runner@test.local"],
        cwd=repo_dir, capture_output=True, check=True,
    )

    # Create initial files
    harness_score = repo_dir / "src" / "soundthing" / "harness" / "score.py"
    harness_score.parent.mkdir(parents=True, exist_ok=True)
    harness_score.write_text("PINNED_HARNESS_SCORE = 1\n", encoding="utf-8")

    event_detector = repo_dir / "src" / "soundthing" / "events" / "detector.py"
    event_detector.parent.mkdir(parents=True, exist_ok=True)
    event_detector.write_text("PINNED_EVENT_DETECTOR = 1\n", encoding="utf-8")

    config_dir = repo_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    pipeline_yaml = config_dir / "pipeline.yaml"
    pipeline_yaml.write_text(
        yaml.safe_dump({
            "eval": {"metric": "pinned_eval", "threshold": 0.5},
            "events": {"postprocess": {"smoothing": 0.2}},
        }),
        encoding="utf-8",
    )

    (config_dir / "environments.yaml").write_text("pinned: true\n", encoding="utf-8")
    (config_dir / "vocabulary.yaml").write_text("pinned: true\n", encoding="utf-8")
    (config_dir / "priors.yaml").write_text("pinned: true\n", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text("[project]\nname = 'soundthing'\n", encoding="utf-8")

    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial pinned commit"], cwd=repo_dir, capture_output=True, check=True)

    pinned_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Create candidate worktree directory with modified files
    worktree_dir = tmp_path / "candidate_worktree"
    worktree_dir.mkdir(parents=True, exist_ok=True)

    cand_harness_score = worktree_dir / "src" / "soundthing" / "harness" / "score.py"
    cand_harness_score.parent.mkdir(parents=True, exist_ok=True)
    cand_harness_score.write_text("TAMPERED_HARNESS = 1\n", encoding="utf-8")

    cand_event_detector = worktree_dir / "src" / "soundthing" / "events" / "detector.py"
    cand_event_detector.parent.mkdir(parents=True, exist_ok=True)
    cand_event_detector.write_text("CANDIDATE_EVENT_DETECTOR = 2\n", encoding="utf-8")

    cand_config_dir = worktree_dir / "config"
    cand_config_dir.mkdir(parents=True, exist_ok=True)

    (cand_config_dir / "pipeline.yaml").write_text(
        yaml.safe_dump({
            "eval": {"metric": "tampered_eval", "threshold": 0.99},
            "events": {"postprocess": {"smoothing": 0.8}},
        }),
        encoding="utf-8",
    )

    (cand_config_dir / "environments.yaml").write_text("tampered: true\n", encoding="utf-8")
    (cand_config_dir / "vocabulary.yaml").write_text("tampered: true\n", encoding="utf-8")
    (cand_config_dir / "priors.yaml").write_text("tampered: true\n", encoding="utf-8")

    return {
        "main_repo": repo_dir,
        "worktree": worktree_dir,
        "pinned_sha": pinned_sha,
    }


def test_trusted_score_overlay_isolation(temp_git_repo: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Overlay preserves pinned harness and protected configs, incorporates candidate code and events config."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))

    repo_dir = temp_git_repo["main_repo"]
    worktree_dir = temp_git_repo["worktree"]
    pinned_sha = temp_git_repo["pinned_sha"]

    captured_runs = []

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        captured_runs.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
        })
        overlay_path = Path(cwd)

        # 1. Harness files stay pinned
        harness_score = overlay_path / "src" / "soundthing" / "harness" / "score.py"
        assert harness_score.is_file()
        assert harness_score.read_text(encoding="utf-8").strip() == "PINNED_HARNESS_SCORE = 1"

        # 2. Component code comes from candidate worktree
        event_detector = overlay_path / "src" / "soundthing" / "events" / "detector.py"
        assert event_detector.is_file()
        assert event_detector.read_text(encoding="utf-8").strip() == "CANDIDATE_EVENT_DETECTOR = 2"

        # 3. Pipeline yaml: eval section restored from pinned, events section from candidate
        pipe_yaml = overlay_path / "config" / "pipeline.yaml"
        assert pipe_yaml.is_file()
        pipe_data = yaml.safe_load(pipe_yaml.read_text(encoding="utf-8"))
        assert pipe_data["eval"]["metric"] == "pinned_eval"
        assert pipe_data["eval"]["threshold"] == 0.5
        assert pipe_data["events"]["postprocess"]["smoothing"] == 0.8

        # 4. Protected shared config files come from pinned
        env_yaml = overlay_path / "config" / "environments.yaml"
        assert env_yaml.read_text(encoding="utf-8").strip() == "pinned: true"
        vocab_yaml = overlay_path / "config" / "vocabulary.yaml"
        assert vocab_yaml.read_text(encoding="utf-8").strip() == "pinned: true"
        priors_yaml = overlay_path / "config" / "priors.yaml"
        assert priors_yaml.read_text(encoding="utf-8").strip() == "pinned: true"

        # Write output json to --json-out
        json_out_idx = cmd.index("--json-out") + 1
        out_file = Path(cmd[json_out_idx])
        out_file.write_text(
            json.dumps({
                "run_id": "test1234",
                "component": "events",
                "objective": 0.85,
                "passed": True,
                "metrics": {"objective": 0.85},
            }),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    result = trusted_score(
        worktree=worktree_dir,
        component="events",
        overrides={"events.postprocess.smoothing": 0.8},
        pinned=pinned_sha,
        run=fake_run,
        main=repo_dir,
    )

    assert len(captured_runs) == 1
    call = captured_runs[0]

    # Verify command argv, cwd, and PYTHONPATH
    assert call["cmd"][0] == sys.executable
    assert call["cmd"][1:6] == ["-m", "soundthing.harness", "score", "--component", "events"]
    assert "--overrides" in call["cmd"]
    assert "--json-out" in call["cmd"]
    assert "--no-log" in call["cmd"]

    assert call["env"]["PYTHONPATH"] == str(Path(call["cwd"]) / "src")
    assert call["env"]["SOUNDTHING_HARNESS_HOME"] == str(tmp_path)

    # Verify result structure and trusted metadata
    assert result["objective"] == 0.85
    assert result["trusted"]["pinned"] == pinned_sha
    assert result["trusted"]["code_paths"] == ["src/soundthing/events"]

    # Verify temp overlay dir was cleaned up
    assert not Path(call["cwd"]).exists()


def test_trusted_score_nonzero_exit_raises(temp_git_repo: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Subprocess failure raises RuntimeError with stderr tail and cleans temp overlay."""
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))

    repo_dir = temp_git_repo["main_repo"]
    worktree_dir = temp_git_repo["worktree"]
    pinned_sha = temp_git_repo["pinned_sha"]

    captured_cwd = []

    def failing_run(cmd, cwd, env, capture_output, text, timeout):
        captured_cwd.append(cwd)
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout="",
            stderr="Traceback (most recent call last):\nValueError: bad score parameter",
        )

    with pytest.raises(RuntimeError) as exc_info:
        trusted_score(
            worktree=worktree_dir,
            component="events",
            overrides={},
            pinned=pinned_sha,
            run=failing_run,
            main=repo_dir,
        )

    assert "trusted_score failed with exit code 1" in str(exc_info.value)
    assert "ValueError: bad score parameter" in str(exc_info.value)

    assert len(captured_cwd) == 1
    assert not Path(captured_cwd[0]).exists()


def test_patch_validation():
    """validate_patch permits component code_paths and config, but rejects external and shared files."""
    code_paths = ("src/soundthing/events",)

    # Valid patch touching component code
    valid_patch = """diff --git a/src/soundthing/events/detector.py b/src/soundthing/events/detector.py
--- a/src/soundthing/events/detector.py
+++ b/src/soundthing/events/detector.py
@@ -1 +1 @@
-old
+new
"""
    assert parse_patch_paths(valid_patch) == {"src/soundthing/events/detector.py"}
    validate_patch(valid_patch, code_paths)

    # Valid patch touching config/pipeline.yaml
    valid_cfg_patch = """diff --git a/config/pipeline.yaml b/config/pipeline.yaml
--- a/config/pipeline.yaml
+++ b/config/pipeline.yaml
@@ -1 +1 @@
-old
+new
"""
    validate_patch(valid_cfg_patch, code_paths)

    # Invalid patch touching protected shared config
    shared_patch = """diff --git a/config/environments.yaml b/config/environments.yaml
--- a/config/environments.yaml
+++ b/config/environments.yaml
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError, match="protected shared config file"):
        validate_patch(shared_patch, code_paths)

    # Invalid patch touching contract or harness
    outside_patch = """diff --git a/src/soundthing/contracts.py b/src/soundthing/contracts.py
--- a/src/soundthing/contracts.py
+++ b/src/soundthing/contracts.py
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError, match="outside component code_paths"):
        validate_patch(outside_patch, code_paths)
