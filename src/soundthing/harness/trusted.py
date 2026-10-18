"""Trusted scoring inside an isolated overlay tree.

Builds an overlay in a temporary directory combining pinned repo files with
only the component's code_paths and allowed config files from the candidate
worktree, then invokes the harness scorer in a subprocess.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import yaml

from soundthing.harness import paths
from soundthing.harness.patches import PROTECTED_CONFIG_FILES


def _get_component(name: str):
    from soundthing.harness import components

    return components.get(name)


def _find_main_repo(worktree: Path) -> Path:
    try:
        out = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        common_git = Path(out)
        return common_git.parent
    except Exception:
        return paths.main_checkout()


def trusted_score(
    worktree: Path | str,
    component: str,
    overrides: dict | None = None,
    *,
    pinned: str | None = None,
    split: str = "dev",
    timeout: float = 3600,
    run: Any = subprocess.run,
    main: Path | None = None,
) -> dict:
    """Score a candidate worktree inside an isolated overlay directory.

    Args:
        worktree: Candidate worktree path.
        component: Component name (e.g. 'events').
        overrides: Parameter overrides dict.
        pinned: Commit SHA or ref to pin the harness to (default main checkout HEAD).
        split: Dataset split ('dev').
        timeout: Subprocess timeout in seconds.
        run: Callable executing the subprocess (default subprocess.run).
        main: Optional root repo directory (default resolved from worktree/paths).

    Returns:
        Score result dict plus 'trusted' metadata dict.

    Raises:
        RuntimeError: If scoring subprocess exits with non-zero status.
    """
    wt_path = Path(worktree).resolve()
    main_dir = Path(main).resolve() if main is not None else _find_main_repo(wt_path)

    pinned_sha = subprocess.run(
        ["git", "-C", str(main_dir), "rev-parse", pinned or "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    comp = _get_component(component)
    temp_overlay = Path(tempfile.mkdtemp(prefix="trusted_overlay_"))

    try:
        # 1. Archive pinned tree for src, config, pyproject.toml
        ls_proc = subprocess.run(
            ["git", "-C", str(main_dir), "ls-tree", "--name-only", pinned_sha, "src", "config", "pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        )
        available_paths = [p.strip() for p in ls_proc.stdout.splitlines() if p.strip()]
        archive_cmd = ["git", "-C", str(main_dir), "archive", pinned_sha]
        if available_paths:
            archive_cmd.extend(available_paths)

        archive_proc = subprocess.run(archive_cmd, capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(archive_proc.stdout)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=temp_overlay, filter="data")
            else:
                tar.extractall(path=temp_overlay)

        # 2. Copy component's code_paths from candidate worktree
        for rel_path in comp.code_paths:
            src_item = wt_path / rel_path
            dst_item = temp_overlay / rel_path
            if src_item.is_dir():
                if dst_item.exists():
                    shutil.rmtree(dst_item)
                shutil.copytree(src_item, dst_item)
            elif src_item.is_file():
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, dst_item)

        # 3. Read pinned eval section from pipeline.yaml
        pinned_pipe_path = temp_overlay / "config" / "pipeline.yaml"
        pinned_eval = None
        if pinned_pipe_path.is_file():
            with open(pinned_pipe_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            pinned_eval = data.get("eval")

        # 4. Copy candidate config files over overlay (restoring pinned eval and protected yaml)
        cand_config = wt_path / "config"
        if cand_config.is_dir():
            overlay_config = temp_overlay / "config"
            overlay_config.mkdir(parents=True, exist_ok=True)
            for root_dir, _dirs, files in os.walk(cand_config):
                rel_root = Path(root_dir).relative_to(cand_config)
                target_root = overlay_config / rel_root
                target_root.mkdir(parents=True, exist_ok=True)
                for fname in files:
                    if rel_root == Path(".") and fname in PROTECTED_CONFIG_FILES:
                        continue
                    src_file = Path(root_dir) / fname
                    dst_file = target_root / fname
                    if rel_root == Path(".") and fname == "pipeline.yaml":
                        with open(src_file, "r", encoding="utf-8") as f:
                            cand_pipe = yaml.safe_load(f) or {}
                        if pinned_eval is not None:
                            cand_pipe["eval"] = pinned_eval
                        elif "eval" in cand_pipe:
                            del cand_pipe["eval"]
                        with open(dst_file, "w", encoding="utf-8") as f:
                            yaml.safe_dump(cand_pipe, f)
                    else:
                        shutil.copy2(src_file, dst_file)

        # 5. Build scoring subprocess command and environment
        out_file = temp_overlay / "score_result.json"
        cmd = [
            sys.executable,
            "-m",
            "soundthing.harness",
            "score",
            "--component",
            component,
            "--overrides",
            json.dumps(overrides or {}),
            "--json-out",
            str(out_file),
            "--no-log",
        ]
        if split != "dev":
            cmd.extend(["--split", split])

        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(temp_overlay / "src")
        if "SOUNDTHING_HARNESS_HOME" not in child_env:
            child_env["SOUNDTHING_HARNESS_HOME"] = str(paths.harness_home())

        proc = run(
            cmd,
            cwd=str(temp_overlay),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            tail = "\n".join(stderr.splitlines()[-20:])
            raise RuntimeError(f"trusted_score failed with exit code {proc.returncode}:\n{tail}")

        if out_file.is_file():
            result_data = json.loads(out_file.read_text(encoding="utf-8"))
        elif proc.stdout and proc.stdout.strip().startswith("{"):
            result_data = json.loads(proc.stdout)
        else:
            raise RuntimeError(f"trusted_score produced no output file at {out_file}")

        result_data["trusted"] = {
            "pinned": pinned_sha,
            "code_paths": list(comp.code_paths),
        }
        return result_data

    finally:
        shutil.rmtree(temp_overlay, ignore_errors=True)
