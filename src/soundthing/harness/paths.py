"""Where the evaluation harness keeps its state (cache, ledger, dev data, holdout).

Default: `data/harness/` in the main checkout, shared by every git worktree so parallel
agents see one ledger. A pre-existing `data/tuning/` is used until it's renamed.
Override with SOUNDTHING_HARNESS_HOME (tests always do); SOUNDTHING_TUNING_HOME is the
old name and still works.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV_VAR = "SOUNDTHING_HARNESS_HOME"
OLD_ENV_VAR = "SOUNDTHING_TUNING_HOME"


def main_checkout() -> Path:
    """Root of the main checkout, even when called from a linked worktree."""
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=here, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out).parent
    except (OSError, subprocess.CalledProcessError):
        return here.parents[2]


def harness_home() -> Path:
    env = os.environ.get(ENV_VAR) or os.environ.get(OLD_ENV_VAR)
    if env:
        root = Path(env)
    else:
        data = main_checkout() / "data"
        root = data / "harness"
        if not root.exists() and (data / "tuning").is_dir():
            root = data / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    return root


tuning_home = harness_home


def cache_root() -> Path:
    return harness_home() / "cache"


def dev_labels_dir() -> Path:
    return harness_home() / "dev" / "labels"


def dev_audio_dir() -> Path:
    return harness_home() / "dev" / "audio"


def holdout_dir() -> Path:
    return harness_home() / "holdout"


def ledger_path() -> Path:
    return harness_home() / "ledger.jsonl"


def best_path(component: str = "events") -> Path:
    if component == "events":
        return harness_home() / "best.json"
    return harness_home() / f"best-{component}.json"


def harness_role() -> str:
    return os.environ.get("SOUNDTHING_ROLE", "person1")


def proposals_dir() -> Path:
    return harness_home() / "proposals"


def patches_dir() -> Path:
    d = harness_home() / "patches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiments_root() -> Path:
    env = os.environ.get("SOUNDTHING_EXPERIMENTS_ROOT")
    root = Path(env) if env else main_checkout().parent / ".wt"
    root.mkdir(parents=True, exist_ok=True)
    return root
