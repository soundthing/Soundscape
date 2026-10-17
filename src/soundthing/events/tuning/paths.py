"""Where the tuning harness keeps its state (cache, ledger, dev data, holdout).

Default: `data/tuning/` in the main checkout, shared by every git worktree so parallel
agents see one ledger. Override with SOUNDTHING_TUNING_HOME (tests always do).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV_VAR = "SOUNDTHING_TUNING_HOME"


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
        return here.parents[3]


def tuning_home() -> Path:
    env = os.environ.get(ENV_VAR)
    root = Path(env) if env else main_checkout() / "data" / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_root() -> Path:
    return tuning_home() / "cache"


def dev_labels_dir() -> Path:
    return tuning_home() / "dev" / "labels"


def holdout_dir() -> Path:
    return tuning_home() / "holdout"


def ledger_path() -> Path:
    return tuning_home() / "ledger.jsonl"


def best_path() -> Path:
    return tuning_home() / "best.json"


def proposals_dir() -> Path:
    return tuning_home() / "proposals"
