"""Experiment ledger and best result tracking for the tuning harness."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from soundthing.harness import paths


def append(entry: dict) -> None:
    """Append one experiment entry as a single JSON line to ledger_path()."""
    target = paths.ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    data = dict(entry)
    if "ts" not in data:
        data["ts"] = datetime.now(timezone.utc).isoformat()
    if "component" not in data and data.get("kind") in ("score", "promote"):
        data["component"] = "events"

    line = json.dumps(data, sort_keys=True, default=str) + "\n"
    line_bytes = line.encode("utf-8")

    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    fd = os.open(str(target), flags, 0o666)
    try:
        os.write(fd, line_bytes)
    finally:
        os.close(fd)


def read(component: str | None = None) -> list[dict]:
    """Read all entries from ledger_path(), skipping blank lines and filtering by component if given."""
    target = paths.ledger_path()
    if not target.exists():
        return []

    entries: list[dict] = []
    with open(target, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                raise ValueError(
                    f"Corrupt line {line_no} in ledger {target}: {e}"
                ) from e
            if component is not None:
                if item.get("component", "events") != component:
                    continue
            entries.append(item)
    return entries


def best(component: str = "events") -> dict | None:
    """Return JSON contents of best_path(component) or None if absent."""
    target = paths.best_path(component)
    if not target.exists():
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def set_best(entry: dict, component: str = "events") -> None:
    """Write entry to best_path(component) atomically via a temp file in the same directory."""
    target = paths.best_path(component)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = dict(entry)
    if component != "events" and "component" not in data:
        data["component"] = component

    tmp_file = tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f"best_{component}_",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    tmp_path = Path(tmp_file.name)
    try:
        json.dump(data, tmp_file, indent=2, sort_keys=True, default=str)
        tmp_file.flush()
        tmp_file.close()
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Return repository git state {"sha", "dirty", "branch"} or None for all three on error."""
    target_cwd = cwd or Path.cwd()
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        sha = sha_proc.stdout.strip() or None

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=target_cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status_proc.stdout.strip())

        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=target_cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        branch = branch_proc.stdout.strip() or None

        return {"sha": sha, "dirty": dirty, "branch": branch}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None, "branch": None}
