"""Score on the dev clips whose caches exist, while a cache build is still running.

A clip set is a frozen snapshot of the dev labels that have a finished events cache. While
one is active, dev scoring uses it instead of the full dev split. `rebaseline` takes a new
snapshot and re-scores the current best (main checkout code + best overrides) on it, so the
best entry and every later run compare on the same clips. Once every dev clip is cached the
snapshot is dropped and scoring goes back to the full split.

State: `<harness home>/clipset/<component>/current` names the active snapshot directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from soundthing.harness import paths

# Ignore caches written in the last few seconds; the builder may still be writing them.
STABLE_SEC = 15.0
# Auto-rebaseline once the cache has this many more clips than the active snapshot.
MIN_GROWTH = 200


def _root(component: str) -> Path:
    return paths.harness_home() / "clipset" / component


def _pointer(component: str) -> Path:
    return _root(component) / "current"


def active_labels_dir(component: str = "events") -> Path | None:
    """Labels dir of the active snapshot, or None when scoring uses the full dev split."""
    ptr = _pointer(component)
    if not ptr.is_file():
        return None
    d = _root(component) / ptr.read_text(encoding="utf-8").strip() / "labels"
    return d if d.is_dir() else None


def active_ids(component: str = "events") -> list[str]:
    d = active_labels_dir(component)
    return sorted(p.stem for p in d.glob("*.json")) if d else []


def cached_dev_ids(cfg: Any, component: str = "events") -> tuple[list[str], int]:
    """(dev clip ids with a finished cache, total dev clips)."""
    from soundthing.harness import components

    comp = components.get(component)
    key = comp.cache_key(cfg)
    if key is None:
        raise ValueError(f"component {component!r} has no cache; clip sets don't apply")
    cdir = paths.cache_root() / key
    now = time.time()
    ids = sorted(p.stem for p in paths.dev_labels_dir().glob("*.json"))
    done = []
    for vid in ids:
        f = cdir / f"{vid}.npz"
        try:
            if now - f.stat().st_mtime >= STABLE_SEC:
                done.append(vid)
        except FileNotFoundError:
            pass
    return done, len(ids)


def refresh(cfg: Any, component: str = "events") -> dict[str, Any]:
    """Point the active snapshot at the currently cached dev clips."""
    done, total = cached_dev_ids(cfg, component)
    root = _root(component)
    if total and len(done) == total:
        _pointer(component).unlink(missing_ok=True)
        return {"mode": "full", "clips": total, "total": total, "changed": True}
    if not done:
        raise RuntimeError("no dev clips have a finished cache yet")
    if done == active_ids(component):
        return {"mode": "partial", "clips": len(done), "total": total, "changed": False}

    name = f"{len(done):05d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    labels = root / name / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    src = paths.dev_labels_dir()
    for vid in done:
        shutil.copy2(src / f"{vid}.json", labels / f"{vid}.json")
    tmp = root / "current.tmp"
    tmp.write_text(name, encoding="utf-8")
    os.replace(tmp, _pointer(component))
    return {"mode": "partial", "clips": len(done), "total": total, "changed": True}


def rebaseline(component: str = "events", *, force: bool = False) -> dict[str, Any]:
    """Refresh the snapshot and re-score the best on it with the main checkout's code.

    Without `force`, does nothing unless the snapshot changed.
    """
    from soundthing.config import load_config
    from soundthing.harness import ledger
    from soundthing.harness.trusted import trusted_score

    main = paths.main_checkout()
    cfg = load_config(main / "config")
    info = refresh(cfg, component)
    if not info["changed"] and not force:
        return {**info, "rebaselined": False}

    best = ledger.best(component) or {}
    overrides = best.get("overrides") or {}
    res = trusted_score(main, component, overrides)
    git = ledger.git_state(main)
    entry = {
        "kind": "rebaseline",
        "component": component,
        "run_id": res.get("run_id"),
        "hypothesis": f"rebaseline on {info['clips']}/{info['total']} cached dev clips",
        "overrides": overrides,
        "objective": res.get("objective"),
        "metrics": res.get("metrics", {}),
        "limits": res.get("limits", {}),
        "passed": True,
        "per_clip": res.get("per_clip", {}),
        "clipset": info["clips"],
        "git_sha": git.get("sha"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ledger.set_best(entry, component=component)
    ledger.append({k: v for k, v in entry.items() if k != "per_clip"})
    return {**info, "rebaselined": True, "objective": entry["objective"], "metrics": entry["metrics"]}


def maybe_rebaseline(component: str = "events") -> dict[str, Any] | None:
    """Rebaseline when the cache grew by MIN_GROWTH clips (or finished). One process at a
    time; others skip and score on the current snapshot. Never raises."""
    if component != "events":
        return None
    lock = _root(component) / "rebaseline.lock"
    try:
        from soundthing.config import load_config

        has_best = paths.best_path(component).is_file()
        if has_best and _full_marker(component).is_file():
            return None  # the full dev split is cached and the best was scored on it
        cfg = load_config(paths.main_checkout() / "config")
        done, total = cached_dev_ids(cfg, component)
        finished = total > 0 and len(done) == total
        current = active_ids(component)
        if has_best and current and not finished and len(done) - len(current) < MIN_GROWTH:
            return None
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            if lock.exists() and time.time() - lock.stat().st_mtime > 3600:
                lock.unlink()
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        try:
            os.close(fd)
            res = rebaseline(component, force=not has_best)
            if res.get("mode") == "full":
                _full_marker(component).write_text(json.dumps(res, default=str), encoding="utf-8")
            return res
        finally:
            lock.unlink(missing_ok=True)
    except Exception as exc:  # scoring must go on with the snapshot it has
        return {"rebaselined": False, "error": str(exc)}


def _full_marker(component: str) -> Path:
    return _root(component) / "full.json"


def status(component: str = "events") -> dict[str, Any]:
    from soundthing.config import load_config

    cfg = load_config(paths.main_checkout() / "config")
    done, total = cached_dev_ids(cfg, component)
    return {
        "active_clips": len(active_ids(component)),
        "mode": "partial" if active_labels_dir(component) else "full",
        "cached": len(done),
        "total": total,
    }
