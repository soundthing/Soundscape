"""Clip sets: dev scoring on the clips whose caches are already built."""

from __future__ import annotations

import json
import os
import time

from soundthing.config import load_config
from soundthing.harness import clipset, paths


def _setup(tmp_path, monkeypatch, n_labels: int, cached: list[int]):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    cfg = load_config()
    from soundthing.harness import components

    key = components.get("events").cache_key(cfg)
    labels = paths.dev_labels_dir()
    labels.mkdir(parents=True)
    cdir = paths.cache_root() / key
    cdir.mkdir(parents=True)
    old = time.time() - 60
    for i in range(n_labels):
        (labels / f"clip{i}.json").write_text(json.dumps({"video_id": f"clip{i}"}), encoding="utf-8")
    for i in cached:
        f = cdir / f"clip{i}.npz"
        f.write_bytes(b"")
        os.utime(f, (old, old))
    return cfg, cdir


def test_refresh_snapshots_only_finished_caches(tmp_path, monkeypatch):
    cfg, cdir = _setup(tmp_path, monkeypatch, 4, [0, 2])
    (cdir / "clip3.npz").write_bytes(b"")  # just written: not stable yet

    info = clipset.refresh(cfg)

    assert info == {"mode": "partial", "clips": 2, "total": 4, "changed": True}
    assert clipset.active_ids() == ["clip0", "clip2"]
    assert clipset.refresh(cfg)["changed"] is False


def test_refresh_goes_back_to_full_dev_when_everything_is_cached(tmp_path, monkeypatch):
    cfg, _ = _setup(tmp_path, monkeypatch, 2, [0])
    clipset.refresh(cfg)
    assert clipset.active_labels_dir() is not None

    _setup_more = paths.cache_root() / next(p.name for p in paths.cache_root().iterdir()) / "clip1.npz"
    _setup_more.write_bytes(b"")
    old = time.time() - 60
    os.utime(_setup_more, (old, old))

    assert clipset.refresh(cfg)["mode"] == "full"
    assert clipset.active_labels_dir() is None


def test_score_uses_active_clipset(tmp_path, monkeypatch):
    cfg, _ = _setup(tmp_path, monkeypatch, 3, [1])
    clipset.refresh(cfg)
    seen = {}

    from soundthing.harness import components, score as score_mod

    comp = components.get("events")

    def fake_load(cfg, ldir):
        seen["ldir"] = ldir
        raise RuntimeError("stop")

    monkeypatch.setattr(comp, "load_clips", fake_load)
    try:
        score_mod.score({}, split="dev", cfg=cfg)
    except RuntimeError:
        pass
    assert seen["ldir"] == clipset.active_labels_dir()


def test_rebaseline_best_has_top_level_metrics(tmp_path, monkeypatch):
    cfg, _ = _setup(tmp_path, monkeypatch, 2, [0])
    from soundthing.harness import ledger, trusted

    fake = {"run_id": "r1", "objective": 0.5, "per_clip": {"clip0": {}},
            "metrics": {"objective": 0.5, "safety_recall": 0.4, "fp_per_min": 3.0}}
    monkeypatch.setattr(trusted, "trusted_score", lambda *a, **k: fake)

    clipset.rebaseline(force=True)

    best = ledger.best()
    assert best["safety_recall"] == 0.4 and best["fp_per_min"] == 3.0
    assert best["clipset"] == 1
