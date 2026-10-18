"""Kaggle kernel script to build harness caches on GPU.

Runs the code embedded in BUNDLE (the pusher's working tree), so nothing is cloned.
Builds AudioSet dev + holdout caches when the AudioSet dataset is attached, and ESC-50
synthetic mixes when ESC-50 is attached. Only caches, dev labels and meta.json are
written to the output; holdout labels never leave the kernel.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import glob
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

PARAMS = {}  # filled in by kaggle_io.push_cache_kernel
BUNDLE = ""  # filled in by kaggle_io.push_cache_kernel


def _find(pattern: str) -> Path | None:
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return Path(matches[0]).resolve() if matches else None


def _esc50_dir(csv_path: Path) -> Path:
    for d in (csv_path.parent, csv_path.parent.parent):
        if (d / "audio").is_dir():
            return d
    return csv_path.parent.parent if csv_path.parent.name == "meta" else csv_path.parent


log = logging.getLogger("cache_kernel")
_T0 = time.monotonic()


def _stage(name: str) -> None:
    log.info("=== %s (t=%.0fs)", name, time.monotonic() - _T0)


def _log_env() -> None:
    import torch

    log.info("python %s, torch %s, cuda %s", sys.version.split()[0], torch.__version__, torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        log.info("gpu %d: %s", i, torch.cuda.get_device_name(i))
    log.info("params: %s", {k: v for k, v in PARAMS.items()})
    log.info("inputs: %s", sorted(glob.glob("/kaggle/input/*/*"))[:50])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)
    window = float(PARAMS["window_sec"])
    hop = float(PARAMS["hop_sec"])
    model = str(PARAMS["tagger_model"])

    src_dir = Path("/tmp/soundthing")
    home_dir = Path("/tmp/harness")
    out_dir = Path("/kaggle/working/tuning")
    os.environ["SOUNDTHING_HARNESS_HOME"] = str(home_dir)

    _stage("install bundle")
    src_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as tar:
        tar.extractall(src_dir)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".[events]", "transformers>=4.40,<5"],
        cwd=src_dir,
        check=True,
    )
    sys.path.insert(0, str(src_dir / "src"))
    _log_env()

    from soundthing.config import load_config
    from soundthing.events.cache import CACHE_VERSION, cache_key
    from soundthing.events.detector import get_models
    from soundthing.harness import datasets, paths
    from soundthing.harness.overrides import apply_overrides

    cfg = apply_overrides(
        load_config(src_dir / "config"),
        {"events.window_sec": window, "events.hop_sec": hop, "events.tagger.model": model},
    )
    key = cache_key(cfg)
    _stage(f"load models (cache key {key})")
    models = get_models(cfg)
    counts: dict[str, int] = {}

    tsv = _find("audioset_eval_strong.tsv")
    log.info("audioset tsv: %s", tsv)
    if tsv is not None:
        _stage("audioset labels + split")
        with tempfile.TemporaryDirectory() as tmp:
            datasets.audioset_strong_labels(tsv, cfg, Path(tmp), tsv.parent / "class_labels_indices.csv")
            split = datasets.split_audioset(
                Path(tmp) / "labels", tsv.parent / "audio", holdout_frac=float(PARAMS["holdout_frac"])
            )
        counts["audioset_dev"] = len(split["dev"])
        counts["audioset_holdout"] = len(split["holdout"])
        log.info("audioset split: %d dev, %d holdout", counts["audioset_dev"], counts["audioset_holdout"])
        _stage("holdout caches")
        counts["holdout_caches"] = len(datasets.build_holdout_caches(cfg, models=models))

    esc50_csv = _find("esc50.csv")
    log.info("esc50 csv: %s", esc50_csv)
    if esc50_csv is not None:
        _stage("esc50 mixes")
        mixes = datasets.esc50_mixes(
            _esc50_dir(esc50_csv), cfg, int(PARAMS["n_per_env"]), int(PARAMS["seed"]),
            out_dir=home_dir / "dev",
        )
        counts["esc50_mixes"] = len(mixes)

    _stage("dev caches")
    counts["dev_caches"] = len(
        datasets.build_dev_caches(paths.dev_audio_dir(), paths.dev_labels_dir(), cfg, models=models)
    )

    _stage("copy output")
    shutil.copytree(paths.cache_root() / key, out_dir / "cache" / key)
    shutil.copytree(paths.dev_labels_dir(), out_dir / "dev" / "labels")
    manifest = paths.dev_labels_dir().parent / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, out_dir / "dev" / "manifest.json")

    meta = {
        "cache_key": key,
        "cache_version": CACHE_VERSION,
        "window_sec": window,
        "hop_sec": hop,
        "tagger_model": model,
        "git_sha": PARAMS.get("git_sha"),
        "bundle_sha256": PARAMS.get("bundle_sha256"),
        "counts": counts,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _stage(f"done: built cache {key}: {counts}")


if __name__ == "__main__":
    main()
