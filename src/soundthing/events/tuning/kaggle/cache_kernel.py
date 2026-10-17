"""Kaggle kernel script to build tuning caches on GPU."""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse, urlunparse

PARAMS = {}  # filled in by kaggle_io.push_cache_kernel


def _clone_url(repo_url: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if token and repo_url.startswith("https://"):
            parsed = urlparse(repo_url)
            netloc = f"x-access-token:{token}@{parsed.netloc}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return repo_url


def _find_esc50_dir() -> Path:
    matches = glob.glob("/kaggle/input/**/esc50.csv", recursive=True)
    if not matches:
        raise FileNotFoundError("Could not find esc50.csv under /kaggle/input")
    csv_path = Path(matches[0]).resolve()
    if (csv_path.parent / "audio").is_dir():
        return csv_path.parent
    if (csv_path.parent.parent / "audio").is_dir():
        return csv_path.parent.parent
    if csv_path.parent.name == "meta":
        return csv_path.parent.parent
    return csv_path.parent


def main() -> None:
    repo_url = PARAMS["repo_url"]
    git_sha = PARAMS["git_sha"]
    window = float(PARAMS["window_sec"])
    hop = float(PARAMS["hop_sec"])
    model = str(PARAMS["tagger_model"])
    n_per_env = int(PARAMS.get("n_per_env", 20))
    seed = int(PARAMS.get("seed", 0))

    work_dir = Path("/kaggle/working").resolve()
    clone_dir = work_dir / "repo"
    home_dir = work_dir / "tuning"
    os.environ["SOUNDTHING_TUNING_HOME"] = str(home_dir)

    target_url = _clone_url(repo_url)
    subprocess.run(["git", "clone", target_url, str(clone_dir)], check=True)
    subprocess.run(["git", "checkout", git_sha], cwd=clone_dir, check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[events]"],
        cwd=clone_dir,
        check=True,
    )

    sys.path.insert(0, str(clone_dir / "src"))

    from soundthing.config import load_config
    from soundthing.events.tuning.cache import CACHE_VERSION, cache_key
    from soundthing.events.tuning.datasets import build_dev_caches, esc50_mixes
    from soundthing.events.tuning.overrides import apply_overrides

    cfg = load_config(clone_dir / "config")
    cfg = apply_overrides(
        cfg,
        {
            "events.window_sec": window,
            "events.hop_sec": hop,
            "events.tagger.model": model,
        },
    )

    key = cache_key(cfg)
    esc50_dir = _find_esc50_dir()
    dev_dir = home_dir / "dev"
    audio_dir = dev_dir / "audio"
    labels_dir = dev_dir / "labels"

    mix_paths = esc50_mixes(esc50_dir, cfg, n_per_env, seed, out_dir=dev_dir)
    cache_paths = build_dev_caches(audio_dir, labels_dir, cfg)

    meta = {
        "cache_key": key,
        "cache_version": CACHE_VERSION,
        "window_sec": window,
        "hop_sec": hop,
        "tagger_model": model,
        "git_sha": git_sha,
        "n_clips": len(cache_paths) if cache_paths else len(mix_paths),
        "seed": seed,
        "n_per_env": n_per_env,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    home_dir.mkdir(parents=True, exist_ok=True)
    meta_path = home_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built cache {key} with {meta['n_clips']} clips at {meta_path}")


if __name__ == "__main__":
    main()
