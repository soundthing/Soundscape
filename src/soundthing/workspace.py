"""`soundthing setup`: rebuild the gitignored development workspace on a new machine.

Every step is idempotent: it checks what is already present and only fetches or builds
what is missing, so `soundthing setup` can be rerun after a failure or a partial copy.

    check     report tools, packages, model weights, datasets and harness data
    models    prefetch model weights (HF cache, ~/panns_data, Silero, optional demucs)
    data      download ESC-50 (GitHub) and the AudioSet strong eval clips (Kaggle)
    harness   build the dev/holdout splits, ESC-50 mixes and (with --caches) feature caches
    pack      zip the state that can't be rebuilt (ledger, local configs, test clip)
    unpack    restore such a zip into this checkout
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .harness import paths

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
PANNS_FILES = {
    "Cnn14_mAP=0.431.pth": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "class_labels_indices.csv":
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
}
AUDIOSET_DATASET = "wwilliamx07/soundthing-audioset-eval"
ESC_MIX_N_PER_ENV = 3  # matches the recorded dev-esc runs (harness/plan.md)

# (import name, pip extra, required for the full workspace)
PACKAGES = (
    ("torch", "events", True),
    ("transformers", "events", True),
    ("panns_inference", "events", True),
    ("silero_vad", "events", True),
    ("scenedetect", "environment", True),
    ("mcp", "harness", True),
    ("optuna", "harness", True),
    ("fastapi", "server", True),
    ("pytest", "dev", True),
    ("demucs", "separation", False),
)
INSTALL_HINT = 'pip install -e ".[events,environment,harness,server,dev]"'

# Paths (relative to the main checkout) that `pack` carries; caches are opt-in.
PACK_PATHS = ("data/harness/ledger.jsonl", "data/config-real", "data/config-tagger", "data/clips",
              "data/esc50", "runs")
PACK_CACHE_PATHS = ("data/harness/cache",)


def repo_root() -> Path:
    return paths.main_checkout()


def data_dir() -> Path:
    return repo_root() / "data"


def panns_dir() -> Path:
    return Path.home() / "panns_data"


def audioset_dir() -> Path:
    return data_dir() / "audioset_strong"


def esc50_dir() -> Path:
    return data_dir() / "esc50"


def esc_mix_dir() -> Path:
    return paths.harness_home() / "dev-esc"


# ---------------------------------------------------------------- report helpers

class Report:
    def __init__(self) -> None:
        self.ready = True

    def line(self, status: str, what: str, hint: str = "") -> None:
        if status == "MISSING":
            self.ready = False
        print(f"{status:<8}{what}" + (f" -> {hint}" if hint and status != "OK" else ""))


def _count(directory: Path, pattern: str = "*") -> int:
    return sum(1 for _ in directory.glob(pattern)) if directory.is_dir() else 0


def _hf_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def hf_models() -> list[str]:
    """HF model ids the real backends load, read from config/pipeline.yaml."""
    from .config import load_config

    p = load_config(validate=False).pipeline
    ids = [p["environment"]["visual"]["model"], p["environment"]["audio"]["model"],
           p["events"]["clap"]["model"]]
    return list(dict.fromkeys(ids))


def find_ffmpeg() -> Path | None:
    """ffmpeg's bin dir: PATH first, then a winget install old shells don't see yet."""
    found = shutil.which("ffmpeg")
    if found:
        return Path(found).parent
    local = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local:
        for exe in sorted(Path(local, "Microsoft", "WinGet", "Packages").glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe")):
            os.environ["PATH"] = os.pathsep.join([str(exe.parent), os.environ.get("PATH", "")])
            return exe.parent
    return None


def _ffmpeg_hint() -> str:
    if os.name == "nt":
        return "winget install Gyan.FFmpeg, then open a new shell"
    return "brew install ffmpeg" if sys.platform == "darwin" else "sudo apt install ffmpeg"


def kaggle_ready() -> bool:
    from .harness.kaggle_io import _kaggle_cli

    try:
        _kaggle_cli()
    except RuntimeError:
        return False
    home = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return bool((os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
                or os.environ.get("KAGGLE_API_TOKEN") or (home / "kaggle.json").is_file()
                or (home / "access_token").is_file())


# ---------------------------------------------------------------- steps

def check(args: argparse.Namespace | None = None) -> int:
    r = Report()
    print("== tools")
    ok = sys.version_info >= (3, 10)
    r.line("OK" if ok else "MISSING", f"python {sys.version.split()[0]}", "" if ok else "needs Python >= 3.10")
    ff = find_ffmpeg()
    r.line("OK" if ff and shutil.which("ffprobe") else "MISSING", "ffmpeg/ffprobe" + (f" ({ff})" if ff else ""),
           "" if ff else _ffmpeg_hint())
    r.line("OK" if kaggle_ready() else "INFO", "kaggle CLI + credentials",
           "" if kaggle_ready() else "only needed to download AudioSet: pip install kaggle, then "
                                     "put your API token in ~/.kaggle (kaggle.com > Settings > API)")

    print("== python packages")
    for module, extra, required in PACKAGES:
        found = importlib.util.find_spec(module) is not None
        r.line("OK" if found else ("MISSING" if required else "INFO"), module,
               "" if found else f'pip install -e ".[{extra}]"' + ("" if required else " (optional)"))
    if importlib.util.find_spec("transformers") is not None:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        r.line("OK" if major < 5 else "MISSING", f"transformers {transformers.__version__} < 5",
               "" if major < 5 else 'pip install "transformers<5" (events/clap.py breaks on 5.x)')

    print("== model weights")
    for name in PANNS_FILES:
        r.line("OK" if (panns_dir() / name).is_file() else "MISSING", f"~/panns_data/{name}",
               "soundthing setup models")
    for model_id in hf_models():
        r.line("OK" if _hf_cached(model_id) else "MISSING", f"HF cache {model_id}", "soundthing setup models")

    print("== datasets")
    csv = _esc50_csv()
    n_esc = _count(esc50_dir() / "audio", "*.wav")
    n_rows = len(csv.read_text(encoding="utf-8").splitlines()) - 1 if csv else 0
    r.line("MISSING" if not (csv and n_esc) else "OK" if n_esc >= n_rows else "INFO",
           f"ESC-50 ({n_esc} of {n_rows} clips) in data/esc50",
           "soundthing setup data" if not (csv and n_esc) else "partial copy; keep it to reproduce the "
           "recorded dev-esc mixes, or delete data/esc50 and rerun setup data for the full set")
    n_as = _count(audioset_dir() / "audio")
    r.line("OK" if n_as and (audioset_dir() / "audioset_eval_strong.tsv").is_file() else "MISSING",
           f"AudioSet strong eval ({n_as} clips) in data/audioset_strong", "soundthing setup data")

    print("== harness data")
    from .harness import holdout

    n_dev, n_hold = _count(paths.dev_labels_dir(), "*.json"), _count(holdout.holdout_labels_dir(), "*.json")
    n_mix = _count(esc_mix_dir() / "labels", "*.json")
    r.line("OK" if n_dev and n_hold else "MISSING", f"AudioSet dev/holdout splits ({n_dev}/{n_hold} clips)",
           "soundthing setup harness")
    r.line("OK" if n_mix else "MISSING", f"ESC-50 dev mixes ({n_mix} clips) in dev-esc", "soundthing setup harness")
    n_cache = _count(paths.cache_root(), "*/*.npz")
    r.line("OK" if n_cache else "INFO", f"event feature caches ({n_cache} files)",
           "" if n_cache else "soundthing setup harness --caches (slow on CPU), or unpack a --with-caches zip")
    r.line("OK" if paths.ledger_path().is_file() else "INFO", "experiment ledger",
           "" if paths.ledger_path().is_file() else "starts empty; copy one with soundthing setup unpack")

    print("Workspace is ready." if r.ready else "Fix the MISSING items (run the hinted commands) and rerun.")
    return 0 if r.ready else 1


def download(url: str, dest: Path) -> Path:
    """Stream `url` to `dest` through a .part file, so an interrupted download reruns cleanly."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"downloading {dest.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "soundthing-setup"})
    with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if total and pct // 10 != shown:
                shown = pct // 10
                print(f"  {pct}% of {total >> 20} MB", flush=True)
    part.replace(dest)
    return dest


def models(args: argparse.Namespace) -> int:
    # The csv first: importing panns_inference fetches it with wget, absent on Windows.
    for name, url in PANNS_FILES.items():
        if not (panns_dir() / name).is_file():
            download(url, panns_dir() / name)
    print("OK      PANNs CNN14")

    from transformers import AutoModel, AutoProcessor

    for model_id in hf_models():
        AutoModel.from_pretrained(model_id)
        AutoProcessor.from_pretrained(model_id)
        print(f"OK      {model_id}")

    from silero_vad import load_silero_vad

    load_silero_vad()
    print("OK      Silero VAD")
    if args.separation:
        import demucs.pretrained

        demucs.pretrained.get_model("htdemucs")
        print("OK      demucs htdemucs")
    return 0


def _esc50_csv() -> Path | None:
    for csv in (esc50_dir() / "esc50.csv", esc50_dir() / "meta" / "esc50.csv"):
        if csv.is_file():
            return csv
    return None


def fetch_esc50() -> None:
    if _esc50_csv() and _count(esc50_dir() / "audio", "*.wav"):
        print("OK      ESC-50 already in data/esc50")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = download(ESC50_URL, Path(tmp) / "esc50.zip")
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                parts = Path(info.filename).parts
                if info.is_dir() or len(parts) < 2:
                    continue
                if parts[1] == "audio" and info.filename.endswith(".wav"):
                    target = esc50_dir() / "audio" / parts[-1]
                elif parts[1:] == ("meta", "esc50.csv"):
                    target = esc50_dir() / "esc50.csv"
                else:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"OK      ESC-50 ({_count(esc50_dir() / 'audio', '*.wav')} clips) in data/esc50")


def fetch_audioset() -> None:
    if _count(audioset_dir() / "audio") and (audioset_dir() / "audioset_eval_strong.tsv").is_file():
        print("OK      AudioSet already in data/audioset_strong")
        return
    from .harness.kaggle_io import KaggleClient, audioset_dataset_slug

    slug = audioset_dataset_slug() or AUDIOSET_DATASET
    print(f"downloading Kaggle dataset {slug} (~1.3 GB) ...", flush=True)
    KaggleClient().download_dataset(slug, audioset_dir())
    print(f"OK      AudioSet ({_count(audioset_dir() / 'audio')} clips) in data/audioset_strong")


def data(args: argparse.Namespace) -> int:
    if not args.skip_esc50:
        fetch_esc50()
    if not args.skip_audioset:
        if not kaggle_ready() and not _count(audioset_dir() / "audio"):
            print("MISSING kaggle CLI + credentials -> install kaggle and add your API token to ~/.kaggle, "
                  "or rerun with --skip-audioset", file=sys.stderr)
            return 1
        fetch_audioset()
    return 0


def _class_csv() -> Path:
    for csv in (audioset_dir() / "class_labels_indices.csv", panns_dir() / "class_labels_indices.csv"):
        if csv.is_file():
            return csv
    raise FileNotFoundError("class_labels_indices.csv not found; run soundthing setup models first")


def harness(args: argparse.Namespace) -> int:
    from .config import load_config
    from .harness import datasets, holdout

    cfg = load_config()
    if args.skip_audioset:
        pass
    elif _count(paths.dev_labels_dir(), "*.json") and _count(holdout.holdout_labels_dir(), "*.json"):
        print("OK      AudioSet dev/holdout splits already built")
    else:
        tsv = audioset_dir() / "audioset_eval_strong.tsv"
        with tempfile.TemporaryDirectory() as tmp:
            datasets.audioset_strong_labels(tsv, cfg, Path(tmp), _class_csv())
            split = datasets.split_audioset(Path(tmp) / "labels", audioset_dir() / "audio")
        print(f"OK      AudioSet splits: {len(split['dev'])} dev / {len(split['holdout'])} holdout")

    mix = esc_mix_dir()
    if args.skip_esc50:
        pass
    elif _count(mix / "labels", "*.json"):
        print("OK      ESC-50 dev mixes already built")
    else:
        made = datasets.esc50_mixes(esc50_dir(), cfg, ESC_MIX_N_PER_ENV, 0, mix)
        print(f"OK      ESC-50 dev mixes: {len(made or [])} clips in {mix}")

    if args.caches:
        from .harness.components.environment import build_env_caches

        if _count(mix / "labels", "*.json"):
            datasets.build_dev_caches(mix / "audio", mix / "labels", cfg)
            build_env_caches(mix / "labels", mix / "audio", cfg)
        datasets.build_dev_caches(paths.dev_audio_dir(), paths.dev_labels_dir(), cfg)
        datasets.build_holdout_caches(cfg)
        print(f"OK      feature caches ({_count(paths.cache_root(), '*/*.npz')} files)")
    else:
        print("INFO    skipped feature caches; build them with --caches (hours on CPU) "
              "or copy them with pack --with-caches / unpack")
    return 0


def pack(args: argparse.Namespace) -> int:
    root = repo_root()
    out = Path(args.archive or "soundthing-workspace.zip").resolve()
    picked = PACK_PATHS + (PACK_CACHE_PATHS if args.with_caches else ())
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in picked:
            src = root / rel
            files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
            for f in files:
                if f.resolve() != out:
                    z.write(f, f.relative_to(root).as_posix())
                    n += 1
    print(f"OK      packed {n} files into {out}")
    return 0


def unpack(args: argparse.Namespace) -> int:
    root = repo_root().resolve()
    archive = Path(args.archive or "soundthing-workspace.zip")
    written = skipped = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing to extract outside the checkout: {info.filename}")
            if info.is_dir():
                continue
            if target.exists() and not args.force:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    print(f"OK      restored {written} files" + (f", kept {skipped} existing (use --force to overwrite)"
                                                 if skipped else ""))
    return 0


def run_all(args: argparse.Namespace) -> int:
    if args.archive:
        unpack(args)
    for step in (models, data, harness):
        code = step(args)
        if code:
            return code
    return check(args)


STEPS: dict[str, Callable[[argparse.Namespace], int]] = {
    "all": run_all, "check": check, "models": models, "data": data, "harness": harness,
    "pack": pack, "unpack": unpack,
}


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("setup", help="rebuild the development workspace (models, datasets, harness data)",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", choices=list(STEPS), default="all",
                   help="default: all = [unpack ARCHIVE], models, data, harness, check")
    p.add_argument("--archive", help="zip for pack/unpack (default soundthing-workspace.zip); "
                                     "with all, unpacked first")
    p.add_argument("--with-caches", action="store_true", help="pack: include feature caches")
    p.add_argument("--force", action="store_true", help="unpack: overwrite existing files")
    p.add_argument("--caches", action="store_true", help="harness: also build feature caches (slow on CPU)")
    p.add_argument("--separation", action="store_true", help="models: also fetch demucs")
    p.add_argument("--skip-audioset", action="store_true", help="data/harness: skip AudioSet (no Kaggle)")
    p.add_argument("--skip-esc50", action="store_true", help="data/harness: skip ESC-50")


def main(args: argparse.Namespace) -> int:
    try:
        return STEPS[args.step](args)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"setup {args.step} failed: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print(f"install the extras first: {INSTALL_HINT}", file=sys.stderr)
        return 1
