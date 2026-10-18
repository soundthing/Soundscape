"""Person 3: explicit real-model setup/run/serve; never silently use stubs.

Local configuration copies live under the ignored runs directory. Real outputs
have their own root so completed demo captions cannot masquerade as model output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

from ..config import load_config, DEFAULT_CONFIG_DIR
from .development import prepare_local_path, doctor as base_doctor

ROOT = Path(__file__).resolve().parents[3]
REAL_CONFIG = ROOT / "runs" / ".real-config"
REAL_RUNS = ROOT / "runs" / "real"
# Publisher metadata: https://zenodo.org/api/records/3987831
PANN_SIZE = 327428481
PANN_MD5 = "541141fa2ee191a88f24a3219fff024e"


def ensure_panns_checkpoint(cache: Path | None = None) -> Path:
    """Install the publisher-verified weight atomically; recover interrupted downloads."""
    target = (cache or Path.home() / "panns_data") / "Cnn14_mAP=0.431.pth"

    def valid(path):
        if not path.is_file() or path.stat().st_size != PANN_SIZE:
            return False
        digest = hashlib.md5()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest() == PANN_MD5

    if valid(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pth.tmp")
    print("Downloading PANNs CNN14 checkpoint…", flush=True)
    url = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            downloaded, reported = 0, 0
            while block := response.read(1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded - reported >= 32 * 1024 * 1024:
                    print(f"PANNs: {downloaded * 100 / PANN_SIZE:.0f}%", flush=True)
                    reported = downloaded
        if not valid(temporary):
            raise ValueError("PANNs checkpoint size/checksum does not match the publisher")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def ensure_audioset_labels(cache: Path | None = None) -> Path:
    """Preseed PANNs' standard cache using HTTPS (its legacy fallback uses HTTP)."""
    target = (cache or Path.home() / "panns_data") / "class_labels_indices.csv"

    def valid(path):
        try:
            with path.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            return len(rows) == 527 and all(row.get("display_name") and row.get("mid") for row in rows)
        except (OSError, UnicodeError, csv.Error):
            return False

    if valid(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
    print("Downloading official AudioSet labels over HTTPS…", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not valid(temporary):
            raise ValueError("AudioSet download is not the expected 527-label CSV")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare_config(source: Path | None = None, destination: Path | None = None) -> Path:
    source = Path(source or DEFAULT_CONFIG_DIR).resolve()
    destination = Path(destination or REAL_CONFIG).resolve()
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Real config must be separate from shared config")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.yaml"):
        shutil.copy2(path, destination / path.name)
    pipeline_path = destination / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text())
    for stage in ("environment", "events"):
        pipeline[stage]["backend"] = "real"
    pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    cfg = load_config(destination)
    assert all(cfg.pipeline[stage]["backend"] == "real" for stage in ("environment", "events"))
    return destination


def model_doctor(config_dir: Path) -> int:
    if base_doctor():
        return 1
    ensure_audioset_labels()
    for module in ("torch", "torchaudio", "torchvision", "transformers", "panns_inference", "silero_vad", "PIL", "scenedetect", "sentencepiece", "google.protobuf"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            print(f"FAILED {module}: {exc}", file=sys.stderr)
            return 1
        print(f"OK {module}", flush=True)
    import torch
    from panns_inference import labels
    cfg = load_config(config_dir)
    cfg.validate(set(labels))
    print("OK AudioSet vocabulary mappings")
    print(f"PyTorch {torch.__version__}; device: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    return 0


def load_models(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    from ..environment.audio import _model as audio_model
    from ..environment.visual import _model as visual_model
    from ..events.detector import get_models
    from ..events.vad import _load_silero
    print("Loading environment CLAP…", flush=True)
    audio_model(cfg.pipeline["environment"]["audio"]["model"])
    print("Loading environment SigLIP…", flush=True)
    visual_model(cfg.pipeline["environment"]["visual"]["model"])
    print("Loading event PANNs…", flush=True)
    ensure_panns_checkpoint()
    get_models(cfg)
    print("Loading Silero VAD…", flush=True)
    _load_silero()
    print("All real models loaded.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "doctor", "models"):
        sub.add_parser(name)
    run = sub.add_parser("run")
    run.add_argument("source", type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--cookies", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_local_path()
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    if args.command == "serve" and args.cookies is not None:
        os.environ["SOUNDTHING_YOUTUBE_COOKIES"] = str(args.cookies.expanduser().resolve())
    config_dir = prepare_config()
    if args.command == "prepare":
        print(config_dir)
        return 0
    if model_doctor(config_dir):
        return 1
    if args.command == "doctor":
        return 0
    if args.command == "models":
        load_models(config_dir)
        return 0
    ensure_panns_checkpoint()
    if args.command == "run":
        from ..pipeline import run_video
        from .. import contracts as C
        source = args.source.resolve()
        if source.suffix == ".json":
            # Inputs may point at demo extraction, but never write into that run.
            inputs = C.load_inputs(source)
            target = REAL_RUNS / inputs.video_id / "inputs.json"
            if target.resolve() == source:
                raise ValueError("Pass the original media or an extraction from a different run")
            C.save_inputs(inputs, target)
            source = target
        print(f"Running actual environment and event models; output: {REAL_RUNS}", flush=True)
        result = run_video(source, REAL_RUNS, load_config(config_dir))
        print(result, flush=True)
        return 0
    # The server reads these at import. CLI --config alone does not propagate to
    # uvicorn's imported app, so configure both the app and its cache root explicitly.
    os.environ["SOUNDTHING_CONFIG"] = str(config_dir)
    os.environ["SOUNDTHING_RUNS"] = str(REAL_RUNS)
    if args.cookies is not None:
        from .youtube import _cookie_arguments
        with _cookie_arguments():
            pass
    print(f"Serving REAL models using {config_dir}; outputs: {REAL_RUNS}", flush=True)
    import uvicorn
    uvicorn.run("soundthing.server.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        print(f"Real-model startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
