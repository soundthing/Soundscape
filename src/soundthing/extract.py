"""ffmpeg extraction — OWNER: Person 3.

video -> audio.wav (16 kHz mono), audio_stereo.wav (optional), frames/frame_%06d.jpg (~1 fps)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .contracts import ExtractedInputs


def video_id_for(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem)


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        raise RuntimeError(f"{tool} not found on PATH; install ffmpeg")
    return exe


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract(video_path: str | Path, out_dir: str | Path, cfg: Config, video_id: str | None = None) -> ExtractedInputs:
    ffmpeg = _require("ffmpeg")
    xcfg = cfg.pipeline["extract"]
    video_path = Path(video_path).resolve()
    out = Path(out_dir)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    audio = out / "audio.wav"
    run = lambda *args: subprocess.run([ffmpeg, "-y", "-v", "error", *args], check=True)  # noqa: E731
    run("-i", str(video_path), "-vn", "-ac", "1", "-ar", str(xcfg["sample_rate"]), "-c:a", "pcm_s16le", str(audio))

    stereo = None
    if xcfg.get("keep_stereo"):
        stereo = out / "audio_stereo.wav"
        run("-i", str(video_path), "-vn", "-ac", "2", "-c:a", "pcm_s16le", str(stereo))

    # Frame i is the frame at t = i / fps.
    for old in frames.glob("frame_*.jpg"):
        old.unlink()
    run("-i", str(video_path), "-vf", f"fps={xcfg['fps']}", "-start_number", "0", "-q:v", "3",
        str(frames / "frame_%06d.jpg"))

    return ExtractedInputs(
        video_id=video_id or video_id_for(video_path),
        video_path=str(video_path),
        audio_path=str(audio.resolve()),
        frames_dir=str(frames.resolve()),
        fps=float(xcfg["fps"]),
        duration=probe_duration(video_path),
        sample_rate=int(xcfg["sample_rate"]),
        audio_stereo_path=str(stereo.resolve()) if stereo else None,
    )
