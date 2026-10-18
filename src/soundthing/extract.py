"""ffmpeg extraction — OWNER: Person 3.

video -> audio.wav (16 kHz mono), audio_stereo.wav (optional), frames/frame_%06d.jpg (~1 fps)
"""

from __future__ import annotations

import json
import logging
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


def prepare_analysis_video(source: str | Path, run_dir: Path) -> Path:
    """OpenCV wheels cannot reliably decode AV1; provide a real H.264 copy.

    Audio streams are copied verbatim, retaining stereo/5.1 information. The
    resulting media is referenced through the existing inputs.video_path field.
    """
    source = Path(source).resolve()
    probe = subprocess.run([_require("ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name", "-of", "json", str(source)],
                           check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams or streams[0].get("codec_name") != "av1":
        return source
    target = Path(run_dir) / "video.analysis.mkv"
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.mkv")
    ffmpeg = _require("ffmpeg")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], check=True,
                              capture_output=True, text=True).stdout
    encoder = next((name for name in ("libx264", "libopenh264") if name in encoders), None)
    if encoder is None:
        raise RuntimeError("AV1 scene analysis needs ffmpeg with libx264 or libopenh264")
    logging.getLogger(__name__).info("Preparing H.264 scene-analysis video; preserving original audio: %s", source)
    try:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(source),
                        "-map", "0:v:0", "-map", "0:a?", "-c:v", encoder,
                        "-pix_fmt", "yuv420p", "-b:v", "4M", "-c:a", "copy", str(temporary)],
                       check=True, timeout=900)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


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
