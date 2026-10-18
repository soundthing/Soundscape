"""Person 3: offline speaker-feed activity; never physical source localization.

Read original media, preserving its channel layout. No model or browser audio
capture is needed. PCM is streamed in fixed windows rather than loaded in RAM.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import contracts as C

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16000
WINDOW_SEC = .05


def layout_channels(layout: str, count: int) -> tuple[str, list[str], list[int]]:
    """ffmpeg's canonical PCM channel order; do not infer layout from count."""
    if layout == "stereo" and count == 2:
        return "stereo", ["FL", "FR"], [0, 1]
    if layout in {"5.1", "5.1(side)"} and count == 6:
        return "surround51", ["FL", "FR", "FC", "SL", "SR"], [0, 1, 2, 4, 5]
    return "unavailable", [], []


def channel_strengths(samples: np.ndarray, previous: np.ndarray, seconds: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(samples.astype(np.float64)), axis=0))
    target = np.clip((20 * np.log10(np.maximum(rms, 1e-12)) + 60) / 60, 0, 1)
    tau = np.where(target > previous, .05, .2)
    return previous + (target - previous) * (1 - np.exp(-seconds / tau))


def analyze(source: str | Path, duration: float) -> C.DirectionTrack:
    if not math.isfinite(duration) or duration <= 0:
        return C.DirectionTrack(0, "unavailable", [], [])
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise RuntimeError("Direction analysis requires ffprobe and ffmpeg")
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=channels,channel_layout", "-of", "json", str(source)],
                           check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(probe.stdout).get("streams", [])
    stream = streams[0] if streams else {}
    count = stream.get("channels", 0)
    mode, channels, indices = layout_channels(stream.get("channel_layout", ""), count)
    if mode == "unavailable":
        return C.DirectionTrack(duration, mode, [], [])
    window = round(SAMPLE_RATE * WINDOW_SEC)
    frames = []
    previous = np.zeros(len(channels))
    sample_offset = 0
    # stderr in a disposable file avoids a full stderr pipe blocking the decoder.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-i", str(source),
                                    "-map", "0:a:0", "-vn", "-ar", str(SAMPLE_RATE),
                                    "-t", str(duration), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            while sample_offset / SAMPLE_RATE < duration:
                block = process.stdout.read(window * count * 4)
                if not block:
                    break
                if len(block) % (count * 4):
                    raise RuntimeError("Incomplete direction PCM sample")
                samples = np.frombuffer(block, dtype="<f4").reshape(-1, count)[:, indices]
                previous = channel_strengths(samples, previous, len(samples) / SAMPLE_RATE)
                frames.append(C.DirectionFrame(round(sample_offset / SAMPLE_RATE, 6),
                                               dict(zip(channels, previous.tolist()))))
                sample_offset += len(samples)
            process.stdout.close()
            if process.wait(timeout=30):
                raise RuntimeError("Direction audio decoding failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()
    track = C.DirectionTrack(duration, mode, channels, frames) if frames else C.DirectionTrack(duration, "unavailable", [], [])
    track.validate()
    return track


def ensure_direction(source: str | Path, run_dir: Path, duration: float) -> C.DirectionTrack:
    """Cache success or unavailability. Analysis failure must not break captions."""
    path = Path(run_dir) / "direction.json"
    if path.is_file():
        try:
            return C.load_direction(path)
        except (OSError, ValueError, TypeError, KeyError):
            LOG.warning("Replacing invalid direction track in %s", run_dir)
    try:
        track = analyze(source, duration)
    except Exception:
        LOG.warning("Direction unavailable for %s", source, exc_info=True)
        track = C.DirectionTrack(duration, "unavailable", [], [])
    # Only the completed, validated artifact becomes visible to HTTP readers.
    temporary = path.with_suffix(".json.tmp")
    try:
        C.save_direction(track, temporary)
        temporary.replace(path)
    except OSError:
        LOG.exception("Could not store direction track in %s", run_dir)
    return track
