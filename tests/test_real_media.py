"""Person 3: actual AV1 compatibility conversion preserves audio and original media."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from soundthing.extract import prepare_analysis_video, probe_duration


def test_av1_analysis_copy_is_decodable_and_preserves_surround_audio(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    if "libaom-av1" not in encoders or not any(name in encoders for name in ("libx264", "libopenh264")):
        pytest.skip("AV1 and H.264 software encoders required")
    source = tmp_path / "original.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=5.1", "-t", "0.5",
                    "-c:v", "libaom-av1", "-cpu-used", "8", "-c:a", "pcm_s16le", str(source)], check=True)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = prepare_analysis_video(source, tmp_path / "run")
    assert result != source and result.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    capture = cv2.VideoCapture(str(result))
    try:
        ok, frame = capture.read()
        assert ok and frame.shape[:2] == (48, 64)
    finally:
        capture.release()
    assert probe_duration(result) == pytest.approx(probe_duration(source), abs=.01)
    probe = json.loads(subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(result)], capture_output=True, text=True, check=True).stdout)
    assert probe["streams"][0]["codec_name"] == "h264"
    assert probe["streams"][1]["channels"] == 6

    def audio_hash(path):
        return subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "copy",
                               "-f", "hash", "-hash", "sha256", "pipe:1"], capture_output=True, check=True).stdout

    assert audio_hash(result) == audio_hash(source)
    before = result.stat().st_mtime_ns
    assert prepare_analysis_video(source, result.parent) == result
    assert result.stat().st_mtime_ns == before
    assert prepare_analysis_video(result, result.parent) == result
