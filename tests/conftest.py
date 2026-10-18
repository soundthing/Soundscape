import wave
from pathlib import Path

import pytest

from soundthing import contracts as C
from soundthing.config import load_config

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


@pytest.fixture(scope="session")
def cfg():
    # Tests run the stub backends whatever config/pipeline.yaml selects, so they
    # stay fast, offline and deterministic.
    cfg = load_config()
    for stage in ("events", "environment"):
        cfg.pipeline[stage]["backend"] = "stub"
    return cfg


@pytest.fixture
def fake_inputs(tmp_path) -> Path:
    """A run dir with a silent 30 s WAV and inputs.json — no ffmpeg needed."""
    run = tmp_path / "runs" / "clip01"
    (run / "frames").mkdir(parents=True)
    audio = run / "audio.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 30)
    inputs = C.ExtractedInputs(
        video_id="clip01", video_path=str(run / "clip01.mp4"), audio_path=str(audio),
        frames_dir=str(run / "frames"), fps=1.0, duration=30.0,
    )
    path = run / "inputs.json"
    C.save_inputs(inputs, path)
    return path
