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


@pytest.fixture
def inject_scores(monkeypatch):
    """Feed fixed class scores through detector.detect's cache + replay path.

    Call with a no-arg function returning FrameScores; it runs once per cache build
    (like the tagger), and replay starts from its scores instead of the tagger's.
    VAD is bypassed. The cache keeps real per-window RMS for the activity gate.
    """
    import numpy as np

    from soundthing.events import detector, replay
    from soundthing.events.cache import ClipCache
    from soundthing.events.fallbacks import window_rms_dbfs

    def install(scores_fn):
        built = {}

        def fake_build_cache(audio, sr, video_id, cfg, models=None):
            fs = built["scores"] = scores_fn()
            win = float(cfg.pipeline["events"]["window_sec"])
            return ClipCache(
                video_id=video_id, duration=len(audio) / sr, key="inject_scores",
                window_sec=win, hop=fs.hop, times=fs.times, audioset_names=[],
                tagger_raw=np.zeros((len(fs.times), 0), dtype=np.float16),
                vad_chunk_sec=0.032, vad_probs=np.zeros(0, dtype=np.float32),
                rms_dbfs=window_rms_dbfs(audio, sr, fs.times, win).astype(np.float32),
            )

        def fake_class_scores(cache, cfg):
            fs = built["scores"]
            return fs.copy_with(fs.scores.copy())

        monkeypatch.setattr(detector, "build_cache", fake_build_cache)
        monkeypatch.setattr(replay, "class_scores", fake_class_scores)

    return install
