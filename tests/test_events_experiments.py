from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

from soundthing import contracts as C
from soundthing.config import Config
from soundthing.events.core import EventModels
import soundthing.events.detector as detector
from soundthing.events.experiments import main, wrong_env_report, wrong_env_segments
from soundthing.events.tagger import build_tagger_map


class FakeTagger:
    def __init__(self, names: list[str], default_prob: float = 0.0) -> None:
        self.names = names
        self.default_prob = default_prob
        self.scores_by_name: dict[str, float] = {}

    def set_score(self, name: str, prob: float) -> None:
        self.scores_by_name[name] = prob

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.full((b, len(self.names)), self.default_prob, dtype=np.float32)
        for name, prob in self.scores_by_name.items():
            if name in self.names:
                probs[:, self.names.index(name)] = prob
        return probs


@pytest.fixture(autouse=True)
def clean_detector_caches():
    detector._clear_caches()
    yield
    detector._clear_caches()


@pytest.fixture
def fake_env(cfg: Config, monkeypatch):
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    tagger = FakeTagger(audioset_names, default_prob=0.0)

    def fake_load_models(c: Config) -> EventModels:
        return EventModels(
            tagger=tagger,
            class_ids=list(c.class_ids),
            tagger_map=tagger_map,
        )

    monkeypatch.setattr(detector, "load_models", fake_load_models)
    monkeypatch.setattr(detector, "speech_regions", lambda audio, sr, c: [])
    return tagger


def test_no_torch_imported_at_top_level_experiments():
    import soundthing.events.experiments

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "panns_inference" not in sys.modules
    assert "silero_vad" not in sys.modules


def test_wrong_env_segments_picks_different_env_and_validates(cfg: Config):
    segments = [
        C.EnvSegment(
            start=0.0,
            end=2.0,
            env=cfg.env_ids[0],
            confidence=0.9,
            probs={cfg.env_ids[0]: 0.9, cfg.env_ids[1]: 0.1},
        ),
        C.EnvSegment(
            start=2.0,
            end=5.0,
            env=cfg.env_ids[1],
            confidence=1.0,
            probs={cfg.env_ids[1]: 1.0},
        ),
    ]
    C.validate_segments(segments, 5.0)

    wrong = wrong_env_segments(segments, cfg)

    assert len(wrong) == len(segments)
    C.validate_segments(wrong, 5.0)

    for orig, w in zip(segments, wrong):
        w.validate()
        assert w.start == orig.start
        assert w.end == orig.end
        assert w.confidence == 1.0
        assert sum(w.probs.values()) == pytest.approx(1.0)
        assert w.env != orig.env
        assert w.probs == {w.env: 1.0}


def test_wrong_env_segments_deterministic_lowest_cosine(cfg: Config):
    from soundthing.events.conditioning import prior_weights

    env_vectors = {e: prior_weights(cfg, {e: 1.0}, cfg.class_ids) for e in cfg.env_ids}
    env_norms = {e: np.linalg.norm(vec) for e, vec in env_vectors.items()}

    def cos_sim(e1, e2):
        return np.dot(env_vectors[e1], env_vectors[e2]) / (env_norms[e1] * env_norms[e2])

    top_env = cfg.env_ids[0]
    expected_wrong = min(
        [e for e in cfg.env_ids if e != top_env],
        key=lambda cand: (cos_sim(top_env, cand), cand),
    )

    segs = [C.EnvSegment(start=0.0, end=3.0, env=top_env, confidence=1.0, probs={top_env: 1.0})]
    result = wrong_env_segments(segs, cfg)
    assert result[0].env == expected_wrong


def test_wrong_env_report_tmp_runs(cfg: Config, fake_env, tmp_path: Path):
    runs_dir = tmp_path / "runs"
    labels_dir = tmp_path / "labels"
    clip_id = "test_clip"

    clip_run = runs_dir / clip_id
    clip_run.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    audio_wav = clip_run / "audio.wav"
    sf.write(str(audio_wav), np.zeros(16000 * 3, dtype=np.float32), 16000)

    inputs = C.ExtractedInputs(
        video_id=clip_id,
        video_path=str(clip_run / f"{clip_id}.mp4"),
        audio_path=str(audio_wav),
        frames_dir=str(clip_run / "frames"),
        fps=1.0,
        duration=3.0,
        sample_rate=16000,
    )
    C.save_inputs(inputs, clip_run / "inputs.json")

    segs = [
        C.EnvSegment(
            start=0.0,
            end=3.0,
            env=cfg.env_ids[0],
            confidence=1.0,
            probs={cfg.env_ids[0]: 1.0},
        )
    ]
    C.save_segments(segs, clip_run / "env_segments.json")

    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.set_score(sc.audioset[0], 0.95)

    labels = C.ClipLabels(
        video_id=clip_id,
        duration=3.0,
        labeler="test",
        segments=[C.LabeledSegment(start=0.0, end=3.0, env=cfg.env_ids[0])],
        events=[C.LabeledEvent(start=1.0, end=2.0, class_id=sc.id)],
    )
    C.save_labels(labels, labels_dir / f"{clip_id}.json")

    stored_evs = [
        C.CaptionEvent(
            start=1.0,
            end=2.0,
            label=cfg.caption_for(sc.id),
            class_id=sc.id,
            confidence=0.9,
        )
    ]
    C.save_events(stored_evs, clip_run / "events.baseline.json")
    C.save_events(stored_evs, clip_run / "events.conditioned.json")

    files_before = set(clip_run.iterdir())

    report = wrong_env_report(runs_dir, labels_dir, cfg)

    files_after = set(clip_run.iterdir())
    assert files_before == files_after, "wrong_env_report must not write files into run dirs"

    assert clip_id in report["clips"]
    clip_report = report["clips"][clip_id]

    expected_modes_levels = [
        "baseline.fine",
        "baseline.coarse",
        "conditioned.fine",
        "conditioned.coarse",
        "wrong_env.fine",
        "wrong_env.coarse",
    ]
    for key in expected_modes_levels:
        assert key in clip_report
        assert key in report["totals"]
        summary = report["totals"][key]
        assert "precision" in summary
        assert "recall" in summary
        assert "f1" in summary
        assert "fp_per_min" in summary
        assert summary["minutes"] > 0


def test_experiments_main_with_out(cfg: Config, fake_env, tmp_path: Path):
    runs_dir = tmp_path / "runs"
    labels_dir = tmp_path / "labels"
    clip_id = "test_clip"

    clip_run = runs_dir / clip_id
    clip_run.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    audio_wav = clip_run / "audio.wav"
    sf.write(str(audio_wav), np.zeros(16000 * 2, dtype=np.float32), 16000)

    inputs = C.ExtractedInputs(
        video_id=clip_id,
        video_path=str(clip_run / f"{clip_id}.mp4"),
        audio_path=str(audio_wav),
        frames_dir=str(clip_run / "frames"),
        fps=1.0,
        duration=2.0,
        sample_rate=16000,
    )
    C.save_inputs(inputs, clip_run / "inputs.json")
    segs = [
        C.EnvSegment(
            start=0.0,
            end=2.0,
            env=cfg.env_ids[0],
            confidence=1.0,
            probs={cfg.env_ids[0]: 1.0},
        )
    ]
    C.save_segments(segs, clip_run / "env_segments.json")

    sc = next(c for c in cfg.classes if c.audioset)
    labels = C.ClipLabels(
        video_id=clip_id,
        duration=2.0,
        labeler="test",
        segments=[],
        events=[C.LabeledEvent(start=0.5, end=1.5, class_id=sc.id)],
    )
    C.save_labels(labels, labels_dir / f"{clip_id}.json")

    out_file = tmp_path / "report.json"
    ret = main(["--runs", str(runs_dir), "--labels", str(labels_dir), "--out", str(out_file)])
    assert ret == 0
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "clips" in data
    assert "totals" in data
    assert clip_id in data["clips"]


def test_experiments_main_stdout(cfg: Config, fake_env, tmp_path: Path, capsys):
    runs_dir = tmp_path / "runs"
    labels_dir = tmp_path / "labels"
    runs_dir.mkdir()
    labels_dir.mkdir()

    ret = main(["--runs", str(runs_dir), "--labels", str(labels_dir)])
    assert ret == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "clips" in data
    assert "totals" in data
