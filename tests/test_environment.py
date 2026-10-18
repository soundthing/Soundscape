"""Person 2 environment stage tests that do not download model weights."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from soundthing.environment import audio, classifier, compare, segmentation, visual


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def fake_inference(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=48000)

    def __call__(self, text=None, audio=None, images=None, **kwargs):
        return {"items": text if text is not None else audio if audio is not None else images}


class FakeModel:
    def get_text_features(self, items):
        return FakeTensor([[1, 0] if "first" in item else [0, 1] for item in items])

    def get_audio_features(self, items):
        return FakeTensor([[float(np.mean(item)), 1 - float(np.mean(item))] for item in items])

    def get_image_features(self, items):
        return FakeTensor([[1 - image.getpixel((0, 0))[0] / 255,
                            image.getpixel((0, 0))[0] / 255] for image in items])


def test_scene_spans_merge_short_shots_and_split_long_ones(cfg, monkeypatch):
    class Time:
        def __init__(self, seconds):
            self.seconds = seconds

        def get_seconds(self):
            return self.seconds

    class Manager:
        def add_detector(self, detector):
            pass

        def detect_scenes(self, video):
            pass

        def get_scene_list(self):
            return [(None, Time(t)) for t in (4, 9, 45)]

    fake = SimpleNamespace(ContentDetector=lambda **kw: object(), SceneManager=Manager,
                           open_video=lambda path: object())
    monkeypatch.setitem(sys.modules, "scenedetect", fake)
    spans = segmentation.detect_spans("movie.mp4", 70, cfg)
    assert spans[0] == (0, 9)  # the four-second shot joins its neighbour
    assert all(end - start >= 5 for start, end in spans)
    assert all(end - start <= 30 for start, end in spans)
    assert spans[-1][1] == 70
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_classify_audio_only_and_missing_frames(cfg, fake_inputs, monkeypatch):
    from soundthing.contracts import load_inputs, validate_segments

    inputs = load_inputs(fake_inputs)
    monkeypatch.setattr(classifier, "detect_spans", lambda *args: [(0, 10), (10, 30)])
    monkeypatch.setattr(classifier, "audio_env_logits", lambda *args: np.tile(
        np.array([1.0] + [0.0] * (len(cfg.environments) - 2)), (2, 1)))
    monkeypatch.setattr(classifier, "visual_env_logits", lambda *args: pytest.fail("visual branch ran"))
    segments = classifier.classify(inputs, cfg)
    validate_segments(segments, 30)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (0, 30)
    assert segments[0].env == cfg.environments[1].id
    with pytest.raises(FileNotFoundError):
        classifier.classify(inputs, cfg, use_audio=False)
    with pytest.raises(ValueError, match="at least one"):
        classifier.classify(inputs, cfg, use_audio=False, use_visual=False)


def test_classify_merges_only_adjacent_matching_environments(cfg, fake_inputs, monkeypatch):
    from soundthing.contracts import load_inputs

    inputs = load_inputs(fake_inputs)
    monkeypatch.setattr(classifier, "detect_spans", lambda *args: [(0, 5), (5, 12), (12, 30)])
    logits = np.zeros((3, len(cfg.environments) - 1))
    logits[:2, 0] = 1
    logits[2, 1] = 1
    monkeypatch.setattr(classifier, "audio_env_logits", lambda *args: logits)

    segments = classifier.classify(inputs, cfg, use_visual=False)

    assert [(s.start, s.end, s.env) for s in segments] == [
        (0, 12, cfg.environments[1].id),
        (12, 30, cfg.environments[2].id),
    ]


def test_classify_uses_general_when_no_visual_prompt_matches(cfg, fake_inputs, monkeypatch):
    from pathlib import Path
    from soundthing.contracts import load_inputs

    inputs = load_inputs(fake_inputs)
    frames = Path(inputs.frames_dir)
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "frame_000000.jpg").write_bytes(b"placeholder")
    spans = [(0, 10), (10, 20), (20, 30)]
    env_ids = [env.id for env in cfg.environments if env.id != "general"]
    audio_logits = np.full((3, len(env_ids)), -1.0)
    audio_logits[:, env_ids.index("sports_venue")] = 1.0
    visual_logits = np.full((3, len(env_ids)), -1.0)
    visual_logits[:, env_ids.index("kitchen")] = [0.2, -0.1, 0.2]
    monkeypatch.setattr(classifier, "detect_spans", lambda *args: spans)
    monkeypatch.setattr(classifier, "audio_env_logits", lambda *args: audio_logits)
    monkeypatch.setattr(classifier, "visual_env_logits", lambda *args: visual_logits)

    segments = classifier.classify(inputs, cfg)

    assert [(s.start, s.end, s.env) for s in segments] == [
        (0, 10, "kitchen"), (10, 20, "general"), (20, 30, "kitchen")
    ]
    assert segments[1].confidence == pytest.approx(1 / len(env_ids))
    assert segments[1].probs["general"] > 0.9


def test_clap_audio_prefers_original_track_and_resamples(cfg, fake_inputs, tmp_path, monkeypatch):
    from soundthing.contracts import load_inputs

    inputs = load_inputs(fake_inputs)
    stereo = tmp_path / "original.wav"
    stereo.write_bytes(b"placeholder")
    inputs.audio_stereo_path = str(stereo)
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        return SimpleNamespace(stdout=np.ones(48000, dtype="<f4").tobytes())

    monkeypatch.setattr(audio.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(audio.subprocess, "run", run)
    samples = audio._audio_at_clap_rate(inputs, 48000)
    assert len(samples) == 48000
    assert str(stereo) in seen[0]
    assert seen[0][seen[0].index("-ar") + 1] == "48000"


def test_audio_branch_pools_ten_second_chunks(cfg, fake_inputs, monkeypatch):
    from soundthing.config import Environment
    from soundthing.contracts import load_inputs

    fake_inference(monkeypatch)
    monkeypatch.setattr(audio, "_model", lambda model_id: (FakeModel(), FakeProcessor()))
    samples = np.concatenate([np.ones(15 * 48000), np.zeros(5 * 48000)]).astype(np.float32)
    monkeypatch.setattr(audio, "_audio_at_clap_rate", lambda inputs, rate: samples)
    envs = [Environment("first", "First", audio_prompts=["first"]),
            Environment("second", "Second", audio_prompts=["second"])]
    scores = audio.audio_env_logits(load_inputs(fake_inputs), [(0, 15), (15, 20)], envs, cfg)
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_visual_branch_uses_nearest_frame_for_empty_span(cfg, fake_inputs, monkeypatch):
    PIL = pytest.importorskip("PIL.Image")
    from soundthing.config import Environment
    from soundthing.contracts import load_inputs

    fake_inference(monkeypatch)
    inputs = load_inputs(fake_inputs)
    PIL.new("RGB", (2, 2), (0, 0, 0)).save(inputs.frames_dir + "/frame_000000.jpg")
    PIL.new("RGB", (2, 2), (255, 255, 255)).save(inputs.frames_dir + "/frame_000002.jpg")
    monkeypatch.setattr(visual, "_model", lambda model_id: (FakeModel(), FakeProcessor()))
    envs = [Environment("first", "First", visual_prompts=["first"]),
            Environment("second", "Second", visual_prompts=["second"])]
    scores = visual.visual_env_logits(inputs, [(0, 0.5), (0.5, 1.5), (1.5, 3)], envs, cfg)
    assert scores.shape == (3, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 0] > scores[1, 1]
    assert scores[2, 1] > scores[2, 0]


def test_modality_report_uses_labeled_time(cfg, fake_inputs, tmp_path, monkeypatch):
    from soundthing import contracts as C

    inputs = C.load_inputs(fake_inputs)
    labels_dir = tmp_path / "labels"
    C.save_labels(C.ClipLabels(
        video_id=inputs.video_id, duration=30, labeler="test",
        segments=[C.LabeledSegment(0, 10, "kitchen"), C.LabeledSegment(10, 30, "street")],
        events=[],
    ), labels_dir / "clip01.json")

    def classify_fake(inputs, cfg, use_audio=True, use_visual=True):
        env = "street" if use_audio else "kitchen"
        if use_audio and use_visual:
            return [C.EnvSegment(0, 10, "kitchen", 1, {"kitchen": 1}),
                    C.EnvSegment(10, 30, "street", 1, {"street": 1})]
        return [C.EnvSegment(0, 30, env, 1, {env: 1})]

    monkeypatch.setattr(compare, "classify", classify_fake)
    report = compare.compare(fake_inputs.parents[1], labels_dir)
    assert report["mean_accuracy"] == {"audio": 2 / 3, "visual": 1 / 3, "fused": 1.0}
    assert report["overall_accuracy"] == report["mean_accuracy"]
    assert report["confusion_seconds"]["audio"]["kitchen"]["street"] == 10
