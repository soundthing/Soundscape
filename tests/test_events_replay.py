from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, EnvSegment, ExtractedInputs
from soundthing.events import clap, detector, vad
from soundthing.events.core import EventModels
from soundthing.events.tagger import build_tagger_map
from soundthing.events.cache import (
    ClipCache,
    build_cache,
    load_cache,
    save_cache,
)
from soundthing.harness.paths import cache_root
from soundthing.events.replay import (
    _clear_text_emb_memo,
    class_scores,
    replay,
    text_embeddings_for,
)


class ControllableTagger:
    """Tagger whose output varies with audio energy and chunk index."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.active_class_name: str | None = None
        self.call_count = 0

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.zeros((b, len(self.names)), dtype=np.float32)

        # High score if chunk has substantial energy (> 0.1 RMS)
        rms = float(np.sqrt(np.mean(audio**2)))
        if self.active_class_name and self.active_class_name in self.names and rms > 0.1:
            idx = self.names.index(self.active_class_name)
            probs[:, idx] = 0.95
        return probs


class ControllableClap:
    """CLAP mock with deterministic embeddings matching prompt text and audio."""

    def __init__(self, dim: int = 16, center: float = 0.2, scale: float = 20.0) -> None:
        self.dim = dim
        self.center = center
        self.scale = scale
        self.target_prompt: str | None = None

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        out = []
        for p in prompts:
            vec = np.zeros(self.dim, dtype=np.float32)
            if self.target_prompt and p == self.target_prompt:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            out.append(vec)
        return np.array(out, dtype=np.float32)

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Match target_prompt embedding (vec[0]=1.0) when audio has energy in second half
        vec = np.zeros(self.dim, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms > 0.1:
            vec[0] = 1.0
        else:
            vec[-1] = 1.0
        return vec


@pytest.fixture(autouse=True)
def setup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))
    detector._clear_caches()
    _clear_text_emb_memo()
    yield
    detector._clear_caches()
    _clear_text_emb_memo()


def test_equivalence_detect_and_replay(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Select an audioset class and a clap class
    sc_tagger = next(c for c in cfg.classes if c.audioset and not c.safety)
    sc_clap = next(c for c in cfg.classes if c.clap and not c.audioset and not c.safety)

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)

    fake_tagger = ControllableTagger(audioset_names)
    fake_tagger.active_class_name = sc_tagger.audioset[0]

    fake_clap = ControllableClap(dim=16)
    fake_clap.target_prompt = sc_clap.clap[0]

    text_emb = clap.text_embeddings(fake_clap, cfg)

    models = EventModels(
        tagger=fake_tagger,
        clap=fake_clap,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=text_emb,
        audioset_names=audioset_names,
    )

    monkeypatch.setattr(detector, "load_models", lambda c: models)
    monkeypatch.setattr(detector, "get_models", lambda c: models)

    # 4s audio: first 2s has energy (triggers tagger and clap)
    sr = 16000
    duration = 4.0
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    audio[: int(2.0 * sr)] = 0.5  # high energy in 0..2s

    clip_dir = tmp_path / "clip01"
    clip_dir.mkdir(parents=True, exist_ok=True)
    audio_path = clip_dir / "audio.wav"
    sf.write(str(audio_path), audio, sr)

    inputs = ExtractedInputs(
        video_id="clip01",
        video_path=str(clip_dir / "clip01.mp4"),
        audio_path=str(audio_path),
        frames_dir=str(clip_dir / "frames"),
        fps=1.0,
        duration=duration,
        sample_rate=sr,
    )

    # Fake VAD: speech run between 0.0 and 1.5s
    n_vad_chunks = int(duration / vad.VAD_CHUNK_SEC)
    vad_probs = np.zeros(n_vad_chunks, dtype=np.float32)
    for i in range(n_vad_chunks):
        if i * vad.VAD_CHUNK_SEC < 1.5:
            vad_probs[i] = 0.95

    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: vad_probs)

    # Cache
    cache = build_cache(audio, sr, "clip01", cfg, models=models)

    # 1. Equivalence in unconditioned baseline (segments=None)
    strip_mods = lambda evs: [replace(e, modifiers=[]) for e in evs]

    det_base = strip_mods(detector.detect(inputs, None, cfg))
    rep_base = replay(cache, None, cfg, text_emb=models.clap_text_emb)

    assert len(det_base) > 0, "Detections must be non-empty"
    assert len(rep_base) > 0, "Replay must be non-empty"
    assert det_base == rep_base

    # Check that at least one CLAP-driven event is present
    clap_events = [e for e in det_base if e.class_id == sc_clap.id]
    assert len(clap_events) > 0, "Must include at least one CLAP-driven event"

    # 2. Equivalence in conditioned mode
    segs = [
        EnvSegment(
            start=0.0,
            end=duration,
            env=cfg.env_ids[0],
            confidence=1.0,
            probs={cfg.env_ids[0]: 1.0},
        )
    ]
    detector._clear_caches()
    det_cond = strip_mods(detector.detect(inputs, segs, cfg))
    rep_cond = replay(cache, segs, cfg, text_emb=models.clap_text_emb)

    assert len(det_cond) > 0
    assert det_cond == rep_cond

    # 3. Speech suppression effect: assert that suppressed event differs when VAD is off
    vad_probs_off = np.zeros(n_vad_chunks, dtype=np.float32)
    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: vad_probs_off)
    cache_novad = build_cache(audio, sr, "clip01", cfg, models=models)
    rep_novad = replay(cache_novad, None, cfg, text_emb=models.clap_text_emb)

    # Scores under speech are multiplied by 0.3 when VAD is on, so rep_base and rep_novad must differ
    assert rep_base != rep_novad, "Speech suppression must have a demonstrable effect"


def test_replay_on_loaded_cache_equals_original(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    sc = next(c for c in cfg.classes if c.audioset)

    tagger = ControllableTagger(audioset_names)
    tagger.active_class_name = sc.audioset[0]

    models = EventModels(
        tagger=tagger,
        clap=None,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
    )

    sr = 16000
    audio = np.zeros(sr * 3, dtype=np.float32)
    audio[:sr] = 0.8
    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: np.zeros(10, dtype=np.float32))

    cache = build_cache(audio, sr, "roundtrip", cfg, models=models)

    cache_file = tmp_path / "cache.npz"
    save_cache(cache, cache_file)
    loaded = load_cache(cache_file)

    orig_events = replay(cache, None, cfg)
    loaded_events = replay(loaded, None, cfg)

    assert len(orig_events) > 0
    assert orig_events == loaded_events


def test_onset_shift_sec(cfg: Config):
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_raw = np.zeros((10, len(audioset_names)), dtype=np.float16)
    # Event active from window 2 to 5 (approx 0.5s to 1.5s)
    tagger_raw[2:6, 0] = 0.95
    times = np.arange(10) * 0.25 + 0.125
    vad_probs = np.zeros(10, dtype=np.float32)

    cache = ClipCache(
        video_id="shift_clip",
        duration=3.0,
        key="testkey1234",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=0.032,
        vad_probs=vad_probs,
    )

    # Base config: onset_shift_sec = 0.0
    p0 = copy.deepcopy(cfg.pipeline)
    p0["events"]["postprocess"]["onset_shift_sec"] = 0.0
    cfg0 = replace(cfg, pipeline=p0)
    evs0 = replay(cache, None, cfg0)
    assert len(evs0) > 0
    base_ev = evs0[0]

    # Positive shift: +0.2 s
    p_pos = copy.deepcopy(cfg.pipeline)
    p_pos["events"]["postprocess"]["onset_shift_sec"] = 0.2
    cfg_pos = replace(cfg, pipeline=p_pos)
    evs_pos = replay(cache, None, cfg_pos)
    assert len(evs_pos) == len(evs0)
    assert evs_pos[0].start == pytest.approx(base_ev.start + 0.2)
    assert evs_pos[0].end == pytest.approx(min(base_ev.end + 0.2, cache.duration))

    # Negative shift clamps start at 0
    p_neg = copy.deepcopy(cfg.pipeline)
    p_neg["events"]["postprocess"]["onset_shift_sec"] = -(base_ev.start + 0.5)
    cfg_neg = replace(cfg, pipeline=p_neg)
    evs_neg = replay(cache, None, cfg_neg)
    if evs_neg:
        assert evs_neg[0].start == 0.0
        assert evs_neg[0].end > 0.0


def test_combine_modes(cfg: Config):
    sc = next(c for c in cfg.classes if c.audioset and c.clap)
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})

    times = np.array([0.5], dtype=np.float64)
    # Tagger gives low score (0.2)
    tagger_raw = np.full((1, len(audioset_names)), 0.2, dtype=np.float16)

    # CLAP audio embedding aligned with text embedding -> CLAP similarity = 1.0 -> score ~1.0
    dim = 8
    clap_emb = np.zeros((1, dim), dtype=np.float16)
    clap_emb[0, 0] = 1.0

    text_emb = np.zeros((len(cfg.classes), dim), dtype=np.float32)
    sc_idx = cfg.class_ids.index(sc.id)
    text_emb[sc_idx, 0] = 1.0

    cache = ClipCache(
        video_id="combine_test",
        duration=1.0,
        key="key123",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=clap_emb,
        vad_chunk_sec=0.032,
        vad_probs=np.zeros(1, dtype=np.float32),
    )

    # 1. combine: max -> class gets high CLAP score (~1.0)
    p_max = copy.deepcopy(cfg.pipeline)
    p_max["events"]["clap"]["combine"] = "max"
    cfg_max = replace(cfg, pipeline=p_max)
    scores_max = class_scores(cache, cfg_max, text_emb=text_emb)
    val_max = scores_max.column(sc.id)[0]
    assert val_max > 0.9

    # 2. combine: fallback -> class has AudioSet labels, so CLAP score is ignored, tagger score (0.2) used
    p_fallback = copy.deepcopy(cfg.pipeline)
    p_fallback["events"]["clap"]["combine"] = "fallback"
    cfg_fallback = replace(cfg, pipeline=p_fallback)
    scores_fallback = class_scores(cache, cfg_fallback, text_emb=text_emb)
    val_fallback = scores_fallback.column(sc.id)[0]
    assert abs(val_fallback - 0.2) < 1e-4

    assert val_max != val_fallback

    # 3. Unknown combine mode raises ValueError
    p_bad = copy.deepcopy(cfg.pipeline)
    p_bad["events"]["clap"]["combine"] = "invalid_mode"
    cfg_bad = replace(cfg, pipeline=p_bad)
    with pytest.raises(ValueError, match="Unknown clap combine mode"):
        class_scores(cache, cfg_bad, text_emb=text_emb)


def test_text_embeddings_for_memo_and_disk(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    fake_clap = ControllableClap(dim=8)
    load_count = 0

    def fake_load_clap(c):
        nonlocal load_count
        load_count += 1
        return fake_clap

    monkeypatch.setattr(clap, "load_clap", fake_load_clap)

    # First call: computes via fake_load_clap and writes to disk
    emb1 = text_embeddings_for(cfg)
    assert emb1 is not None
    assert emb1.shape == (len(cfg.classes), 8)
    assert load_count == 1

    # In-process memo hit: no load_clap call
    emb2 = text_embeddings_for(cfg)
    assert emb2 is emb1
    assert load_count == 1

    # Clear in-process memo: disk hit, no load_clap call
    _clear_text_emb_memo()
    emb3 = text_embeddings_for(cfg)
    assert load_count == 1
    assert np.array_equal(emb3, emb1)

    # Disabled clap returns None
    p_disabled = copy.deepcopy(cfg.pipeline)
    p_disabled["events"]["clap"]["enabled"] = False
    cfg_disabled = replace(cfg, pipeline=p_disabled)
    assert text_embeddings_for(cfg_disabled) is None


def test_replay_separation_disabled_with_sep_arrays_matches_without_them(cfg: Config):
    """When separation is disabled in cfg, caches with sep arrays match caches without them."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    times = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    tagger_raw = np.full((4, len(audioset_names)), 0.2, dtype=np.float16)
    tagger_raw[1:3, 0] = 0.95
    tagger_sep = np.full((4, len(audioset_names)), 0.8, dtype=np.float16)
    sep_mask = np.array([False, True, True, False], dtype=bool)

    cache_with_sep = ClipCache(
        video_id="disabled_test",
        duration=2.0,
        key="key123",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=0.032,
        vad_probs=np.zeros(4, dtype=np.float32),
        sep_mask=sep_mask,
        tagger_sep=tagger_sep,
        clap_emb_sep=None,
    )
    cache_no_sep = replace(cache_with_sep, sep_mask=None, tagger_sep=None, clap_emb_sep=None)

    # Separation is disabled in cfg by default
    assert not cfg.pipeline["events"]["separation"]["enabled"]

    scores_with = class_scores(cache_with_sep, cfg)
    scores_no = class_scores(cache_no_sep, cfg)
    np.testing.assert_array_equal(scores_with.scores, scores_no.scores)

    events_with = replay(cache_with_sep, None, cfg)
    events_no = replay(cache_no_sep, None, cfg)
    assert events_with == events_no


def test_replay_separation_fusion_modes(cfg: Config):
    """Enabled separation fuses masked rows according to mode and leaves unmasked rows as raw."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    sc0 = cfg.classes[0]
    sc1 = cfg.classes[1]
    name0 = sc0.audioset[0] if sc0.audioset else audioset_names[0]
    name1 = sc1.audioset[0] if sc1.audioset else audioset_names[1]
    idx0 = audioset_names.index(name0)
    idx1 = audioset_names.index(name1)

    times = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    tagger_raw = np.full((3, len(audioset_names)), 0.1, dtype=np.float16)
    tagger_sep = np.full((3, len(audioset_names)), 0.1, dtype=np.float16)

    # Window 0: unmasked
    tagger_raw[0, idx0] = 0.5
    tagger_sep[0, idx0] = 0.2
    # Window 1: masked
    tagger_raw[1, idx0] = 0.6  # raw > sep
    tagger_sep[1, idx0] = 0.2
    tagger_raw[1, idx1] = 0.3  # sep > raw
    tagger_sep[1, idx1] = 0.8
    # Window 2: unmasked
    tagger_raw[2, idx1] = 0.4
    tagger_sep[2, idx1] = 0.9

    sep_mask = np.array([False, True, False], dtype=bool)

    cache = ClipCache(
        video_id="fusion_test",
        duration=1.5,
        key="key_fusion",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=0.032,
        vad_probs=np.zeros(4, dtype=np.float32),
        sep_mask=sep_mask,
        tagger_sep=tagger_sep,
        clap_emb_sep=None,
    )

    # Base raw scores for comparison
    p_off = copy.deepcopy(cfg.pipeline)
    p_off["events"]["separation"]["enabled"] = False
    cfg_off = replace(cfg, pipeline=p_off)
    raw_scores = class_scores(cache, cfg_off).scores

    # 1. fusion: max
    p_max = copy.deepcopy(cfg.pipeline)
    p_max["events"]["separation"]["enabled"] = True
    p_max["events"]["separation"]["fusion"] = "max"
    cfg_max = replace(cfg, pipeline=p_max)
    scores_max = class_scores(cache, cfg_max).scores

    # Unmasked rows equal raw
    np.testing.assert_array_equal(scores_max[0], raw_scores[0])
    np.testing.assert_array_equal(scores_max[2], raw_scores[2])
    # Masked row 1 fused with max
    assert scores_max[1, cfg.class_ids.index(sc0.id)] == pytest.approx(0.6, abs=1e-3)
    assert scores_max[1, cfg.class_ids.index(sc1.id)] == pytest.approx(0.8, abs=1e-3)

    # 2. fusion: sep
    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["separation"]["enabled"] = True
    p_sep["events"]["separation"]["fusion"] = "sep"
    cfg_sep = replace(cfg, pipeline=p_sep)
    scores_sep = class_scores(cache, cfg_sep).scores

    np.testing.assert_array_equal(scores_sep[0], raw_scores[0])
    np.testing.assert_array_equal(scores_sep[2], raw_scores[2])
    assert scores_sep[1, cfg.class_ids.index(sc0.id)] == pytest.approx(0.2, abs=1e-3)
    assert scores_sep[1, cfg.class_ids.index(sc1.id)] == pytest.approx(0.8, abs=1e-3)

    # 3. fusion: weighted
    p_weighted = copy.deepcopy(cfg.pipeline)
    p_weighted["events"]["separation"]["enabled"] = True
    p_weighted["events"]["separation"]["fusion"] = "weighted"
    p_weighted["events"]["separation"]["sep_weight"] = 0.25
    cfg_weighted = replace(cfg, pipeline=p_weighted)
    scores_weighted = class_scores(cache, cfg_weighted).scores

    np.testing.assert_array_equal(scores_weighted[0], raw_scores[0])
    np.testing.assert_array_equal(scores_weighted[2], raw_scores[2])
    expected_sc0 = 0.25 * 0.2 + 0.75 * 0.6  # 0.50
    expected_sc1 = 0.25 * 0.8 + 0.75 * 0.3  # 0.425
    assert scores_weighted[1, cfg.class_ids.index(sc0.id)] == pytest.approx(expected_sc0, abs=1e-3)
    assert scores_weighted[1, cfg.class_ids.index(sc1.id)] == pytest.approx(expected_sc1, abs=1e-3)


def test_replay_separation_enabled_with_none_sep_arrays(cfg: Config):
    """When separation is enabled but cache sep arrays are None, output is identical to raw."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    times = np.array([0.25, 0.5], dtype=np.float64)
    tagger_raw = np.full((2, len(audioset_names)), 0.3, dtype=np.float16)

    cache = ClipCache(
        video_id="none_sep_arrays",
        duration=1.0,
        key="key_none",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=0.032,
        vad_probs=np.zeros(4, dtype=np.float32),
        sep_mask=None,
        tagger_sep=None,
        clap_emb_sep=None,
    )

    p_enabled = copy.deepcopy(cfg.pipeline)
    p_enabled["events"]["separation"]["enabled"] = True
    cfg_enabled = replace(cfg, pipeline=p_enabled)

    p_disabled = copy.deepcopy(cfg.pipeline)
    p_disabled["events"]["separation"]["enabled"] = False
    cfg_disabled = replace(cfg, pipeline=p_disabled)

    scores_enabled = class_scores(cache, cfg_enabled)
    scores_disabled = class_scores(cache, cfg_disabled)
    np.testing.assert_array_equal(scores_enabled.scores, scores_disabled.scores)
    assert replay(cache, None, cfg_enabled) == replay(cache, None, cfg_disabled)


def test_replay_separation_vad_suppression(cfg: Config):
    """Speech region overlapping masked and unmasked windows uses appropriate factors, leaving safety untouched."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    sc_safety = next(c for c in cfg.classes if c.safety and c.audioset)
    sc_nonsafety = next(c for c in cfg.classes if not c.safety and c.audioset)
    idx_safety = cfg.class_ids.index(sc_safety.id)
    idx_nonsafety = cfg.class_ids.index(sc_nonsafety.id)

    times = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    duration = 4.0
    vad_chunk_sec = 0.5
    # Chunks 0..3 (0.0 to 2.0s) have speech (speech region [0.0, 2.1s] covering window 0 and 1)
    vad_probs = np.array([0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Window 0 (0.5s): speech, unmasked
    # Window 1 (1.5s): speech, masked
    # Window 2 (2.5s): no speech, unmasked
    # Window 3 (3.5s): no speech, masked
    sep_mask = np.array([False, True, False, True], dtype=bool)

    tagger_raw = np.full((4, len(audioset_names)), 0.0, dtype=np.float16)
    tagger_sep = np.full((4, len(audioset_names)), 0.0, dtype=np.float16)

    # Map audioset classes to 0.8 in both raw and sep so fusion doesn't change baseline score
    tagger_raw[:, audioset_names.index(sc_safety.audioset[0])] = 0.8
    tagger_sep[:, audioset_names.index(sc_safety.audioset[0])] = 0.8
    tagger_raw[:, audioset_names.index(sc_nonsafety.audioset[0])] = 0.8
    tagger_sep[:, audioset_names.index(sc_nonsafety.audioset[0])] = 0.8

    cache = ClipCache(
        video_id="vad_supp_test",
        duration=duration,
        key="key_vad",
        window_sec=1.0,
        hop=1.0,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=vad_chunk_sec,
        vad_probs=vad_probs,
        sep_mask=sep_mask,
        tagger_sep=tagger_sep,
        clap_emb_sep=None,
    )

    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"]["enabled"] = True
    p["events"]["separation"]["fusion"] = "max"
    p["events"]["vad"]["suppress_under_speech"] = 0.2
    p["events"]["separation"]["suppress_under_speech"] = 0.7
    cfg_test = replace(cfg, pipeline=p)

    scores = class_scores(cache, cfg_test).scores

    # Window 0: speech + unmasked -> vad.suppress_under_speech (0.2)
    assert scores[0, idx_nonsafety] == pytest.approx(0.8 * 0.2, abs=1e-3)
    assert scores[0, idx_safety] == pytest.approx(0.8, abs=1e-3)  # safety untouched

    # Window 1: speech + masked -> separation.suppress_under_speech (0.7)
    assert scores[1, idx_nonsafety] == pytest.approx(0.8 * 0.7, abs=1e-3)
    assert scores[1, idx_safety] == pytest.approx(0.8, abs=1e-3)  # safety untouched

    # Window 2: no speech + unmasked -> no suppression (1.0)
    assert scores[2, idx_nonsafety] == pytest.approx(0.8, abs=1e-3)
    assert scores[2, idx_safety] == pytest.approx(0.8, abs=1e-3)

    # Window 3: no speech + masked -> no suppression (1.0)
    assert scores[3, idx_nonsafety] == pytest.approx(0.8, abs=1e-3)
    assert scores[3, idx_safety] == pytest.approx(0.8, abs=1e-3)


def test_replay_separation_validation_errors(cfg: Config):
    """Unknown fusion mode or mismatched tagger_sep / sep_mask shapes raise ValueError."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    times = np.array([0.25, 0.5], dtype=np.float64)
    tagger_raw = np.zeros((2, len(audioset_names)), dtype=np.float16)

    cache = ClipCache(
        video_id="val_err_test",
        duration=1.0,
        key="key_err",
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=audioset_names,
        tagger_raw=tagger_raw,
        clap_emb=None,
        vad_chunk_sec=0.032,
        vad_probs=np.zeros(4, dtype=np.float32),
        sep_mask=np.array([False, True], dtype=bool),
        tagger_sep=np.zeros((2, len(audioset_names)), dtype=np.float16),
        clap_emb_sep=None,
    )

    # 1. Unknown fusion mode
    p_bad_fusion = copy.deepcopy(cfg.pipeline)
    p_bad_fusion["events"]["separation"]["enabled"] = True
    p_bad_fusion["events"]["separation"]["fusion"] = "unknown_mode"
    cfg_bad_fusion = replace(cfg, pipeline=p_bad_fusion)

    with pytest.raises(ValueError, match="Unknown separation fusion mode"):
        class_scores(cache, cfg_bad_fusion)

    # 2. Mismatched tagger_sep shape (e.g. 3 rows instead of 2)
    p_valid = copy.deepcopy(cfg.pipeline)
    p_valid["events"]["separation"]["enabled"] = True
    p_valid["events"]["separation"]["fusion"] = "max"
    cfg_valid = replace(cfg, pipeline=p_valid)

    cache_bad_tagger = replace(
        cache,
        tagger_sep=np.zeros((3, len(audioset_names)), dtype=np.float16),
    )
    with pytest.raises(ValueError, match="tagger_sep shape"):
        class_scores(cache_bad_tagger, cfg_valid)

    # 3. Mismatched sep_mask length (e.g. length 3 instead of 2)
    cache_bad_mask = replace(
        cache,
        sep_mask=np.array([True, False, True], dtype=bool),
    )
    with pytest.raises(ValueError, match="sep_mask length"):
        class_scores(cache_bad_mask, cfg_valid)

