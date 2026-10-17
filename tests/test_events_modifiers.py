import copy
import numpy as np
import pytest

from soundthing.contracts import CaptionEvent, MODIFIERS
from soundthing.events.postprocess import add_modifiers


def _make_event(cfg, start: float = 0.0, end: float = 2.0, modifiers: list[str] | None = None) -> CaptionEvent:
    class_id = cfg.classes[0].id
    caption = cfg.caption_for(class_id)
    return CaptionEvent(
        start=start,
        end=end,
        label=caption,
        class_id=class_id,
        confidence=0.9,
        modifiers=list(modifiers) if modifiers is not None else [],
    )


def test_right_panned_noise(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 1] = noise  # Channel 1 (R) only

    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "right" in res[0].modifiers
    assert "left" not in res[0].modifiers
    res[0].validate()


def test_left_panned_noise(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 0] = noise  # Channel 0 (L) only

    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "left" in res[0].modifiers
    assert "right" not in res[0].modifiers
    res[0].validate()


def test_centred_noise_gets_neither_left_nor_right(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.column_stack([noise, noise])

    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "left" not in res[0].modifiers
    assert "right" not in res[0].modifiers
    res[0].validate()


def test_approaching_ramp(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(3.0 * sr)
    noise = rng.standard_normal(n) * 0.05
    gain_up = 10.0 ** np.linspace(-1.0, 0.0, n)  # 20 dB increase
    audio = noise * gain_up
    stereo = np.column_stack([audio, audio])

    ev = _make_event(cfg, 0.0, 3.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "approaching" in res[0].modifiers
    assert "receding" not in res[0].modifiers
    res[0].validate()


def test_receding_ramp(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(3.0 * sr)
    noise = rng.standard_normal(n) * 0.05
    gain_down = 10.0 ** np.linspace(0.0, -1.0, n)  # 20 dB decrease
    audio = noise * gain_down
    stereo = np.column_stack([audio, audio])

    ev = _make_event(cfg, 0.0, 3.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "receding" in res[0].modifiers
    assert "approaching" not in res[0].modifiers
    res[0].validate()


def test_short_ramp_gets_no_trend(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(1.0 * sr)
    noise = rng.standard_normal(n) * 0.05
    gain_up = 10.0 ** np.linspace(-1.0, 0.0, n)
    audio = noise * gain_up
    stereo = np.column_stack([audio, audio])

    ev = _make_event(cfg, 0.0, 1.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "approaching" not in res[0].modifiers
    assert "receding" not in res[0].modifiers
    res[0].validate()


def test_loud_burst_above_quiet_background(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(10.0 * sr)
    audio = rng.standard_normal(n) * 0.01  # Quiet background (~-40 dB)
    burst_start = int(4.0 * sr)
    burst_end = int(5.0 * sr)
    audio[burst_start:burst_end] = rng.standard_normal(burst_end - burst_start) * 0.10  # 20 dB above
    stereo = np.column_stack([audio, audio])

    ev = _make_event(cfg, 4.0, 5.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "loud" in res[0].modifiers
    assert "quiet" not in res[0].modifiers
    res[0].validate()


def test_quiet_dip_below_background(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(10.0 * sr)
    audio = rng.standard_normal(n) * 0.01  # Baseline (~-40 dB)
    dip_start = int(4.0 * sr)
    dip_end = int(5.0 * sr)
    audio[dip_start:dip_end] = rng.standard_normal(dip_end - dip_start) * 0.001  # 20 dB below
    stereo = np.column_stack([audio, audio])

    ev = _make_event(cfg, 4.0, 5.0)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert "quiet" in res[0].modifiers
    assert "loud" not in res[0].modifiers
    res[0].validate()


def test_mono_input_gives_no_left_right(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    mono = rng.standard_normal(n) * 0.1

    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], mono, sr, cfg)

    assert len(res) == 1
    assert "left" not in res[0].modifiers
    assert "right" not in res[0].modifiers
    res[0].validate()


def test_none_audio_returns_events_unchanged(cfg):
    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], None, 16000, cfg)

    assert res == [ev]
    assert res[0] is ev


def test_disabled_config_returns_events_unchanged(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 1] = noise

    cfg_disabled = copy.deepcopy(cfg)
    cfg_disabled.pipeline["events"]["modifiers"]["enabled"] = False

    ev = _make_event(cfg, 0.0, 2.0)
    res = add_modifiers([ev], stereo, sr, cfg_disabled)

    assert res == [ev]
    assert res[0] is ev
    assert "right" not in res[0].modifiers


def test_existing_modifiers_preserved_no_duplicates(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 1] = noise

    ev = _make_event(cfg, 0.0, 2.0, modifiers=["distant", "right"])
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert res[0].modifiers.count("right") == 1
    assert "distant" in res[0].modifiers
    res[0].validate()


def test_input_not_mutated(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 1] = noise

    ev = _make_event(cfg, 0.0, 2.0, modifiers=["distant"])
    orig_modifiers = list(ev.modifiers)

    res = add_modifiers([ev], stereo, sr, cfg)

    assert ev.modifiers == orig_modifiers
    assert res[0] is not ev
    res[0].validate()


def test_short_frame_event_gets_no_modifiers(cfg):
    sr = 16000
    rng = np.random.default_rng(42)
    n = int(2.0 * sr)
    noise = rng.standard_normal(n) * 0.1
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[:, 1] = noise

    # Event shorter than 1 frame (0.1 s = 1600 samples)
    ev = _make_event(cfg, 0.0, 0.05)
    res = add_modifiers([ev], stereo, sr, cfg)

    assert len(res) == 1
    assert res[0].modifiers == []
    res[0].validate()
