from copy import deepcopy

import numpy as np
import pytest

from soundthing.config import CoarseGroup, SoundClass, Thresholds
from soundthing.events.fallbacks import add_generic_scores, window_rms_dbfs
from soundthing.events.hierarchy import suppress_ancestors
from soundthing.events.postprocess import threshold_to_events
from soundthing.events.types import FrameScores


@pytest.fixture
def fallback_config(cfg):
    copy = deepcopy(cfg)
    # Fakes exercise existing contracts while exact live vocabulary edits await
    # their separate shared-interface approval.
    copy.classes = [SoundClass('specific_one', 'precise one', audioset=['one'], parents=['broad_one']),
                    SoundClass('specific_two', 'precise two', audioset=['two'], parents=['broad_one']),
                    SoundClass('scored_parent', 'scored parent', audioset=['parent'])]
    copy.coarse_groups = {'broad_one': CoarseGroup('broad_one', 'general sound')}
    copy.thresholds = Thresholds(.5, .6, {}, {'generic': .35, 'fine': .8},
                                 {'broad_one': 'generic', 'specific_one': 'fine', 'specific_two': 'fine'})
    copy.pipeline['events']['generic_fallbacks'] = {
        'enabled': True, 'groups': ['broad_one'], 'exclude_classes': [], 'min_rms_dbfs': -60,
    }
    return copy


def test_generic_support_uses_max_not_sum_and_keeps_native_scores(fallback_config):
    cfg = fallback_config
    scores = FrameScores(np.array([.5, .75]), cfg.class_ids, np.array([[.3, .3, .1], [.7, .5, .1]]), .25)
    original = scores.scores.copy()
    result = add_generic_scores(scores, np.array([-20, -20]), cfg)
    assert result.class_ids == [*scores.class_ids, 'broad_one']
    np.testing.assert_equal(result.column('broad_one'), [.3, .7])
    np.testing.assert_array_equal(result.scores[:, :3], original)
    np.testing.assert_array_equal(scores.scores, original)
    np.testing.assert_array_equal(result.column('scored_parent'), [.1, .1])


def test_uncertain_specific_guess_can_display_generic_but_strong_guess_suppresses_it(fallback_config):
    cfg = fallback_config
    for confidence, expected in [(.3, set()), (.7, {'broad_one'}), (.9, {'specific_one'})]:
        scores = FrameScores(np.array([.5, .75]), cfg.class_ids, np.tile([confidence, 0, 0], (2, 1)), .25)
        result = add_generic_scores(scores, np.array([-20, -20]), cfg)
        events = suppress_ancestors(threshold_to_events(result, cfg), cfg.hierarchy())
        assert {e.class_id for e in events} == expected
        assert all(e.confidence == confidence for e in events)


def test_audio_activity_gate_and_exclusions_prevent_false_generic_captions(fallback_config):
    cfg = fallback_config
    scores = FrameScores(np.array([.5, .75, 1.]), cfg.class_ids,
                         np.tile([.9, .1, 0], (3, 1)), .25)
    result = add_generic_scores(scores, np.array([-240, -80, -40]), cfg)
    np.testing.assert_equal(result.column('broad_one'), [0, 0, .9])
    cfg.pipeline['events']['generic_fallbacks']['exclude_classes'] = ['specific_one']
    result = add_generic_scores(scores, np.array([-20, -20, -20]), cfg)
    np.testing.assert_equal(result.column('broad_one'), [.1, .1, .1])
    cfg.pipeline['events']['generic_fallbacks']['min_rms_dbfs'] = -10
    result = add_generic_scores(scores, np.array([-20, -20, -20]), cfg)
    assert not result.column('broad_one').any()


def test_optional_fallback_settings_preserve_legacy_behavior_and_handle_empty_scores(fallback_config):
    cfg = fallback_config
    empty = FrameScores(np.array([]), cfg.class_ids, np.empty((0, 3)), .25)
    assert add_generic_scores(empty, np.array([]), cfg).scores.shape == (0, 4)
    cfg.pipeline['events'].pop('generic_fallbacks')
    assert add_generic_scores(empty, np.array([]), cfg) is empty


def test_activity_windows_match_recording_level_and_short_clip_padding():
    sr = 16000
    np.testing.assert_allclose(window_rms_dbfs(np.full(sr * 2, .01), sr, np.array([.5, 1, 1.5]), 1), -40)
    levels = window_rms_dbfs(np.full(sr // 2, .01), sr, np.array([.5]), 1)
    np.testing.assert_allclose(levels, [-40 + 10 * np.log10(.5)])
    assert window_rms_dbfs(np.zeros(sr), sr, np.array([.5]), 1)[0] < -120


def test_activity_shape_mismatch_is_rejected(fallback_config):
    scores = FrameScores(np.array([.5, .75]), fallback_config.class_ids, np.zeros((2, 3)), .25)
    with pytest.raises(ValueError, match='RMS must match'):
        add_generic_scores(scores, np.array([-20]), fallback_config)


def test_stub_uses_generic_support_without_loading_or_measuring_models(fallback_config, fake_inputs):
    from soundthing.contracts import load_inputs
    from soundthing.events.stub import detect_stub

    cfg = fallback_config
    cfg.classes = [cfg.classes[0]]
    cfg.thresholds.tiers['fine'] = .99
    inputs = load_inputs(fake_inputs)
    events = detect_stub(inputs, None, cfg)
    assert events and all(e.class_id == 'broad_one' for e in events)
    cfg.pipeline['events']['generic_fallbacks']['enabled'] = False
    assert detect_stub(inputs, None, cfg) == []
    cfg.pipeline['events']['generic_fallbacks']['enabled'] = True
    cfg.thresholds.tiers['fine'] = .4
    assert {e.class_id for e in detect_stub(inputs, None, cfg)} == {'specific_one'}


def test_real_detector_uses_activity_gate_and_can_retune_cached_scores(cfg, tmp_path, monkeypatch, inject_scores):
    import soundfile as sf
    from soundthing.contracts import EnvSegment, ExtractedInputs
    from soundthing.events import detector

    copy = deepcopy(cfg)
    copy.coarse_groups['broad_one'] = CoarseGroup('broad_one', 'general sound')
    copy.sound_class('typing').parents.append('broad_one')
    copy.thresholds.class_tiers['broad_one'] = 'generic'
    copy.thresholds.tiers['specific'] = .65  # controlled fixture, independent of live policy
    copy.pipeline['events']['generic_fallbacks'] = {
        'enabled': True, 'groups': ['broad_one'], 'min_rms_dbfs': -60,
    }
    copy.pipeline['events']['modifiers']['enabled'] = False
    copy.validate()
    audio = tmp_path / 'audio.wav'
    sf.write(audio, np.full(32000, .01), 16000)
    inputs = ExtractedInputs('clip', 'video', str(audio), str(tmp_path), 1, 2)
    calls = []

    def score(*args):
        calls.append(True)
        return FrameScores(np.array([.5, .75, 1, 1.25, 1.5]), ['typing'], np.full((5, 1), .6), .25)

    monkeypatch.setattr(detector, 'get_models', lambda cfg: None)
    inject_scores(score)
    detector._clear_caches()
    try:
        assert {e.class_id for e in detector.detect(inputs, None, copy)} == {'broad_one'}
        segments = [EnvSegment(0, 2, 'general', 1, {'general': 1})]
        assert {e.class_id for e in detector.detect(inputs, segments, copy)} == {'broad_one'}
        copy.thresholds.tiers['specific'] = .5
        assert {e.class_id for e in detector.detect(inputs, None, copy)} == {'typing'}
        copy.thresholds.tiers['specific'] = .65
        copy.pipeline['events']['generic_fallbacks']['min_rms_dbfs'] = -10
        assert detector.detect(inputs, None, copy) == []
        assert len(calls) == 1  # postprocessing tuning does not rerun PANNs
        detector._clear_caches()
        copy.pipeline['events']['generic_fallbacks']['min_rms_dbfs'] = -60
        sf.write(audio, np.zeros(32000), 16000)
        assert detector.detect(inputs, None, copy) == []
    finally:
        detector._clear_caches()
