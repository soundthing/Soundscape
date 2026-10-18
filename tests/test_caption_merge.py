"""Person 3: subtitle parsing, overlap rendering, and CLI integration."""
from copy import deepcopy

import pytest

from soundthing import contracts as C
from soundthing.captions import Cue, merge_subtitle_files, merge_with_speech, read_subtitles, write_captions
from soundthing.pipeline import main, run_video


def test_read_srt_bom_crlf_multiline_and_hours(tmp_path):
    path = tmp_path / "speech.srt"
    path.write_bytes(b'\xef\xbb\xbf1\r\n01:02:03,004 --> 01:02:05,006\r\nHello\r\nworld\r\n\r\n2\r\n01:03:00,000 --> 01:03:01,000\r\nGoodbye\r\n')
    assert read_subtitles(path) == [Cue(3723.004, 3725.006, "Hello\nworld"), Cue(3780, 3781, "Goodbye")]


def test_read_vtt_identifiers_settings_metadata_and_unsorted_cues(tmp_path):
    path = tmp_path / "speech.vtt"
    path.write_text('WEBVTT subtitles\nLanguage: en\n\nNOTE ignored comment\n\nSTYLE\n::cue {color:white}\n\nREGION\nid:sample\n\nsecond\n00:03.000 --> 00:04.000 align:right\nLater\n\nfirst\n00:01.000 --> 00:02.000\nEarlier\n')
    assert read_subtitles(path) == [Cue(1, 2, "Earlier"), Cue(3, 4, "Later")]


@pytest.mark.parametrize("name,text", [
    ("bad.txt", ""), ("bad.vtt", "not WEBVTT"),
    ("bad.vtt", "WEBVTT\n00:01.000 --> 00:02.000\nmissing separator"),
    ("bad.srt", "1\nno timing\nHello"),
    ("bad.srt", "1\n00:65:00,000 --> 00:66:00,000\nHello"),
    ("bad.vtt", "WEBVTT\n\n00:03.000 --> 00:02.000\nWrong duration"),
])
def test_invalid_subtitles_are_reported(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ValueError):
        read_subtitles(path)


def test_merge_preserves_timing_and_stacks_speech_before_sounds(tmp_path):
    speech = tmp_path / "speech.srt"
    speech.write_text('1\n00:00:01,000 --> 00:00:05,000\nHello\nworld\n')
    sounds = [Cue(2, 4, "[door closes]"), Cue(3, 6, "[footsteps]")]
    before = deepcopy(sounds)
    assert merge_with_speech(sounds, speech) == [
        Cue(1, 2, "Hello\nworld"),
        Cue(2, 3, "Hello\nworld\n[door closes]"),
        Cue(3, 4, "Hello\nworld\n[door closes]\n[footsteps]"),
        Cue(4, 5, "Hello\nworld\n[footsteps]"),
        Cue(5, 6, "[footsteps]"),
    ]
    assert sounds == before


def test_merge_gaps_empty_tracks_identical_text_and_touching_boundaries(tmp_path):
    speech = tmp_path / "speech.vtt"
    speech.write_text("WEBVTT\n\n")
    assert merge_with_speech([], speech) == []
    assert merge_with_speech([Cue(1, 2, "[bell]"), Cue(2, 3, "[bell]"), Cue(4, 5, "[bell]")], speech) == [Cue(1, 3, "[bell]"), Cue(4, 5, "[bell]")]
    speech.write_text("WEBVTT\n\n00:01.000 --> 00:03.000\n[bell]\n")
    assert merge_with_speech([Cue(1, 3, "[bell]")], speech) == [Cue(1, 3, "[bell]")]
    with pytest.raises(ValueError, match="finite"):
        merge_with_speech([Cue(float("nan"), 3, "invalid")], speech)


@pytest.mark.parametrize("suffix", ["vtt", "srt"])
def test_merge_command_roundtrips_and_does_not_need_project_config(tmp_path, monkeypatch, suffix):
    speech = tmp_path / "speech.srt"
    sounds = tmp_path / "sounds.vtt"
    output = tmp_path / "nested" / f"combined.{suffix}"
    speech.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n")
    sounds.write_text("WEBVTT\n\n00:02.000 --> 00:04.000\n[door]\n")
    monkeypatch.chdir(tmp_path)
    assert main(["merge-captions", str(sounds), str(speech), "--out", str(output)]) == 0
    assert read_subtitles(output) == [Cue(1, 2, "Hello"), Cue(2, 3, "Hello\n[door]"), Cue(3, 4, "[door]")]
    assert main(["merge-captions", str(sounds), str(speech), "--out", str(sounds)]) == 1
    assert read_subtitles(sounds) == [Cue(2, 4, "[door]")]
    with pytest.raises(ValueError, match="output"):
        merge_subtitle_files(sounds, speech, tmp_path / "combined.txt")


def test_pipeline_writes_additional_merged_tracks_and_preserves_contracts(cfg, fake_inputs, tmp_path):
    run = run_video(fake_inputs, tmp_path / "runs", cfg)
    original = {mode: (run / f"captions.{mode}.vtt").read_text() for mode in ("baseline", "conditioned")}
    events = {mode: C.load_events(run / f"events.{mode}.json") for mode in original}
    speech = tmp_path / "speech.srt"
    speech.write_text("1\n00:00:00,000 --> 00:00:30,000\nHello\n")
    run_video(fake_inputs, tmp_path / "runs", cfg, speech_subtitles=speech)
    for mode in original:
        assert (run / f"captions.{mode}.vtt").read_text() == original[mode]
        assert C.load_events(run / f"events.{mode}.json") == events[mode]
        assert all(cue.text.startswith("Hello") for cue in read_subtitles(run / f"captions.{mode}.merged.vtt"))
    output = tmp_path / "direct.vtt"
    write_captions(events["baseline"], output, speech_subtitle_path=speech)
    assert output.read_text() == (run / "captions.baseline.merged.vtt").read_text()
    with pytest.raises(ValueError, match="different filename"):
        run_video(fake_inputs, tmp_path / "runs", cfg, speech_subtitles=run / "captions.baseline.vtt")


def test_cli_rejects_ambiguous_speech_input_and_missing_files(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["run", "one.mp4", "two.mp4", "--speech", "speech.srt"])
    assert error.value.code == 2
    assert main(["merge-captions", "missing.vtt", "missing.srt", "--out", str(tmp_path / "combined.vtt")]) == 1
