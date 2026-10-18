"""Person 3: real PCM channel isolation, smoothing, and direction failure boundaries."""
import importlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from soundthing import contracts as C
from soundthing import direction as D


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe needed for real channel-preservation checks")


def wav(tmp_path, layout, samples):
    path = tmp_path / "channels.wav"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "f32le", "-ar", "16000",
                    "-ac", str(samples.shape[1]), "-channel_layout", layout, "-i", "pipe:0",
                    "-c:a", "pcm_s24le", str(path)], input=samples.astype("<f4").tobytes(), check=True)
    return path


@pytest.mark.parametrize("layout", ["5.1", "5.1(side)"])
@pytest.mark.parametrize("channel", range(6))
def test_real_surround_channel_isolation_and_lfe_exclusion(tmp_path, tools, layout, channel):
    samples = np.zeros((8000, 6), dtype=np.float32)
    samples[:, channel] = .5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
    track = D.analyze(wav(tmp_path, layout, samples), .5)
    assert track.mode == "surround51"
    assert len(track.frames) == 10
    expected = {0: "FL", 1: "FR", 2: "FC", 4: "SL", 5: "SR"}.get(channel)
    for name, level in track.frames[-1].levels.items():
        assert (level > .8) if name == expected else (level == 0)


@pytest.mark.parametrize("right", [0, .5])
def test_stereo_balance_silence_and_media_timestamps(tmp_path, tools, right):
    tone = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
    samples = np.column_stack([.5 * tone, right * tone])
    samples[:800] = 0
    track = D.analyze(wav(tmp_path, "stereo", samples), .5)
    assert track.mode == "stereo"
    assert track.frames[0].levels == {"FL": 0, "FR": 0}
    assert track.frames[-1].time == .45
    assert track.frames[-1].levels["FL"] > .8
    if right:
        assert track.frames[-1].levels["FL"] == pytest.approx(track.frames[-1].levels["FR"])
    else:
        assert track.frames[-1].levels["FR"] == 0


def test_mono_and_unknown_layout_do_not_claim_direction(tmp_path, tools):
    assert D.analyze(wav(tmp_path, "mono", np.ones((800, 1))), .05).mode == "unavailable"
    assert D.layout_channels("", 6)[0] == "unavailable"
    assert D.layout_channels("7.1", 8)[0] == "unavailable"
    assert D.layout_channels("stereo", 6)[0] == "unavailable"


def test_smoothing_preserves_relative_energy_and_uses_attack_release():
    samples = np.column_stack([np.ones(800), np.full(800, 10 ** (-30 / 20))])
    attack = D.channel_strengths(samples, np.zeros(2), .05)
    assert attack[0] == pytest.approx(1 - np.exp(-1))
    assert attack[1] == pytest.approx(.5 * (1 - np.exp(-1)))
    release = D.channel_strengths(np.zeros((800, 2)), attack, .05)
    assert release[0] == pytest.approx(attack[0] * np.exp(-.25))


def test_analysis_failure_is_cached_and_does_not_break_caption_outputs(tmp_path, monkeypatch):
    analyze = Mock(side_effect=RuntimeError("decoder unavailable"))
    monkeypatch.setattr(D, "analyze", analyze)
    caption = tmp_path / "captions.conditioned.vtt"
    caption.write_text("WEBVTT\n\n")
    assert D.ensure_direction("missing.mp4", tmp_path, 2).mode == "unavailable"
    assert D.ensure_direction("missing.mp4", tmp_path, 2).mode == "unavailable"
    assert analyze.call_count == 1
    assert caption.read_text() == "WEBVTT\n\n"
    assert not (tmp_path / "direction.json.tmp").exists()


def test_audio_format_prefers_surround_only_in_selected_language():
    from soundthing.server.youtube import audio_format
    metadata = {"requested_formats": [{"acodec": "aac", "language": "en"}], "formats": [
        {"format_id": "surround-fr", "language": "fr", "audio_channels": 6, "vcodec": "none", "acodec": "aac", "abr": 999},
        {"format_id": "surround-en", "language": "en", "audio_channels": 6, "vcodec": "none", "acodec": "aac", "abr": 384}]}
    assert "surround-en" in audio_format(metadata)
    metadata["formats"].pop()
    assert "surround-fr" not in audio_format(metadata)
    assert "audio_channels<=?2" in audio_format(metadata)
    # Non-linguistic uploads often omit language entirely. They still support 5.1
    # when no ambiguous alternate language is exposed.
    assert "51" in audio_format({"formats": [{"format_id": "51", "audio_channels": 6, "vcodec": "none", "acodec": "aac"}]})


def test_direction_endpoint_backfills_once_without_reprocessing_captions(tmp_path, cfg, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from soundthing.server.youtube import YouTubeJobs
    app = importlib.import_module("soundthing.server.app")
    key = "youtube_aB3dE_fG-12"
    run = tmp_path / "runs" / key
    run.mkdir(parents=True)
    C.save_inputs(C.ExtractedInputs(key, "old.mp4", "audio.wav", str(run), 1, 2), run / "inputs.json")
    for mode in ("baseline", "conditioned"):
        C.save_events([], run / f"events.{mode}.json")
        (run / f"captions.{mode}.vtt").write_text("WEBVTT\n\n")
    entered, release = threading.Event(), threading.Event()

    def download(*args):
        entered.set()
        assert release.wait(3)
        return Path("new-surround.mkv")

    analyze = Mock(return_value=C.DirectionTrack(2, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"FL": 1, "FR": 0})]))
    monkeypatch.setattr(D, "analyze", analyze)
    processor = Mock()
    jobs = YouTubeJobs(tmp_path / "runs", tmp_path / "uploads", cfg, {}, downloader=download, processor=processor)
    monkeypatch.setattr(app, "RUNS", jobs.runs)
    monkeypatch.setattr(app, "youtube_jobs", jobs)
    with TestClient(app.app) as client:
        try:
            assert client.get(f"/api/videos/{key}/direction").status_code == 404
            assert entered.wait(3)
            assert client.get(f"/api/videos/{key}/direction").status_code == 404
            assert client.get(f"/api/videos/{key}").json()["status"] == "done"
            release.set()
            jobs.executor.submit(lambda: None).result(timeout=3)
            response = client.get(f"/api/videos/{key}/direction")
            assert response.status_code == 200 and response.json()["mode"] == "stereo"
            assert analyze.call_count == 1
            processor.assert_not_called()
        finally:
            release.set()
