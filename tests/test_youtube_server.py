"""Person 3: downloader boundaries, job lifecycle, and extension API integration."""
import importlib
import json
import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from soundthing import contracts as C
from soundthing.server import youtube
from soundthing.server.youtube import YouTubeJobs, complete_run

ID = "aB3dE_fG-12"
OTHER_ID = "xY3dE_fG-12"


@pytest.fixture(autouse=True)
def isolated_cookie_configuration(monkeypatch):
    # Tests must never read the server operator's real exported session.
    monkeypatch.delenv(youtube.COOKIE_ENV, raising=False)


def artifacts(source, runs, cfg):
    run = runs / source.stem
    run.mkdir(parents=True, exist_ok=True)
    C.save_inputs(C.ExtractedInputs(video_id=source.stem, video_path=str(source), audio_path=str(source), frames_dir=str(run), fps=1, duration=2), run / "inputs.json")
    for mode in ("baseline", "conditioned"):
        C.save_events([C.CaptionEvent(0, 2, "footsteps", "footsteps", .9)], run / f"events.{mode}.json")
        (run / f"captions.{mode}.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n[footsteps]\n")


def make_source(uploads, youtube_id):
    uploads.mkdir(parents=True, exist_ok=True)
    source = uploads / f"youtube_{youtube_id}.mp4"
    source.write_bytes(b"fixture video")
    return source


def test_jobs_deduplicate_serialize_and_cache(tmp_path, cfg):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def download(video_id, uploads):
        calls.append(video_id)
        entered.set()
        assert release.wait(3)
        return make_source(uploads, video_id)

    status = {}
    jobs = YouTubeJobs(tmp_path / "runs", tmp_path / "uploads", cfg, status, downloader=download, processor=artifacts)
    try:
        first = jobs.submit(ID)
        assert entered.wait(3)
        assert status[first]["status"] == "downloading"
        assert jobs.submit(ID) == first
        second = jobs.submit(OTHER_ID)
        assert status[second]["status"] == "queued"
        assert calls == [ID]
        release.set()
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert calls == [ID, OTHER_ID]
        assert status[first]["status"] == status[second]["status"] == "done"
        jobs.submit(ID)
        assert calls == [ID, OTHER_ID]
        status.clear()  # Like a restart: complete artifacts are reusable.
        jobs.submit(ID)
        assert status[first]["status"] == "done"
        assert calls == [ID, OTHER_ID]
    finally:
        release.set()
        jobs.close()


def test_failed_jobs_retry_and_incomplete_runs_are_not_ready(tmp_path, cfg):
    download = Mock(side_effect=RuntimeError("YouTube unavailable"))
    jobs = YouTubeJobs(tmp_path / "runs", tmp_path / "uploads", cfg, {}, downloader=download, processor=artifacts)
    try:
        video_id = jobs.submit(ID)
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert jobs.status[video_id] == {"status": "error", "error": "YouTube unavailable"}
        download.side_effect = lambda video_id, uploads: make_source(uploads, video_id)
        jobs.submit(ID)
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert jobs.status[video_id]["status"] == "done"
        (jobs.runs / video_id / "captions.conditioned.vtt").unlink()
        jobs.status.clear()
        assert not complete_run(jobs.runs / video_id)
        jobs.submit(ID)
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert jobs.status[video_id]["status"] == "done"
    finally:
        jobs.close()


@pytest.mark.parametrize('stale', ['inputs.json', 'events.baseline.json', 'events.conditioned.json'])
def test_previous_schema_cache_is_reprocessed(tmp_path, cfg, stale):
    source = make_source(tmp_path / 'uploads', ID)
    runs = tmp_path / 'runs'
    artifacts(source, runs, cfg)
    run = runs / source.stem
    old = json.loads((run / stale).read_text())
    old['schema_version'] = 1
    (run / stale).write_text(json.dumps(old))
    assert not complete_run(run)
    processed = Mock(side_effect=artifacts)
    jobs = YouTubeJobs(runs, tmp_path / 'uploads', cfg, {},
                       downloader=lambda *args: source, processor=processed)
    try:
        jobs.submit(ID)
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert processed.call_count == 1
        assert complete_run(run)
        assert jobs.status[source.stem]['status'] == 'done'
    finally:
        jobs.close()


def test_missing_pipeline_outputs_are_not_reported_as_success(tmp_path, cfg):
    jobs = YouTubeJobs(tmp_path / "runs", tmp_path / "uploads", cfg, {}, downloader=lambda video_id, uploads: make_source(uploads, video_id), processor=lambda *args: None)
    try:
        key = jobs.submit(ID)
        jobs.executor.submit(lambda: None).result(timeout=3)
        assert jobs.status[key]["status"] == "error"
    finally:
        jobs.close()


def tool_stubs(monkeypatch):
    monkeypatch.setattr(youtube.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(youtube.shutil, "which", lambda name: f"/tools/{name}")


def test_downloader_validates_id_dependencies_and_live_metadata(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        youtube.download_video("../not-a-video", tmp_path)
    monkeypatch.setattr(youtube.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install"):
        youtube.download_video(ID, tmp_path)
    tool_stubs(monkeypatch)
    monkeypatch.setattr(youtube.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        youtube.download_video(ID, tmp_path)
    tool_stubs(monkeypatch)
    invoke = Mock(return_value=json.dumps({"id": ID, "is_live": True}))
    monkeypatch.setattr(youtube, "_invoke", invoke)
    with pytest.raises(RuntimeError, match="Live and upcoming"):
        youtube.download_video(ID, tmp_path)
    assert invoke.call_count == 1  # Metadata only: never download a live stream.


def test_downloader_uses_canonical_url_and_fixed_output(monkeypatch, tmp_path):
    tool_stubs(monkeypatch)
    calls = []

    def invoke(args, timeout):
        calls.append((args, timeout))
        assert args[-2:] == ["--", f"https://www.youtube.com/watch?v={ID}"]
        if "--dump-single-json" in args:
            return json.dumps({"id": ID, "duration": 2})
        template = args[args.index("--output") + 1]
        path = Path(template.replace("%(ext)s", "webm"))
        path.write_bytes(b"video")
        return f"{path}\n"

    monkeypatch.setattr(youtube, "_invoke", invoke)
    path = youtube.download_video(ID, tmp_path)
    assert path.name == f"youtube_{ID}.webm"
    assert calls[1][1] == 900
    assert "height<=480" in calls[1][0][calls[1][0].index("--format") + 1]
    monkeypatch.setattr(youtube, "_invoke", Mock(side_effect=[json.dumps({"id": ID, "duration": 2}), "/etc/passwd\n"]))
    with pytest.raises(RuntimeError, match="expected video file"):
        youtube.download_video(ID, tmp_path)


def test_subprocess_ignores_user_config_and_handles_failures(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "metadata", ""))
    monkeypatch.setattr(youtube.subprocess, "run", run)
    assert youtube._invoke(["--", "https://www.youtube.com/watch?v=" + ID], 60) == "metadata"
    args, kwargs = run.call_args
    assert "--ignore-config" in args[0] and "--no-playlist" in args[0]
    assert not kwargs.get("shell")
    run.side_effect = subprocess.TimeoutExpired("yt-dlp", 60)
    with pytest.raises(RuntimeError, match="timed out"):
        youtube._invoke([], 60)
    run.side_effect = subprocess.CalledProcessError(1, "yt-dlp", stderr="Sign in to confirm")
    with pytest.raises(RuntimeError, match="Sign in to confirm"):
        youtube._invoke([], 60)


def test_api_submission_status_vtt_and_restart_detection(monkeypatch, tmp_path, cfg):
    app = importlib.import_module("soundthing.server.app")
    states = {}
    jobs = YouTubeJobs(tmp_path / "runs", tmp_path / "uploads", cfg, states, downloader=lambda video_id, uploads: make_source(uploads, video_id), processor=artifacts)
    monkeypatch.setattr(app, "cfg", cfg)
    monkeypatch.setattr(app, "youtube_jobs", jobs)
    monkeypatch.setattr(app, "status", states)
    monkeypatch.setattr(app, "RUNS", jobs.runs)
    with TestClient(app.app) as client:
        for bad in ["", "../etc/passwd", "https://youtube.com/watch?v=" + ID, ID + "\n"]:
            assert client.post("/api/youtube", json={"youtube_id": bad}).status_code == 422
        response = client.post("/api/youtube", json={"youtube_id": ID})
        assert response.status_code == 202
        key = response.json()["video_id"]
        jobs.executor.submit(lambda: None).result(timeout=3)
        state = client.get(f"/api/videos/{key}").json()
        assert state["status"] == "done" and state["demo"] is True
        assert client.get(f"/api/videos/{key}/captions/conditioned.vtt").text.startswith("WEBVTT")
        states.clear()
        assert client.get(f"/api/videos/{key}").json()["status"] == "done"
        (jobs.runs / key / "captions.conditioned.vtt").unlink()
        assert client.get(f"/api/videos/{key}").status_code == 404


def test_api_does_not_serve_previous_schema_caption_cache(monkeypatch, tmp_path, cfg):
    app = importlib.import_module('soundthing.server.app')
    source = make_source(tmp_path / 'uploads', ID)
    runs = tmp_path / 'runs'
    artifacts(source, runs, cfg)
    stale = runs / source.stem / 'events.conditioned.json'
    payload = json.loads(stale.read_text())
    payload['schema_version'] = 1
    stale.write_text(json.dumps(payload))
    jobs = YouTubeJobs(runs, tmp_path / 'uploads', cfg, {})
    monkeypatch.setattr(app, 'youtube_jobs', jobs)
    monkeypatch.setattr(app, 'RUNS', runs)
    monkeypatch.setattr(app, 'status', jobs.status)
    with TestClient(app.app) as client:
        assert client.get(f'/api/videos/{source.stem}').status_code == 404
        assert client.get(f'/api/videos/{source.stem}/captions/conditioned.vtt').status_code == 404


def test_cookie_configuration_is_optional():
    with youtube._cookie_arguments() as args:
        assert args == []


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid", "unreadable"])
def test_cookie_configuration_errors_are_actionable(monkeypatch, tmp_path, kind):
    source = tmp_path / "export.txt"
    if kind == "directory":
        source.mkdir()
    elif kind != "missing":
        source.write_text("not an export" if kind == "invalid" else "# Netscape HTTP Cookie File\n")
    if kind == "unreadable":
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path == source:
                raise PermissionError("fixture")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    monkeypatch.setenv(youtube.COOKIE_ENV, str(source))
    with pytest.raises(RuntimeError, match="SOUNDTHING_YOUTUBE_COOKIES") as error:
        with youtube._cookie_arguments():
            pytest.fail("invalid cookie source should be rejected")
    assert str(source) not in str(error.value)


@pytest.mark.parametrize("header", ["# Netscape HTTP Cookie File", "# HTTP Cookie File"])
def test_cookie_copy_is_private_disposable_and_does_not_modify_export(monkeypatch, tmp_path, header):
    source = tmp_path / "export.txt"
    data = (header + "\r\n.youtube.com\tTRUE\t/\tTRUE\t0\tTEST\tfixture-secret\r\n").encode()
    source.write_bytes(data)
    monkeypatch.setenv(youtube.COOKIE_ENV, str(source))
    with youtube._cookie_arguments() as args:
        assert args[0] == "--cookies"
        working = Path(args[1])
        folder = working.parent
        assert working != source and working.read_bytes() == data
        if os.name == "posix":
            # Windows has no owner/group/other bits; chmod only toggles read-only there.
            assert working.stat().st_mode & 0o777 == 0o600
            assert folder.stat().st_mode & 0o777 == 0o700
        working.write_text("yt-dlp updated the working cookie jar")
    assert source.read_bytes() == data
    assert not working.exists() and not folder.exists()


def test_both_download_steps_share_the_temporary_cookie_jar(monkeypatch, tmp_path):
    tool_stubs(monkeypatch)
    source = tmp_path / "export.txt"
    source.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv(youtube.COOKIE_ENV, str(source))
    jars = []

    def invoke(args, timeout):
        working = Path(args[args.index("--cookies") + 1])
        jars.append(working)
        assert working.is_file()
        if "--dump-single-json" in args:
            working.write_text("# Netscape HTTP Cookie File\n# refreshed by metadata step\n")
            return json.dumps({"id": ID, "duration": 2})
        assert "refreshed" in working.read_text()
        output = Path(args[args.index("--output") + 1].replace("%(ext)s", "mp4"))
        output.write_bytes(b"fixture video")
        return str(output)

    monkeypatch.setattr(youtube, "_invoke", invoke)
    assert youtube.download_video(ID, tmp_path / "uploads").is_file()
    assert len(jars) == 2 and jars[0] == jars[1]
    assert not jars[0].exists()
    assert source.read_text() == "# Netscape HTTP Cookie File\n"


@pytest.mark.parametrize("step", ["metadata", "download", "timeout"])
def test_cookie_jar_cleanup_on_failure(monkeypatch, tmp_path, step):
    tool_stubs(monkeypatch)
    source = tmp_path / "export.txt"
    source.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv(youtube.COOKIE_ENV, str(source))
    jars = []

    def run(args, **kwargs):
        working = Path(args[args.index("--cookies") + 1])
        jars.append(working)
        if step == "download" and "--dump-single-json" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps({"id": ID, "duration": 2}), "")
        if step == "timeout":
            raise subprocess.TimeoutExpired(args, 60)
        raise subprocess.CalledProcessError(1, args, stderr="download failed")

    monkeypatch.setattr(youtube.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        youtube.download_video(ID, tmp_path / "uploads")
    assert jars and not jars[0].parent.exists()
    assert source.read_text() == "# Netscape HTTP Cookie File\n"


@pytest.mark.parametrize("authenticated", [False, True])
def test_bot_challenges_explain_cookie_setup_without_raw_stderr(monkeypatch, authenticated):
    stderr = "WARNING: No title found\nERROR: Sign in to confirm you’re not a bot. private-debug-details"
    monkeypatch.setattr(youtube.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "yt-dlp", stderr=stderr)))
    args = ["--cookies", "/private/session.txt"] if authenticated else []
    with pytest.raises(RuntimeError) as error:
        youtube._invoke(args, 60)
    message = str(error.value)
    assert ("Refresh your exported" if authenticated else youtube.COOKIE_ENV) in message
    assert "No title" not in message and "private-debug-details" not in message


def test_authenticated_parse_errors_never_expose_cookie_rows(monkeypatch):
    secret = ".youtube.com\tTRUE\t/\tTRUE\t0\tSESSION\tfixture-secret"
    failure = subprocess.CalledProcessError(1, "yt-dlp", stderr=f"invalid Netscape cookies file '/private/session.txt': {secret}")
    monkeypatch.setattr(youtube.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as error:
        youtube._invoke(["--cookies", "/private/session.txt"], 60)
    assert "configured cookies" in str(error.value)
    assert secret not in str(error.value) and "/private/session.txt" not in str(error.value)
