"""Person 3: setup diagnostics and safe Chrome extension file synchronization."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soundthing.pipeline import main
from soundthing.server import development as dev


@pytest.fixture(autouse=True)
def no_real_cookie_source(monkeypatch):
    monkeypatch.setenv("SOUNDTHING_YOUTUBE_COOKIES", "")


def test_local_tools_work_without_activation(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    dev.prepare_local_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    before = os.environ["PATH"]
    dev.prepare_local_path()
    assert os.environ["PATH"] == before


def test_doctor_reports_missing_dependencies_and_tools(monkeypatch, capsys):
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: None if name == "yt_dlp" else object())
    monkeypatch.setattr(dev.shutil, "which", lambda name: None if name == "ffprobe" else f"/bin/{name}")
    assert dev.doctor() == 1
    output = capsys.readouterr().out
    assert "MISSING yt_dlp" in output and "MISSING ffprobe" in output
    assert "no cookie export configured" in output


def test_doctor_validates_cookies_without_exposing_contents(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(dev.shutil, "which", lambda name: f"/bin/{name}")
    source = tmp_path / "cookies.txt"
    source.write_text("# Netscape HTTP Cookie File\n# private-session-fixture\n")
    monkeypatch.setenv("SOUNDTHING_YOUTUBE_COOKIES", str(source))
    assert dev.doctor() == 0
    assert "private-session-fixture" not in capsys.readouterr().out
    source.write_text("invalid fixture")
    assert dev.doctor() == 1


def extension_source(tmp_path):
    source = tmp_path / "repo" / "extension"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"name": "soundthing"}))
    (source / "popup.js").write_text("console.log('fixture');")
    (source / "tests").mkdir()
    (source / "tests" / "test.mjs").write_text("test fixture")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "test.pyc").write_bytes(b"cache")
    return source


def test_sync_copies_runtime_and_preserves_unrelated_existing_files(tmp_path):
    source = extension_source(tmp_path)
    target = tmp_path / "Windows Downloads" / "soundthing-extension"
    assert dev.sync_extension(target, source=source) == target
    assert (target / "popup.js").read_bytes() == (source / "popup.js").read_bytes()
    assert not (target / "tests").exists() and not (target / "__pycache__").exists()
    (target / "keep.txt").write_text("my file")
    for name in ("index.html", "app.js", "styles.css", "tokens.css"):
        (target / name).write_text("retired library")
    (source / "popup.js").write_text("updated")
    dev.sync_extension(target, source=source)
    assert (target / "popup.js").read_text() == "updated"
    assert (target / "keep.txt").read_text() == "my file"
    assert not any((target / name).exists() for name in ("index.html", "app.js", "styles.css", "tokens.css"))


def test_sync_rejects_recursive_or_unrelated_destinations(tmp_path):
    source = extension_source(tmp_path)
    for target in [source, source / "child", source.parent]:
        with pytest.raises(ValueError, match="separate"):
            dev.sync_extension(target, source=source)
    target = tmp_path / "not-an-extension"
    target.mkdir()
    (target / "important.txt").write_text("untouched")
    with pytest.raises(ValueError, match="empty folder"):
        dev.sync_extension(target, source=source)
    (target / "manifest.json").write_text(json.dumps({"name": "another-extension"}))
    with pytest.raises(ValueError, match="different extension"):
        dev.sync_extension(target, source=source)


def test_wsl_sync_uses_windows_profile_not_linux_username(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        text = "C:\\Users\\Windows Person\r\n" if args[0] == "cmd.exe" else "/mnt/c/Users/Windows Person\n"
        return subprocess.CompletedProcess(args, 0, text, "")

    monkeypatch.setattr(dev.subprocess, "run", run)
    monkeypatch.delenv("USERPROFILE", raising=False)
    assert dev.windows_extension_destination() == Path("/mnt/c/Users/Windows Person/Downloads/soundthing-extension")
    assert calls[1] == ["wslpath", "-u", "C:\\Users\\Windows Person"]


def test_serve_rejects_invalid_cookie_path_before_startup(tmp_path, capsys):
    assert main(["serve", "--cookies", str(tmp_path / "missing.txt")]) == 1
    assert "Cannot start server" in capsys.readouterr().err
