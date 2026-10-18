"""`soundthing setup`: offline checks of the workspace bootstrap steps."""
import argparse
import io
import zipfile
from pathlib import Path

import pytest

from soundthing import workspace as ws
from soundthing.pipeline import main


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(ws.paths, "main_checkout", lambda: root)
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(root / "data" / "harness"))
    monkeypatch.setattr(ws.Path, "home", lambda: tmp_path / "home")
    return root


def _args(**kw):
    base = dict(step="all", archive=None, with_caches=False, force=False, caches=False,
                separation=False, skip_audioset=False, skip_esc50=False)
    return argparse.Namespace(**{**base, **kw})


def test_pack_and_unpack_round_trip(checkout, tmp_path):
    (checkout / "data" / "harness" / "cache" / "k").mkdir(parents=True)
    (checkout / "data" / "harness" / "ledger.jsonl").write_text("{}\n")
    (checkout / "data" / "harness" / "cache" / "k" / "a.npz").write_bytes(b"x")
    (checkout / "data" / "clips").mkdir()
    (checkout / "data" / "clips" / "c.wav").write_bytes(b"w")
    archive = tmp_path / "ws.zip"

    assert ws.pack(_args(archive=str(archive))) == 0
    names = set(zipfile.ZipFile(archive).namelist())
    assert names == {"data/harness/ledger.jsonl", "data/clips/c.wav"}

    (checkout / "data" / "clips" / "c.wav").unlink()
    (checkout / "data" / "harness" / "ledger.jsonl").write_text("local\n")
    assert ws.unpack(_args(archive=str(archive))) == 0
    assert (checkout / "data" / "clips" / "c.wav").read_bytes() == b"w"
    assert (checkout / "data" / "harness" / "ledger.jsonl").read_text() == "local\n"  # kept without --force

    ws.pack(_args(archive=str(archive), with_caches=True))
    assert "data/harness/cache/k/a.npz" in zipfile.ZipFile(archive).namelist()


def test_unpack_refuses_paths_outside_checkout(checkout, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "x")
    assert ws.main(_args(step="unpack", archive=str(archive))) == 1
    assert not (tmp_path / "outside.txt").exists()


def test_download_writes_through_part_file(tmp_path, monkeypatch):
    class Resp(io.BytesIO):
        headers = {"Content-Length": "5"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ws.urllib.request, "urlopen", lambda req: Resp(b"hello"))
    dest = ws.download("https://example.invalid/f", tmp_path / "sub" / "f.bin")
    assert dest.read_bytes() == b"hello"
    assert not (tmp_path / "sub" / "f.bin.part").exists()


def test_fetch_esc50_extracts_audio_and_csv(checkout, monkeypatch):
    def fake_download(url, dest):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("ESC-50-master/audio/1-1-A-0.wav", b"wav")
            z.writestr("ESC-50-master/meta/esc50.csv", "filename\n1-1-A-0.wav\n")
            z.writestr("ESC-50-master/README.md", "skip")
        return dest

    monkeypatch.setattr(ws, "download", fake_download)
    ws.fetch_esc50()
    assert (checkout / "data" / "esc50" / "audio" / "1-1-A-0.wav").read_bytes() == b"wav"
    assert (checkout / "data" / "esc50" / "esc50.csv").is_file()
    assert not (checkout / "data" / "esc50" / "README.md").exists()


def test_data_without_kaggle_fails_with_hint(checkout, monkeypatch, capsys):
    monkeypatch.setattr(ws, "kaggle_ready", lambda: False)
    assert ws.data(_args(skip_esc50=True)) == 1
    assert "--skip-audioset" in capsys.readouterr().err


def test_check_reports_missing_workspace(checkout, monkeypatch, capsys):
    monkeypatch.setattr(ws, "kaggle_ready", lambda: False)
    monkeypatch.setattr(ws, "_hf_cached", lambda model_id: False)
    assert ws.check() == 1
    out = capsys.readouterr().out
    assert "MISSING ~/panns_data/Cnn14_mAP=0.431.pth -> soundthing setup models" in out
    assert "MISSING ESC-50" in out and "MISSING AudioSet dev/holdout" in out


def test_setup_is_a_soundthing_subcommand(checkout, monkeypatch, capsys):
    monkeypatch.setattr(ws, "kaggle_ready", lambda: False)
    monkeypatch.setattr(ws, "_hf_cached", lambda model_id: False)
    assert main(["setup", "check"]) == 1
    assert "== model weights" in capsys.readouterr().out
