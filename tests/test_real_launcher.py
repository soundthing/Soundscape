"""Person 3: real launcher never reuses demo output or mutates shared config."""
import os
from pathlib import Path

import pytest
from soundthing import contracts as C
from soundthing.config import DEFAULT_CONFIG_DIR, load_config
from soundthing.server import real


def test_local_config_enables_both_models_without_changing_shared_files(tmp_path):
    before = {path.name: path.read_bytes() for path in DEFAULT_CONFIG_DIR.glob("*.yaml")}
    target = real.prepare_config(destination=tmp_path / "config")
    cfg = load_config(target)
    assert cfg.pipeline["events"]["backend"] == cfg.pipeline["environment"]["backend"] == "real"
    assert {path.name: path.read_bytes() for path in DEFAULT_CONFIG_DIR.glob("*.yaml")} == before
    assert len(cfg.classes) == len(load_config().classes)
    with pytest.raises(ValueError, match="separate"):
        real.prepare_config(destination=DEFAULT_CONFIG_DIR)


def test_server_gets_real_config_and_separate_output_root(tmp_path, monkeypatch):
    import uvicorn
    captured = []
    monkeypatch.setattr(real, "REAL_CONFIG", tmp_path / "config")
    monkeypatch.setattr(real, "REAL_RUNS", tmp_path / "real")
    monkeypatch.setattr(real, "model_doctor", lambda directory: 0)
    monkeypatch.setattr(real, "ensure_panns_checkpoint", lambda: None)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.append((args, kwargs)))
    monkeypatch.setenv("SOUNDTHING_CONFIG", "old-demo-config")
    monkeypatch.setenv("SOUNDTHING_RUNS", "old-demo-runs")
    assert real.main(["serve", "--port", "8767"]) == 0
    assert os.environ["SOUNDTHING_CONFIG"] == str(tmp_path / "config")
    assert os.environ["SOUNDTHING_RUNS"] == str(tmp_path / "real")
    assert captured == [(("soundthing.server.app:app",), {"host": "127.0.0.1", "port": 8767})]


def test_reusing_demo_extraction_still_writes_into_real_run(tmp_path, fake_inputs, monkeypatch):
    from soundthing import pipeline
    before = fake_inputs.read_bytes()
    original = C.load_inputs(fake_inputs)
    monkeypatch.setattr(real, "REAL_CONFIG", tmp_path / "local-config")
    monkeypatch.setattr(real, "REAL_RUNS", tmp_path / "real-runs")
    monkeypatch.setattr(real, "model_doctor", lambda directory: 0)
    monkeypatch.setattr(real, "ensure_panns_checkpoint", lambda: None)
    calls = []

    def run(source, runs, cfg):
        assert source != fake_inputs
        assert source.parent.parent == runs
        assert C.load_inputs(source) == original
        assert cfg.pipeline["events"]["backend"] == cfg.pipeline["environment"]["backend"] == "real"
        calls.append(source)
        return source.parent

    monkeypatch.setattr(pipeline, "run_video", run)
    assert real.main(["run", str(fake_inputs)]) == 0
    assert len(calls) == 1
    assert fake_inputs.read_bytes() == before


def test_missing_ml_dependencies_prevent_server_start(tmp_path, monkeypatch):
    import uvicorn
    monkeypatch.setattr(real, "REAL_CONFIG", tmp_path / "config")
    monkeypatch.setattr(real, "model_doctor", lambda directory: 1)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: pytest.fail("server started with incomplete real setup"))
    assert real.main(["serve"]) == 1


def test_serve_cookie_argument_overrides_old_environment_before_diagnostics(tmp_path, monkeypatch):
    import uvicorn
    cookie = tmp_path / "export.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(real, "REAL_CONFIG", tmp_path / "config")
    monkeypatch.setattr(real, "REAL_RUNS", tmp_path / "real-runs")
    monkeypatch.setenv("SOUNDTHING_YOUTUBE_COOKIES", "missing-old-export.txt")
    monkeypatch.setattr(real, "ensure_panns_checkpoint", lambda: None)

    def doctor(directory):
        assert os.environ["SOUNDTHING_YOUTUBE_COOKIES"] == str(cookie)
        return 0

    monkeypatch.setattr(real, "model_doctor", doctor)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    assert real.main(["serve", "--cookies", str(cookie)]) == 0


def test_panns_labels_download_uses_https_and_is_validated_and_cached(tmp_path, monkeypatch):
    import io
    calls = []
    payload = "index,mid,display_name\n" + "".join(f"{i},/m/{i},label{i}\n" for i in range(527))

    def download(url, **kwargs):
        assert url.startswith("https://storage.googleapis.com/")
        calls.append(url)
        return io.BytesIO(payload.encode())

    monkeypatch.setattr(real.urllib.request, "urlopen", download)
    path = real.ensure_audioset_labels(tmp_path)
    assert real.ensure_audioset_labels(tmp_path) == path
    assert len(calls) == 1
    path.write_text("invalid truncated download")
    real.ensure_audioset_labels(tmp_path)
    assert len(calls) == 2
    assert not path.with_suffix(".csv.tmp").exists()


def test_invalid_panns_labels_are_not_installed(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(real.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"error page"))
    with pytest.raises(ValueError, match="527-label"):
        real.ensure_audioset_labels(tmp_path)
    assert not list(tmp_path.glob("*.csv*"))


def test_checkpoint_checksum_recovers_partial_or_corrupt_download(tmp_path, monkeypatch):
    import hashlib
    import io
    payload = b"verified model fixture"
    monkeypatch.setattr(real, "PANN_SIZE", len(payload))
    monkeypatch.setattr(real, "PANN_MD5", hashlib.md5(payload).hexdigest())
    calls = []

    def download(url, **kwargs):
        assert url.startswith("https://zenodo.org/")
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(real.urllib.request, "urlopen", download)
    target = tmp_path / "Cnn14_mAP=0.431.pth"
    target.write_bytes(b"x" * len(payload))
    assert real.ensure_panns_checkpoint(tmp_path) == target
    assert target.read_bytes() == payload
    real.ensure_panns_checkpoint(tmp_path)
    assert len(calls) == 1
    assert not target.with_suffix(".pth.tmp").exists()


def test_bad_checkpoint_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(real, "PANN_SIZE", 100)
    monkeypatch.setattr(real.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"error page"))
    target = tmp_path / "Cnn14_mAP=0.431.pth"
    target.write_bytes(b"previous incomplete model")
    with pytest.raises(ValueError, match="checksum"):
        real.ensure_panns_checkpoint(tmp_path)
    assert target.read_bytes() == b"previous incomplete model"
    assert not target.with_suffix(".pth.tmp").exists()
