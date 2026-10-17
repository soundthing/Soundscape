"""Tests for Kaggle cache runner client and kernel scripts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import types
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import load_config
import soundthing.events.tuning as tuning_pkg
from soundthing.events.tuning import kaggle_io, paths


def _inject(monkeypatch: pytest.MonkeyPatch, name: str, fake: types.ModuleType) -> None:
    """Fake a tuning submodule; the package attribute wins over sys.modules once imported."""
    monkeypatch.setitem(sys.modules, f"soundthing.events.tuning.{name}", fake)
    monkeypatch.setattr(tuning_pkg, name, fake, raising=False)


@pytest.fixture
def fake_wav(tmp_path: Path) -> Path:
    """Create a tiny fake mono WAV using soundfile in tmp_path."""
    wav_path = tmp_path / "audio" / "test_clip.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    data = np.zeros(sr, dtype=np.float32)
    sf.write(str(wav_path), data, sr)
    return wav_path


def test_fake_audio_creation(fake_wav: Path) -> None:
    data, sr = sf.read(str(fake_wav))
    assert sr == 16000
    assert len(data) == 16000


def test_importing_kaggle_io_does_not_import_kaggle() -> None:
    # Ensure neither kaggle nor any submodules are in sys.modules
    for mod in list(sys.modules):
        if mod == "kaggle" or mod.startswith("kaggle."):
            sys.modules.pop(mod)

    # Force re-importing kaggle_io
    if "soundthing.events.tuning.kaggle_io" in sys.modules:
        del sys.modules["soundthing.events.tuning.kaggle_io"]

    import soundthing.events.tuning.kaggle_io as kio  # noqa: F401

    assert "kaggle" not in sys.modules
    assert not any(m.startswith("kaggle.") for m in sys.modules)


def test_slug_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))

    # Overrides via env
    monkeypatch.setenv("SOUNDTHING_KAGGLE_KERNEL", "custom-org/custom-kernel")
    monkeypatch.setenv("SOUNDTHING_KAGGLE_DATASET", "custom-org/custom-dataset")
    assert kaggle_io.kernel_slug() == "custom-org/custom-kernel"
    assert kaggle_io.dataset_slug() == "custom-org/custom-dataset"

    # Fallback to KAGGLE_USERNAME
    monkeypatch.delenv("SOUNDTHING_KAGGLE_KERNEL", raising=False)
    monkeypatch.delenv("SOUNDTHING_KAGGLE_DATASET", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "mockuser")
    assert kaggle_io.kernel_slug() == "mockuser/soundthing-tuning-cache"
    assert kaggle_io.dataset_slug() == "mockuser/soundthing-tuning-cache-data"

    # Missing both -> RuntimeError
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="SOUNDTHING_KAGGLE_KERNEL|KAGGLE_USERNAME"):
        kaggle_io.kernel_slug()
    with pytest.raises(RuntimeError, match="SOUNDTHING_KAGGLE_DATASET|KAGGLE_USERNAME"):
        kaggle_io.dataset_slug()


def test_push_cache_kernel_missing_repo_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    monkeypatch.delenv("SOUNDTHING_REPO_URL", raising=False)
    with pytest.raises(RuntimeError, match="SOUNDTHING_REPO_URL"):
        kaggle_io.push_cache_kernel(1.0, 0.5, "panns_cnn14")


def test_push_cache_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    monkeypatch.setenv("SOUNDTHING_REPO_URL", "https://github.com/example/repo.git")
    monkeypatch.setenv("KAGGLE_USERNAME", "mockuser")
    monkeypatch.delenv("SOUNDTHING_KAGGLE_KERNEL", raising=False)
    monkeypatch.delenv("SOUNDTHING_KAGGLE_ESC50", raising=False)

    fake_ledger = types.ModuleType("soundthing.events.tuning.ledger")
    fake_ledger.git_state = lambda cwd=None: {"sha": "sha_feedcafe", "dirty": False, "branch": "main"}
    _inject(monkeypatch, "ledger", fake_ledger)

    recorded_files: dict[str, str] = {}

    class FakeApi:
        def kernels_push(self, folder: str) -> dict[str, Any]:
            p = Path(folder)
            for f in p.iterdir():
                if f.is_file():
                    recorded_files[f.name] = f.read_text(encoding="utf-8")
            return {"ref": "pushed_kernel_ref"}

    client = kaggle_io.KaggleClient(api=FakeApi())
    result = kaggle_io.push_cache_kernel(
        window=1.5,
        hop=0.75,
        model="panns_cnn14",
        n_per_env=15,
        seed=42,
        client=client,
    )

    assert result["kernel"] == "mockuser/soundthing-tuning-cache"
    assert result["push"] == {"ref": "pushed_kernel_ref"}
    assert "kernel-metadata.json" in recorded_files
    assert "cache_kernel.py" in recorded_files

    meta = json.loads(recorded_files["kernel-metadata.json"])
    assert meta["id"] == "mockuser/soundthing-tuning-cache"
    assert meta["title"] == "soundthing-tuning-cache"
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True
    assert len(meta["dataset_sources"]) == 1
    assert not any(k.startswith("_soundthing") for k in meta)

    code = recorded_files["cache_kernel.py"]
    match = re.search(r"^PARAMS\s*=\s*(\{.*?\})\s*(?:#.*)?$", code, flags=re.MULTILINE)
    assert match is not None
    params = json.loads(match.group(1))
    assert params["window_sec"] == 1.5
    assert params["hop_sec"] == 0.75
    assert params["tagger_model"] == "panns_cnn14"
    assert params["git_sha"] == "sha_feedcafe"
    assert params["repo_url"] == "https://github.com/example/repo.git"
    assert params["n_per_env"] == 15
    assert params["seed"] == 42
    assert params["esc50_dataset"] == meta["dataset_sources"][0]
    assert result["params"] == params


def test_kernel_status_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    monkeypatch.setenv("SOUNDTHING_KAGGLE_KERNEL", "mockuser/test-kernel")

    class DictApi:
        def kernels_status(self, slug: str) -> dict[str, str]:
            return {"status": "COMPLETE"}

    assert kaggle_io.kernel_status(client=kaggle_io.KaggleClient(api=DictApi())) == "complete"

    class ObjStatus:
        status = "RUNNING"

    class ObjApi:
        def kernels_status(self, slug: str) -> Any:
            return ObjStatus()

    assert kaggle_io.kernel_status(client=kaggle_io.KaggleClient(api=ObjApi())) == "running"

    class StrApi:
        def kernels_status(self, slug: str) -> str:
            return "QUEUED"

    assert kaggle_io.kernel_status(client=kaggle_io.KaggleClient(api=StrApi())) == "queued"


def test_download_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "mockuser")

    dataset_calls: list[tuple[str, str, bool]] = []
    kernel_calls: list[tuple[str, str]] = []

    class FakeApi:
        def dataset_download_files(self, slug: str, path: str, unzip: bool) -> None:
            dataset_calls.append((slug, path, unzip))

        def kernels_output(self, slug: str, path: str) -> None:
            kernel_calls.append((slug, path))

    client = kaggle_io.KaggleClient(api=FakeApi())

    # Default dataset download
    dest_ds = kaggle_io.download_cache(client=client)
    assert dest_ds == tmp_path / "downloads" / "soundthing-tuning-cache-data"
    assert dataset_calls == [("mockuser/soundthing-tuning-cache-data", str(dest_ds), True)]

    # From kernel download
    dest_k = kaggle_io.download_cache(from_kernel=True, client=client)
    assert dest_k == tmp_path / "downloads" / "soundthing-tuning-cache"
    assert kernel_calls == [("mockuser/soundthing-tuning-cache", str(dest_k))]


def test_verify_cache_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    fake_cache = types.ModuleType("soundthing.events.tuning.cache")
    fake_cache.cache_key = lambda c: "expected_key_abc"
    _inject(monkeypatch, "cache", fake_cache)

    # 1. Matching meta in root
    cache_dir = tmp_path / "dl1"
    cache_dir.mkdir()
    (cache_dir / "meta.json").write_text(json.dumps({"cache_key": "expected_key_abc"}), encoding="utf-8")
    assert kaggle_io.verify_cache_key(cache_dir, cfg) == "expected_key_abc"

    # 2. Matching meta in tuning/ subdir
    cache_dir_nested = tmp_path / "dl2"
    (cache_dir_nested / "tuning").mkdir(parents=True)
    (cache_dir_nested / "tuning" / "meta.json").write_text(
        json.dumps({"cache_key": "expected_key_abc"}), encoding="utf-8"
    )
    assert kaggle_io.verify_cache_key(cache_dir_nested, cfg) == "expected_key_abc"

    # 3. Mismatched meta -> ValueError naming both keys
    cache_dir_mismatch = tmp_path / "dl3"
    cache_dir_mismatch.mkdir()
    (cache_dir_mismatch / "meta.json").write_text(json.dumps({"cache_key": "wrong_key_xyz"}), encoding="utf-8")
    with pytest.raises(ValueError) as exc_mismatch:
        kaggle_io.verify_cache_key(cache_dir_mismatch, cfg)
    assert "wrong_key_xyz" in str(exc_mismatch.value)
    assert "expected_key_abc" in str(exc_mismatch.value)

    # 4. Missing meta -> ValueError naming expected key
    cache_dir_empty = tmp_path / "dl4"
    cache_dir_empty.mkdir()
    with pytest.raises(ValueError) as exc_missing:
        kaggle_io.verify_cache_key(cache_dir_empty, cfg)
    assert "expected_key_abc" in str(exc_missing.value)


def test_install_cache_holdout_overlap_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    fake_cache = types.ModuleType("soundthing.events.tuning.cache")
    fake_cache.cache_key = lambda c: "cache_123"
    _inject(monkeypatch, "cache", fake_cache)

    fake_holdout = types.ModuleType("soundthing.events.tuning.holdout")

    def raise_overlap(ids: list[str]) -> None:
        raise ValueError(f"Holdout overlap detected: {ids}")

    fake_holdout.assert_disjoint = raise_overlap
    _inject(monkeypatch, "holdout", fake_holdout)

    src = tmp_path / "source_cache"
    (src / "tuning" / "cache" / "cache_123").mkdir(parents=True)
    (src / "tuning" / "dev" / "labels").mkdir(parents=True)
    (src / "tuning" / "meta.json").write_text(json.dumps({"cache_key": "cache_123"}), encoding="utf-8")
    (src / "tuning" / "cache" / "cache_123" / "clip01.npz").write_bytes(b"npz_data")
    (src / "tuning" / "dev" / "labels" / "clip01.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Holdout overlap detected"):
        kaggle_io.install_cache(src, cfg)

    # Verify nothing was copied
    dest_cache = paths.cache_root() / "cache_123"
    assert not dest_cache.exists() or not list(dest_cache.glob("*.npz"))
    dest_labels = paths.dev_labels_dir()
    assert not dest_labels.exists() or not list(dest_labels.glob("*.json"))


def test_install_cache_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    fake_cache = types.ModuleType("soundthing.events.tuning.cache")
    fake_cache.cache_key = lambda c: "cache_123"
    _inject(monkeypatch, "cache", fake_cache)

    disjoint_calls: list[list[str]] = []
    fake_holdout = types.ModuleType("soundthing.events.tuning.holdout")
    fake_holdout.assert_disjoint = lambda ids: disjoint_calls.append(ids)
    _inject(monkeypatch, "holdout", fake_holdout)

    src = tmp_path / "source_cache"
    (src / "tuning" / "cache" / "cache_123").mkdir(parents=True)
    (src / "tuning" / "dev" / "labels").mkdir(parents=True)
    (src / "tuning" / "meta.json").write_text(json.dumps({"cache_key": "cache_123"}), encoding="utf-8")
    (src / "tuning" / "cache" / "cache_123" / "clip01.npz").write_bytes(b"npz_data")
    (src / "tuning" / "dev" / "labels" / "clip01.json").write_text('{"video_id": "clip01"}', encoding="utf-8")
    (src / "tuning" / "dev" / "manifest.json").write_text('{"count": 1}', encoding="utf-8")

    installed = kaggle_io.install_cache(src, cfg)
    assert installed == paths.cache_root() / "cache_123"
    assert (paths.cache_root() / "cache_123" / "clip01.npz").read_bytes() == b"npz_data"
    assert (paths.dev_labels_dir() / "clip01.json").exists()
    assert (paths.dev_labels_dir().parent / "manifest.json").exists()
    assert disjoint_calls == [["clip01"]]


def test_request_recache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    monkeypatch.setenv("SOUNDTHING_REPO_URL", "https://github.com/example/repo.git")
    monkeypatch.setenv("KAGGLE_USERNAME", "mockuser")

    fake_cache = types.ModuleType("soundthing.events.tuning.cache")
    fake_cache.cache_key = lambda c: "key_expected_456"
    _inject(monkeypatch, "cache", fake_cache)

    fake_overrides = types.ModuleType("soundthing.events.tuning.overrides")
    fake_overrides.apply_overrides = lambda c, o: c
    _inject(monkeypatch, "overrides", fake_overrides)

    # 1. Dirty state
    fake_ledger = types.ModuleType("soundthing.events.tuning.ledger")
    fake_ledger.git_state = lambda cwd=None: {"sha": "sha_dirty_999", "dirty": True, "branch": "dev"}
    _inject(monkeypatch, "ledger", fake_ledger)

    class FakeApi:
        def kernels_push(self, folder: str) -> dict[str, str]:
            return {"status": "ok"}

    client = kaggle_io.KaggleClient(api=FakeApi())
    res_dirty = kaggle_io.request_recache(2.0, 1.0, "panns_cnn14", client=client)

    assert res_dirty["kernel"] == "mockuser/soundthing-tuning-cache"
    assert res_dirty["dataset"] == "mockuser/soundthing-tuning-cache-data"
    assert res_dirty["expected_cache_key"] == "key_expected_456"
    assert res_dirty["git_sha"] == "sha_dirty_999"
    assert res_dirty["dirty"] is True
    assert "uncommitted" in res_dirty["note"].lower()
    assert res_dirty["status"] == "pushed"
    assert res_dirty["window_sec"] == 2.0
    assert res_dirty["hop_sec"] == 1.0
    assert res_dirty["tagger_model"] == "panns_cnn14"

    # 2. Clean state
    fake_ledger.git_state = lambda cwd=None: {"sha": "sha_clean_000", "dirty": False, "branch": "dev"}
    res_clean = kaggle_io.request_recache(2.0, 1.0, "panns_cnn14", client=client)
    assert res_clean["dirty"] is False
    assert "uncommitted" not in res_clean["note"].lower()


def test_cache_kernel_script_compiles() -> None:
    kernel_file = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "soundthing"
        / "events"
        / "tuning"
        / "kaggle"
        / "cache_kernel.py"
    )
    src = kernel_file.read_text(encoding="utf-8")
    assert "PARAMS = {}  # filled in by kaggle_io.push_cache_kernel" in src
    lines = src.splitlines()
    assert len(lines) < 120
    # Compiles cleanly
    compile(src, str(kernel_file), "exec")
