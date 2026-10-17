from pathlib import Path

from soundthing.events.tuning import paths


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "t"))
    assert paths.tuning_home() == tmp_path / "t"
    assert paths.ledger_path() == tmp_path / "t" / "ledger.jsonl"
    assert paths.cache_root().parent == tmp_path / "t"


def test_main_checkout_has_pyproject():
    assert (paths.main_checkout() / "pyproject.toml").exists()


def test_default_is_data_tuning(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: None)
    assert paths.tuning_home() == paths.main_checkout() / "data" / "tuning"
