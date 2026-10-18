from pathlib import Path

from soundthing.harness import paths


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "t"))
    assert paths.harness_home() == tmp_path / "t"
    assert paths.ledger_path() == tmp_path / "t" / "ledger.jsonl"
    assert paths.cache_root().parent == tmp_path / "t"


def test_old_env_var_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setenv(paths.OLD_ENV_VAR, str(tmp_path / "old"))
    assert paths.harness_home() == tmp_path / "old"


def test_main_checkout_has_pyproject():
    assert (paths.main_checkout() / "pyproject.toml").exists()


def test_default_is_data_harness(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.delenv(paths.OLD_ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "main_checkout", lambda: tmp_path)
    assert paths.harness_home() == tmp_path / "data" / "harness"


def test_falls_back_to_existing_data_tuning(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.delenv(paths.OLD_ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "main_checkout", lambda: tmp_path)
    (tmp_path / "data" / "tuning").mkdir(parents=True)
    assert paths.harness_home() == tmp_path / "data" / "tuning"
