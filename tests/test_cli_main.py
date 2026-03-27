from pathlib import Path

from histoclass import project_root
from histoclass_cli.main import resolve_config_path


def test_resolve_config_path_named_without_suffix() -> None:
    resolved = resolve_config_path("default")

    assert resolved == (project_root() / "configs" / "default.json").resolve()


def test_resolve_config_path_named_with_json_suffix() -> None:
    resolved = resolve_config_path("default.json")

    assert resolved == (project_root() / "configs" / "default.json").resolve()


def test_resolve_config_path_explicit_relative_path(tmp_path, monkeypatch) -> None:
    cfg_dir = tmp_path / "nested"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "custom.json"
    cfg_file.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    resolved = resolve_config_path("nested/custom.json")

    assert resolved == cfg_file.resolve()
