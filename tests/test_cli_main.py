from pathlib import Path

from histoclass import project_root
from histoclass_cli.main import main, resolve_config_path
from histoclass_cli.pipeline import BatchPipelineResult, PipelineMode


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


def test_main_always_uses_batch_requests(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def _run_batch_pipeline_stub(requests):  # noqa: ANN001
        captured["requests"] = tuple(requests)
        return BatchPipelineResult(results=(object(), object()))

    monkeypatch.setattr("histoclass_cli.main.run_batch_pipeline", _run_batch_pipeline_stub)
    monkeypatch.setattr(
        "histoclass_cli.main._serialize_result_for_console",
        lambda **_: {"ok": True},
    )
    exit_code = main(["default", "train", "--mode", "train_eval"])

    assert exit_code == 0
    requests = captured["requests"]
    assert isinstance(requests, tuple)
    assert len(requests) == 2
    assert requests[0].mode == PipelineMode.TRAIN_EVAL
    assert requests[1].mode == PipelineMode.TRAIN_EVAL
    assert requests[0].config_path == (project_root() / "configs" / "default.json").resolve()
    assert requests[1].config_path == (project_root() / "configs" / "train.json").resolve()

    output = capsys.readouterr().out
    docs = [part.strip() for part in output.split("\n\n") if part.strip()]
    assert docs == ['{\n    "ok": true\n}', '{\n    "ok": true\n}']
