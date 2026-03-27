import json
from pathlib import Path

from histoclass.config import AppConfig, load_config


def test_load_default_config_smoke() -> None:
    config = load_config()

    assert isinstance(config, AppConfig)
    assert config.data.image_root == Path("data/raw").resolve()
    assert config.model.backbone_name == "resnet18"
    assert config.trainer.epochs == 10
    assert config.evaluator.threshold == 0.5
    assert config.seed.seed == 42
    assert config.data.lmdb.enabled is True
    assert config.logging.level == "INFO"
    assert config.logging.streams["INFO"] == "stdout"
    assert config.logging.streams["ERROR"] == "stderr"
    assert config.logging.file_path is None


def test_load_config_with_partial_override(tmp_path) -> None:
    custom = {
        "data": {
            "image_root": "custom/raw",
            "loader": {"batch_size": 8, "num_workers": 0},
            "lmdb": {"enabled": False},
        },
        "trainer": {"epochs": 2, "checkpoint_dir": "custom/ckpt"},
        "seed": {"seed": 123},
        "logging": {
            "level": "DEBUG",
            "streams": {
                "DEBUG": "file",
                "INFO": "stdout",
                "WARNING": "stderr",
                "ERROR": "file",
            },
            "file_path": "outputs/logs/test.log",
        },
    }
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(custom), encoding="utf-8")

    config = load_config(config_path)

    project_root = Path(__file__).resolve().parents[1]
    assert config.data.image_root == (project_root / "custom/raw").resolve()
    assert config.data.loader.batch_size == 8
    assert config.data.loader.num_workers == 0
    assert config.data.loader.pin_memory is True
    assert config.data.lmdb.enabled is False
    assert config.trainer.epochs == 2
    assert config.trainer.checkpoint_dir == (project_root / "custom/ckpt").resolve()
    assert config.trainer.learning_rate == 1e-3
    assert config.seed.seed == 123
    assert config.seed.benchmark is False
    assert config.logging.level == "DEBUG"
    assert config.logging.streams["DEBUG"] == "file"
    assert config.logging.streams["ERROR"] == "file"
    assert config.logging.file_path == (project_root / "outputs/logs/test.log").resolve()
