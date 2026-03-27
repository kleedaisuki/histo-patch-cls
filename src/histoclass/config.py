"""Unified JSON config loader for IDC patch classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .data import DataModuleConfig, ImageSchema, LmdbSchema, LoaderSchema, SplitSchema
from .engine import EvaluatorConfig, TrainerConfig
from .model import ModelConfig
from .utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """@brief 随机种子配置；Random seed configuration.

    @param seed 主随机种子；Primary random seed.
    @param deterministic 是否启用确定性算法；Whether to enable deterministic algorithms.
    @param benchmark 是否启用 cuDNN benchmark；Whether to enable cuDNN benchmark.
    """

    seed: int = 42
    deterministic: bool = False
    benchmark: bool = False


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """@brief 日志系统配置；Logging system configuration.

    @param level logger 最低级别；Global logger threshold.
    @param streams 各日志级别到输出流的映射；Per-level output stream mapping.
    @param file_path 文件流路径（仅 file 目标时需要）；File target path when needed.
    """

    level: str = "INFO"
    streams: dict[str, str] = None  # type: ignore[assignment]
    file_path: Path | None = None

    def __post_init__(self) -> None:
        if self.streams is None:
            object.__setattr__(
                self,
                "streams",
                {
                    "DEBUG": "stdout",
                    "INFO": "stdout",
                    "WARNING": "stderr",
                    "ERROR": "stderr",
                    "CRITICAL": "stderr",
                },
            )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """@brief 顶层应用配置；Top-level application config.

    @param data 数据模块配置；Data module configuration.
    @param model 模型配置；Model configuration.
    @param trainer 训练配置；Trainer configuration.
    @param evaluator 评估配置；Evaluator configuration.
    @param seed 随机种子配置；Seed configuration.
    """

    data: DataModuleConfig
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    seed: SeedConfig = SeedConfig()
    logging: LoggingConfig = LoggingConfig()


def project_root() -> Path:
    """@brief 获取项目根目录；Get project root path.

    @return 项目根目录绝对路径；Absolute project root path.
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """@brief 获取默认配置路径；Get default config path.

    @return 默认配置文件绝对路径；Absolute path to default config file.
    """
    return project_root() / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> AppConfig:
    """@brief 加载并解析配置；Load and parse configuration.

    @param path 用户配置路径，None 时读取默认配置；User config path, default when None.
    @return 解析后的应用配置；Parsed application config.
    """
    default_path = default_config_path().resolve()
    target_path = default_path if path is None else Path(path).expanduser().resolve()

    if path is None:
        payload = _load_json_object(default_path)
        return parse_config_dict(payload, config_path=default_path)

    default_payload = _load_json_object(default_path)
    override_payload = _load_json_object(target_path)
    merged_payload = _deep_merge_dict(default_payload, override_payload)
    return parse_config_dict(merged_payload, config_path=target_path)


def parse_config_dict(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> AppConfig:
    """@brief 从字典解析配置；Parse configuration from mapping.

    @param payload 配置字典；Configuration payload.
    @param config_path 配置文件路径，用于解析相对路径；Config file path for resolving relative paths.
    @return 解析后的应用配置；Parsed application config.
    """
    root = _expect_mapping(payload, "config")
    _ensure_allowed_keys(
        root,
        allowed={"data", "model", "trainer", "evaluator", "seed", "logging"},
        scope="config",
    )

    del config_path
    config_dir = project_root()
    data_section = _expect_mapping(root.get("data"), "config.data")
    data_config = _parse_data_config(data_section, config_dir=config_dir)

    model_section = _expect_mapping(root.get("model", {}), "config.model")
    trainer_section = _expect_mapping(root.get("trainer", {}), "config.trainer")
    evaluator_section = _expect_mapping(root.get("evaluator", {}), "config.evaluator")
    seed_section = _expect_mapping(root.get("seed", {}), "config.seed")
    logging_section = _expect_mapping(root.get("logging", {}), "config.logging")

    app_config = AppConfig(
        data=data_config,
        model=_parse_model_config(model_section),
        trainer=_parse_trainer_config(trainer_section, config_dir=config_dir),
        evaluator=_parse_evaluator_config(evaluator_section),
        seed=_parse_seed_config(seed_section),
        logging=_parse_logging_config(logging_section, config_dir=config_dir),
    )
    configure_logging(
        level=app_config.logging.level,
        level_targets=app_config.logging.streams,
        file_path=app_config.logging.file_path,
    )
    return app_config


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """@brief 配置对象转字典；Convert config object to dictionary.

    @param config 顶层配置对象；Top-level config object.
    @return 可 JSON 序列化字典；JSON-serializable dictionary.
    """
    return _to_json_compatible(asdict(config))


def save_config(config: AppConfig, path: str | Path) -> Path:
    """@brief 保存配置到 JSON；Save config to JSON.

    @param config 顶层配置对象；Top-level config object.
    @param path 输出文件路径；Output file path.
    @return 输出文件绝对路径；Absolute path to saved file.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config_to_dict(config)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Config saved to %s", output_path)
    return output_path


def _parse_data_config(section: Mapping[str, Any], *, config_dir: Path) -> DataModuleConfig:
    _ensure_allowed_keys(
        section,
        allowed={
            "image_root",
            "image",
            "split",
            "loader",
            "lmdb",
        },
        scope="config.data",
    )
    image_root_raw = section.get("image_root")
    if image_root_raw is None:
        raise ValueError("config.data.image_root is required.")

    image_root = _resolve_path(Path(str(image_root_raw)), base_dir=config_dir)
    image_section = _expect_mapping(section.get("image", {}), "config.data.image")
    split_section = _expect_mapping(section.get("split", {}), "config.data.split")
    loader_section = _expect_mapping(section.get("loader", {}), "config.data.loader")
    lmdb_section = _expect_mapping(section.get("lmdb", {}), "config.data.lmdb")

    return DataModuleConfig(
        image_root=image_root,
        image=_parse_image_schema(image_section),
        split=_parse_split_schema(split_section),
        loader=_parse_loader_schema(loader_section),
        lmdb=_parse_lmdb_schema(lmdb_section, config_dir=config_dir),
    )


def _parse_image_schema(section: Mapping[str, Any]) -> ImageSchema:
    _ensure_allowed_keys(
        section,
        allowed={"image_size", "mean", "std"},
        scope="config.data.image",
    )
    return ImageSchema(
        image_size=_read_int_pair(section.get("image_size", (50, 50)), "image_size"),
        mean=_read_float_triplet(section.get("mean", (0.5, 0.5, 0.5)), "mean"),
        std=_read_float_triplet(section.get("std", (0.25, 0.25, 0.25)), "std"),
    )


def _parse_split_schema(section: Mapping[str, Any]) -> SplitSchema:
    _ensure_allowed_keys(section, allowed={"val_ratio", "seed"}, scope="config.data.split")
    return SplitSchema(
        val_ratio=float(section.get("val_ratio", 0.2)),
        seed=int(section.get("seed", 42)),
    )


def _parse_loader_schema(section: Mapping[str, Any]) -> LoaderSchema:
    _ensure_allowed_keys(
        section,
        allowed={
            "batch_size",
            "num_workers",
            "pin_memory",
            "train_drop_last",
            "eval_drop_last",
            "use_weighted_sampler",
        },
        scope="config.data.loader",
    )
    return LoaderSchema(
        batch_size=int(section.get("batch_size", 64)),
        num_workers=int(section.get("num_workers", 4)),
        pin_memory=bool(section.get("pin_memory", True)),
        train_drop_last=bool(section.get("train_drop_last", True)),
        eval_drop_last=bool(section.get("eval_drop_last", False)),
        use_weighted_sampler=bool(section.get("use_weighted_sampler", False)),
    )


def _parse_lmdb_schema(section: Mapping[str, Any], *, config_dir: Path) -> LmdbSchema:
    _ensure_allowed_keys(
        section,
        allowed={"enabled", "path", "map_size_bytes"},
        scope="config.data.lmdb",
    )
    path_raw = section.get("path")
    path = None if path_raw is None else _resolve_path(Path(str(path_raw)), base_dir=config_dir)
    return LmdbSchema(
        enabled=bool(section.get("enabled", True)),
        path=path,
        map_size_bytes=int(section.get("map_size_bytes", 8 * 1024 * 1024 * 1024)),
    )


def _parse_model_config(section: Mapping[str, Any]) -> ModelConfig:
    _ensure_allowed_keys(
        section,
        allowed={
            "backbone_name",
            "hidden_dim",
            "dropout",
            "pretrained",
            "freeze_backbone",
        },
        scope="config.model",
    )
    return ModelConfig(
        backbone_name=str(section.get("backbone_name", "resnet18")),
        hidden_dim=int(section.get("hidden_dim", 256)),
        dropout=float(section.get("dropout", 0.2)),
        pretrained=bool(section.get("pretrained", True)),
        freeze_backbone=bool(section.get("freeze_backbone", False)),
    )


def _parse_trainer_config(section: Mapping[str, Any], *, config_dir: Path) -> TrainerConfig:
    _ensure_allowed_keys(
        section,
        allowed={
            "epochs",
            "learning_rate",
            "weight_decay",
            "threshold",
            "device",
            "use_amp",
            "grad_clip_norm",
            "pos_weight",
            "log_every_n_steps",
            "checkpoint_dir",
            "save_checkpoint_each_epoch",
        },
        scope="config.trainer",
    )

    checkpoint_raw = section.get("checkpoint_dir", "outputs/checkpoints")
    checkpoint_dir = _resolve_path(Path(str(checkpoint_raw)), base_dir=config_dir)
    return TrainerConfig(
        epochs=int(section.get("epochs", 10)),
        learning_rate=float(section.get("learning_rate", 1e-3)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
        threshold=float(section.get("threshold", 0.5)),
        device=_read_optional_str(section.get("device")),
        use_amp=bool(section.get("use_amp", False)),
        grad_clip_norm=_read_optional_float(section.get("grad_clip_norm")),
        pos_weight=_read_optional_float(section.get("pos_weight")),
        log_every_n_steps=int(section.get("log_every_n_steps", 50)),
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_each_epoch=bool(section.get("save_checkpoint_each_epoch", True)),
    )


def _parse_evaluator_config(section: Mapping[str, Any]) -> EvaluatorConfig:
    _ensure_allowed_keys(
        section,
        allowed={"threshold", "device", "use_amp"},
        scope="config.evaluator",
    )
    return EvaluatorConfig(
        threshold=float(section.get("threshold", 0.5)),
        device=_read_optional_str(section.get("device")),
        use_amp=bool(section.get("use_amp", False)),
    )


def _parse_seed_config(section: Mapping[str, Any]) -> SeedConfig:
    _ensure_allowed_keys(
        section,
        allowed={"seed", "deterministic", "benchmark"},
        scope="config.seed",
    )
    return SeedConfig(
        seed=int(section.get("seed", 42)),
        deterministic=bool(section.get("deterministic", False)),
        benchmark=bool(section.get("benchmark", False)),
    )


def _parse_logging_config(section: Mapping[str, Any], *, config_dir: Path) -> LoggingConfig:
    _ensure_allowed_keys(
        section,
        allowed={"level", "streams", "file_path"},
        scope="config.logging",
    )

    streams_value = section.get("streams")
    if streams_value is None:
        streams: dict[str, str] | None = None
    else:
        streams_mapping = _expect_mapping(streams_value, "config.logging.streams")
        streams = {str(key): str(value) for key, value in streams_mapping.items()}

    file_path_raw = section.get("file_path")
    file_path = (
        None
        if file_path_raw is None
        else _resolve_path(Path(str(file_path_raw)), base_dir=config_dir)
    )

    return LoggingConfig(
        level=str(section.get("level", "INFO")),
        streams=streams,
        file_path=file_path,
    )


def _expect_mapping(value: Any, scope: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{scope} must be a mapping, got {type(value).__name__}.")
    return value


def _ensure_allowed_keys(
    data: Mapping[str, Any], *, allowed: set[str], scope: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise KeyError(f"Unknown keys in {scope}: {unknown}")


def _load_json_object(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc

    if not isinstance(payload, Mapping):
        raise TypeError(f"Top-level JSON must be an object in: {path}")
    return payload


def _deep_merge_dict(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dict(
                _expect_mapping(merged[key], f"base.{key}"),
                _expect_mapping(value, f"override.{key}"),
            )
        else:
            merged[key] = value
    return merged


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_int_pair(value: Any, field_name: str) -> tuple[int, int]:
    sequence = _read_sequence(value, field_name, expected_len=2)
    return (int(sequence[0]), int(sequence[1]))


def _read_float_triplet(value: Any, field_name: str) -> tuple[float, float, float]:
    sequence = _read_sequence(value, field_name, expected_len=3)
    return (float(sequence[0]), float(sequence[1]), float(sequence[2]))


def _read_sequence(value: Any, field_name: str, *, expected_len: int) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{field_name} must be a list/tuple with length {expected_len}, "
            f"got {type(value).__name__}."
        )
    if len(value) != expected_len:
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(value)}."
        )
    return list(value)


def _read_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _read_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    return value


__all__ = [
    "AppConfig",
    "LoggingConfig",
    "SeedConfig",
    "config_to_dict",
    "default_config_path",
    "load_config",
    "parse_config_dict",
    "project_root",
    "save_config",
]
