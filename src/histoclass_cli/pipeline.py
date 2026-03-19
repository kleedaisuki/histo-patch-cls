"""Application-level pipeline orchestration for histoclass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from histoclass import (
    AppConfig,
    EvaluationResult,
    Evaluator,
    Trainer,
    TrainSummary,
    build_data_module,
    build_model,
    config_to_dict,
    load_config,
)
from histoclass.utils import SeedState, get_logger, seed_everything


LOGGER = get_logger(__name__)


class PipelineMode(str, Enum):
    """@brief Pipeline 运行模式；Pipeline execution mode."""

    TRAIN = "train"
    EVAL = "eval"
    TRAIN_EVAL = "train_eval"


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """@brief Pipeline 请求参数；Pipeline request parameters.

    @param config_path 配置文件路径，None 表示使用默认配置；Config path, None to use default config.
    @param mode 运行模式；Execution mode.
    @param checkpoint_path 可选 checkpoint 路径；Optional checkpoint path.
    """

    config_path: str | Path | None = None
    mode: PipelineMode = PipelineMode.TRAIN_EVAL
    checkpoint_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """@brief Pipeline 执行结果；Pipeline execution result.

    @param config 生效配置；Resolved application config.
    @param seed_state 随机种子状态；Applied seed state.
    @param train_summary 训练结果；Training summary.
    @param evaluation 评估结果；Evaluation result.
    @param checkpoint_path 本次使用或生成的 checkpoint 路径；Used or generated checkpoint path.
    """

    config: AppConfig
    seed_state: SeedState
    train_summary: TrainSummary | None
    evaluation: EvaluationResult | None
    checkpoint_path: Path | None


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    """@brief 运行应用层 pipeline；Run application-level pipeline.

    @param request pipeline 请求；Pipeline request.
    @return pipeline 执行结果；Pipeline result.
    """
    config = load_config(request.config_path)
    LOGGER.info(
        "Pipeline started: mode=%s, config_path=%s",
        request.mode.value,
        request.config_path,
    )

    seed_state = seed_everything(
        seed=config.seed.seed,
        deterministic=config.seed.deterministic,
        benchmark=config.seed.benchmark,
    )

    data_module = build_data_module(config.data)
    model = build_model(config.model)

    checkpoint_path = _resolve_checkpoint_path(
        request.checkpoint_path,
        mode=request.mode,
    )
    if checkpoint_path is not None:
        _load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)

    train_summary: TrainSummary | None = None
    evaluation: EvaluationResult | None = None

    if request.mode in (PipelineMode.TRAIN, PipelineMode.TRAIN_EVAL):
        trainer = Trainer(model=model, config=config.trainer)
        train_summary = trainer.fit(data_module.train_loader)
        if train_summary.final_checkpoint is not None:
            checkpoint_path = train_summary.final_checkpoint

    if request.mode in (PipelineMode.EVAL, PipelineMode.TRAIN_EVAL):
        evaluator = Evaluator(model=model, config=config.evaluator)
        evaluation = evaluator.evaluate(data_module.val_loader)

    _log_pipeline_summary(
        mode=request.mode,
        train_summary=train_summary,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path,
    )
    return PipelineResult(
        config=config,
        seed_state=seed_state,
        train_summary=train_summary,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path,
    )


def run_pipeline_from_paths(
    *,
    config_path: str | Path | None,
    mode: PipelineMode,
    checkpoint_path: str | Path | None,
) -> PipelineResult:
    """@brief 面向 main.py 的便捷入口；Convenient entrypoint for main.py.

    @param config_path 配置路径；Config path.
    @param mode 运行模式；Execution mode.
    @param checkpoint_path checkpoint 路径；Checkpoint path.
    @return pipeline 执行结果；Pipeline result.
    """
    request = PipelineRequest(
        config_path=config_path,
        mode=mode,
        checkpoint_path=checkpoint_path,
    )
    return run_pipeline(request)


def format_result_for_console(result: PipelineResult) -> str:
    """@brief 将 pipeline 结果序列化为终端文本；Serialize pipeline result to console text.

    @param result pipeline 执行结果；Pipeline result.
    @return 可打印文本；Printable text.
    """
    payload: dict[str, Any] = {
        "config": config_to_dict(result.config),
        "seed": {
            "seed": result.seed_state.seed,
            "deterministic": result.seed_state.deterministic,
            "benchmark": result.seed_state.benchmark,
            "cuda_available": result.seed_state.cuda_available,
        },
        "checkpoint_path": (
            result.checkpoint_path.as_posix()
            if result.checkpoint_path is not None
            else None
        ),
    }

    if result.train_summary is not None:
        payload["train"] = {
            "epochs": len(result.train_summary.history),
            "final_checkpoint": (
                result.train_summary.final_checkpoint.as_posix()
                if result.train_summary.final_checkpoint is not None
                else None
            ),
            "final_train_loss": result.train_summary.history[-1].train.loss,
            "final_train_metrics": result.train_summary.history[-1].train.metrics.to_dict(),
        }

    if result.evaluation is not None:
        payload["eval"] = {
            "samples": result.evaluation.samples,
            "steps": result.evaluation.steps,
            "loss": result.evaluation.loss,
            "metrics": result.evaluation.metrics.to_dict(),
        }

    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    *,
    mode: PipelineMode,
) -> Path | None:
    if checkpoint_path is None:
        if mode == PipelineMode.EVAL:
            raise ValueError("checkpoint_path is required when mode='eval'.")
        return None
    return Path(checkpoint_path).expanduser().resolve()


def _load_model_checkpoint(*, model: torch.nn.Module, checkpoint_path: Path) -> None:
    """@brief 加载模型 checkpoint；Load model checkpoint into model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint payload misses key 'model_state_dict'.")

    model.load_state_dict(state_dict)
    LOGGER.info("Checkpoint loaded: %s", checkpoint_path)


def _log_pipeline_summary(
    *,
    mode: PipelineMode,
    train_summary: TrainSummary | None,
    evaluation: EvaluationResult | None,
    checkpoint_path: Path | None,
) -> None:
    if train_summary is not None:
        final_epoch = train_summary.history[-1]
        LOGGER.info(
            "Pipeline train summary | mode=%s epoch=%d train_loss=%.6f train_f1=%.4f",
            mode.value,
            final_epoch.epoch,
            final_epoch.train.loss,
            final_epoch.train.metrics.f1,
        )

    if evaluation is not None:
        LOGGER.info(
            "Pipeline eval summary | mode=%s val_loss=%s val_f1=%.4f val_auc=%s",
            mode.value,
            (f"{evaluation.loss:.6f}" if evaluation.loss is not None else "None"),
            evaluation.metrics.f1,
            (
                f"{evaluation.metrics.roc_auc:.4f}"
                if evaluation.metrics.roc_auc is not None
                else "None"
            ),
        )

    LOGGER.info("Pipeline completed | mode=%s checkpoint=%s", mode.value, checkpoint_path)


__all__ = [
    "PipelineMode",
    "PipelineRequest",
    "PipelineResult",
    "format_result_for_console",
    "run_pipeline",
    "run_pipeline_from_paths",
]

