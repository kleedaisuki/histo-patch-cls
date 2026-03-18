"""Evaluation and inference engine for IDC patch classification."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from ..data import Batch
from ..utils import BinaryMetrics, compute_binary_metrics, get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """@brief 评估器配置；Evaluator configuration.

    @param threshold 二分类阈值；Decision threshold for binary classification.
    @param device 推理设备，None 表示自动选择；Inference device, None for auto-select.
    @param use_amp 是否启用自动混合精度；Enable automatic mixed precision.
    """

    threshold: float = 0.5
    device: str | None = None
    use_amp: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """@brief 数据集评估结果；Dataset-level evaluation result.

    @param metrics 二分类指标；Aggregated binary metrics.
    @param samples 样本总数；Total number of processed samples.
    @param steps 批次数；Total number of evaluation steps.
    @param loss 平均损失，若未提供 criterion 则为 None；Average loss, None if no criterion.
    """

    metrics: BinaryMetrics
    samples: int
    steps: int
    loss: float | None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """@brief 批量推理结果；Batch inference result.

    @param logits 原始 logits；Raw logits.
    @param probabilities 正类概率；Positive-class probabilities.
    @param predictions 阈值化预测标签；Thresholded predicted labels.
    @param labels 真值标签；Ground-truth labels.
    @param patient_ids 患者 ID 序列；Patient ID sequence.
    @param paths 样本路径序列；Sample path sequence.
    """

    logits: Tensor
    probabilities: Tensor
    predictions: Tensor
    labels: Tensor
    patient_ids: tuple[str, ...]
    paths: tuple[Path, ...]


class Evaluator:
    """@brief 评估与推理器；Evaluator for validation and inference."""

    def __init__(
        self,
        model: nn.Module,
        config: EvaluatorConfig = EvaluatorConfig(),
        *,
        criterion: nn.Module | None = None,
    ) -> None:
        """@brief 初始化评估器；Initialize evaluator.

        @param model 待评估模型；Model to evaluate.
        @param config 评估器配置；Evaluator configuration.
        @param criterion 可选损失函数；Optional loss function.
        """
        _validate_config(config)
        self.model = model
        self.config = config
        self.criterion = criterion

        self.device = _resolve_device(config.device)
        self.amp_enabled = config.use_amp and self.device.type == "cuda"
        self.model.to(self.device)

        LOGGER.info(
            "Evaluator initialized: device=%s, threshold=%.3f, use_amp=%s, has_criterion=%s",
            self.device,
            config.threshold,
            config.use_amp,
            criterion is not None,
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch]) -> EvaluationResult:
        """@brief 运行完整评估；Run full evaluation on a loader.

        @param loader 评估数据加载器；Evaluation data loader.
        @return 评估结果对象；Evaluation result.
        """
        self.model.eval()

        total_loss = 0.0
        has_loss = self.criterion is not None
        step_count = 0
        sample_count = 0
        logits_all: list[Tensor] = []
        labels_all: list[Tensor] = []

        for batch in loader:
            prepared = batch.to(self.device)
            targets = prepared.labels.float().view(-1, 1)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(prepared.images)
                if self.criterion is not None:
                    loss = self.criterion(logits, targets)
                else:
                    loss = None

            logits_all.append(logits.detach().reshape(-1).cpu())
            labels_all.append(prepared.labels.detach().reshape(-1).to(dtype=torch.long).cpu())

            batch_size = int(prepared.labels.numel())
            sample_count += batch_size
            step_count += 1
            if loss is not None:
                total_loss += float(loss.detach().item()) * batch_size

        if step_count == 0 or sample_count == 0:
            raise RuntimeError("Loader produced zero batches; cannot evaluate.")

        merged_logits = torch.cat(logits_all, dim=0)
        merged_labels = torch.cat(labels_all, dim=0)
        metrics = compute_binary_metrics(
            targets=merged_labels,
            scores=merged_logits,
            threshold=self.config.threshold,
            from_logits=True,
        )

        average_loss = (total_loss / float(sample_count)) if has_loss else None
        return EvaluationResult(
            metrics=metrics,
            samples=sample_count,
            steps=step_count,
            loss=average_loss,
        )

    @torch.no_grad()
    def predict(self, loader: Iterable[Batch]) -> PredictionResult:
        """@brief 运行批量推理；Run batch inference on a loader.

        @param loader 推理数据加载器；Inference data loader.
        @return 预测结果对象；Prediction result.
        """
        self.model.eval()

        logits_all: list[Tensor] = []
        labels_all: list[Tensor] = []
        patient_ids: list[str] = []
        paths: list[Path] = []

        for batch in loader:
            prepared = batch.to(self.device)
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(prepared.images)

            logits_all.append(logits.detach().reshape(-1).cpu())
            labels_all.append(prepared.labels.detach().reshape(-1).to(dtype=torch.long).cpu())
            patient_ids.extend(prepared.patient_ids)
            paths.extend(prepared.paths)

        if not logits_all:
            raise RuntimeError("Loader produced zero batches; cannot predict.")

        merged_logits = torch.cat(logits_all, dim=0)
        merged_labels = torch.cat(labels_all, dim=0)
        probabilities = torch.sigmoid(merged_logits)
        predictions = (probabilities >= self.config.threshold).to(dtype=torch.long)

        return PredictionResult(
            logits=merged_logits,
            probabilities=probabilities,
            predictions=predictions,
            labels=merged_labels,
            patient_ids=tuple(patient_ids),
            paths=tuple(paths),
        )


def _resolve_device(device: str | None) -> torch.device:
    """@brief 解析设备；Resolve execution device.

    @param device 设备字符串；Device string.
    @return torch 设备对象；Resolved torch.device.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_config(config: EvaluatorConfig) -> None:
    """@brief 校验评估配置；Validate evaluator config.

    @param config 评估器配置；Evaluator configuration.
    """
    if not (0.0 <= config.threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {config.threshold}.")


__all__ = [
    "EvaluationResult",
    "Evaluator",
    "EvaluatorConfig",
    "PredictionResult",
]
