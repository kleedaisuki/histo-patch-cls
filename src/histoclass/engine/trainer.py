"""Training engine for IDC patch classification."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..data import Batch
from ..utils import BinaryMetrics, compute_binary_metrics, get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """@brief 训练器配置；Trainer configuration.

    @param epochs 训练轮数；Number of training epochs.
    @param learning_rate 学习率；Learning rate.
    @param weight_decay 权重衰减；Weight decay factor.
    @param threshold 二分类阈值；Decision threshold for positive class.
    @param device 训练设备，None 表示自动选择；Training device, None for auto-select.
    @param use_amp 是否启用自动混合精度；Enable automatic mixed precision.
    @param grad_clip_norm 梯度裁剪阈值，None 表示不裁剪；Gradient clip norm, None to disable.
    @param pos_weight 正类权重，None 表示不加权；Positive class weight, None for no weighting.
    @param log_every_n_steps 训练日志间隔步数；Log interval in steps.
    @param checkpoint_dir checkpoint 输出目录；Checkpoint output directory.
    @param save_checkpoint_each_epoch 是否每个 epoch 保存 checkpoint；Save checkpoint at each epoch.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.5
    device: str | None = None
    use_amp: bool = False
    grad_clip_norm: float | None = None
    pos_weight: float | None = None
    log_every_n_steps: int = 50
    checkpoint_dir: Path = Path("outputs/checkpoints")
    save_checkpoint_each_epoch: bool = True


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """@brief 单阶段结果；Result summary of one phase.

    @param loss 平均损失；Average loss.
    @param metrics 二分类指标；Binary metrics.
    @param steps 实际迭代步数；Number of optimization steps.
    @param samples 实际样本数；Number of processed samples.
    """

    loss: float
    metrics: BinaryMetrics
    steps: int
    samples: int


@dataclass(frozen=True, slots=True)
class EpochResult:
    """@brief 单轮训练结果；One-epoch training result.

    @param epoch 当前轮次（从 1 开始）；Current epoch index (1-based).
    @param train 训练阶段结果；Train phase result.
    @param learning_rate 当前学习率；Current learning rate.
    @param checkpoint_path 本轮保存的 checkpoint 路径；Checkpoint path saved for this epoch.
    """

    epoch: int
    train: PhaseResult
    learning_rate: float
    checkpoint_path: Path | None


@dataclass(frozen=True, slots=True)
class TrainSummary:
    """@brief 训练总结果；Overall training summary.

    @param history 每轮结果历史；Per-epoch result history.
    @param final_checkpoint 最后一轮 checkpoint 路径；Last epoch checkpoint path.
    """

    history: tuple[EpochResult, ...]
    final_checkpoint: Path | None


class Trainer:
    """@brief 训练器；Pure trainer for binary IDC classification."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig = TrainerConfig(),
        *,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        """@brief 初始化训练器；Initialize trainer.

        @param model 待训练模型；Model to train.
        @param config 训练配置；Trainer configuration.
        @param optimizer 可选优化器；Optional optimizer.
        @param scheduler 可选学习率调度器；Optional LR scheduler.
        @param criterion 可选损失函数；Optional loss function.
        """
        _validate_config(config)
        self.model = model
        self.config = config

        self.device = _resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler
        self.criterion = criterion or _build_default_criterion(config, self.device)
        self.amp_enabled = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        LOGGER.info(
            "Trainer initialized: device=%s, epochs=%d, lr=%.6f, weight_decay=%.6f, use_amp=%s",
            self.device,
            config.epochs,
            config.learning_rate,
            config.weight_decay,
            config.use_amp,
        )

    def fit(self, train_loader: Iterable[Batch]) -> TrainSummary:
        """@brief 执行完整训练流程；Run full training loop.

        @param train_loader 训练集加载器；Training data loader.
        @return 训练结果汇总；Training summary.
        """
        history: list[EpochResult] = []
        final_checkpoint: Path | None = None

        for epoch in range(1, self.config.epochs + 1):
            train_result = self.train_one_epoch(train_loader, epoch)
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            checkpoint_path: Path | None = None
            if self.config.save_checkpoint_each_epoch:
                checkpoint_path = self.save_checkpoint(epoch, train_result)
                final_checkpoint = checkpoint_path

            history.append(
                EpochResult(
                    epoch=epoch,
                    train=train_result,
                    learning_rate=current_lr,
                    checkpoint_path=checkpoint_path,
                )
            )

            if self.scheduler is not None:
                self.scheduler.step()

            LOGGER.info(
                "Epoch %d/%d | train_loss=%.6f train_f1=%.4f train_auc=%s",
                epoch,
                self.config.epochs,
                train_result.loss,
                train_result.metrics.f1,
                (
                    f"{train_result.metrics.roc_auc:.4f}"
                    if train_result.metrics.roc_auc is not None
                    else "None"
                ),
            )

        return TrainSummary(history=tuple(history), final_checkpoint=final_checkpoint)

    def train_one_epoch(self, train_loader: Iterable[Batch], epoch: int) -> PhaseResult:
        """@brief 训练一个 epoch；Train for one epoch.

        @param train_loader 训练集加载器；Training data loader.
        @param epoch 当前轮次；Current epoch index.
        @return 训练阶段统计结果；Train-phase result.
        """
        del epoch
        self.model.train()

        total_loss = 0.0
        step_count = 0
        sample_count = 0
        all_logits: list[Tensor] = []
        all_labels: list[Tensor] = []

        for step_idx, batch in enumerate(train_loader, start=1):
            prepared = batch.to(self.device)
            targets = prepared.labels.float().view(-1, 1)

            self.optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(prepared.images)
                loss = self.criterion(logits, targets)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.optimizer.step()

            detached_logits = logits.detach().reshape(-1).cpu()
            detached_labels = targets.detach().reshape(-1).to(dtype=torch.long).cpu()
            all_logits.append(detached_logits)
            all_labels.append(detached_labels)

            batch_size = int(prepared.labels.numel())
            sample_count += batch_size
            step_count += 1
            total_loss += float(loss.detach().item()) * batch_size

            if step_idx % self.config.log_every_n_steps == 0:
                LOGGER.info(
                    "Train step %d | batch_size=%d loss=%.6f",
                    step_idx,
                    batch_size,
                    float(loss.detach().item()),
                )

        return _finalize_phase_result(
            total_loss=total_loss,
            step_count=step_count,
            sample_count=sample_count,
            threshold=self.config.threshold,
            labels=all_labels,
            logits=all_logits,
        )

    def save_checkpoint(self, epoch: int, train_result: PhaseResult) -> Path:
        """@brief 保存 checkpoint；Save checkpoint to disk.

        @param epoch 当前轮次；Current epoch.
        @param train_result 训练阶段结果；Train-phase result.
        @return checkpoint 文件路径；Saved checkpoint path.
        """
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"epoch{epoch:03d}.pt"

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "trainer_config": asdict(self.config),
            "train_loss": train_result.loss,
            "train_metrics": train_result.metrics.to_dict(),
        }
        torch.save(payload, checkpoint_path)
        LOGGER.info("Checkpoint saved: %s", checkpoint_path)
        return checkpoint_path


def _finalize_phase_result(
    *,
    total_loss: float,
    step_count: int,
    sample_count: int,
    threshold: float,
    labels: list[Tensor],
    logits: list[Tensor],
) -> PhaseResult:
    """@brief 聚合单阶段统计；Aggregate one phase result.

    @param total_loss 总损失（按样本累计）；Total sample-weighted loss.
    @param step_count 步数；Number of steps.
    @param sample_count 样本数；Number of samples.
    @param threshold 指标阈值；Metric threshold.
    @param labels 标签集合；Collected label tensors.
    @param logits logits 集合；Collected logit tensors.
    @return 单阶段结果；Phase result.
    """
    if step_count == 0 or sample_count == 0:
        raise RuntimeError("Loader produced zero batches; cannot finalize phase result.")

    merged_labels = torch.cat(labels, dim=0)
    merged_logits = torch.cat(logits, dim=0)
    metrics = compute_binary_metrics(
        targets=merged_labels,
        scores=merged_logits,
        threshold=threshold,
        from_logits=True,
    )
    average_loss = total_loss / float(sample_count)
    return PhaseResult(
        loss=average_loss,
        metrics=metrics,
        steps=step_count,
        samples=sample_count,
    )


def _resolve_device(device: str | None) -> torch.device:
    """@brief 解析训练设备；Resolve training device.

    @param device 设备字符串；Device string.
    @return 解析后的 torch.device；Resolved torch.device.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_default_criterion(config: TrainerConfig, device: torch.device) -> nn.Module:
    """@brief 构建默认损失函数；Build default loss function.

    @param config 训练配置；Trainer config.
    @param device 训练设备；Training device.
    @return BCEWithLogitsLoss 实例；BCEWithLogitsLoss instance.
    """
    if config.pos_weight is None:
        return nn.BCEWithLogitsLoss()

    weight_tensor = torch.as_tensor(
        [config.pos_weight], dtype=torch.float32, device=device
    )
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


def _validate_config(config: TrainerConfig) -> None:
    """@brief 校验训练配置；Validate trainer configuration.

    @param config 训练配置；Trainer configuration.
    """
    if config.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {config.epochs}.")
    if config.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}.")
    if config.weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay}.")
    if not (0.0 <= config.threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {config.threshold}.")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0.0:
        raise ValueError(
            f"grad_clip_norm must be > 0 when set, got {config.grad_clip_norm}."
        )
    if config.pos_weight is not None and config.pos_weight <= 0.0:
        raise ValueError(f"pos_weight must be > 0 when set, got {config.pos_weight}.")
    if config.log_every_n_steps <= 0:
        raise ValueError(
            f"log_every_n_steps must be > 0, got {config.log_every_n_steps}."
        )


__all__ = [
    "EpochResult",
    "PhaseResult",
    "TrainSummary",
    "Trainer",
    "TrainerConfig",
]
