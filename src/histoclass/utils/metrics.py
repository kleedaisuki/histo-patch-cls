"""Metric utilities for binary classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .logger import get_logger


LOGGER = get_logger(__name__)
EPSILON: float = 1e-12


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """@brief 二分类指标集合；Aggregated metric values for binary classification.

    @param threshold 分类阈值；Decision threshold for positive class.
    @param support 样本总数；Total number of samples.
    @param positives 正样本数量；Number of positive targets.
    @param negatives 负样本数量；Number of negative targets.
    @param true_positive 真阳性数量；Count of true positives.
    @param true_negative 真阴性数量；Count of true negatives.
    @param false_positive 假阳性数量；Count of false positives.
    @param false_negative 假阴性数量；Count of false negatives.
    @param accuracy 准确率；Accuracy score.
    @param precision 精确率；Precision score.
    @param recall 召回率；Recall score.
    @param specificity 特异度；Specificity score.
    @param f1 F1 分数；F1 score.
    @param balanced_accuracy 平衡准确率；Balanced accuracy score.
    @param roc_auc ROC 曲线下面积；Area under ROC curve.
    """

    threshold: float
    support: int
    positives: int
    negatives: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """@brief 指标对象转字典；Convert metrics dataclass to dictionary.

        @return 适合日志与序列化的键值结构；A mapping suitable for logging/serialization.
        """
        return {
            "threshold": self.threshold,
            "support": self.support,
            "positives": self.positives,
            "negatives": self.negatives,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
        }


def compute_binary_metrics(
    targets: Tensor | Sequence[int],
    scores: Tensor | Sequence[float],
    *,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> BinaryMetrics:
    """@brief 计算二分类指标；Compute binary classification metrics.

    @param targets 二分类标签（0/1）；Binary labels (0/1).
    @param scores 预测分数或概率；Prediction scores or probabilities.
    @param threshold 判定正类的阈值；Decision threshold for positive class.
    @param from_logits 若为 True 则先做 sigmoid；If True, applies sigmoid first.
    @return 完整二分类指标对象；Complete binary metric bundle.
    @note 当标签全为同一类时，ROC-AUC 返回 None。
    """
    target_tensor = _as_binary_targets(targets)
    score_tensor = _as_probabilities(scores, from_logits=from_logits)

    if target_tensor.numel() != score_tensor.numel():
        raise ValueError(
            "targets and scores must have same length. "
            f"Got {target_tensor.numel()} vs {score_tensor.numel()}."
        )

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

    target_long = target_tensor.to(dtype=torch.long)
    prediction_tensor = (score_tensor >= threshold).to(dtype=torch.long)

    true_positive_tensor = ((prediction_tensor == 1) & (target_long == 1)).sum()
    true_negative_tensor = ((prediction_tensor == 0) & (target_long == 0)).sum()
    false_positive_tensor = ((prediction_tensor == 1) & (target_long == 0)).sum()
    false_negative_tensor = ((prediction_tensor == 0) & (target_long == 1)).sum()

    positives_tensor = (target_long == 1).sum()
    negatives_tensor = (target_long == 0).sum()
    support_tensor = torch.as_tensor(
        target_long.numel(),
        device=target_long.device,
        dtype=torch.int64,
    )

    tp = true_positive_tensor.to(dtype=torch.float64)
    tn = true_negative_tensor.to(dtype=torch.float64)
    fp = false_positive_tensor.to(dtype=torch.float64)
    fn = false_negative_tensor.to(dtype=torch.float64)
    support_float = support_tensor.to(dtype=torch.float64)

    accuracy_tensor = _safe_divide_tensor(tp + tn, support_float)
    precision_tensor = _safe_divide_tensor(tp, tp + fp)
    recall_tensor = _safe_divide_tensor(tp, tp + fn)
    specificity_tensor = _safe_divide_tensor(tn, tn + fp)
    f1_tensor = _safe_divide_tensor(
        2.0 * precision_tensor * recall_tensor, precision_tensor + recall_tensor
    )
    balanced_accuracy_tensor = (recall_tensor + specificity_tensor) / 2.0
    roc_auc = _binary_roc_auc(target_long, score_tensor)

    true_positive = int(true_positive_tensor.item())
    true_negative = int(true_negative_tensor.item())
    false_positive = int(false_positive_tensor.item())
    false_negative = int(false_negative_tensor.item())
    positives = int(positives_tensor.item())
    negatives = int(negatives_tensor.item())
    support = int(support_tensor.item())
    accuracy = float(accuracy_tensor.item())
    precision = float(precision_tensor.item())
    recall = float(recall_tensor.item())
    specificity = float(specificity_tensor.item())
    f1 = float(f1_tensor.item())
    balanced_accuracy = float(balanced_accuracy_tensor.item())

    if roc_auc is None:
        LOGGER.warning(
            "ROC-AUC is undefined because targets contain only one class. "
            "positives=%d, negatives=%d",
            positives,
            negatives,
        )

    return BinaryMetrics(
        threshold=threshold,
        support=support,
        positives=positives,
        negatives=negatives,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        balanced_accuracy=balanced_accuracy,
        roc_auc=roc_auc,
    )


def _as_binary_targets(targets: Tensor | Sequence[int]) -> Tensor:
    """@brief 标签转一维 0/1 张量；Convert labels to a 1D binary tensor.

    @param targets 标签序列；Target label sequence.
    @return 一维长整型标签张量；1D int64 tensor.
    """
    target_tensor = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    invalid_mask = (target_tensor != 0) & (target_tensor != 1)
    if bool(invalid_mask.any()):
        raise ValueError("targets must be binary labels in {0, 1}.")
    return target_tensor


def _as_probabilities(scores: Tensor | Sequence[float], *, from_logits: bool) -> Tensor:
    """@brief 分数转概率张量；Convert score inputs into probability tensor.

    @param scores 分数或概率序列；Input scores/probabilities.
    @param from_logits 是否从 logits 映射；Whether to map from logits.
    @return 一维 float 概率张量；1D float probability tensor.
    """
    score_tensor = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if from_logits:
        return torch.sigmoid(score_tensor)

    lower_ok = bool((score_tensor >= 0.0).all())
    upper_ok = bool((score_tensor <= 1.0).all())
    if not (lower_ok and upper_ok):
        raise ValueError(
            "scores must be probabilities in [0, 1] when from_logits=False."
        )
    return score_tensor


def _binary_roc_auc(targets: Tensor, probabilities: Tensor) -> float | None:
    """@brief 计算 ROC-AUC；Compute ROC-AUC via rank statistics.

    @param targets 二分类标签；Binary labels tensor.
    @param probabilities 正类概率；Positive class probabilities.
    @return ROC-AUC 值；ROC-AUC value.
    @note 使用平均秩（average rank）处理分数并列。
    """
    total_positives = int((targets == 1).sum().item())
    total_negatives = int((targets == 0).sum().item())
    if total_positives == 0 or total_negatives == 0:
        return None

    sorted_indices = torch.argsort(probabilities, stable=True)
    sorted_scores = probabilities[sorted_indices]
    sorted_targets = targets[sorted_indices]

    counts = torch.unique_consecutive(sorted_scores, return_counts=True)[1]
    end_positions = torch.cumsum(counts, dim=0)
    start_positions = end_positions - counts
    average_ranks = (
        start_positions.to(dtype=torch.float64)
        + end_positions.to(dtype=torch.float64)
        - 1.0
    ) / 2.0 + 1.0
    ranks = torch.repeat_interleave(average_ranks, counts)

    positive_rank_sum = float(ranks[sorted_targets == 1].sum().item())
    u_statistic = positive_rank_sum - (total_positives * (total_positives + 1) / 2.0)
    auc = u_statistic / (total_positives * total_negatives)
    return float(max(0.0, min(1.0, auc)))


def _safe_divide_tensor(numerator: Tensor, denominator: Tensor) -> Tensor:
    """@brief 张量安全除法；Tensor divide with zero-denominator guard.

    @param numerator 分子张量；Numerator tensor.
    @param denominator 分母张量；Denominator tensor.
    @return 商值张量，分母为 0 时返回 0；Quotient tensor or 0 when denominator is zero.
    """
    zero = torch.zeros((), dtype=numerator.dtype, device=numerator.device)
    return torch.where(denominator == 0, zero, numerator / denominator)


__all__ = ["BinaryMetrics", "compute_binary_metrics"]
