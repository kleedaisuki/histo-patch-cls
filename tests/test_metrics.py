from __future__ import annotations

import torch

from histoclass.utils.metrics import compute_binary_metrics


def test_compute_binary_metrics_handles_all_tied_scores() -> None:
    metrics = compute_binary_metrics(
        targets=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        scores=torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32),
    )

    assert metrics.roc_auc is not None
    assert metrics.roc_auc == 0.5


def test_compute_binary_metrics_handles_tied_score_groups() -> None:
    metrics = compute_binary_metrics(
        targets=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        scores=torch.tensor([0.2, 0.2, 0.8, 0.8], dtype=torch.float32),
    )

    assert metrics.roc_auc is not None
    assert metrics.roc_auc == 0.5


def test_compute_binary_metrics_large_tied_input_regression() -> None:
    sample_count = 200_000
    targets = torch.cat(
        [
            torch.zeros(sample_count // 2, dtype=torch.long),
            torch.ones(sample_count - sample_count // 2, dtype=torch.long),
        ]
    )
    scores = torch.full((sample_count,), 0.5, dtype=torch.float32)

    metrics = compute_binary_metrics(targets=targets, scores=scores)

    assert metrics.support == sample_count
    assert metrics.roc_auc is not None
    assert metrics.roc_auc == 0.5
