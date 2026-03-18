import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from histoclass.data import Batch
from histoclass.engine import Evaluator, EvaluatorConfig


class _DummyBatchDataset(Dataset[Batch]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> Batch:
        _ = index
        images = torch.randn(3, 3, 8, 8)
        labels = torch.randint(low=0, high=2, size=(3,), dtype=torch.long)
        patient_ids = tuple(f"p{i}" for i in range(3))
        paths = tuple()
        return Batch(images=images, labels=labels, patient_ids=patient_ids, paths=paths)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _identity_collate(batch: list[Batch]) -> Batch:
    return batch[0]


def test_evaluator_evaluate_and_predict_smoke() -> None:
    loader = DataLoader(
        _DummyBatchDataset(),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
    )

    model = _TinyModel()
    criterion = nn.BCEWithLogitsLoss()
    evaluator = Evaluator(
        model=model,
        config=EvaluatorConfig(threshold=0.5, device="cpu", use_amp=False),
        criterion=criterion,
    )

    eval_result = evaluator.evaluate(loader)
    pred_result = evaluator.predict(loader)

    assert eval_result.steps == 4
    assert eval_result.samples == 12
    assert eval_result.loss is not None
    assert 0.0 <= eval_result.metrics.accuracy <= 1.0

    assert pred_result.logits.shape == torch.Size([12])
    assert pred_result.probabilities.shape == torch.Size([12])
    assert pred_result.predictions.shape == torch.Size([12])
    assert pred_result.labels.shape == torch.Size([12])
    assert len(pred_result.patient_ids) == 12
