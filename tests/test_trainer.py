import torch
from torch.utils.data import DataLoader, Dataset

from histoclass.data import Batch
from histoclass.engine import Trainer, TrainerConfig
from histoclass.model import IDCResNetClassifier, ModelConfig


class _DummyBatchDataset(Dataset[Batch]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> Batch:
        _ = index
        images = torch.randn(4, 3, 50, 50)
        labels = torch.randint(low=0, high=2, size=(4,), dtype=torch.long)
        patient_ids = tuple(f"p{i}" for i in range(4))
        paths = tuple()
        return Batch(images=images, labels=labels, patient_ids=patient_ids, paths=paths)


def _identity_collate(batch: list[Batch]) -> Batch:
    return batch[0]


def test_trainer_fit_smoke(tmp_path) -> None:
    train_loader = DataLoader(
        _DummyBatchDataset(),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
    )

    model = IDCResNetClassifier(ModelConfig(pretrained=False, hidden_dim=64, dropout=0.1))
    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            device="cpu",
            use_amp=False,
            checkpoint_dir=tmp_path,
            save_checkpoint_each_epoch=True,
            log_every_n_steps=1000,
        ),
    )

    summary = trainer.fit(train_loader=train_loader)

    assert len(summary.history) == 1
    assert summary.final_checkpoint is not None
    assert summary.final_checkpoint.exists()
    assert summary.history[0].train.samples > 0
