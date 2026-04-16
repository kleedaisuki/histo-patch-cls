from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from histoclass.config import AppConfig, SeedConfig
from histoclass.data import DataModuleConfig
from histoclass.engine import EvaluationResult, EvaluatorConfig, TrainerConfig
from histoclass.model import ModelConfig
from histoclass.utils import SeedState, compute_binary_metrics
from histoclass_cli.pipeline import (
    PipelineMode,
    PipelineRequest,
    _run_broadcast_stage,
    run_batch_pipeline,
    run_pipeline,
)


@dataclass(slots=True)
class _DummyDataModule:
    train_loader: object
    val_loader: object


class _MetricsStub:
    def __init__(self) -> None:
        self.f1 = 1.0
        self.roc_auc = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"f1": self.f1, "roc_auc": self.roc_auc}


def _build_test_config() -> AppConfig:
    return AppConfig(
        data=DataModuleConfig(image_root=Path("data/raw")),
        model=ModelConfig(pretrained=False),
        trainer=TrainerConfig(device="cpu", use_amp=False),
        evaluator=EvaluatorConfig(device="cpu", use_amp=False),
        seed=SeedConfig(seed=7, deterministic=True, benchmark=False),
    )


def test_pipeline_train_mode_wires_trainer_interface(monkeypatch, tmp_path) -> None:
    config = _build_test_config()
    model_sentinel = object()
    train_loader = object()
    val_loader = object()
    data_module = _DummyDataModule(train_loader=train_loader, val_loader=val_loader)

    monkeypatch.setattr("histoclass_cli.pipeline.load_config", lambda _: config)
    monkeypatch.setattr("histoclass_cli.pipeline.seed_everything", lambda **_: SeedState(7, True, False, False))
    monkeypatch.setattr("histoclass_cli.pipeline.build_data_module", lambda _: data_module)
    monkeypatch.setattr("histoclass_cli.pipeline.build_model", lambda _: model_sentinel)

    captured: dict[str, object] = {}

    class _TrainerStub:
        def __init__(self, *, model, config):  # noqa: ANN001
            captured["trainer_model"] = model
            captured["trainer_config"] = config

        def fit(self, loader):  # noqa: ANN001
            captured["fit_loader"] = loader
            checkpoint = tmp_path / "epoch001.pt"
            checkpoint.write_text("ok", encoding="utf-8")
            epoch_result = type(
                "EpochResultLike",
                (),
                {
                    "epoch": 1,
                    "train": type(
                        "TrainLike", (), {"loss": 0.1, "metrics": _MetricsStub()}
                    )(),
                },
            )()
            return type(
                "TrainSummaryLike",
                (),
                {"history": (epoch_result,), "final_checkpoint": checkpoint},
            )()

    monkeypatch.setattr("histoclass_cli.pipeline.Trainer", _TrainerStub)

    result = run_pipeline(PipelineRequest(mode=PipelineMode.TRAIN))

    assert captured["trainer_model"] is model_sentinel
    assert captured["trainer_config"] == config.trainer
    assert captured["fit_loader"] is train_loader
    assert result.train_summary is not None
    assert result.evaluation is None
    assert result.checkpoint_path == result.train_summary.final_checkpoint


def test_pipeline_eval_mode_wires_evaluator_interface(monkeypatch, tmp_path) -> None:
    config = _build_test_config()
    model_sentinel = object()
    train_loader = object()
    val_loader = object()
    data_module = _DummyDataModule(train_loader=train_loader, val_loader=val_loader)
    ckpt_path = tmp_path / "model.pt"
    ckpt_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("histoclass_cli.pipeline.load_config", lambda _: config)
    monkeypatch.setattr("histoclass_cli.pipeline.seed_everything", lambda **_: SeedState(7, True, False, False))
    monkeypatch.setattr("histoclass_cli.pipeline.build_data_module", lambda _: data_module)
    monkeypatch.setattr("histoclass_cli.pipeline.build_model", lambda _: model_sentinel)

    captured: dict[str, object] = {}

    def _load_ckpt_stub(*, model, checkpoint_path):  # noqa: ANN001
        captured["loaded_model"] = model
        captured["loaded_checkpoint_path"] = checkpoint_path

    class _EvaluatorStub:
        def __init__(self, *, model, config):  # noqa: ANN001
            captured["evaluator_model"] = model
            captured["evaluator_config"] = config

        def evaluate(self, loader):  # noqa: ANN001
            captured["eval_loader"] = loader
            return EvaluationResult(
                metrics=compute_binary_metrics(
                    targets=[0, 1, 1, 0],
                    scores=[0.2, 0.9, 0.8, 0.1],
                ),
                samples=10,
                steps=2,
                loss=0.2,
            )

    monkeypatch.setattr("histoclass_cli.pipeline._load_model_checkpoint", _load_ckpt_stub)
    monkeypatch.setattr("histoclass_cli.pipeline.Evaluator", _EvaluatorStub)

    result = run_pipeline(
        PipelineRequest(mode=PipelineMode.EVAL, checkpoint_path=ckpt_path)
    )

    assert captured["loaded_model"] is model_sentinel
    assert captured["loaded_checkpoint_path"] == ckpt_path.resolve()
    assert captured["evaluator_model"] is model_sentinel
    assert captured["evaluator_config"] == config.evaluator
    assert captured["eval_loader"] is val_loader
    assert result.train_summary is None
    assert result.evaluation is not None
    assert result.checkpoint_path == ckpt_path.resolve()


def test_run_batch_pipeline_reuses_data_module_for_same_data_config(
    monkeypatch, tmp_path
) -> None:
    config = _build_test_config()
    model_sentinel = object()
    train_loader = [object(), object()]
    val_loader = [object()]
    data_module = _DummyDataModule(train_loader=train_loader, val_loader=val_loader)

    monkeypatch.setattr("histoclass_cli.pipeline.load_config", lambda _: config)
    monkeypatch.setattr(
        "histoclass_cli.pipeline.seed_everything",
        lambda **_: SeedState(7, True, False, False),
    )
    monkeypatch.setattr("histoclass_cli.pipeline.build_model", lambda _: model_sentinel)

    counters: dict[str, int] = {"build_data_module": 0, "fit": 0}

    def _build_data_module_stub(_):  # noqa: ANN001
        counters["build_data_module"] += 1
        return data_module

    class _TrainerStub:
        def __init__(self, *, model, config):  # noqa: ANN001
            assert model is model_sentinel
            assert config == config_obj.trainer

        def fit(self, loader):  # noqa: ANN001
            counters["fit"] += 1
            assert hasattr(loader, "__iter__")
            checkpoint = tmp_path / f"epoch{counters['fit']:03d}.pt"
            checkpoint.write_text("ok", encoding="utf-8")
            epoch_result = type(
                "EpochResultLike",
                (),
                {
                    "epoch": 1,
                    "train": type(
                        "TrainLike", (), {"loss": 0.1, "metrics": _MetricsStub()}
                    )(),
                },
            )()
            return type(
                "TrainSummaryLike",
                (),
                {"history": (epoch_result,), "final_checkpoint": checkpoint},
            )()

    config_obj = config
    monkeypatch.setattr("histoclass_cli.pipeline.build_data_module", _build_data_module_stub)
    monkeypatch.setattr("histoclass_cli.pipeline.Trainer", _TrainerStub)

    batch_result = run_batch_pipeline(
        (
            PipelineRequest(config_path="configs/train.json", mode=PipelineMode.TRAIN),
            PipelineRequest(config_path="configs/default.json", mode=PipelineMode.TRAIN),
        )
    )

    assert len(batch_result.results) == 2
    assert counters["build_data_module"] == 1
    assert counters["fit"] == 2


def test_run_broadcast_stage_raises_without_deadlock_on_consumer_error() -> None:
    source_loader = [1, 2, 3, 4, 5]

    def _bad_consumer(_):  # noqa: ANN001
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="Broadcast stage 'train' failed"):
        _run_broadcast_stage(
            stage_name="train",
            source_loader=source_loader,
            jobs=(("bad", _bad_consumer),),
        )
