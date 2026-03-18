"""Data pipeline for IDC patch classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms as T


IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TransformFactory = Callable[["ImageSchema"], T.Compose]


@dataclass(frozen=True, slots=True)
class ImageSchema:
    """@brief 图像预处理配置；Image preprocessing schema."""

    image_size: tuple[int, int] = (50, 50)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.25, 0.25, 0.25)


@dataclass(frozen=True, slots=True)
class SplitSchema:
    """@brief 按患者切分配置；Patient-wise split schema."""

    val_ratio: float = 0.2
    seed: int = 42


@dataclass(frozen=True, slots=True)
class LoaderSchema:
    """@brief DataLoader 配置；DataLoader schema."""

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    train_drop_last: bool = True
    eval_drop_last: bool = False
    use_weighted_sampler: bool = False


@dataclass(frozen=True, slots=True)
class DataModuleConfig:
    """@brief 数据模块顶层配置；Top-level data module config."""

    image_root: Path
    train_transform: str = "train_basic"
    eval_transform: str = "eval_basic"
    image: ImageSchema = ImageSchema()
    split: SplitSchema = SplitSchema()
    loader: LoaderSchema = LoaderSchema()


@dataclass(frozen=True, slots=True)
class PatchRecord:
    """@brief 单个样本记录；One discovered patch record."""

    path: Path
    label: int
    patient_id: str


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """@brief 训练/验证样本切分；Train/validation split."""

    train: tuple[PatchRecord, ...]
    val: tuple[PatchRecord, ...]


@dataclass(frozen=True, slots=True)
class PatchExample:
    """@brief 数据集单样本输出；Dataset item schema."""

    image: Tensor
    label: int
    patient_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class Batch:
    """@brief 训练批次结构；Unified training batch schema."""

    images: Tensor
    labels: Tensor
    patient_ids: tuple[str, ...]
    paths: tuple[Path, ...]

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "Batch":
        """@brief 将批次迁移到目标设备；Move tensors in batch to target device."""
        return Batch(
            images=self.images.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            patient_ids=self.patient_ids,
            paths=self.paths,
        )


@dataclass(frozen=True, slots=True)
class DataModule:
    """@brief 数据模块产物；Materialized data module."""

    split: DatasetSplit
    train_dataset: "IDCPatchDataset"
    val_dataset: "IDCPatchDataset"
    train_loader: DataLoader[Batch]
    val_loader: DataLoader[Batch]


class IDCPatchDataset(Dataset[PatchExample]):
    """@brief IDC patch 数据集；IDC patch dataset."""

    def __init__(
        self,
        records: Sequence[PatchRecord],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self._records: tuple[PatchRecord, ...] = tuple(records)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PatchExample:
        record = self._records[index]
        with Image.open(record.path) as image:
            rgb_image = image.convert("RGB")
            tensor = self._transform(rgb_image)
        return PatchExample(
            image=tensor,
            label=record.label,
            patient_id=record.patient_id,
            path=record.path,
        )


class TransformRegistry:
    """@brief 变换工厂注册表；Transform factory registry."""

    def __init__(self) -> None:
        self._registry: dict[str, TransformFactory] = {
            "train_basic": _build_train_basic,
            "eval_basic": _build_eval_basic,
        }

    def register(self, name: str, factory: TransformFactory) -> None:
        """@brief 注册新变换；Register a new transform factory."""
        self._registry[name] = factory

    def build(self, name: str, schema: ImageSchema) -> T.Compose:
        """@brief 根据名字构建变换；Build transform by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Unknown transform '{name}'. Available: [{available}]")
        return self._registry[name](schema)


def discover_records(image_root: Path) -> tuple[PatchRecord, ...]:
    """@brief 扫描数据目录生成样本记录；Discover records from dataset tree.

    @param image_root 数据根目录，期望结构为 `<patient>/<label>/<image>`.
    @return 按路径稳定排序的样本记录元组。
    """
    root = image_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image_root not found: {root}")

    records: list[PatchRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 3:
            continue

        patient_id = parts[0]
        label_token = next((p for p in parts if p in {"0", "1"}), None)
        if label_token is None:
            continue

        records.append(
            PatchRecord(path=path, label=int(label_token), patient_id=patient_id)
        )

    if not records:
        raise RuntimeError(f"No images discovered under: {root}")

    return tuple(records)


def split_by_patient(
    records: Sequence[PatchRecord], schema: SplitSchema
) -> DatasetSplit:
    """@brief 按患者切分，避免信息泄漏；Patient-wise split to avoid leakage."""
    patients = sorted({record.patient_id for record in records})
    if len(patients) < 2:
        raise ValueError("At least 2 patients are required for train/val split.")

    if not (0.0 < schema.val_ratio < 1.0):
        raise ValueError("split.val_ratio must be in (0, 1).")

    rng = Random(schema.seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    val_count = max(
        1, min(len(shuffled) - 1, int(round(len(shuffled) * schema.val_ratio)))
    )
    val_patients = set(shuffled[:val_count])

    train_records = tuple(
        record for record in records if record.patient_id not in val_patients
    )
    val_records = tuple(
        record for record in records if record.patient_id in val_patients
    )

    if not train_records or not val_records:
        raise RuntimeError("Invalid split produced an empty train or val set.")

    return DatasetSplit(train=train_records, val=val_records)


def collate_patch_examples(examples: Sequence[PatchExample]) -> Batch:
    """@brief 自定义拼接函数；Custom collate function producing Batch."""
    images = torch.stack([example.image for example in examples], dim=0)
    labels = torch.as_tensor([example.label for example in examples], dtype=torch.long)
    patient_ids = tuple(example.patient_id for example in examples)
    paths = tuple(example.path for example in examples)
    return Batch(images=images, labels=labels, patient_ids=patient_ids, paths=paths)


def build_data_module(
    config: DataModuleConfig,
    transform_registry: TransformRegistry | None = None,
) -> DataModule:
    """@brief 构建可直接供训练使用的数据模块；Build a ready-to-use data module."""
    registry = transform_registry or TransformRegistry()

    records = discover_records(config.image_root)
    split = split_by_patient(records, config.split)

    train_tf = registry.build(config.train_transform, config.image)
    eval_tf = registry.build(config.eval_transform, config.image)

    train_dataset = IDCPatchDataset(split.train, transform=train_tf)
    val_dataset = IDCPatchDataset(split.val, transform=eval_tf)

    sampler = _build_train_sampler(split.train, config.loader)

    worker_count = config.loader.num_workers
    persistent_workers = worker_count > 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.loader.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=config.loader.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=config.loader.train_drop_last,
        collate_fn=collate_patch_examples,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.loader.batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=config.loader.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=config.loader.eval_drop_last,
        collate_fn=collate_patch_examples,
    )

    return DataModule(
        split=split,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def _build_train_basic(schema: ImageSchema) -> T.Compose:
    """@brief 默认训练增强；Default training transform."""
    return T.Compose(
        [
            T.Resize(schema.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=schema.mean, std=schema.std),
        ]
    )


def _build_eval_basic(schema: ImageSchema) -> T.Compose:
    """@brief 默认验证/推理变换；Default eval/inference transform."""
    return T.Compose(
        [
            T.Resize(schema.image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=schema.mean, std=schema.std),
        ]
    )


def _build_train_sampler(
    records: Sequence[PatchRecord],
    loader_schema: LoaderSchema,
) -> Sampler[int] | None:
    """@brief 构建训练采样器；Build optional train sampler."""
    if not loader_schema.use_weighted_sampler:
        return None

    class_counts = _count_by_label(records)
    if len(class_counts) < 2:
        return None

    weights = [1.0 / class_counts[record.label] for record in records]
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


def _count_by_label(records: Iterable[PatchRecord]) -> dict[int, int]:
    """@brief 统计类别样本数；Count samples per class label."""
    counts: dict[int, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


__all__ = [
    "Batch",
    "DataModule",
    "DataModuleConfig",
    "DatasetSplit",
    "IDCPatchDataset",
    "ImageSchema",
    "LoaderSchema",
    "PatchExample",
    "PatchRecord",
    "SplitSchema",
    "TransformRegistry",
    "build_data_module",
    "collate_patch_examples",
    "discover_records",
    "split_by_patient",
]
