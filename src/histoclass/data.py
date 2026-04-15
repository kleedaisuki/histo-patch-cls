"""Data pipeline for IDC patch classification."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import lmdb
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms as T

from .utils import get_logger


IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ImageSchema:
    """@brief 图像预处理配置；Image preprocessing schema."""

    image_size: tuple[int, int] = (50, 50)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.25, 0.25, 0.25)


@dataclass(frozen=True, slots=True)
class SplitSchema:
    """@brief 按患者切分配置；Patient-wise split schema."""

    strategy: str = "patient"
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
class LmdbSchema:
    """@brief LMDB 缓存配置；LMDB cache schema."""

    enabled: bool = True
    use_caches: bool = True
    path: Path | None = None
    map_size_bytes: int = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DataModuleConfig:
    """@brief 数据模块顶层配置；Top-level data module config."""

    image_root: Path
    image: ImageSchema = ImageSchema()
    split: SplitSchema = SplitSchema()
    loader: LoaderSchema = LoaderSchema()
    lmdb: LmdbSchema = LmdbSchema()


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
        transform: T.Compose,
        lmdb_path: Path | None = None,
    ) -> None:
        self._records: tuple[PatchRecord, ...] = tuple(records)
        self._transform = transform
        self._lmdb_path = lmdb_path
        self._lmdb_env: lmdb.Environment | None = None
        self._warned_missing_lmdb_key = False

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PatchExample:
        record = self._records[index]
        payload = self._read_image_bytes(record.path)

        if payload is None:
            with Image.open(record.path) as image:
                rgb_image = image.convert("RGB")
                tensor = self._transform(rgb_image)
        else:
            with Image.open(BytesIO(payload)) as image:
                rgb_image = image.convert("RGB")
                tensor = self._transform(rgb_image)

        return PatchExample(
            image=tensor,
            label=record.label,
            patient_id=record.patient_id,
            path=record.path,
        )

    def __del__(self) -> None:
        if self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_lmdb_env"] = None
        return state

    def _read_image_bytes(self, path: Path) -> bytes | None:
        env = self._ensure_lmdb_env()
        if env is None:
            return None

        with env.begin(write=False) as txn:
            payload = txn.get(_record_key(path))

        if payload is None:
            if not self._warned_missing_lmdb_key:
                LOGGER.warning(
                    "LMDB key miss detected. Falling back to filesystem reads. Example key path=%s",
                    path,
                )
                self._warned_missing_lmdb_key = True
            return None
        return bytes(payload)

    def _ensure_lmdb_env(self) -> lmdb.Environment | None:
        if self._lmdb_path is None:
            return None
        if self._lmdb_env is not None:
            return self._lmdb_env

        self._lmdb_env = lmdb.open(
            str(self._lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False,
            max_readers=2048,
        )
        return self._lmdb_env


def discover_records(image_root: Path) -> tuple[PatchRecord, ...]:
    """@brief 扫描数据目录生成样本记录；Discover records from dataset tree.

    @param image_root 数据根目录，期望结构为 `<patient>/<label>/<image>`.
    @return 按路径稳定排序的样本记录元组。
    """
    root = image_root.expanduser().resolve()
    LOGGER.info("Discovering IDC patch records under: %s", root)
    if not root.exists():
        LOGGER.error("image_root not found: %s", root)
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
        LOGGER.error("No images discovered under: %s", root)
        raise RuntimeError(f"No images discovered under: {root}")

    label_counts = _count_by_label(records)
    patient_count = len({record.patient_id for record in records})
    LOGGER.info(
        "Discovered %d records from %d patients. Label counts=%s",
        len(records),
        patient_count,
        label_counts,
    )
    return tuple(records)


def split_by_patient(
    records: Sequence[PatchRecord], schema: SplitSchema
) -> DatasetSplit:
    """@brief 按患者切分，避免信息泄漏；Patient-wise split to avoid leakage."""
    LOGGER.info(
        "Splitting %d records by patient with val_ratio=%.4f, seed=%d",
        len(records),
        schema.val_ratio,
        schema.seed,
    )
    patients = sorted({record.patient_id for record in records})
    if len(patients) < 2:
        LOGGER.error("Split failed: less than 2 patients found (%d)", len(patients))
        raise ValueError("At least 2 patients are required for train/val split.")

    if not (0.0 < schema.val_ratio < 1.0):
        LOGGER.error("Invalid split.val_ratio: %.6f", schema.val_ratio)
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
        LOGGER.error(
            "Invalid split produced empty set. train=%d, val=%d",
            len(train_records),
            len(val_records),
        )
        raise RuntimeError("Invalid split produced an empty train or val set.")

    LOGGER.info(
        "Patient-wise split done: train=%d, val=%d, train_patients=%d, val_patients=%d",
        len(train_records),
        len(val_records),
        len(patients) - len(val_patients),
        len(val_patients),
    )
    return DatasetSplit(train=train_records, val=val_records)


def split_by_patch_random(
    records: Sequence[PatchRecord], schema: SplitSchema
) -> DatasetSplit:
    """@brief Patch 级随机切分；Patch-level random split."""
    LOGGER.info(
        "Splitting %d records by patch-level random with val_ratio=%.4f, seed=%d",
        len(records),
        schema.val_ratio,
        schema.seed,
    )
    if len(records) < 2:
        LOGGER.error("Split failed: less than 2 records found (%d)", len(records))
        raise ValueError("At least 2 records are required for train/val split.")

    if not (0.0 < schema.val_ratio < 1.0):
        LOGGER.error("Invalid split.val_ratio: %.6f", schema.val_ratio)
        raise ValueError("split.val_ratio must be in (0, 1).")

    rng = Random(schema.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    val_count = max(
        1, min(len(shuffled) - 1, int(round(len(shuffled) * schema.val_ratio)))
    )
    val_records = tuple(shuffled[:val_count])
    train_records = tuple(shuffled[val_count:])

    if not train_records or not val_records:
        LOGGER.error(
            "Invalid split produced empty set. train=%d, val=%d",
            len(train_records),
            len(val_records),
        )
        raise RuntimeError("Invalid split produced an empty train or val set.")

    train_patients = {record.patient_id for record in train_records}
    val_patients = {record.patient_id for record in val_records}
    overlap = train_patients & val_patients
    LOGGER.info(
        "Patch-level random split done: train=%d, val=%d, patient_overlap=%d",
        len(train_records),
        len(val_records),
        len(overlap),
    )
    return DatasetSplit(train=train_records, val=val_records)


def split_records(records: Sequence[PatchRecord], schema: SplitSchema) -> DatasetSplit:
    """@brief 按策略执行数据切分；Dispatch split strategy."""
    if schema.strategy == "patient":
        return split_by_patient(records, schema)
    if schema.strategy == "patch_random":
        return split_by_patch_random(records, schema)

    LOGGER.error("Unknown split.strategy: %s", schema.strategy)
    raise ValueError("split.strategy must be one of {'patient', 'patch_random'}.")


def collate_patch_examples(examples: Sequence[PatchExample]) -> Batch:
    """@brief 自定义拼接函数；Custom collate function producing Batch."""
    images = torch.stack([example.image for example in examples], dim=0)
    labels = torch.as_tensor([example.label for example in examples], dtype=torch.long)
    patient_ids = tuple(example.patient_id for example in examples)
    paths = tuple(example.path for example in examples)
    return Batch(images=images, labels=labels, patient_ids=patient_ids, paths=paths)


def build_data_module(config: DataModuleConfig) -> DataModule:
    """@brief 构建可直接供训练使用的数据模块；Build a ready-to-use data module."""
    LOGGER.info(
        "Building data module (image_root=%s, lmdb_enabled=%s)",
        config.image_root,
        config.lmdb.enabled,
    )

    records = discover_records(config.image_root)
    split = split_records(records, config.split)

    train_tf = _build_train_basic(config.image)
    eval_tf = _build_eval_basic(config.image)
    lmdb_path = _prepare_lmdb_cache(records=records, config=config)

    train_dataset = IDCPatchDataset(
        split.train, transform=train_tf, lmdb_path=lmdb_path
    )
    val_dataset = IDCPatchDataset(split.val, transform=eval_tf, lmdb_path=lmdb_path)

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

    LOGGER.info(
        "Data module ready: train_size=%d, val_size=%d, batch_size=%d, num_workers=%d, "
        "weighted_sampler=%s, lmdb_path=%s",
        len(train_dataset),
        len(val_dataset),
        config.loader.batch_size,
        worker_count,
        config.loader.use_weighted_sampler,
        lmdb_path,
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
        LOGGER.debug("Weighted sampler disabled; using default shuffle strategy.")
        return None

    class_counts = _count_by_label(records)
    if len(class_counts) < 2:
        LOGGER.warning(
            "Weighted sampler requested but only one class present: %s. Fallback to shuffle.",
            class_counts,
        )
        return None

    weights = [1.0 / class_counts[record.label] for record in records]
    LOGGER.info("Using weighted sampler with class counts: %s", class_counts)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


def _count_by_label(records: Iterable[PatchRecord]) -> dict[int, int]:
    """@brief 统计类别样本数；Count samples per class label."""
    counts: dict[int, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


def _prepare_lmdb_cache(
    records: Sequence[PatchRecord], config: DataModuleConfig
) -> Path | None:
    """@brief 构建并预热 LMDB 缓存（多线程加速版）；Build and warm LMDB cache with ThreadPool."""
    if not config.lmdb.enabled:
        LOGGER.info("LMDB cache disabled by config.")
        return None

    lmdb_path = (
        config.lmdb.path
        if config.lmdb.path is not None
        else config.image_root.parent / "processed" / "idc_patches.lmdb"
    )
    resolved_path = lmdb_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = config.image_root.parent / "processed" / "lmdb_meta.json"
    meta_path = meta_path.expanduser().resolve()

    if config.lmdb.use_caches and _is_lmdb_cache_reusable(
        meta_path=meta_path,
        lmdb_path=resolved_path,
        records=records,
        map_size_bytes=config.lmdb.map_size_bytes,
    ):
        LOGGER.info(
            "LMDB cache hit: reusing existing cache without warmup. lmdb_path=%s, meta_path=%s",
            resolved_path,
            meta_path,
        )
        return resolved_path

    LOGGER.info("Preparing LMDB cache at: %s", resolved_path)
    env = lmdb.open(
        str(resolved_path),
        map_size=config.lmdb.map_size_bytes,
        subdir=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    def _read_worker(record: PatchRecord) -> tuple[bytes, bytes]:
        key = _record_key(record.path)
        data = record.path.read_bytes()
        return key, data

    inserted = 0
    max_workers = min(32, multiprocessing.cpu_count() * 4)
    BATCH_SIZE = 5000

    LOGGER.info(
        "Starting LMDB ingestion with max_workers=%d, batch_size=%d",
        max_workers,
        BATCH_SIZE,
    )

    try:
        txn = env.begin(write=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_read_worker, record) for record in records]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                key, data = future.result()

                is_new = txn.put(key, data, overwrite=False)
                if is_new:
                    inserted += 1

                if (i + 1) % BATCH_SIZE == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    LOGGER.debug("Committed batch %d / %d", i + 1, len(records))

        txn.commit()

    except Exception as e:
        LOGGER.error("Error occurred during LMDB ingestion, aborting transaction!")
        txn.abort()
        raise e
    finally:
        env.sync()
        env.close()

    LOGGER.info(
        "LMDB cache ready: total_records=%d, inserted_new=%d, path=%s",
        len(records),
        inserted,
        resolved_path,
    )
    _write_lmdb_meta(
        meta_path=meta_path,
        lmdb_path=resolved_path,
        records=records,
        map_size_bytes=config.lmdb.map_size_bytes,
    )
    return resolved_path


def _is_lmdb_cache_reusable(
    *,
    meta_path: Path,
    lmdb_path: Path,
    records: Sequence[PatchRecord],
    map_size_bytes: int,
) -> bool:
    """@brief 判断 LMDB 缓存是否可复用；Check whether LMDB cache can be reused."""
    if not lmdb_path.exists() or not meta_path.exists():
        return False

    payload = _load_lmdb_meta(meta_path)
    if payload is None:
        return False

    expected = {
        "schema_version": 1,
        "lmdb_path": str(lmdb_path),
        "map_size_bytes": map_size_bytes,
        "records_count": len(records),
        "records_fingerprint": _records_fingerprint(records),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True


def _write_lmdb_meta(
    *,
    meta_path: Path,
    lmdb_path: Path,
    records: Sequence[PatchRecord],
    map_size_bytes: int,
) -> None:
    """@brief 写入 LMDB 元数据缓存；Persist LMDB metadata cache."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "lmdb_path": str(lmdb_path),
        "map_size_bytes": map_size_bytes,
        "records_count": len(records),
        "records_fingerprint": _records_fingerprint(records),
    }
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("LMDB metadata cache updated at: %s", meta_path)


def _load_lmdb_meta(meta_path: Path) -> dict[str, object] | None:
    """@brief 读取 LMDB 元数据缓存；Load LMDB metadata cache."""
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        LOGGER.warning("Failed to read LMDB metadata cache: %s", meta_path)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Invalid LMDB metadata payload type: %s", type(payload).__name__)
        return None
    return payload


def _records_fingerprint(records: Sequence[PatchRecord]) -> str:
    """@brief 样本集合指纹；Build fingerprint for record collection."""
    hasher = hashlib.sha256()
    for record in records:
        stat = record.path.stat()
        hasher.update(str(record.path).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(b"|")
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _record_key(path: Path) -> bytes:
    """@brief 样本路径转 LMDB 键；Convert sample path to LMDB key."""
    return str(path).encode("utf-8")


__all__ = [
    "Batch",
    "DataModule",
    "DataModuleConfig",
    "DatasetSplit",
    "IDCPatchDataset",
    "ImageSchema",
    "LmdbSchema",
    "LoaderSchema",
    "PatchExample",
    "PatchRecord",
    "SplitSchema",
    "build_data_module",
    "collate_patch_examples",
    "discover_records",
    "split_by_patch_random",
    "split_by_patient",
    "split_records",
]
