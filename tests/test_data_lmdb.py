from pathlib import Path

from PIL import Image
import pytest
import torch

from histoclass.data import Batch, DataModuleConfig, LmdbSchema, LoaderSchema, SplitSchema, build_data_module
from histoclass.utils import seed_worker


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_build_data_module_reads_from_lmdb_when_source_file_missing(tmp_path) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "patient_a" / "0" / "a.png", (255, 0, 0))
    _write_image(image_root / "patient_a" / "1" / "b.png", (0, 255, 0))
    _write_image(image_root / "patient_b" / "0" / "c.png", (0, 0, 255))
    _write_image(image_root / "patient_b" / "1" / "d.png", (255, 255, 0))

    lmdb_path = tmp_path / "cache" / "patches.lmdb"
    config = DataModuleConfig(
        image_root=image_root,
        split=SplitSchema(val_ratio=0.5, seed=1),
        loader=LoaderSchema(batch_size=2, num_workers=0),
        lmdb=LmdbSchema(enabled=True, path=lmdb_path, map_size_bytes=64 * 1024 * 1024),
    )

    data_module = build_data_module(config)
    assert lmdb_path.exists()

    first_example = data_module.train_dataset[0]
    first_example.path.unlink()

    cached_example = data_module.train_dataset[0]
    assert cached_example.image.shape[0] == 3
    assert cached_example.label in (0, 1)


def test_build_data_module_reuses_lmdb_cache_when_use_caches_enabled(
    tmp_path, monkeypatch
) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "patient_a" / "0" / "a.png", (255, 0, 0))
    _write_image(image_root / "patient_a" / "1" / "b.png", (0, 255, 0))
    _write_image(image_root / "patient_b" / "0" / "c.png", (0, 0, 255))
    _write_image(image_root / "patient_b" / "1" / "d.png", (255, 255, 0))

    lmdb_path = tmp_path / "cache" / "patches.lmdb"
    config = DataModuleConfig(
        image_root=image_root,
        split=SplitSchema(val_ratio=0.5, seed=1),
        loader=LoaderSchema(batch_size=2, num_workers=0),
        lmdb=LmdbSchema(
            enabled=True,
            use_caches=True,
            path=lmdb_path,
            map_size_bytes=64 * 1024 * 1024,
        ),
    )

    build_data_module(config)
    meta_path = image_root.parent / "processed" / "lmdb_meta.json"
    assert meta_path.exists()

    original_read_bytes = Path.read_bytes

    def _failing_read_bytes(path_obj: Path) -> bytes:
        if path_obj.suffix.lower() == ".png":
            raise RuntimeError("source image should not be read when cache is hit")
        return original_read_bytes(path_obj)

    monkeypatch.setattr(Path, "read_bytes", _failing_read_bytes)
    reused_module = build_data_module(config)
    assert len(reused_module.train_dataset) > 0


def test_build_data_module_rewarms_when_use_caches_disabled(tmp_path, monkeypatch) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "patient_a" / "0" / "a.png", (255, 0, 0))
    _write_image(image_root / "patient_a" / "1" / "b.png", (0, 255, 0))
    _write_image(image_root / "patient_b" / "0" / "c.png", (0, 0, 255))
    _write_image(image_root / "patient_b" / "1" / "d.png", (255, 255, 0))

    lmdb_path = tmp_path / "cache" / "patches.lmdb"
    config = DataModuleConfig(
        image_root=image_root,
        split=SplitSchema(val_ratio=0.5, seed=1),
        loader=LoaderSchema(batch_size=2, num_workers=0),
        lmdb=LmdbSchema(
            enabled=True,
            use_caches=False,
            path=lmdb_path,
            map_size_bytes=64 * 1024 * 1024,
        ),
    )

    build_data_module(config)

    original_read_bytes = Path.read_bytes

    def _failing_read_bytes(path_obj: Path) -> bytes:
        if path_obj.suffix.lower() == ".png":
            raise RuntimeError("expected warmup to read source files")
        return original_read_bytes(path_obj)

    monkeypatch.setattr(Path, "read_bytes", _failing_read_bytes)
    with pytest.raises(RuntimeError, match="expected warmup"):
        build_data_module(config)


def test_batch_pin_memory_pins_custom_batch_tensors() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Pinned memory assertions require CUDA runtime support.")

    batch = Batch(
        images=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        labels=torch.zeros((2,), dtype=torch.long),
        patient_ids=("p1", "p2"),
        paths=(Path("a.png"), Path("b.png")),
    )

    pinned = batch.pin_memory()

    assert pinned.images.is_pinned()
    assert pinned.labels.is_pinned()
    assert pinned.patient_ids == batch.patient_ids
    assert pinned.paths == batch.paths


def test_build_data_module_wires_worker_seed_and_loader_generators(tmp_path) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "patient_a" / "0" / "a.png", (255, 0, 0))
    _write_image(image_root / "patient_a" / "1" / "b.png", (0, 255, 0))
    _write_image(image_root / "patient_b" / "0" / "c.png", (0, 0, 255))
    _write_image(image_root / "patient_b" / "1" / "d.png", (255, 255, 0))

    torch.manual_seed(123)
    config = DataModuleConfig(
        image_root=image_root,
        split=SplitSchema(val_ratio=0.5, seed=1),
        loader=LoaderSchema(batch_size=2, num_workers=0),
        lmdb=LmdbSchema(enabled=False),
    )

    data_module = build_data_module(config)

    assert data_module.train_loader.worker_init_fn is seed_worker
    assert data_module.val_loader.worker_init_fn is seed_worker
    assert data_module.train_loader.generator is not None
    assert data_module.val_loader.generator is not None
    assert data_module.train_loader.generator.initial_seed() == 123
    assert data_module.val_loader.generator.initial_seed() == 124
