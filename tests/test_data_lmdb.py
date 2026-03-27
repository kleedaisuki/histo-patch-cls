from pathlib import Path

from PIL import Image

from histoclass.data import DataModuleConfig, LmdbSchema, LoaderSchema, SplitSchema, build_data_module


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
