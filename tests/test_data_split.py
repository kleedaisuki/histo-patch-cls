from pathlib import Path

import pytest

from histoclass.data import PatchRecord, SplitSchema, split_by_patch_random, split_by_patient


def _build_records() -> tuple[PatchRecord, ...]:
    return (
        PatchRecord(path=Path("p1_0.png"), label=0, patient_id="p1"),
        PatchRecord(path=Path("p1_1.png"), label=1, patient_id="p1"),
        PatchRecord(path=Path("p2_0.png"), label=0, patient_id="p2"),
        PatchRecord(path=Path("p2_1.png"), label=1, patient_id="p2"),
    )


def test_split_by_patch_random_returns_non_empty_sets() -> None:
    records = _build_records()
    split = split_by_patch_random(records, SplitSchema(strategy="patch_random", val_ratio=0.5, seed=7))
    assert len(split.train) == 2
    assert len(split.val) == 2
    assert {record.path for record in split.train} | {record.path for record in split.val} == {
        record.path for record in records
    }
    assert {record.path for record in split.train} & {record.path for record in split.val} == set()


def test_patch_random_works_with_single_patient_records() -> None:
    records = (
        PatchRecord(path=Path("single_0.png"), label=0, patient_id="single"),
        PatchRecord(path=Path("single_1.png"), label=1, patient_id="single"),
        PatchRecord(path=Path("single_2.png"), label=0, patient_id="single"),
    )
    split = split_by_patch_random(records, SplitSchema(strategy="patch_random", val_ratio=0.34, seed=1))
    assert len(split.train) == 2
    assert len(split.val) == 1


def test_split_by_patient_requires_at_least_two_patients() -> None:
    records = (
        PatchRecord(path=Path("single_0.png"), label=0, patient_id="single"),
        PatchRecord(path=Path("single_1.png"), label=1, patient_id="single"),
    )
    with pytest.raises(ValueError, match="At least 2 patients"):
        split_by_patient(records, SplitSchema(strategy="patient", val_ratio=0.5, seed=1))
