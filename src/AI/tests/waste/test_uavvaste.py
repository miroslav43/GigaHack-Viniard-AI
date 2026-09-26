"""UAVVaste source (Zenodo 8214061, CC BY 4.0): download log, md5, safe extraction, COCO parsing."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from vineyard.perception.waste.uavvaste import (
    LICENCE,
    SOURCE_FILE,
    DownloadState,
    UavvasteError,
    batch_of,
    download_state,
    extract_dataset,
    load_dataset,
    verify_zip,
)

COCO = {
    "images": [
        {"id": 0, "width": 40, "height": 30, "file_name": "BATCH_d07_img_1.jpg"},
        {"id": 1, "width": 40, "height": 30, "file_name": "batch_s02_img_9.jpg"},
    ],
    "categories": [{"id": 0, "name": "rubbish", "supercategory": ""}],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 0,
            "bbox": [2, 3, 10, 5],
            "area": 40.0,
            "segmentation": [[2, 3, 12, 3, 12, 8, 2, 8]],
            "iscrowd": 0,
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "bbox": [20, 10, 4, 12],
            "area": 30.0,
            "segmentation": [[20, 10, 24, 10, 24, 22]],
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 0,
            "bbox": [1, 1, 6, 6],
            "area": 36.0,
            "segmentation": [[1, 1, 7, 1, 7, 7, 1, 7]],
            "iscrowd": 0,
        },
    ],
    "licenses": [],
    "info": {"description": "UAVVaste dataset"},
}


def _jpg(h: int = 30, w: int = 40) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def make_zip(path: Path, coco: dict = COCO, extra: dict[str, bytes] | None = None) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("annotations/", b"")
        zf.writestr("annotations/annotations.json", json.dumps(coco))
        for img in coco["images"]:
            zf.writestr(f"images/{img['file_name']}", _jpg())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zenodo publishes md5


# ------------------------------------------------------------------ download log


@pytest.mark.parametrize(
    ("text", "state"),
    [
        (None, DownloadState.PENDING),
        ("", DownloadState.PENDING),
        ("UAV EXIT 0\n", DownloadState.OK),
        ("UAV EXIT 22\n", DownloadState.FAILED),
    ],
)
def test_download_state(tmp_path: Path, text: str | None, state: DownloadState) -> None:
    log = tmp_path / "uavvaste_download.log"
    if text is not None:
        log.write_text(text)
    assert download_state(log) == state


# ------------------------------------------------------------------ verification and extraction


def test_verify_zip_md5_mismatch_names_file_and_hashes(tmp_path: Path) -> None:
    z = tmp_path / "UAVVasteDataset.zip"
    md5 = make_zip(z)
    verify_zip(z, md5)
    with pytest.raises(UavvasteError) as err:
        verify_zip(z, "0" * 32)
    msg = str(err.value)
    assert "UAVVasteDataset.zip" in msg and md5 in msg and "0" * 32 in msg


def test_verify_zip_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UavvasteError, match="missing"):
        verify_zip(tmp_path / "nope.zip", "0" * 32)


def test_extract_writes_source_and_is_idempotent(tmp_path: Path) -> None:
    z = tmp_path / "UAVVasteDataset.zip"
    md5 = make_zip(z)
    root = extract_dataset(z, tmp_path / "out", md5)
    assert (root / "annotations" / "annotations.json").is_file()
    assert (root / "images" / "BATCH_d07_img_1.jpg").is_file()
    source = json.loads((root / SOURCE_FILE).read_text())
    assert source["licence"] == LICENCE == "CC-BY-4.0"
    assert source["md5"] == md5 and source["doi"] == "10.5281/zenodo.8214061"
    assert "Kraft" in source["attribution"]
    stamp = (root / SOURCE_FILE).stat().st_mtime_ns
    assert extract_dataset(z, tmp_path / "out", md5) == root
    assert (root / SOURCE_FILE).stat().st_mtime_ns == stamp  # not re-extracted


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    md5 = make_zip(z, extra={"../escape.txt": b"x"})
    with pytest.raises(UavvasteError, match="unsafe"):
        extract_dataset(z, tmp_path / "out", md5)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "out" / "dataset").exists()


# ------------------------------------------------------------------ COCO parsing


def test_load_dataset_parses_objects(tmp_path: Path) -> None:
    z = tmp_path / "UAVVasteDataset.zip"
    root = extract_dataset(z, tmp_path / "out", make_zip(z))
    ds = load_dataset(root)
    assert len(ds.images) == 2 and len(ds.objects) == 3
    by_image = ds.objects_by_image()
    assert [o.ann_id for o in by_image[0]] == [0, 1] and [o.ann_id for o in by_image[1]] == [2]
    obj = by_image[0][1]
    assert obj.bbox == (20.0, 10.0, 4.0, 12.0) and obj.long_side == 12.0
    assert obj.polygons == ((20.0, 10.0, 24.0, 10.0, 24.0, 22.0),)
    assert ds.image_path(ds.images[1]).name == "batch_s02_img_9.jpg"
    assert ds.images[0].batch == "batch_d07"


def test_load_dataset_rejects_bad_annotation(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(COCO))
    bad["annotations"][2]["bbox"] = [1, 2, 3]
    z = tmp_path / "UAVVasteDataset.zip"
    root = extract_dataset(z, tmp_path / "out", make_zip(z, bad))
    with pytest.raises(UavvasteError, match="bbox") as err:
        load_dataset(root)
    assert "ann_id=2" in str(err.value)


def test_load_dataset_unknown_image(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(COCO))
    bad["annotations"][0]["image_id"] = 99
    z = tmp_path / "UAVVasteDataset.zip"
    root = extract_dataset(z, tmp_path / "out", make_zip(z, bad))
    with pytest.raises(UavvasteError, match="unknown image"):
        load_dataset(root)


def test_load_dataset_missing_annotations(tmp_path: Path) -> None:
    with pytest.raises(UavvasteError, match="annotations"):
        load_dataset(tmp_path)


def test_image_path_missing_file(tmp_path: Path) -> None:
    z = tmp_path / "UAVVasteDataset.zip"
    root = extract_dataset(z, tmp_path / "out", make_zip(z))
    (root / "images" / "batch_s02_img_9.jpg").unlink()
    ds = load_dataset(root)
    with pytest.raises(UavvasteError, match="image file"):
        ds.image_path(ds.images[1])


@pytest.mark.parametrize(
    ("name", "batch"),
    [
        ("BATCH_d07_img_6400.jpg", "batch_d07"),
        ("batch_s02_img_1.jpg", "batch_s02"),
        ("GOPR0038.JPG", "misc"),
        ("photo_2020-10-23_14-02-02.jpg", "misc"),
    ],
)
def test_batch_of(name: str, batch: str) -> None:
    assert batch_of(name) == batch
