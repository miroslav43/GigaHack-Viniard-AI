"""UAVVaste positives source (A§4.9 step 3, design 03 W7).

Zenodo record 8214061 (DOI 10.5281/zenodo.8214061), licence CC BY 4.0 (the Zenodo record; the design's
"Apache-2.0" refers to the GitHub code). One archive, UAVVasteDataset.zip (3 010 727 259 B), holding
`annotations/annotations.json` (COCO: 772 images, 3 718 `rubbish` annotations, one polygon each) and the
images. The download itself is done outside this module (curl, log line "UAV EXIT <code>"); here: log
state, md5 check, safe extraction with SOURCE.json (attribution), COCO parsing. Images and crops are never
redistributed (work/ is gitignored); only probe coefficients ship.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Final

from vineyard.errors import VineyardError
from vineyard.pipeline.atomic import atomic_write_json

ZENODO_RECORD: Final = 8214061
DOI: Final = "10.5281/zenodo.8214061"
URL: Final = "https://zenodo.org/records/8214061"
LICENCE: Final = "CC-BY-4.0"
ATTRIBUTION: Final = (
    "UAVVaste dataset v1 by M. Kraft, M. Piechocki, B. Ptak, K. Walas (Institute of Robotics and Machine "
    "Intelligence, Poznan University of Technology), Zenodo, doi:10.5281/zenodo.8214061, licensed CC BY 4.0"
)
ZIP_NAME: Final = "UAVVasteDataset.zip"
ZIP_MD5: Final = "1575c32c04bdf944047563e4a1786c2a"
ZIP_SIZE: Final = 3_010_727_259
ANNOTATIONS: Final = Path("annotations") / "annotations.json"
SOURCE_FILE: Final = "SOURCE.json"
DATASET_DIR: Final = "dataset"
LOG_MARKER: Final = "UAV EXIT"
MISC_BATCH: Final = "misc"
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png"})
BBOX_LEN: Final = 4
_BATCH_RE: Final = re.compile(r"^(batch_[a-z0-9]+)_img_", re.IGNORECASE)
_HASH_CHUNK: Final = 1 << 22


class UavvasteError(VineyardError):
    """UAVVaste download, archive or annotation problem."""


class DownloadState(StrEnum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"


def download_state(log_path: Path, marker: str = LOG_MARKER) -> DownloadState:
    """State of the background download from its log ("<marker> <exit code>" is written at the end)."""
    path = Path(log_path)
    if not path.is_file():
        return DownloadState.PENDING
    codes = re.findall(rf"{re.escape(marker)}\s+(\d+)", path.read_text(encoding="utf-8", errors="replace"))
    if not codes:
        return DownloadState.PENDING
    return DownloadState.OK if codes[-1] == "0" else DownloadState.FAILED


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the md5 Zenodo publishes
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(zip_path: Path, expected_md5: str) -> None:
    path = Path(zip_path)
    if not path.is_file():
        raise UavvasteError("archive missing", path=str(path))
    actual = md5_file(path)
    if actual != expected_md5:
        raise UavvasteError(
            f"md5 mismatch for {path.name}: expected {expected_md5}, got {actual}",
            path=str(path),
            size=path.stat().st_size,
        )


def _safe_members(zf: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    root = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if info.filename.startswith("/") or not target.is_relative_to(root):
            raise UavvasteError("unsafe archive member (path traversal)", member=info.filename)
    return zf.infolist()


def _source_doc(zip_path: Path, md5: str, n_members: int) -> dict[str, Any]:
    return {
        "dataset": "UAVVaste",
        "zenodo_record": ZENODO_RECORD,
        "doi": DOI,
        "url": URL,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "archive": zip_path.name,
        "md5": md5,
        "size": zip_path.stat().st_size,
        "n_members": n_members,
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "redistribution": "images and crops are not redistributed; only probe coefficients ship",
    }


def _already_extracted(root: Path, md5: str) -> bool:
    source = root / SOURCE_FILE
    if not source.is_file():
        return False
    try:
        return json.loads(source.read_text(encoding="utf-8")).get("md5") == md5
    except json.JSONDecodeError as exc:
        raise UavvasteError("corrupt SOURCE.json", path=str(source), error=str(exc)) from exc


def extract_dataset(zip_path: Path, dest_dir: Path, expected_md5: str = ZIP_MD5) -> Path:
    """Verify, then extract into <dest>/dataset (via a temporary sibling); idempotent per md5."""
    zpath, root = Path(zip_path), Path(dest_dir) / DATASET_DIR
    if _already_extracted(root, expected_md5):
        return root
    verify_zip(zpath, expected_md5)
    tmp = Path(dest_dir) / f"{DATASET_DIR}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zpath) as zf:
            members = _safe_members(zf, tmp)
            zf.extractall(tmp, members=members)
        atomic_write_json(tmp / SOURCE_FILE, _source_doc(zpath, expected_md5, len(members)))
        shutil.rmtree(root, ignore_errors=True)
        os.replace(tmp, root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return root


# ------------------------------------------------------------------ COCO


def batch_of(file_name: str) -> str:
    """Flight batch of an image ("BATCH_d07_img_6400.jpg" -> "batch_d07"); others -> "misc"."""
    m = _BATCH_RE.match(file_name)
    return m.group(1).lower() if m else MISC_BATCH


@dataclass(frozen=True)
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int

    @property
    def batch(self) -> str:
        return batch_of(self.file_name)


@dataclass(frozen=True)
class CocoObject:
    ann_id: int
    image_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h in image px (COCO)
    area: float
    polygons: tuple[tuple[float, ...], ...]  # flat x0, y0, x1, y1, ...

    @property
    def long_side(self) -> float:
        return max(self.bbox[2], self.bbox[3])


@dataclass(frozen=True)
class UavvasteDataset:
    root: Path
    images: Mapping[int, CocoImage]
    objects: tuple[CocoObject, ...]

    def objects_by_image(self) -> dict[int, tuple[CocoObject, ...]]:
        out: dict[int, list[CocoObject]] = {}
        for obj in self.objects:
            out.setdefault(obj.image_id, []).append(obj)
        return {k: tuple(v) for k, v in sorted(out.items())}

    @cached_property
    def _files(self) -> dict[str, Path]:
        return {p.name: p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES}

    def image_path(self, img: CocoImage) -> Path:
        path = self._files.get(img.file_name)
        if path is None or not path.is_file():
            raise UavvasteError(
                "image file not found under the dataset root", file=img.file_name, root=str(self.root)
            )
        return path


def _image(d: Mapping[str, Any]) -> CocoImage:
    return CocoImage(int(d["id"]), str(d["file_name"]), int(d["width"]), int(d["height"]))


def _object(d: Mapping[str, Any], images: Mapping[int, CocoImage]) -> CocoObject:
    ann_id = int(d["id"])
    bbox = d.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != BBOX_LEN or min(bbox[2:]) <= 0:
        raise UavvasteError("annotation bbox must be [x, y, w, h] with w, h > 0", ann_id=ann_id, bbox=bbox)
    if int(d["image_id"]) not in images:
        raise UavvasteError("annotation refers to an unknown image", ann_id=ann_id, image_id=d["image_id"])
    seg = d.get("segmentation") or []
    polys = tuple(tuple(float(v) for v in poly) for poly in seg if isinstance(poly, list))
    return CocoObject(
        ann_id, int(d["image_id"]), tuple(float(v) for v in bbox), float(d.get("area", 0.0)), polys
    )


def load_dataset(root: Path) -> UavvasteDataset:
    """Parse <root>/annotations/annotations.json (COCO) into frozen images and objects."""
    path = Path(root) / ANNOTATIONS
    if not path.is_file():
        raise UavvasteError("UAVVaste annotations file missing", path=str(path))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        images = {img.image_id: img for img in map(_image, doc["images"])}
        objects = tuple(_object(a, images) for a in doc["annotations"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise UavvasteError("malformed UAVVaste annotations", path=str(path), error=repr(exc)) from exc
    return UavvasteDataset(root=Path(root), images=images, objects=objects)
