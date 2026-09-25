"""cvat.packer: part planning (greedy count + balanced contiguous split) and deterministic ZIPs."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from vineyard.cvat.packer import (
    ANNOTATIONS_NAME,
    ZIP_DATE_TIME,
    PartPlan,
    estimate_tile_bytes,
    greedy_count,
    image_arcname,
    plan_parts,
    write_upload_zip,
    zip_base_bytes,
)
from vineyard.errors import ExportBlocked


def _sizes(values: list[int]) -> list[tuple[str, int]]:
    return [(f"siret3_r{i:03d}_c000", v) for i, v in enumerate(values)]


def test_greedy_count_is_minimal_for_contiguous_packing() -> None:
    assert greedy_count([5, 5, 5, 5], 10) == 2
    assert greedy_count([6, 5, 5], 10) == 2
    assert greedy_count([10, 1], 10) == 2
    assert greedy_count([], 10) == 0


def test_greedy_count_rejects_item_over_budget() -> None:
    with pytest.raises(ExportBlocked):
        greedy_count([11], 10)


def test_plan_is_contiguous_sorted_and_under_limit() -> None:
    sizes = list(reversed(_sizes([3, 4, 2, 7, 1, 5, 6, 2, 3])))  # unsorted input
    plans = plan_parts(sizes, max_bytes=12, n_parts=None, balance=False)
    flat = [t for p in plans for t in p.tile_ids]
    assert flat == sorted(t for t, _ in sizes)
    assert all(p.est_bytes <= 12 for p in plans)
    assert [p.index for p in plans] == list(range(1, len(plans) + 1))
    assert len(plans) == greedy_count([v for _, v in sorted(sizes)], 12)


def test_balanced_plan_keeps_count_and_evens_parts() -> None:
    sizes = _sizes([10] * 10)
    greedy = plan_parts(sizes, max_bytes=40, n_parts=None, balance=False)
    balanced = plan_parts(sizes, max_bytes=40, n_parts=None, balance=True)
    assert [len(p.tile_ids) for p in greedy] == [4, 4, 2]
    assert len(balanced) == 3
    part_sums = [p.est_bytes for p in balanced]
    assert max(part_sums) - min(part_sums) <= 10
    assert all(s <= 40 for s in part_sums)


def test_balanced_plan_on_uneven_sizes_within_one_tile_of_mean() -> None:
    values = [7, 1, 9, 3, 3, 8, 2, 6, 5, 4, 9, 1, 2, 8]
    plans = plan_parts(_sizes(values), max_bytes=25, n_parts=None, balance=True)
    mean = sum(values) / len(plans)
    assert all(abs(p.est_bytes - mean) <= max(values) for p in plans)
    assert all(p.est_bytes <= 25 for p in plans)


def test_base_bytes_count_per_part() -> None:
    plans = plan_parts(_sizes([5, 5, 5, 5]), max_bytes=12, n_parts=None, balance=False, base_bytes=2)
    assert [p.est_bytes for p in plans] == [12, 12]


def test_explicit_n_parts_and_errors() -> None:
    plans = plan_parts(_sizes([5, 5, 5, 5]), max_bytes=100, n_parts=3, balance=True)
    assert len(plans) == 3 and all(p.tile_ids for p in plans)
    with pytest.raises(ExportBlocked):
        plan_parts(_sizes([5, 5, 5, 5]), max_bytes=10, n_parts=1, balance=True)
    with pytest.raises(ExportBlocked):
        plan_parts(_sizes([5, 5]), max_bytes=100, n_parts=3, balance=True)
    with pytest.raises(ExportBlocked):
        plan_parts([("a", 1), ("a", 2)], max_bytes=100, n_parts=None, balance=False)


@pytest.mark.parametrize(
    ("values", "max_bytes", "n_parts", "expected"),
    [
        ([2, 7, 2, 6], 8, 3, [1, 1, 2]),  # nearest-to-mean cuts overflow -> min-max capacity search
        ([5, 4, 7, 2, 6, 2], 10, 4, [1, 1, 2, 2]),  # min-max gives 3 parts -> widest part split to reach n
    ],
)
def test_balanced_plan_falls_back_to_minmax(values: list[int], max_bytes: int, n_parts: int,
                                            expected: list[int]) -> None:
    plans = plan_parts(_sizes(values), max_bytes=max_bytes, n_parts=n_parts, balance=True)
    assert [len(p.tile_ids) for p in plans] == expected
    assert all(p.est_bytes <= max_bytes for p in plans)


def test_plan_is_deterministic() -> None:
    sizes = _sizes([3, 9, 4, 4, 7, 1, 8])
    assert plan_parts(sizes, max_bytes=15, n_parts=None, balance=True) == plan_parts(
        list(reversed(sizes)), max_bytes=15, n_parts=None, balance=True
    )
    assert isinstance(plan_parts(sizes, max_bytes=15, n_parts=None, balance=True)[0], PartPlan)


def test_estimate_tile_bytes_counts_headers() -> None:
    name = "siret3_r021_c012.tif"
    arc = image_arcname(name)
    assert arc == "images/siret3_r021_c012.tif"
    assert estimate_tile_bytes(1000, name, 50) == 1000 + 76 + 2 * len(arc) + 50
    assert zip_base_bytes(500) == 500 + 76 + 2 * len(ANNOTATIONS_NAME) + 22


def _tif(tmp_path: Path, name: str, payload: bytes) -> tuple[str, Path]:
    path = tmp_path / "src" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return name, path


def test_write_upload_zip_layout_and_determinism(tmp_path: Path) -> None:
    images = [
        _tif(tmp_path, "siret3_r021_c012.tif", b"B" * 300),
        _tif(tmp_path, "siret3_r006_c004.tif", b"A" * 200),
    ]
    xml = b"<annotations>" + b"x" * 1000 + b"</annotations>\n"
    out1 = tmp_path / "a" / "part.zip"
    out2 = tmp_path / "b" / "part.zip"
    shas = write_upload_zip(out1, xml, images, deflate_level=9)
    write_upload_zip(out2, xml, list(reversed(images)), deflate_level=9)
    assert out1.read_bytes() == out2.read_bytes()
    assert shas == {
        "siret3_r006_c004.tif": hashlib.sha256(b"A" * 200).hexdigest(),
        "siret3_r021_c012.tif": hashlib.sha256(b"B" * 300).hexdigest(),
    }
    with zipfile.ZipFile(out1) as zf:
        infos = zf.infolist()
        assert [i.filename for i in infos] == [
            "annotations.xml", "images/siret3_r006_c004.tif", "images/siret3_r021_c012.tif",
        ]
        assert infos[0].compress_type == zipfile.ZIP_DEFLATED
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos[1:])
        assert all(i.date_time == ZIP_DATE_TIME for i in infos)
        assert all((i.external_attr >> 16) == 0o100644 for i in infos)
        assert zf.read("annotations.xml") == xml
        assert zf.testzip() is None
    assert not list(out1.parent.glob("*.tmp-*"))


def test_write_upload_zip_rejects_bad_names_and_duplicates(tmp_path: Path) -> None:
    good = _tif(tmp_path, "siret3_r021_c012.tif", b"B")
    with pytest.raises(ExportBlocked):
        write_upload_zip(tmp_path / "x.zip", b"<a/>", [good, good], deflate_level=9)
    with pytest.raises(ExportBlocked):
        write_upload_zip(tmp_path / "y.zip", b"<a/>", [("sub/siret3_r021_c012.tif", good[1])], deflate_level=9)
    with pytest.raises(ExportBlocked):
        write_upload_zip(tmp_path / "z.zip", b"<a/>", [("siret3_r021_c012.tif", tmp_path / "nope.tif")],
                         deflate_level=9)
    assert not (tmp_path / "x.zip").exists()
