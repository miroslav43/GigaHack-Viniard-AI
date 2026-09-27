import io
import zipfile

import pytest

from fte.data import zip_range


def _archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a/keep1.txt", "hello " * 50, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("a/skip.bin", b"\x00" * 100, compress_type=zipfile.ZIP_STORED)
        zf.writestr("a/keep2.txt", "world", compress_type=zipfile.ZIP_STORED)
        zf.writestr("a/dir/", "")
    return buf.getvalue()


def _fetch(blob: bytes):
    return lambda url, s, e: blob[s:e + 1]


def test_list_select_and_fetch_members_roundtrip():
    blob = _archive()
    members = zip_range.list_members("u", fetch=_fetch(blob), total=len(blob))
    assert {m.name for m in members} == {"a/keep1.txt", "a/skip.bin", "a/keep2.txt", "a/dir/"}
    wanted = zip_range.select(members, ["a/keep*.txt", "a/*/"])
    assert [m.name for m in wanted] == ["a/keep1.txt", "a/keep2.txt"]
    files = zip_range.fetch_members("u", wanted, total=len(blob), fetch=_fetch(blob))
    assert files["a/keep1.txt"] == ("hello " * 50).encode()
    assert files["a/keep2.txt"] == b"world"


def test_parse_eocd_missing_signature():
    with pytest.raises(zip_range.ZipRangeError):
        zip_range.parse_eocd(b"not a zip")


def test_fetch_members_empty():
    assert zip_range.fetch_members("u", [], total=0) == {}
