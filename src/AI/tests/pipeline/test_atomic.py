"""Atomic write helpers: tmp file + os.replace, nothing left behind on failure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vineyard.pipeline.atomic import (
    atomic_path,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    tmp_path_for,
)


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if ".tmp-" in p.name)


def test_tmp_name_uses_pid(tmp_path: Path) -> None:
    target = tmp_path / "a.parquet"
    assert tmp_path_for(target) == tmp_path / f"a.parquet.tmp-{os.getpid()}"


def test_atomic_path_replaces_on_success(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out.bin"
    with atomic_path(target) as tmp:
        assert tmp.parent == target.parent
        assert tmp.parent.is_dir()
        assert not target.exists()
        tmp.write_bytes(b"hello")
    assert target.read_bytes() == b"hello"
    assert _leftovers(target.parent) == []


def test_atomic_path_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old")
    with atomic_path(target) as tmp:
        tmp.write_text("new")
    assert target.read_text() == "new"


def test_atomic_path_exception_leaves_nothing(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    with pytest.raises(RuntimeError, match="boom"), atomic_path(target) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")
    assert not target.exists()
    assert _leftovers(tmp_path) == []


def test_atomic_path_exception_keeps_previous_version(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    target.write_bytes(b"v1")
    with pytest.raises(ValueError), atomic_path(target) as tmp:
        tmp.write_bytes(b"v2-partial")
        raise ValueError("fail")
    assert target.read_bytes() == b"v1"


def test_atomic_path_without_write_raises(tmp_path: Path) -> None:
    target = tmp_path / "never.bin"
    with pytest.raises(FileNotFoundError), atomic_path(target):
        pass
    assert not target.exists()


def test_atomic_write_bytes_and_text(tmp_path: Path) -> None:
    b = atomic_write_bytes(tmp_path / "x" / "b.bin", b"\x00\x01")
    t = atomic_write_text(tmp_path / "t.txt", "șțăîâ\n")
    assert b.read_bytes() == b"\x00\x01"
    assert t.read_text(encoding="utf-8") == "șțăîâ\n"


def test_atomic_write_json_is_deterministic(tmp_path: Path) -> None:
    obj_a = {"b": 1, "a": {"z": [1, 2], "y": "ș"}}
    obj_b = {"a": {"y": "ș", "z": [1, 2]}, "b": 1}
    pa = atomic_write_json(tmp_path / "a.json", obj_a)
    pb = atomic_write_json(tmp_path / "b.json", obj_b)
    assert pa.read_bytes() == pb.read_bytes()
    text = pa.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == obj_a


def test_atomic_write_json_converts_numpy_paths_and_sets(tmp_path: Path) -> None:
    obj = {
        "n": np.int64(3),
        "f": np.float32(0.5),
        "arr": np.arange(3),
        "p": Path("/x/y"),
        "s": frozenset({"b", "a"}),
        "flag": np.bool_(True),
    }
    out = json.loads(atomic_write_json(tmp_path / "n.json", obj).read_text())
    assert out == {"n": 3, "f": 0.5, "arr": [0, 1, 2], "p": "/x/y", "s": ["a", "b"], "flag": True}


def test_atomic_write_json_rejects_nan_and_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "nan.json", {"x": float("nan")})
    with pytest.raises(TypeError):
        atomic_write_json(tmp_path / "obj.json", {"x": object()})
    assert _leftovers(tmp_path) == []
    assert not (tmp_path / "nan.json").exists()
