"""pipeline.cache: cache keys, freshness and key files."""

from __future__ import annotations

from pathlib import Path

import pytest

from vineyard.errors import StageError
from vineyard.pipeline.atomic import atomic_path
from vineyard.pipeline.cache import (
    all_fresh,
    cache_key,
    is_fresh,
    key_path,
    read_key,
    remove_key,
    write_key,
    write_keys,
)
from vineyard.pipeline.tile_cache import key_path as tile_cache_key_path


def test_cache_key_is_hex_sha1_and_deterministic() -> None:
    key = cache_key("tile_prep", "1", "abc", ["x", "y"])
    assert len(key) == 40
    assert key == cache_key("tile_prep", "1", "abc", ["x", "y"])


def test_cache_key_ignores_input_order() -> None:
    assert cache_key("s", "1", "c", ["a", "b"]) == cache_key("s", "1", "c", ["b", "a"])


@pytest.mark.parametrize(
    "other",
    [("s2", "1", "c", ["a"]), ("s", "2", "c", ["a"]), ("s", "1", "d", ["a"]), ("s", "1", "c", ["b"]),
     ("s", "1", "c", ["a", "a2"])],
)
def test_cache_key_sensitive_to_every_part(other: tuple[str, str, str, list[str]]) -> None:
    assert cache_key("s", "1", "c", ["a"]) != cache_key(*other)


def test_cache_key_no_concatenation_ambiguity() -> None:
    assert cache_key("s", "1", "c", ["ab", "c"]) != cache_key("s", "1", "c", ["a", "bc"])


def test_key_path_matches_tile_cache_convention(tmp_path: Path) -> None:
    artifact = tmp_path / "veg" / "siret3_r006_c004.png"
    assert key_path(artifact) == tile_cache_key_path(artifact)
    assert key_path(artifact).name == "siret3_r006_c004.png.key"


def test_fresh_only_with_artifact_and_matching_key(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    assert not is_fresh(artifact, "k1")
    artifact.write_bytes(b"x")
    assert not is_fresh(artifact, "k1")
    write_key(artifact, "k1")
    assert is_fresh(artifact, "k1")
    assert not is_fresh(artifact, "k2")
    assert read_key(artifact) == "k1"


def test_key_without_artifact_is_not_fresh(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"x")
    write_key(artifact, "k")
    artifact.unlink()
    assert not is_fresh(artifact, "k")


def test_write_key_requires_existing_artifact(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="artifact missing"):
        write_key(tmp_path / "missing.bin", "k")


def test_write_key_rejects_blank_key(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"x")
    with pytest.raises(StageError, match="empty"):
        write_key(artifact, "  ")


def test_read_key_missing_is_none(tmp_path: Path) -> None:
    assert read_key(tmp_path / "nothing") is None


def test_all_fresh_and_write_keys(tmp_path: Path) -> None:
    arts = [tmp_path / "b.png", tmp_path / "a.json"]
    for art in arts:
        art.write_bytes(b"1")
    assert not all_fresh(arts, "k")
    write_keys(arts, "k")
    assert all_fresh(arts, "k")
    assert not all_fresh([], "k")
    remove_key(arts[0])
    assert not all_fresh(arts, "k")
    remove_key(arts[0])  # idempotent


def test_exception_inside_atomic_path_leaves_no_file_and_no_key(tmp_path: Path) -> None:
    artifact = tmp_path / "out.bin"
    with pytest.raises(RuntimeError), atomic_path(artifact) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")
    assert not artifact.exists()
    assert not key_path(artifact).exists()
    assert list(tmp_path.iterdir()) == []
