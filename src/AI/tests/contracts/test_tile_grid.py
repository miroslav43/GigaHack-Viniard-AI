import zipfile
from pathlib import Path

import pytest

from vineyard.contracts.ids import tile_grid_ids, tile_id_from_file_name

N_PARTS = 5
TILE_ZIP = "01_tiles/siret3_challenge_tiles_part{i}of5.zip"


@pytest.mark.needs_tiles
def test_tile_grid_equals_zip_names(data_root: Path) -> None:
    names: list[str] = []
    for i in range(1, N_PARTS + 1):
        with zipfile.ZipFile(data_root / TILE_ZIP.format(i=i)) as zf:
            names.extend(tile_id_from_file_name(n) for n in zf.namelist())
    assert len(names) == len(set(names)) == 311
    assert tuple(sorted(names)) == tile_grid_ids()
