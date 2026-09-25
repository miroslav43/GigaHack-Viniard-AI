"""P0-GEO modules run inside spawn workers: importing them must never pull torch."""

from __future__ import annotations

import subprocess
import sys

MODULES = (
    "vineyard.geo.tiling",
    "vineyard.geo.raster",
    "vineyard.geo.vector_io",
    "vineyard.geo.ops",
    "vineyard.geo.polygonize",
    "vineyard.perception.types",
    "vineyard.perception.vegmask",
    "vineyard.perception.corridor",
    "vineyard.pipeline.atomic",
    "vineyard.pipeline.tile_cache",
)


def test_no_torch_after_import(project_root) -> None:
    code = "import sys\n" + "".join(f"import {m}\n" for m in MODULES) + "print('torch' in sys.modules)\n"
    out = subprocess.run([sys.executable, "-c", code], cwd=project_root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stderr
