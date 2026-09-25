from __future__ import annotations

import subprocess
import sys

MODULES = (
    "types",
    "candidates",
    "lattice",
    "filters",
    "nms",
    "crops",
    "decide",
    "assign",
    "confirm",
    "review",
    "verify",
    "cli",
)


def test_waste_modules_never_import_torch() -> None:
    imports = "; ".join(f"import vineyard.perception.waste.{m}" for m in MODULES)
    code = f"import sys; {imports}; import vineyard.pipeline.stages.waste; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
