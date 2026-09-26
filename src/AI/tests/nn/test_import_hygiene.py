"""The torch-free NN modules and every pipeline stage import without torch (spawned pool workers stay light)."""

from __future__ import annotations

import subprocess
import sys

TORCH_FREE = (
    "vineyard.nn.probs",
    "vineyard.nn.fusion",
    "vineyard.nn.pseudolabels",
    "vineyard.nn.store_format",
    "vineyard.nn.patch_store",
    "vineyard.nn.validate",
    "vineyard.nn.weights",
    "vineyard.nn.ablation",
    "vineyard.nn.panels",
    "vineyard.nn.workflows",
    "vineyard.nn.cli",
    "vineyard.cli",
)


def _loaded_heavy(modules: tuple[str, ...], extra: str = "") -> str:
    imports = "; ".join(f"import {m}" for m in modules)
    code = f"import sys; {imports}; {extra}print(sorted(m for m in ('torch', 'open_clip', 'transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_torch_free_nn_modules() -> None:
    assert _loaded_heavy(TORCH_FREE) == "[]"


def test_every_pipeline_stage_imports_without_torch() -> None:
    load_all = (
        "from vineyard.pipeline.registry import ALL_STAGES, load_stage, stage_available; "
        "[load_stage(n) for n in ALL_STAGES if stage_available(n)]; "
    )
    assert _loaded_heavy(("vineyard.pipeline.registry",), load_all) == "[]"
