"""Filesystem layout shared by every fte module (all paths absolute, spaces-safe)."""

from __future__ import annotations

from pathlib import Path

FTE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FTE_ROOT.parents[1]
AI_ROOT = REPO_ROOT / "src" / "AI"
AI_WORK = AI_ROOT / "work"
AI_RUNS = AI_WORK / "runs"
AI_TILES = AI_WORK / "tiles"
AI_CACHE = AI_WORK / "cache"
WORK = FTE_ROOT / "work"
RAW = WORK / "raw"
DATA = WORK / "data"
STORES = WORK / "stores"
PREDS = WORK / "preds"
LOGS = WORK / "logs"
MODELS = FTE_ROOT / "models"
EXAMPLE_TILES = ("siret3_r006_c004", "siret3_r021_c012")
BASE_RUN = "complete-v4"
