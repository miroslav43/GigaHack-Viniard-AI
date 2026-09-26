"""Fixtures of the waste tests (helpers live in tests/waste/synth_waste.py)."""

from __future__ import annotations

import pytest

from tests.waste.synth_waste import MakeCandidate, make_candidate
from vineyard.config import AppConfig, load_config


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    # These tests exercise the ML verification path; the default pipeline run keeps SAM 3 off.
    return load_config(overrides=("waste.sam3.enabled=true",))


@pytest.fixture
def cand() -> MakeCandidate:
    return make_candidate
