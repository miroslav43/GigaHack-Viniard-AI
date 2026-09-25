"""Fixtures of the waste tests (helpers live in tests/waste/synth_waste.py)."""

from __future__ import annotations

import pytest

from tests.waste.synth_waste import MakeCandidate, make_candidate
from vineyard.config import AppConfig, load_config


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def cand() -> MakeCandidate:
    return make_candidate
