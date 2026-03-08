"""Shared pytest fixtures for SynthDoc test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    """A deterministic NumPy RNG for reproducible tests."""
    return np.random.default_rng(42)
