"""Shared fixtures. The full pipeline runs once per session so the
integration tests all read from the same measured result."""

from __future__ import annotations

import pytest

from fdo.pipeline import run_pipeline


@pytest.fixture(scope="session")
def results() -> dict:
    return run_pipeline(seed=7)
