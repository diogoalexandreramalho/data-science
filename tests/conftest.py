"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repo root path (= parent of tests/)."""
    return Path(__file__).resolve().parents[1]
