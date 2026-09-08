"""Session-scoped fixtures shared across all test files.

Training the full model (including 100-seed stability analysis) takes ~200s.
A session-scoped fixture ensures this cost is paid once per ``pytest`` run
rather than once per test file.
"""
from __future__ import annotations

import pytest

from app.modeling import fit_credit_model
from app.preprocessing import build_preprocessed_data


@pytest.fixture(scope="session")
def credit_data():
    """Build and return the preprocessed dataset (session-scoped)."""
    return build_preprocessed_data()


@pytest.fixture(scope="session")
def model_result(credit_data):
    """Train the full model with stability analysis (session-scoped)."""
    return fit_credit_model(credit_data)
