"""Session-scoped fixtures shared across all test files.

Production training uses 100 seed fits, which is intentionally expensive. The
test fixture uses the explicit compact test configuration and is session-scoped
so ordinary regression checks remain practical.
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
    """Train a compact validation model once; production defaults remain 100 seeds/5 folds."""
    return fit_credit_model(credit_data, stability_n_seeds=5, cross_validation_folds=2)
