"""Shared fixtures.

The only interesting one is :func:`matched_aero`; ``matched_coefficients.py``
explains at length why it exists and why no frozen-engine comparison should use
``naval_5in38_coefficients()`` instead.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

from matched_coefficients import coefficients_from_frozen  # noqa: E402,F401
from sixdof.paths import AERO_SOURCE_5IN38  # noqa: E402


@pytest.fixture(scope="session")
def frozen_aero():
    """The frozen engine's coefficient object, built from the source workbook."""
    import motor_original as frozen

    with contextlib.redirect_stdout(io.StringIO()):
        return frozen.RealAerodynamicCoefficients(str(AERO_SOURCE_5IN38))


@pytest.fixture(scope="session")
def matched_aero(frozen_aero):
    """The package's coefficients, carrying the frozen engine's own grids.

    Use this -- never ``naval_5in38_coefficients()`` -- on the package side of
    any comparison against the frozen engine.  See this module's docstring.
    """
    return coefficients_from_frozen(frozen_aero)
