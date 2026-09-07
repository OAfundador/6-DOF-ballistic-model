"""Feed the frozen engine and the package the *same* coefficient arrays.

Why this file exists
--------------------

The suite's central claim is that the package reproduces the frozen engine
*exactly*.  Testing that honestly means feeding both sides the **same numbers**,
and until this file existed they were not.

The frozen engine reads ``data/aero_coefficients_5in38.xlsx`` and builds its
100x100 coefficient grid at import time, on whatever machine is running the
test.  The package loads ``data/aero_coefficients_5in38_spin73.npz``, a bake of
that same grid that ships in the repository.  Same grid in principle -- but the
bake happened once, on one machine, and three of the seven coefficients are
built with trigonometry::

    sin_alpha_mesh = np.sin(alpha_mesh)
    cos_alpha_mesh = np.cos(alpha_mesh)
    CD_total  = CX_total*cos_alpha_mesh - CNA_grid*sin_alpha_2_mesh
    CLA_total = CNA_grid*cos_alpha_mesh - CX_total
    CNP_total = CNPA_grid*sign(alpha) + CNPA3_grid*sin^3 + CNPA5_grid*sin^5

``sin`` and ``cos`` are libm calls, and libm implementations differ in the last
place between platforms -- glibc and the MSVC runtime disagree by up to one ULP.
So on a machine other than the one that produced the ``.npz``, the two sides get
inputs that differ in the last bit, ``solve_ivp`` picks a different but equally
valid step sequence, and a test that means to ask "is the physics the same?"
fails with 1594 samples against 1592.

That failure is real, but it is not about the refactor, and the test was not
measuring what it said it measured.  It was measuring two claims at once:

1. the package's physics equals the frozen engine's physics, and
2. the shipped ``.npz`` equals a fresh bake of the ``.xlsx``.

The first is exact and must hold everywhere.  The second is inherently
platform-dependent and cannot be an equality.  :func:`coefficients_from_frozen`
separates them: it hands the package the frozen engine's *own* arrays, so any
remaining difference is code rather than provenance.  Claim 2 keeps its own
test, with a tolerance and this explanation, in ``test_coefficients.py``.

The shipped ``.npz`` is a convenience: it saves rebuilding the grid on every
import and it is what ``naval_5in38_coefficients()`` returns for ordinary use.
Nothing about the physics depends on which of the two you load.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

from sixdof import AerodynamicCoefficients  # noqa: E402
from sixdof.paths import AERO_SOURCE_5IN38  # noqa: E402

#: How the frozen engine names the seven, against how the package does.
#: The frozen engine also carries the source table's intermediate columns
#: (``CX0``, ``CX2``, ``CNA``, the Magnus series); the package never sees them.
FROZEN_TO_PACKAGE = {
    "CD_total": "CD",
    "CLA_total": "CLA",
    "CNP_total": "CNP",
    "CYP": "CYP",
    "CLP": "CLP",
    "CMA": "CMA",
    "CMQ": "CMQ",
}


def coefficients_from_frozen(frozen) -> AerodynamicCoefficients:
    """Wrap a frozen engine's own grids in the package's coefficient object.

    Both sides then hold literally the same arrays -- the same objects' values,
    not a re-derivation of them -- so an exact-equality comparison between the
    engines tests the code and nothing else.  Portable to any platform, because
    nothing is recomputed on the way across.

    Parameters
    ----------
    frozen:
        A ``motor_original.RealAerodynamicCoefficients`` (or the anti-air
        engine's equivalent): anything exposing ``mach_grid``, ``alpha_grid``
        and a ``grid_2d`` dict keyed as :data:`FROZEN_TO_PACKAGE` expects.
    """
    grids = {
        package_name: frozen.grid_2d[frozen_name]
        for frozen_name, package_name in FROZEN_TO_PACKAGE.items()
    }
    return AerodynamicCoefficients(
        mach_grid=frozen.mach_grid,
        alpha_grid=frozen.alpha_grid,
        **grids,
    )
