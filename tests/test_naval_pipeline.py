"""The naval-drone pipeline — the case the thesis is actually about.

The anti-air layer is newer work; the thesis proper is surface fire against
naval drones, and that runs through the four Monte Carlo stages.  This file
checks them against the artefacts the thesis published.

One result here is worth stating plainly: the published tables are **not**
bit-reproducible on a different machine, and that is a property of the original
code, not of the refactor.  ``test_published_sweep_differs_from_both_engines_alike``
pins that down — the frozen engine and the package agree with each other
exactly, and both differ from the published workbook by about the integrator's
own tolerance.  See the module docstring of ``scripts/proof_of_equivalence.py``
and the README section on reproducibility.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

import motor_original as frozen_engine  # noqa: E402
from compat import load_repl_function, patch_legacy_coefficients  # noqa: E402

patch_legacy_coefficients(frozen_engine)

from sixdof import (  # noqa: E402
    BallisticSimulator,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.montecarlo import expected_engagement_cost  # noqa: E402
from sixdof.paths import AERO_SOURCE_5IN38, OPTIMAL_AZIMUTHS  # noqa: E402

CUSTO_PASTE = REPO_ROOT / "tests" / "reference" / "custo_original.py.txt"

#: The integrator runs at rtol=1e-7, so agreement with a table produced on
#: another machine cannot be asked for beyond that.
INTEGRATOR_RTOL = 1e-7

#: Empirical bound on how far the published workbook sits from this machine,
#: measured over the whole envelope: median ~1e-9, p90 ~1e-8, worst ~1.6e-7,
#: and never more than a millimetre in absolute terms.  Anything beyond this
#: would be a modelling difference, not floating-point drift.
PUBLISHED_TABLE_RTOL = 1e-6

#: Worst absolute range disagreement with the published workbook, in metres.
PUBLISHED_TABLE_ABS_M = 0.005

#: Cost parameters hard-coded in the original Custo.py.
ROUND_COST = 2000
TARGET_VALUE = 289_000_000


@pytest.fixture(scope="module")
def coefficients():
    return naval_5in38_coefficients()


@pytest.fixture(scope="module")
def simulator(coefficients):
    return BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(), standard_atmosphere(), coefficients
    )


@pytest.fixture(scope="module")
def frozen_simulator():
    with contextlib.redirect_stdout(io.StringIO()):
        coefficients = frozen_engine.RealAerodynamicCoefficients(str(AERO_SOURCE_5IN38))
        projectile = frozen_engine.Projectile.from_imperial(
            'Projétil Naval 5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0
        )
        weapon = frozen_engine.Weapon(
            position=(0.0, 10.0, 0.0), elevation_deg=45.0, azimuth_deg=0.0,
            muzzle_velocity_mps=807.0,
        )
        return frozen_engine.BallisticSimulator(
            projectile, weapon, frozen_engine.Environment(), coefficients
        )


@pytest.fixture(scope="module")
def published_sweep():
    return pd.read_excel(OPTIMAL_AZIMUTHS)


def _sample_rows(sweep, elevations):
    return [sweep[np.isclose(sweep["Elevacao_deg"], e)].iloc[0] for e in elevations]


def _run_frozen(frozen_simulator, elevation, azimuth):
    frozen_simulator.weapon.set_firing_angles(
        elevation_deg=elevation, azimuth_deg=azimuth
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return frozen_simulator.simulate(
            max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0, w_j0=5.0, w_k0=5.0
        )


def _run_refactored(simulator, elevation, azimuth):
    simulator.weapon.set_firing_angles(elevation, azimuth)
    return simulator.simulate(
        max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0, w_j0=5.0, w_k0=5.0, verbose=False
    )


# ----------------------------------------------------------------------
# stage 1 -- the angle sweep
# ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("elevation", [39.6, 30.0, 20.0, 10.0, 5.0])
def test_sweep_row_identical_between_engines(
    simulator, frozen_simulator, published_sweep, elevation
):
    """On the sweep's own aim points, the two engines agree bit for bit."""
    row = _sample_rows(published_sweep, [elevation])[0]
    azimuth = float(row["Azimute_otimo_deg"])

    old = _run_frozen(frozen_simulator, elevation, azimuth)
    new = _run_refactored(simulator, elevation, azimuth)

    assert np.array_equal(old.t, new.t)
    assert np.array_equal(old.solution.y, new.solution.y)
    assert old.alcance_max == new.max_range
    assert old.tempo_voo == new.flight_time


@pytest.mark.slow
@pytest.mark.parametrize("elevation", [39.6, 30.0, 20.0, 10.0, 5.0])
def test_published_sweep_differs_from_both_engines_alike(
    simulator, frozen_simulator, published_sweep, elevation
):
    """The published table drifts from *both* engines by the same tiny amount.

    This is the test that assigns blame correctly.  If the refactor had changed
    the physics, the package would differ from the frozen engine — it does not.
    What differs is the workbook, produced years ago on Windows with older
    SciPy/NumPy: an adaptive integrator lands on a different but equally valid
    step sequence when the platform's ``sin``/``cos`` differ in the last bit.
    """
    row = _sample_rows(published_sweep, [elevation])[0]
    azimuth = float(row["Azimute_otimo_deg"])
    published_range = float(row["Alcance_x_m"])

    old = _run_frozen(frozen_simulator, elevation, azimuth)
    new = _run_refactored(simulator, elevation, azimuth)

    # The two engines agree with each other exactly...
    assert old.alcance_max == new.max_range

    # ...and both sit the same distance from the published value.
    drift_old = abs(old.alcance_max - published_range) / published_range
    drift_new = abs(new.max_range - published_range) / published_range
    assert drift_old == drift_new

    # That distance is within the integrator's own tolerance.
    assert drift_new < INTEGRATOR_RTOL


@pytest.mark.slow
def test_published_sweep_agrees_across_the_envelope(simulator, published_sweep):
    """Across the envelope the package tracks the published sweep to under a mm.

    Typical agreement is ~1e-9 relative; the worst cases sit near the
    integrator's own rtol of 1e-7 and are always sub-millimetre.  A real
    modelling change would show up orders of magnitude above this.
    """
    sample = published_sweep.iloc[::75]
    relative, absolute = [], []
    for _, row in sample.iterrows():
        trajectory = _run_refactored(
            simulator, float(row["Elevacao_deg"]), float(row["Azimute_otimo_deg"])
        )
        published = float(row["Alcance_x_m"])
        absolute.append(abs(trajectory.max_range - published))
        relative.append(absolute[-1] / max(abs(published), 1.0))

    relative, absolute = np.array(relative), np.array(absolute)
    assert relative.max() < PUBLISHED_TABLE_RTOL
    assert absolute.max() < PUBLISHED_TABLE_ABS_M
    assert np.median(relative) < 1e-7


# ----------------------------------------------------------------------
# stage 4 -- engagement cost
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def original_cost_function():
    """The cost function recovered from the frozen Custo.py IDLE paste."""
    return load_repl_function(
        CUSTO_PASTE, "calcular_valor_esperado", {"np": np, "pd": pd}
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.5],
        [0.063, 0.029, 0.024, 0.021],
        [0.0] * 5,
        [1.0, 0.5, 0.5],
        list(np.linspace(0.01, 0.3, 40)),
    ],
)
def test_cost_matches_the_original(original_cost_function, probabilities):
    """Every field of the cost result matches the original function exactly."""
    old = original_cost_function(np.asarray(probabilities), ROUND_COST, TARGET_VALUE)
    new = expected_engagement_cost(probabilities, ROUND_COST, TARGET_VALUE).to_dict()

    assert old.keys() == new.keys()
    for key in old:
        assert old[key] == new[key], key


def test_cost_curve_matches_the_original(original_cost_function):
    """The whole E[cost] vs allocated-rounds curve matches, not just the endpoint."""
    probabilities = np.linspace(0.02, 0.25, 60)
    for n in range(1, len(probabilities) + 1):
        old = original_cost_function(probabilities[:n], ROUND_COST, TARGET_VALUE)
        new = expected_engagement_cost(probabilities[:n], ROUND_COST, TARGET_VALUE)
        assert old["custo_total_esperado"] == new.total_expected_cost
        assert old["num_esperado_disparos"] == new.expected_rounds
        assert old["prob_falha_total"] == new.total_failure_probability


def test_repl_paste_recovers_real_source():
    """The de-REPL step must produce parseable source, not a mangled file."""
    from compat import strip_repl_prompts

    source = strip_repl_prompts(CUSTO_PASTE.read_text(encoding="utf-8", errors="replace"))
    assert "def calcular_valor_esperado" in source

    # No line may still carry a continuation prompt.  ("... " inside the
    # docstring's formula is prose, not a prompt, so only line starts count.)
    assert not [line for line in source.split("\n") if line.startswith(("... ", ">>> "))]

    import ast

    ast.parse(source)  # must not raise
