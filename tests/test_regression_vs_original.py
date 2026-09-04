"""The refactor must reproduce the original engine bit for bit.

``tests/reference/motor_original.py`` is a frozen, unmodified copy of the
single-file ``Motor.py`` from the ``legacy`` branch.  These tests build the same
scenario with both engines and compare the integrator output exactly -- not to a
tolerance.  Any change to the equations of motion, to the coefficient grid or to
the integrator settings breaks them, which is the point: this file is what lets
the refactored package inherit the original's verification.

Run with ``pytest tests/`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

import motor_original as legacy  # noqa: E402
from compat import patch_legacy_coefficients  # noqa: E402

# The reference file is frozen at the upstream bytes, so on NumPy 2.x it needs a
# plumbing-only shim to run at all; see tests/reference/compat.py.
LEGACY_PATCHED = patch_legacy_coefficients(legacy)

from sixdof import (  # noqa: E402
    BallisticSimulator,
    Environment,
    naval_5in38_coefficients,
    Projectile,
    Weapon,
)
from sixdof.paths import AERO_SOURCE_5IN38  # noqa: E402

#: The seven coefficients the equations read, as keyed inside the right-hand side.
RHS_KEYS = ("CD_total", "CLA_total", "CNP_total", "CYP", "CLP", "CMA", "CMQ")

#: The scenario of the original ``Exemplo.py``: 5"/38 ashore, 43.3 deg, no wind.
SCENARIO = dict(
    elevation_deg=43.3,
    azimuth_deg=0.0,
    height_m=10.0,
    muzzle_velocity_mps=807.0,
    max_time=100.0,
    alpha0_deg=0.0,
    beta0_deg=0.0,
    w_j0=5.0,
    w_k0=5.0,
    rtol=1e-7,
    atol=1e-8,
)


@pytest.fixture(scope="module")
def legacy_coefficients():
    return legacy.RealAerodynamicCoefficients(str(AERO_SOURCE_5IN38))


@pytest.fixture(scope="module")
def refactored_coefficients():
    return naval_5in38_coefficients()


def _legacy_run(coefficients, **overrides):
    scenario = {**SCENARIO, **overrides}
    projectile = legacy.Projectile.from_imperial(
        name='Projétil Naval 5"/38',
        mass_lb=68.10,
        diameter_in=5.0,
        I_P_lbin2=240.9,
        I_T_lbin2=2619.0,
        rifling_twist_calibers=25.0,
    )
    weapon = legacy.Weapon(
        name='Canhão Naval 5"/38',
        position=(0.0, scenario["height_m"], 0.0),
        elevation_deg=scenario["elevation_deg"],
        azimuth_deg=scenario["azimuth_deg"],
        rate_of_fire_rpm=15.0,
        muzzle_velocity_mps=scenario["muzzle_velocity_mps"],
        mounted_on_vessel=None,
    )
    environment = legacy.Environment(rho=1.225, g=9.81, W1=0.0, W2=0.0, W3=0.0)
    simulator = legacy.BallisticSimulator(projectile, weapon, environment, coefficients)
    return simulator.simulate(
        max_time=scenario["max_time"],
        alpha0_deg=scenario["alpha0_deg"],
        beta0_deg=scenario["beta0_deg"],
        w_j0=scenario["w_j0"],
        w_k0=scenario["w_k0"],
        rtol=scenario["rtol"],
        atol=scenario["atol"],
    )


def _refactored_run(coefficients, **overrides):
    scenario = {**SCENARIO, **overrides}
    projectile = Projectile.from_imperial(
        name='Projétil Naval 5"/38',
        mass_lb=68.10,
        diameter_in=5.0,
        I_P_lbin2=240.9,
        I_T_lbin2=2619.0,
        rifling_twist_calibers=25.0,
    )
    weapon = Weapon(
        name='Canhão Naval 5"/38',
        position=(0.0, scenario["height_m"], 0.0),
        elevation_deg=scenario["elevation_deg"],
        azimuth_deg=scenario["azimuth_deg"],
        rate_of_fire_rpm=15.0,
        muzzle_velocity_mps=scenario["muzzle_velocity_mps"],
        mounted_on_vessel=None,
    )
    environment = Environment(rho=1.225, g=9.81, W1=0.0, W2=0.0, W3=0.0)
    simulator = BallisticSimulator(projectile, weapon, environment, coefficients)
    return simulator.simulate(
        max_time=scenario["max_time"],
        alpha0_deg=scenario["alpha0_deg"],
        beta0_deg=scenario["beta0_deg"],
        w_j0=scenario["w_j0"],
        w_k0=scenario["w_k0"],
        rtol=scenario["rtol"],
        atol=scenario["atol"],
        verbose=False,
    )


# ----------------------------------------------------------------------
# coefficient interpolation
# ----------------------------------------------------------------------
def test_coefficients_agree_over_the_whole_grid(legacy_coefficients, refactored_coefficients):
    """Every node of the pre-computed grid returns the same seven values.

    This compares the *contract* -- what the equations get asked for -- rather
    than the internals, which is the point of the refactor: the package no
    longer carries the source table's intermediate columns at all, so there are
    no internal grids to line up.  10 000 nodes, exact equality.
    """
    mach_grid = refactored_coefficients.mach_grid
    alpha_grid = refactored_coefficients.alpha_grid
    assert (len(mach_grid), len(alpha_grid)) == (100, 100)

    for mach in mach_grid:
        for alpha in alpha_grid:
            old = legacy_coefficients.get_coefficients(mach, alpha)
            new = refactored_coefficients.get_coefficients(mach, alpha)
            for key in RHS_KEYS:
                assert old[key] == new[key], f"{key} at mach={mach}, alpha={alpha}"


@pytest.mark.parametrize(
    "mach, alpha_deg",
    [(0.5, 0.0), (0.9, 1.5), (1.0, -2.0), (1.8, 3.25), (2.4, -7.0), (5.0, 9.9)],
)
def test_coefficient_lookup_is_identical(
    legacy_coefficients, refactored_coefficients, mach, alpha_deg
):
    """Point lookups must agree exactly across the transonic and supersonic range."""
    alpha = np.radians(alpha_deg)
    old = legacy_coefficients.get_coefficients(mach, alpha)
    new = refactored_coefficients.get_coefficients(mach, alpha)
    # The frozen engine also returns CNA, which the equations never read; the
    # package does not carry it.  Compare what the right-hand side asks for.
    for key in RHS_KEYS:
        assert old[key] == new[key], f"{key} differs at mach={mach}, alpha={alpha_deg}"


# ----------------------------------------------------------------------
# initial conditions and right-hand side
# ----------------------------------------------------------------------
def test_initial_state_is_identical(legacy_coefficients, refactored_coefficients):
    """The launch state must match before any integration happens."""
    legacy_projectile = legacy.Projectile.from_imperial(
        '5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0
    )
    legacy_weapon = legacy.Weapon(position=(0.0, 10.0, 0.0), elevation_deg=43.3, azimuth_deg=0.0)
    legacy_sim = legacy.BallisticSimulator(
        legacy_projectile, legacy_weapon, legacy.Environment(), legacy_coefficients
    )

    projectile = Projectile.from_imperial('5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0)
    weapon = Weapon(position=(0.0, 10.0, 0.0), elevation_deg=43.3, azimuth_deg=0.0)
    simulator = BallisticSimulator(
        projectile, weapon, Environment(), refactored_coefficients
    )

    old = legacy_sim.build_initial_conditions(0.0, 0.0, 5.0, 5.0)
    new = simulator.build_initial_conditions(0.0, 0.0, 5.0, 5.0)
    assert np.array_equal(old, new)


def test_rhs_is_identical(legacy_coefficients, refactored_coefficients):
    """The derivative must match exactly at the launch state and downrange."""
    legacy_projectile = legacy.Projectile.from_imperial(
        '5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0
    )
    legacy_weapon = legacy.Weapon(position=(0.0, 10.0, 0.0), elevation_deg=43.3, azimuth_deg=0.0)
    legacy_sim = legacy.BallisticSimulator(
        legacy_projectile, legacy_weapon, legacy.Environment(), legacy_coefficients
    )

    projectile = Projectile.from_imperial('5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0)
    weapon = Weapon(position=(0.0, 10.0, 0.0), elevation_deg=43.3, azimuth_deg=0.0)
    simulator = BallisticSimulator(
        projectile, weapon, Environment(), refactored_coefficients
    )

    y0 = legacy_sim.build_initial_conditions(0.0, 0.0, 5.0, 5.0)
    for t in (0.0, 3.7, 21.0):
        state = y0.copy()
        state[9] += 100.0 * t  # nudge downrange so the states differ between calls
        assert np.array_equal(legacy_sim.rhs(t, state), simulator.rhs(t, state))


# ----------------------------------------------------------------------
# whole trajectories
# ----------------------------------------------------------------------
def test_trajectory_is_bit_identical(legacy_coefficients, refactored_coefficients):
    """Same sample times and same state history, to the last bit."""
    old = _legacy_run(legacy_coefficients)
    new = _refactored_run(refactored_coefficients)

    assert old.solution.success and new.solution.success
    assert np.array_equal(old.t, new.t)
    assert np.array_equal(old.solution.y, new.solution.y)


def test_derived_quantities_are_identical(legacy_coefficients, refactored_coefficients):
    """Speed, Mach, spin and angle of attack histories must match exactly."""
    old = _legacy_run(legacy_coefficients)
    new = _refactored_run(refactored_coefficients)

    assert np.array_equal(old.V_mag, new.V_mag)
    assert np.array_equal(old.mach, new.mach)
    assert np.array_equal(old.h_mag, new.h_mag)
    assert np.array_equal(old.spin_rate, new.spin_rate)
    assert np.array_equal(old.alpha_traj, new.alpha_traj)


def test_summary_statistics_are_identical(legacy_coefficients, refactored_coefficients):
    """The four headline numbers, and the Portuguese aliases that expose them."""
    old = _legacy_run(legacy_coefficients)
    new = _refactored_run(refactored_coefficients)

    assert old.alcance_max == new.max_range == new.alcance_max
    assert old.altura_max == new.max_altitude == new.altura_max
    assert old.desvio_lateral_max == new.max_lateral_drift == new.desvio_lateral_max
    assert old.tempo_voo == new.flight_time == new.tempo_voo


@pytest.mark.parametrize(
    "elevation_deg, azimuth_deg",
    [(39.6, -1.35), (20.0, -1.0), (5.0, 0.0), (-1.5, -0.5), (45.0, -1.65)],
)
def test_trajectory_identical_across_the_firing_envelope(
    legacy_coefficients, refactored_coefficients, elevation_deg, azimuth_deg
):
    """Equality must hold across the whole sweep, not just at one aim point."""
    old = _legacy_run(legacy_coefficients, elevation_deg=elevation_deg, azimuth_deg=azimuth_deg)
    new = _refactored_run(
        refactored_coefficients, elevation_deg=elevation_deg, azimuth_deg=azimuth_deg
    )
    assert np.array_equal(old.t, new.t)
    assert np.array_equal(old.solution.y, new.solution.y)


def test_moving_platform_is_identical(legacy_coefficients, refactored_coefficients):
    """A gun on a moving hull inherits the platform velocity the same way."""
    legacy_vessel = legacy.Vessel(
        name="plataforma", center_position=(0.0, 0.0), length=100.0, width=20.0,
        height=30.0, velocity=(8.0, -2.0),
    )
    legacy_projectile = legacy.Projectile.from_imperial(
        '5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0
    )
    legacy_weapon = legacy.Weapon(
        position=(5.0, 10.0, 1.0), elevation_deg=30.0, azimuth_deg=-1.0,
        mounted_on_vessel=legacy_vessel,
    )
    legacy_sim = legacy.BallisticSimulator(
        legacy_projectile, legacy_weapon, legacy.Environment(), legacy_coefficients
    )
    old = legacy_sim.simulate(max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0, w_j0=5.0, w_k0=5.0)

    from sixdof import Vessel

    vessel = Vessel(
        name="plataforma", center_position=(0.0, 0.0), length=100.0, width=20.0,
        height=30.0, velocity=(8.0, -2.0),
    )
    projectile = Projectile.from_imperial('5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0)
    weapon = Weapon(
        position=(5.0, 10.0, 1.0), elevation_deg=30.0, azimuth_deg=-1.0,
        mounted_on_vessel=vessel,
    )
    simulator = BallisticSimulator(
        projectile, weapon, Environment(), refactored_coefficients
    )
    new = simulator.simulate(
        max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0, w_j0=5.0, w_k0=5.0, verbose=False
    )

    assert np.array_equal(old.t, new.t)
    assert np.array_equal(old.solution.y, new.solution.y)


def test_wind_is_identical(legacy_coefficients, refactored_coefficients):
    """A non-zero wind vector reaches the aerodynamics identically."""
    legacy_projectile = legacy.Projectile.from_imperial(
        '5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0
    )
    legacy_weapon = legacy.Weapon(position=(0.0, 10.0, 0.0), elevation_deg=35.0, azimuth_deg=0.0)
    legacy_env = legacy.Environment(rho=1.2, g=9.80665, W1=6.0, W2=0.0, W3=-4.0)
    legacy_sim = legacy.BallisticSimulator(
        legacy_projectile, legacy_weapon, legacy_env, legacy_coefficients
    )
    old = legacy_sim.simulate(max_time=100.0)

    projectile = Projectile.from_imperial('5"/38', 68.10, 5.0, 240.9, 2619.0, 25.0)
    weapon = Weapon(position=(0.0, 10.0, 0.0), elevation_deg=35.0, azimuth_deg=0.0)
    environment = Environment(rho=1.2, g=9.80665, W1=6.0, W2=0.0, W3=-4.0)
    simulator = BallisticSimulator(
        projectile, weapon, environment, refactored_coefficients
    )
    new = simulator.simulate(max_time=100.0, verbose=False)

    assert np.array_equal(old.t, new.t)
    assert np.array_equal(old.solution.y, new.solution.y)
