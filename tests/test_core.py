"""Unit tests for the engine's building blocks.

These are the checks the single-file version had no place for: unit
conversions, the laying-angle transform, platform coupling, the terminal
events, the result container's API and a headless smoke test of the plotter.
"""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixdof import (  # noqa: E402
    AerodynamicCoefficients,
    BallisticSimulator,
    Environment,
    IntegrationSettings,
    load_coefficients,
    make_ground_event,
    make_proximity_fuze_event,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    Projectile,
    standard_atmosphere,
    surface_target_fleet,
    Vessel,
    Weapon,
)
from sixdof.aa import ProximityFuze, shahed_136  # noqa: E402
from sixdof.paths import AERO_COEFFICIENTS_5IN38  # noqa: E402


@pytest.fixture(scope="module")
def coefficients():
    return naval_5in38_coefficients()


# ----------------------------------------------------------------------
# projectile
# ----------------------------------------------------------------------
def test_imperial_conversion():
    """68.10 lb / 5 in converts to the SI values used by the equations."""
    projectile = naval_5in38_projectile()
    assert projectile.mass == pytest.approx(30.89, abs=0.01)
    assert projectile.diameter == pytest.approx(0.127, abs=1e-9)
    assert projectile.S == pytest.approx(pi * 0.0635**2, abs=1e-15)
    assert projectile.I_P < projectile.I_T  # gyroscopically stabilised, not a rod
    assert projectile.inertia_ratio == pytest.approx(240.9 / 2619.0, abs=1e-12)


def test_initial_spin_from_rifling():
    """One turn in n calibres gives ``p0 = 2 pi V0 / (n d)``."""
    projectile = naval_5in38_projectile()
    spin = projectile.calculate_initial_spin(807.0)
    assert spin == pytest.approx(2 * pi * 807.0 / (25.0 * 0.127))
    assert projectile.calculate_initial_spin(1614.0) == pytest.approx(2 * spin)


def test_reference_area_follows_the_diameter():
    """``S`` is recomputed from the calibre, not carried independently."""
    assert Projectile(diameter=0.1).S == pytest.approx(pi * 0.05**2)
    assert Projectile(diameter=0.0).S == 0.0


# ----------------------------------------------------------------------
# weapon
# ----------------------------------------------------------------------
def test_firing_angles_at_zero_azimuth():
    """With no traverse, theta is zero and phi is the elevation."""
    weapon = naval_5in38_gun(elevation_deg=43.3, azimuth_deg=0.0)
    theta0, phi0 = weapon.calculate_firing_angles()
    assert theta0 == pytest.approx(0.0, abs=1e-15)
    assert np.degrees(phi0) == pytest.approx(43.3, abs=1e-12)


def test_firing_angles_with_traverse():
    """A traverse to the right produces a positive out-of-plane angle."""
    weapon = naval_5in38_gun(elevation_deg=0.0, azimuth_deg=10.0)
    theta0, phi0 = weapon.calculate_firing_angles()
    assert np.degrees(theta0) == pytest.approx(10.0, abs=1e-12)
    assert phi0 == pytest.approx(0.0, abs=1e-12)


def test_set_firing_angles_round_trips():
    """Re-laying the gun updates both stored angles."""
    weapon = naval_5in38_gun()
    weapon.set_firing_angles(elevation_deg=12.5, azimuth_deg=-1.35)
    assert weapon.elevation_deg == pytest.approx(12.5)
    assert weapon.azimuth_deg == pytest.approx(-1.35)


def test_land_mount_has_no_platform_velocity():
    """A gun ashore contributes nothing to the muzzle velocity."""
    weapon = naval_5in38_gun()
    assert np.array_equal(weapon.get_velocity(), np.zeros(3))
    assert np.array_equal(weapon.get_absolute_position(), np.array([0.0, 10.0, 0.0]))


def test_shipborne_mount_inherits_hull_motion():
    """A mounted gun picks up the hull's horizontal velocity and offset."""
    vessel = Vessel(center_position=(1000.0, -50.0), length=100.0, width=20.0,
                    height=30.0, velocity=(8.0, -2.0))
    weapon = Weapon(position=(10.0, 12.0, 3.0), mounted_on_vessel=vessel)

    assert np.allclose(weapon.get_velocity(), [8.0, 0.0, -2.0])
    assert np.allclose(weapon.get_absolute_position(0.0), [1010.0, 12.0, -47.0])
    # The hull has moved after 10 s, and the mount moves with it.
    assert np.allclose(weapon.get_absolute_position(10.0), [1090.0, 12.0, -67.0])


# ----------------------------------------------------------------------
# vessel
# ----------------------------------------------------------------------
def test_hull_bounds_and_impact():
    """The bounding box is centred on the hull and impacts are tested against it."""
    vessel = Vessel(center_position=(100.0, 0.0), length=10.0, width=4.0, height=3.0)
    bounds = vessel.get_bounds()
    assert (bounds["x_min"], bounds["x_max"]) == (95.0, 105.0)
    assert (bounds["z_min"], bounds["z_max"]) == (-2.0, 2.0)

    assert vessel.check_impact([100.0, 1.0, 0.0]) is True
    assert vessel.check_impact([100.0, 5.0, 0.0]) is False  # above the deck
    assert vessel.check_impact([100.0, 5.0, 0.0], check_height=False) is True
    assert vessel.check_impact([120.0, 1.0, 0.0]) is False


def test_moving_hull_bounds_advance():
    """A hull under way is scored where it is at impact time, not where it started."""
    vessel = Vessel(center_position=(0.0, 0.0), length=10.0, width=4.0, velocity=(10.0, 0.0))
    assert vessel.check_impact([100.0, 1.0, 0.0], time=10.0) is True
    assert vessel.check_impact([100.0, 1.0, 0.0], time=0.0) is False
    assert vessel.center_at(10.0) == (100.0, 0.0)


def test_target_fleet_shares_one_aim_point():
    """Every hull in the fleet is centred on the same nominal impact point."""
    fleet = surface_target_fleet((16673.0, 4.2))
    assert set(fleet) == {
        "Drone_Sea_Baby", "IRIS_Paykan", "Osa_class",
        "Hayabusa_class", "SMS_V4", "PT_105",
    }
    for vessel in fleet.values():
        assert np.allclose(vessel.center, [16673.0, 4.2])
    assert fleet["Drone_Sea_Baby"].length == 6.0
    assert fleet["SMS_V4"].length == 72.0


# ----------------------------------------------------------------------
# environment and coefficients
# ----------------------------------------------------------------------
def test_standard_atmosphere_values():
    environment = standard_atmosphere()
    assert (environment.rho, environment.g) == (1.225, 9.81)
    assert environment.wind == (0.0, 0.0, 0.0)
    assert environment.sound_speed == 340.0


def test_constant_coefficients_default_to_zero():
    """Unspecified coefficients are zero, so a drag-only case is one keyword."""
    coefficients = AerodynamicCoefficients(CD=0.3)
    values = coefficients.get_coefficients(2.0, 0.1)
    assert values["CD_total"] == 0.3
    assert values["CLA_total"] == 0.0
    assert values["CMA"] == 0.0


def test_coefficients_reject_source_specific_names():
    """Intermediate columns of a tabulation convention are not model inputs."""
    with pytest.raises(ValueError, match="unknown coefficient"):
        AerodynamicCoefficients(CX0=0.3)


def test_load_coefficients_dispatches_on_content():
    """The loader picks a reader from what is in the file, not from its name."""
    from sixdof.paths import AERO_WORKBOOK_5IN38

    for path in (AERO_COEFFICIENTS_5IN38, AERO_WORKBOOK_5IN38):
        loaded = load_coefficients(path)
        values = loaded.get_coefficients(2.0, 0.05)
        assert set(values) >= {"CD_total", "CLA_total", "CMA"}
        assert values["CD_total"] > 0.0


def test_coefficients_clip_outside_the_table(coefficients):
    """Requests beyond the tabulated envelope clamp instead of extrapolating."""
    assert coefficients.get_coefficients(-3.0) == coefficients.get_coefficients(
        coefficients.mach_min
    )
    assert coefficients.get_coefficients(99.0) == coefficients.get_coefficients(
        coefficients.mach_max
    )


def test_drag_rises_through_the_transonic_range(coefficients):
    """Sanity check on the table: drag peaks just above Mach 1."""
    subsonic = coefficients.get_coefficients(0.7)["CD_total"]
    transonic = coefficients.get_coefficients(1.1)["CD_total"]
    supersonic = coefficients.get_coefficients(3.0)["CD_total"]
    assert subsonic < transonic
    assert supersonic < transonic


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------
def test_ground_event_changes_sign_at_the_surface():
    event = make_ground_event()
    state = np.zeros(12)
    state[10] = 5.0
    assert event(0.0, state) > 0
    state[10] = -5.0
    assert event(0.0, state) < 0
    assert event.terminal is True
    assert event.direction == -1


def test_ground_event_respects_a_raised_surface():
    event = make_ground_event(ground_level=100.0)
    state = np.zeros(12)
    state[10] = 150.0
    assert event(0.0, state) > 0
    state[10] = 50.0
    assert event(0.0, state) < 0


def test_fuze_event_is_inert_before_arming():
    event = make_proximity_fuze_event((0.0, 0.0, 0.0), radius_m=10.0, arm_time_s=0.5)
    state = np.zeros(12)  # sitting exactly on the target
    assert event(0.1, state) == 1.0  # not armed yet
    assert event(1.0, state) == pytest.approx(-10.0)  # armed and well inside


# ----------------------------------------------------------------------
# simulation and results
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def trajectory(coefficients):
    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(elevation_deg=43.3),
        environment=standard_atmosphere(),
        aero_coeffs=coefficients,
    )
    return simulator.simulate(verbose=False)


def test_trajectory_is_physically_sane(trajectory):
    """A 43.3 degree shot arcs up, comes down, and slows through the flight."""
    assert trajectory.stop_reason == "ground"
    assert trajectory.max_range > 15_000
    assert trajectory.max_altitude > 3_000
    assert trajectory.flight_time > 50
    assert trajectory.y[-1] == pytest.approx(0.0, abs=1e-6)
    assert trajectory.impact_speed < 807.0
    assert trajectory.V_mag[0] == pytest.approx(807.0, abs=1e-9)


def test_axis_stays_a_unit_vector(trajectory):
    """``|i'|`` must stay at 1: the orientation equation preserves it."""
    norms = np.sqrt(trajectory.i1**2 + trajectory.i2**2 + trajectory.i3**2)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_spin_decays_but_stays_positive(trajectory):
    """Spin damping bleeds the rate off slowly; it never reverses in flight."""
    assert trajectory.spin_rate[0] > trajectory.spin_rate[-1] > 0
    assert trajectory.spin_rate[0] == pytest.approx(
        trajectory.projectile.calculate_initial_spin(807.0), rel=1e-6
    )


def test_trajectory_accessors(trajectory):
    """Index accessors agree with the raw arrays, and the summary is complete."""
    assert np.array_equal(
        trajectory.position_at(3), [trajectory.x[3], trajectory.y[3], trajectory.z[3]]
    )
    assert np.array_equal(
        trajectory.velocity_at(3), [trajectory.V1[3], trajectory.V2[3], trajectory.V3[3]]
    )
    assert np.array_equal(
        trajectory.axis_at(3), [trajectory.i1[3], trajectory.i2[3], trajectory.i3[3]]
    )
    assert len(trajectory.state_at(0)) == 12
    assert len(trajectory) == len(trajectory.t)
    assert set(trajectory.summary()) == {
        "max_range_m", "max_altitude_m", "max_lateral_drift_m", "flight_time_s",
        "impact_speed_mps", "alpha_min_deg", "alpha_max_deg", "alpha_mean_deg",
    }


def test_legacy_aliases_agree(trajectory):
    """Portuguese attribute names still resolve, for scripts from the old branch."""
    assert trajectory.alcance_max == trajectory.max_range
    assert trajectory.altura_max == trajectory.max_altitude
    assert trajectory.desvio_lateral_max == trajectory.max_lateral_drift
    assert trajectory.tempo_voo == trajectory.flight_time


def test_higher_elevation_flies_longer(coefficients):
    """Below the maximum-range angle, more elevation means more time of flight."""
    times = []
    for elevation in (10.0, 20.0, 30.0):
        simulator = BallisticSimulator(
            naval_5in38_projectile(), naval_5in38_gun(elevation_deg=elevation),
            standard_atmosphere(), coefficients,
        )
        times.append(simulator.simulate(verbose=False).flight_time)
    assert times == sorted(times)


def test_headwind_shortens_the_shot(coefficients):
    """A headwind raises the air-relative speed and cuts the range."""
    still = BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(elevation_deg=30.0),
        Environment(), coefficients,
    ).simulate(verbose=False)
    headwind = BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(elevation_deg=30.0),
        Environment(W1=-20.0), coefficients,
    ).simulate(verbose=False)
    assert headwind.max_range < still.max_range


def test_custom_integration_settings_are_used(coefficients):
    """Overriding the settings changes the sampling without breaking the flight."""
    simulator = BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(elevation_deg=20.0),
        standard_atmosphere(), coefficients,
        settings=IntegrationSettings(rtol=1e-6, atol=1e-7, max_step=0.5),
    )
    coarse = simulator.simulate(verbose=False)
    fine = simulator.simulate(rtol=1e-9, atol=1e-10, max_step=0.05, verbose=False)
    assert len(coarse) < len(fine)
    assert coarse.max_range == pytest.approx(fine.max_range, rel=1e-4)


def test_fuze_stops_the_integration(coefficients):
    """Passing a fuze adds a terminal event and reports it as the stop reason."""
    target = shahed_136(center=(16673.0, 200.0, 0.7))
    simulator = BallisticSimulator(
        naval_5in38_projectile(),
        naval_5in38_gun(elevation_deg=39.6, azimuth_deg=-1.35),
        standard_atmosphere(),
        coefficients,
    )
    with_fuze = simulator.simulate(fuze=ProximityFuze(target_center=target.center),
                                   verbose=False)
    assert with_fuze.stop_reason == "fuze"
    assert with_fuze.y[-1] > 0.0  # bursts in the air, not at the water line

    without_fuze = simulator.simulate(verbose=False)
    assert without_fuze.stop_reason == "ground"
    assert without_fuze.flight_time > with_fuze.flight_time


def test_max_time_stop_reason(coefficients):
    """A horizon shorter than the flight ends the run with ``max_time``."""
    simulator = BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(elevation_deg=43.3),
        standard_atmosphere(), coefficients,
    )
    short = simulator.simulate(max_time=5.0, verbose=False)
    assert short.stop_reason == "max_time"
    assert short.flight_time == pytest.approx(5.0)


# ----------------------------------------------------------------------
# plotting (headless)
# ----------------------------------------------------------------------
def test_plotter_writes_every_figure(trajectory, tmp_path):
    """The full figure set is produced with the original file names."""
    import matplotlib

    matplotlib.use("Agg")
    from sixdof.plotting import FIGURE_FILENAMES, TrajectoryPlotter

    plotter = TrajectoryPlotter(trajectory, output_dir=tmp_path, show=False, dpi=50)
    paths = plotter.plot_all()

    assert len(paths) == 18
    assert {path.name for path in paths} == set(FIGURE_FILENAMES.values())
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
