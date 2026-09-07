"""The Monte Carlo pipeline must reproduce the thesis artefacts.

The strongest check here is :func:`test_selection_reproduces_thesis_table`: it
feeds the shipped zero-drift table through the refactored point selection and
compares the result, cell by cell, against the 163-point table the thesis
actually used.  The rest of the file covers the sweep helpers, the cost model
and the confidence intervals, plus a short end-to-end campaign that exercises
the real integrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixdof import (  # noqa: E402
    BallisticSimulator,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
    surface_target_fleet,
)
from sixdof.montecarlo import (  # noqa: E402
    AimPoint,
    AngleSweep,
    closest_approach,
    DispersionSettings,
    expected_engagement_cost,
    inclusive_range,
    margin_of_error,
    max_range_shot,
    MonteCarloCampaign,
    NearestApproach,
    optimal_azimuths,
    select_points_by_spacing,
    SweepGrid,
    wald_interval,
    wilson_interval,
)
from sixdof.paths import (  # noqa: E402
    OPTIMAL_AZIMUTHS,
    SELECTED_POINTS_100M,
)


@pytest.fixture(scope="module")
def coefficients():
    return naval_5in38_coefficients()


@pytest.fixture(scope="module")
def simulator(coefficients):
    return BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(elevation_deg=10.0, azimuth_deg=0.0),
        environment=standard_atmosphere(),
        aero_coeffs=coefficients,
    )


# ----------------------------------------------------------------------
# stage 1: grid helpers
# ----------------------------------------------------------------------
def test_inclusive_range_descending():
    """A descending sweep includes both endpoints and stays on the grid."""
    values = list(inclusive_range(45.0, 44.5, -0.1, decimals=1))
    assert values == [45.0, 44.9, 44.8, 44.7, 44.6, 44.5]


def test_inclusive_range_ascending_avoids_float_drift():
    """Rounding at each step keeps -1.65 exactly representable for later masks."""
    values = list(inclusive_range(-1.65, -1.45, 0.05, decimals=2))
    assert values == [-1.65, -1.6, -1.55, -1.5, -1.45]


def test_default_grid_size():
    """The thesis grid is 601 elevations by 34 azimuths."""
    grid = SweepGrid()
    assert len(grid.elevations()) == 601
    assert len(grid.azimuths()) == 34
    assert len(grid) == 601 * 34


def test_optimal_azimuths_picks_the_minimum_drift():
    """For each elevation, the azimuth with the smallest |drift| wins."""
    sweep = pd.DataFrame(
        {
            "Elevacao_deg": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "Azimute_deg": [-1.0, -0.5, 0.0, -1.0, -0.5, 0.0],
            "Alcance_x_m": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "Desvio_z_m": [-5.0, 0.2, 4.0, -9.0, 3.0, -0.1],
            "Altura_max_m": [10.0] * 6,
            "Tempo_voo_s": [1.0] * 6,
        }
    )
    optimal = optimal_azimuths(sweep)
    assert list(optimal["Elevacao_deg"]) == [10.0, 20.0]
    assert list(optimal["Azimute_otimo_deg"]) == [-0.5, 0.0]
    assert list(optimal["Desvio_z_resultante_m"]) == [0.2, -0.1]


def test_max_range_shot():
    """The longest-range row is reported with all four of its quantities."""
    sweep = pd.DataFrame(
        {
            "Elevacao_deg": [10.0, 39.6, 60.0],
            "Azimute_deg": [0.0, -1.35, 0.0],
            "Alcance_x_m": [100.0, 16796.8, 12000.0],
            "Desvio_z_m": [1.0, 4.26, 9.0],
            "Altura_max_m": [10.0, 3000.0, 6000.0],
            "Tempo_voo_s": [1.0, 60.0, 70.0],
        }
    )
    elevation, azimuth, range_m, drift = max_range_shot(sweep)
    assert (elevation, azimuth) == (39.6, -1.35)
    assert range_m == pytest.approx(16796.8)
    assert drift == pytest.approx(4.26)


def test_sweep_runs_on_a_small_grid(simulator):
    """A 2x2 grid produces a well-formed table with one row per shot."""
    grid = SweepGrid(
        elevation_start=5.0, elevation_stop=4.9, elevation_step=-0.1,
        azimuth_start=-0.05, azimuth_stop=0.0, azimuth_step=0.05,
    )
    frame = AngleSweep(simulator, grid).run(verbose=False)

    assert len(frame) == 4
    assert list(frame.columns) == [
        "Elevacao_deg", "Azimute_deg", "Alcance_x_m",
        "Desvio_z_m", "Altura_max_m", "Tempo_voo_s",
    ]
    assert (frame["Alcance_x_m"] > 0).all()
    # A higher elevation reaches further in this part of the envelope.
    at_zero = frame[frame["Azimute_deg"] == 0.0].sort_values("Elevacao_deg")
    assert at_zero["Alcance_x_m"].is_monotonic_increasing


# ----------------------------------------------------------------------
# stage 2: point selection, against the thesis table
# ----------------------------------------------------------------------
def test_selection_reproduces_thesis_table():
    """The refactored selection must reproduce the published 163 aim points."""
    optimal = pd.read_excel(OPTIMAL_AZIMUTHS)
    published = pd.read_excel(SELECTED_POINTS_100M)

    selected = select_points_by_spacing(optimal, elevation_max=39.6, elevation_min=-1.5)

    assert len(selected) == len(published) == 163
    for column in (
        "Elevacao_deg",
        "Azimute_otimo_deg",
        "Alcance_x_m",
        "Desvio_z_resultante_m",
    ):
        assert np.array_equal(selected[column].values, published[column].values), column


def test_selection_spacing_is_close_to_the_target():
    """Consecutive aim points sit roughly 100 m apart in range."""
    optimal = pd.read_excel(OPTIMAL_AZIMUTHS)
    selected = select_points_by_spacing(optimal, elevation_max=39.6, elevation_min=-1.5)
    gaps = selected["Diferenca_alcance_m"].values[1:]

    assert gaps.min() > 50.0
    assert gaps.max() < 150.0
    assert 80.0 < gaps.mean() < 120.0
    assert selected["Dentro_tolerancia"].all()


def test_selection_starts_at_the_maximum_range_elevation():
    """The ladder starts at 39.6 deg, the maximum-range elevation of the gun."""
    optimal = pd.read_excel(OPTIMAL_AZIMUTHS)
    selected = select_points_by_spacing(optimal, elevation_max=39.6, elevation_min=-1.5)
    assert selected.iloc[0]["Elevacao_deg"] == 39.6
    assert selected["Elevacao_deg"].is_monotonic_decreasing


def test_selection_handles_an_empty_band():
    """Filtering everything away returns an empty frame rather than raising."""
    optimal = pd.read_excel(OPTIMAL_AZIMUTHS)
    assert select_points_by_spacing(optimal, elevation_max=-90.0).empty


# ----------------------------------------------------------------------
# stage 3: dispersion campaign
# ----------------------------------------------------------------------
def test_perturbations_are_reproducible(simulator):
    """The same seed gives the same draws, and they match the requested sigmas."""
    settings = DispersionSettings(n_shots=1000, seed=16184331)
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)

    first_elevation, first_azimuth = campaign.draw_perturbations(2)
    second_elevation, second_azimuth = campaign.draw_perturbations(2)

    assert np.array_equal(first_elevation, second_elevation)
    assert np.array_equal(first_azimuth, second_azimuth)
    assert len(first_elevation) == 2000
    assert first_elevation.std() == pytest.approx(0.1, abs=0.01)
    assert first_azimuth.std() == pytest.approx(0.05, abs=0.01)


def test_perturbations_match_the_legacy_draw(simulator):
    """The draw is byte-compatible with the legacy global-seed sequence."""
    settings = DispersionSettings(n_shots=10, seed=16184331)
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)
    elevation, azimuth = campaign.draw_perturbations(1)

    np.random.seed(16184331)
    expected_elevation = np.random.normal(0, 0.1, 10)
    expected_azimuth = np.random.normal(0, 0.05, 10)

    assert np.array_equal(elevation, expected_elevation)
    assert np.array_equal(azimuth, expected_azimuth)


def test_partial_draw_diverges_without_campaign_size(simulator):
    """Why ``campaign_size`` exists: the azimuth stream depends on the total.

    The legacy generator draws every elevation error first and every azimuth
    error second, so the azimuth stream starts at a position set by the total
    shot count.  Drawing for one point instead of 163 gives the right
    elevations and entirely wrong azimuths.
    """
    settings = DispersionSettings(n_shots=1000, seed=16184331)
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)

    full_elevation, full_azimuth = campaign.draw_perturbations(163)
    slice_elevation, slice_azimuth = campaign.draw_perturbations(1)

    assert np.array_equal(full_elevation[:1000], slice_elevation)
    assert not np.array_equal(full_azimuth[:1000], slice_azimuth)


def test_partial_run_matches_the_same_slice_of_the_full_campaign(simulator):
    """Running points 1 and 2 separately must equal running them together."""
    settings = DispersionSettings(n_shots=4, seed=4242)
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)

    points = [
        AimPoint(5.0, 0.0, 2000.0, 0.0, label="a"),
        AimPoint(6.0, -0.1, 2400.0, 0.0, label="b"),
    ]

    together = campaign.run(points, verbose=False)
    first = campaign.run(points[:1], verbose=False, campaign_size=2)
    second = campaign.run(
        points[1:], verbose=False, campaign_size=2, first_point_index=1
    )

    assert np.array_equal(together[0].errors_x_m, first[0].errors_x_m)
    assert np.array_equal(together[0].errors_z_m, first[0].errors_z_m)
    assert np.array_equal(together[1].errors_x_m, second[0].errors_x_m)
    assert np.array_equal(together[1].errors_z_m, second[0].errors_z_m)

    assert together[0].point_number == 1
    assert together[1].point_number == 2
    assert second[0].point_number == 2


def test_run_rejects_a_slice_outside_the_campaign(simulator):
    """``campaign_size`` must actually cover the points being run."""
    campaign = MonteCarloCampaign(
        simulator, surface_target_fleet, DispersionSettings(n_shots=2)
    )
    points = [AimPoint(5.0, 0.0, 2000.0, 0.0), AimPoint(6.0, 0.0, 2400.0, 0.0)]
    with pytest.raises(ValueError):
        campaign.run(points, verbose=False, campaign_size=2, first_point_index=1)


def test_short_campaign_end_to_end(simulator):
    """A small campaign produces hits, a CEP and a well-formed results table."""
    settings = DispersionSettings(n_shots=12, seed=7, sigma_elevation_deg=0.1,
                                  sigma_azimuth_deg=0.05)
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)

    aim_point = AimPoint(
        elevation_deg=5.0, azimuth_deg=0.0,
        nominal_range_m=2000.0, nominal_drift_m=0.0, label="teste",
    )
    results = campaign.run([aim_point], verbose=False)

    assert len(results) == 1
    result = results[0]
    assert result.n_valid == 12
    assert result.cep(50) >= 0.0
    assert result.cep(90) >= result.cep(50)
    assert set(result.hits) == set(surface_target_fleet((0.0, 0.0)))

    frame = MonteCarloCampaign.to_frame(results)
    assert len(frame) == 1
    assert frame.iloc[0]["N_simulacoes"] == 12
    assert "Taxa_acerto_Drone_Sea_Baby_pct" in frame.columns
    assert 0.0 <= frame.iloc[0]["Taxa_acerto_Drone_Sea_Baby_pct"] <= 100.0
    # The biggest hull cannot be hit less often than the smallest one.
    assert (
        frame.iloc[0]["Taxa_acerto_SMS_V4_pct"]
        >= frame.iloc[0]["Taxa_acerto_Drone_Sea_Baby_pct"]
    )


def test_aim_point_from_row():
    """Aim points can be built straight from a row of the selected-points table."""
    published = pd.read_excel(SELECTED_POINTS_100M)
    aim_point = AimPoint.from_row(published.iloc[0])
    assert aim_point.elevation_deg == 39.6
    assert aim_point.azimuth_deg == -1.35
    assert aim_point.nominal_range_m == pytest.approx(16796.794263, abs=1e-6)


# ----------------------------------------------------------------------
# stage 4: cost and confidence intervals
# ----------------------------------------------------------------------
def test_expected_cost_two_rounds():
    """Hand-checked case: two rounds at p = 0.5 each."""
    result = expected_engagement_cost([0.5, 0.5], round_cost=2000, target_value=1_000_000)
    assert result.expected_rounds == pytest.approx(1.5)
    assert result.ammunition_cost == pytest.approx(3000.0)
    assert result.total_failure_probability == pytest.approx(0.25)
    assert result.failure_cost == pytest.approx(250_000.0)
    assert result.total_expected_cost == pytest.approx(253_000.0)
    assert result.success_probability == pytest.approx(0.75)


def test_expected_cost_certain_first_round():
    """A guaranteed first-round kill fires exactly one round and never fails."""
    result = expected_engagement_cost([1.0, 0.5, 0.5], round_cost=2000, target_value=1e9)
    assert result.expected_rounds == pytest.approx(1.0)
    assert result.total_failure_probability == pytest.approx(0.0)
    assert result.total_expected_cost == pytest.approx(2000.0)


def test_expected_cost_all_misses():
    """With zero kill probability every round is fired and the penalty is paid."""
    result = expected_engagement_cost([0.0] * 5, round_cost=2000, target_value=289_000_000)
    assert result.expected_rounds == pytest.approx(5.0)
    assert result.total_failure_probability == pytest.approx(1.0)
    assert result.total_expected_cost == pytest.approx(5 * 2000 + 289_000_000)


def test_expected_cost_rejects_empty_salvo():
    with pytest.raises(ValueError):
        expected_engagement_cost([], round_cost=1.0, target_value=1.0)


def test_expected_cost_decreases_with_more_rounds():
    """Adding rounds lowers the expected cost while the penalty dominates."""
    costs = [
        expected_engagement_cost([0.3] * n, 2000, 289_000_000).total_expected_cost
        for n in range(1, 12)
    ]
    assert costs == sorted(costs, reverse=True)


def test_margin_of_error_matches_the_thesis():
    """K = 1000, z = 1.96, variance 1/4 gives the quoted +/-3.10 percentage points."""
    assert margin_of_error(1000) == pytest.approx(0.0310, abs=5e-5)


def test_wald_interval_is_symmetric_and_clipped():
    """Fixed-variance interval: constant width, never leaving [0, 1]."""
    low, high = wald_interval(0.5, 1000)
    assert high - 0.5 == pytest.approx(0.5 - low)
    assert wald_interval(0.0, 1000)[0] == 0.0
    assert wald_interval(1.0, 1000)[1] == 1.0


def test_wilson_interval_stays_inside_the_unit_range():
    """Wilson behaves at the boundaries, where the normal approximation does not."""
    low, high = wilson_interval(0.0, 1000)
    assert low == 0.0
    assert 0.0 < high < 0.01
    low, high = wilson_interval(1.0, 1000)
    assert high == 1.0
    assert 0.99 < low < 1.0


def test_wilson_and_wald_agree_near_one_half():
    """Away from the boundaries the two intervals nearly coincide."""
    wald = wald_interval(0.5, 1000, variance=0.25)
    wilson = wilson_interval(0.5, 1000)
    assert wald[0] == pytest.approx(wilson[0], abs=1e-3)
    assert wald[1] == pytest.approx(wilson[1], abs=1e-3)


# ----------------------------------------------------------------------
# closest approach to a list of fixed points
# ----------------------------------------------------------------------
class _Line:
    """A straight flight, so the geometry can be checked in closed form."""

    def __init__(self, offset_y=0.0, offset_z=0.0, n=101):
        self.t = np.linspace(0.0, 10.0, n)
        self.x = np.linspace(0.0, 1000.0, n)
        self.y = np.full(n, float(offset_y))
        self.z = np.full(n, float(offset_z))


def test_closest_approach_finds_the_perpendicular_distance():
    """A point beside a straight line is met at its own abscissa."""
    sample, distance = closest_approach(_Line(), [(300.0, 4.0, 3.0)])
    assert int(sample[0]) == 30                       # x = 300 m
    assert float(distance[0]) == pytest.approx(5.0)   # 3-4-5


def test_closest_approach_scores_every_point_at_once():
    points = [(0.0, 0.0, 0.0), (500.0, 0.0, 0.0), (1000.0, 0.0, 0.0)]
    sample, distance = closest_approach(_Line(), points)
    assert list(sample) == [0, 50, 100]
    assert distance == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_closest_approach_measures_in_three_dimensions():
    """Lateral offset counts as much as vertical -- this is not a ground miss."""
    _, lateral = closest_approach(_Line(offset_z=7.0), [(500.0, 0.0, 0.0)])
    _, vertical = closest_approach(_Line(offset_y=7.0), [(500.0, 0.0, 0.0)])
    assert float(lateral[0]) == pytest.approx(float(vertical[0]))
    assert float(lateral[0]) == pytest.approx(7.0)


def test_closest_approach_rejects_a_bad_point_array():
    with pytest.raises(ValueError):
        closest_approach(_Line(), [(1.0, 2.0)])


def test_nearest_approach_keeps_the_best_per_point():
    """Each point is served by whichever flight came nearest to *it*."""
    tracker = NearestApproach([(500.0, 0.0, 0.0), (500.0, 9.0, 0.0)])
    tracker.absorb(_Line(offset_y=0.0), label="low")
    tracker.absorb(_Line(offset_y=10.0), label="high")

    best = tracker.best()
    assert best[0].label == "low" and best[0].distance_m == pytest.approx(0.0)
    assert best[1].label == "high" and best[1].distance_m == pytest.approx(1.0)


def test_nearest_approach_returns_the_point_each_flight_served():
    """``absorb`` reports the flight's own nearest point, best or not."""
    tracker = NearestApproach([(500.0, 0.0, 0.0), (500.0, 100.0, 0.0)])
    served = tracker.absorb(_Line(offset_y=90.0), label="high")
    assert served.point == 1                       # nearer the high point
    assert served.distance_m == pytest.approx(10.0)
    assert tracker.best()[0].label == "high"       # yet it is also the best so far


def test_nearest_approach_breaks_ties_in_favour_of_the_first():
    """Two equidistant flights: the answer must not depend on the walk order."""
    tracker = NearestApproach([(500.0, 0.0, 0.0)])
    tracker.absorb(_Line(offset_y=5.0), label="first")
    tracker.absorb(_Line(offset_y=-5.0), label="second")
    assert tracker.best()[0].label == "first"


def test_nearest_approach_reports_when_nothing_flew():
    tracker = NearestApproach([(1.0, 2.0, 3.0)])
    assert tracker.unreached() == [0]
    assert np.isinf(tracker.distances()[0])
    tracker.absorb(_Line(), label="x")
    assert tracker.unreached() == []


def test_nearest_approach_records_the_time_of_the_pass():
    tracker = NearestApproach([(300.0, 0.0, 0.0)])
    approach = tracker.absorb(_Line(), label="x")
    assert approach.time_s == pytest.approx(3.0)   # 300 of 1000 m at 10 s total
    assert approach.position_m == pytest.approx([300.0, 0.0, 0.0])


def test_sweep_reduce_hook_sees_the_whole_trajectory(simulator):
    """The published sweep can drive the proximity reduction, one grid walk."""
    grid = SweepGrid(
        elevation_start=20.0, elevation_stop=21.0, elevation_step=1.0,
        azimuth_start=-1.0, azimuth_stop=-0.5, azimuth_step=0.5,
    )
    sweep = AngleSweep(simulator, grid=grid, max_time=100.0)

    seen = []
    table = sweep.run(
        verbose=False,
        progress_every=0,
        reduce=lambda e, a, traj: seen.append((e, a, float(traj.x[-1]))),
    )

    assert len(seen) == len(table)
    # The hook's view and the table's summary describe the same flights.
    assert [row[0] for row in seen] == list(table["Elevacao_deg"])
    assert [row[1] for row in seen] == list(table["Azimute_deg"])


def test_sweep_reduce_hook_is_optional(simulator):
    """Omitting it leaves the thesis pipeline byte for byte as it was."""
    grid = SweepGrid(
        elevation_start=20.0, elevation_stop=21.0, elevation_step=1.0,
        azimuth_start=-1.0, azimuth_stop=-0.5, azimuth_step=0.5,
    )
    sweep = AngleSweep(simulator, grid=grid, max_time=100.0)
    plain = sweep.run(verbose=False, progress_every=0)
    hooked = sweep.run(verbose=False, progress_every=0, reduce=lambda e, a, t: None)
    pd.testing.assert_frame_equal(plain, hooked)
