"""Angle-of-attack warnings (sixdof.diagnostics)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sixdof import (
    AngleOfAttackSingularityWarning,
    BallisticSimulator,
    HighAngleOfAttackWarning,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.diagnostics import check_angle_of_attack


def _fly(elevation_deg, **kwargs):
    simulator = BallisticSimulator(
        naval_5in38_projectile(),
        naval_5in38_gun(elevation_deg=elevation_deg),
        standard_atmosphere(),
        naval_5in38_coefficients(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trajectory = simulator.simulate(verbose=False, **kwargs)
    return trajectory, [type(w.message) for w in caught]


class _Fake:
    def __init__(self, t, alpha):
        self.t, self.alpha_traj = np.asarray(t, float), np.asarray(alpha, float)


def test_below_both_limits_nothing_is_raised():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_angle_of_attack(_Fake([0, 1, 2], [1.0, 9.9, 3.0])) == []


def test_the_finding_carries_the_excursion():
    with pytest.warns(HighAngleOfAttackWarning, match="instabilidade por formulação"):
        (found,) = check_angle_of_attack(_Fake([0, 1, 2, 3, 4], [1, 12, 30, 11, 2]))
    assert found["tipo"] == "angulo_de_ataque_alto"
    assert found["alpha_max_deg"] == 30 and found["t_alpha_max_s"] == 2
    assert (found["t_inicio_s"], found["t_fim_s"]) == (1, 3)
    assert found["tempo_acima_s"] == pytest.approx(3.0)


def test_near_180_both_warnings_are_raised():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = check_angle_of_attack(_Fake([0, 1], [170.0, 176.0]))
    assert [type(w.message) for w in caught] == [HighAngleOfAttackWarning,
                                                 AngleOfAttackSingularityWarning]
    assert "singularidade" in str(caught[1].message)
    assert [f["tipo"] for f in found] == ["angulo_de_ataque_alto", "singularidade_180"]


def test_the_limits_can_be_moved():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_angle_of_attack(_Fake([0, 1], [12.0, 12.0]), high_deg=15.0) == []


@pytest.mark.slow
def test_flat_fire_stays_quiet():
    trajectory, caught = _fly(43.3)
    assert caught == [] and trajectory.diagnostics == []


@pytest.mark.slow
def test_high_angle_fire_warns_near_the_summit():
    trajectory, caught = _fly(82.0)
    assert HighAngleOfAttackWarning in caught
    assert AngleOfAttackSingularityWarning not in caught
    (found,) = trajectory.diagnostics
    summit = float(trajectory.t[int(np.argmax(trajectory.y))])
    assert found["t_inicio_s"] < summit < found["t_fim_s"]


def test_an_axis_reversed_at_the_muzzle_hits_the_singularity():
    trajectory, caught = _fly(43.3, alpha0_deg=178.0, max_time=0.05)
    assert AngleOfAttackSingularityWarning in caught
    assert trajectory.diagnostics[-1]["tipo"] == "singularidade_180"
