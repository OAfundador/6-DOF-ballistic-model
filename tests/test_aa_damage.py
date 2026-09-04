"""The anti-air module must match the appendix it generalises, and stay generic.

Three groups of tests:

* **Regression** -- ``tests/reference/apendice_dano.py`` is a frozen, unmodified
  copy of the procedural appendix the model was refactored from.  Every
  quantity the two produce is compared exactly.
* **Ported assertions** -- the checks from the appendix's own test script,
  rewritten against the class API.
* **Genericity** -- the same machinery driven with a different target shape and
  a different warhead, which is what "make the AA module generic" has to mean
  in practice.
"""

from __future__ import annotations

import sys
from math import exp, radians, sqrt
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

import apendice_dano as appendix  # noqa: E402

from sixdof.aa import (  # noqa: E402
    FragmentationWarhead,
    FragmentDamageModel,
    PolarZone,
    ProximityFuze,
    box_target,
    destruction_probability,
    evaluate_engagement,
    shahed_136,
    triangular_prism_target,
    vt_fcl_mk49,
)


# ----------------------------------------------------------------------
# fake trajectories, copied from the appendix's own test script
# ----------------------------------------------------------------------
class FakeTrajectory:
    """Approaches the target from below, entering the burst radius at t = 1 s."""

    t = [0.0, 1.0, 2.0]
    x = [0.0, 0.0, 0.0]
    y = [-30.0, -24.38, -5.0]
    z = [0.0, 0.0, 0.0]
    V1 = [0.0, 0.0, 0.0]
    V2 = [620.0, 620.0, 620.0]
    V3 = [0.0, 0.0, 0.0]
    i1 = [0.0, 0.0, 0.0]
    i2 = [1.0, 1.0, 1.0]
    i3 = [0.0, 0.0, 0.0]


class FakeTrajectoryPhiNotAlpha1:
    """Axis pointing at the target while the velocity is perpendicular to it.

    Separates ``phi`` (axis to target) from the velocity direction: the polar
    zone must be chosen by the axis, not by where the shell is going.
    """

    t = [1.0]
    x = [0.0]
    y = [-24.38]
    z = [0.0]
    V1 = [620.0]
    V2 = [0.0]
    V3 = [0.0]
    i1 = [0.0]
    i2 = [1.0]
    i3 = [0.0]


class FakeTrajectoryObliquePhi:
    """Burst offset 30 degrees off the axis, on the burst radius."""

    r = 24.38
    t = [1.0]
    x = [-r * 0.5]
    y = [-r * 0.8660254037844386]
    z = [0.0]
    V1 = [0.0]
    V2 = [620.0]
    V3 = [0.0]
    i1 = [0.0]
    i2 = [1.0]
    i3 = [0.0]


FAKE_TRAJECTORIES = [FakeTrajectory, FakeTrajectoryPhiNotAlpha1, FakeTrajectoryObliquePhi]


@pytest.fixture
def target():
    return shahed_136(center=(0.0, 0.0, 0.0))


@pytest.fixture
def warhead():
    return vt_fcl_mk49()


# ----------------------------------------------------------------------
# regression against the frozen appendix
# ----------------------------------------------------------------------
def test_target_geometry_matches_appendix(target):
    """Every area and normal of the prism must match the procedural version."""
    old = appendix.criar_alvo_shahed(centro=(0.0, 0.0, 0.0))

    assert target.geometry_model == old["modelo_geometrico"]
    for key in (
        "comprimento_m",
        "envergadura_m",
        "espessura_m",
        "comprimento_lateral_m",
        "area_superior_m2",
        "area_inferior_m2",
        "area_lateral_m2",
        "area_laterais_total_m2",
        "area_traseira_m2",
        "volume_aproximado_m3",
    ):
        assert target.metadata[key] == old[key], key

    old_normals = {name: normal for name, _, normal in old["faces_projetadas"]}
    old_areas = {name: area for name, area, _ in old["faces_projetadas"]}
    for facet in target.facets:
        assert facet.area_m2 == old_areas[facet.name]
        assert np.allclose(facet.outward_normal, appendix.vetor_unitario(old_normals[facet.name]))


@pytest.mark.parametrize(
    "direction",
    [
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.4, -0.5, 0.3),
        (-0.7, -0.2, 0.6),
    ],
)
def test_projected_area_matches_appendix(target, direction):
    """The projected area and its per-face split must match exactly."""
    old = appendix.criar_alvo_shahed(centro=(0.0, 0.0, 0.0))
    old_area, old_contributions = appendix.area_exposta_shahed(old, direction)
    new_area, new_contributions = target.projected_area(direction)

    assert new_area == old_area
    assert new_contributions == old_contributions


def test_warhead_matches_appendix(warhead):
    """Fragment count, ejection speed and polar zones must match."""
    old = appendix.criar_ogiva_vt_fcl_mk49()
    assert warhead.effective_fragments == old["n_fragmentos_efetivos"]
    assert warhead.fragment_velocity_mps == old["v0_m_s"]
    assert warhead.zones_as_tuples() == old["zonas_polares_hits"]
    assert warhead.total_hits == 1217


@pytest.mark.parametrize("trajectory_class", FAKE_TRAJECTORIES)
def test_full_assessment_matches_appendix(target, warhead, trajectory_class):
    """Burst point and every damage quantity must match the appendix exactly."""
    old_target = appendix.criar_alvo_shahed(centro=(0.0, 0.0, 0.0))
    old_warhead = appendix.criar_ogiva_vt_fcl_mk49()
    old_burst, old_damage = appendix.avaliar_fuze_e_dano(
        trajectory_class(), old_target, old_warhead, raio_fuze_m=24.38, tempo_armar_s=0.5
    )

    burst, damage = evaluate_engagement(trajectory_class(), target, warhead)

    assert burst.triggered == old_burst["acionado"]
    assert burst.source == old_burst["origem"]
    assert burst.index == old_burst["idx"]
    assert burst.time_s == old_burst["tempo_s"]
    assert np.array_equal(burst.position_m, old_burst["posicao_m"])
    assert burst.distance_m == old_burst["distancia_m"]

    assert damage.distance_m == old_damage["distancia_m"]
    assert damage.phi_target_deg == old_damage["phi_alvo_deg"]
    assert damage.phi_velocity_deg == old_damage["phi_velocidade_deg"]
    assert damage.angle_velocity_to_target_deg == old_damage["angulo_vel_fragmento_deg"]
    assert damage.static_zone == old_damage["zona_polar_npg"]
    assert damage.dynamic_zone_deg == old_damage["zona_polar_dinamica_deg"]
    assert damage.obliquity_deg == old_damage["obliquidade_deg"]
    assert damage.dominant_facet == old_damage["face_alvo_dominante"]
    assert damage.exposed_area_m2 == old_damage["area_exposta_m2"]
    assert damage.fragment_speed_mps == old_damage["v_frag_m_s"]
    assert damage.n_fragments_model == old_damage["n_fragmentos_modelo"]
    assert damage.n_fragments_zone == old_damage["n_fragmentos_zona"]
    assert damage.band_area_m2 == old_damage["area_faixa_dinamica_m2"]
    assert damage.density_per_m2 == old_damage["densidade_efetiva"]
    assert damage.expected_fragments == old_damage["fragmentos_esperados_area"]
    assert damage.p_destruction == old_damage["p_destruicao"]
    assert damage.alpha1_zone_center_deg == old_damage["alpha1_zona_centro_deg"]


# ----------------------------------------------------------------------
# assertions ported from the appendix's own test script
# ----------------------------------------------------------------------
def test_prism_areas(target):
    """Closed-form areas of the 3.5 x 2.5 x 0.35 m triangular prism."""
    side = sqrt(3.5**2 + 1.25**2)
    assert target.metadata["area_superior_m2"] == pytest.approx(4.375, abs=1e-12)
    assert target.metadata["area_inferior_m2"] == pytest.approx(4.375, abs=1e-12)
    assert target.metadata["comprimento_lateral_m"] == pytest.approx(side, abs=1e-12)
    assert target.metadata["area_lateral_m2"] == pytest.approx(side * 0.35, abs=1e-12)
    assert target.metadata["area_laterais_total_m2"] == pytest.approx(2 * side * 0.35, abs=1e-12)
    assert target.metadata["area_traseira_m2"] == pytest.approx(0.875, abs=1e-12)
    assert target.metadata["volume_aproximado_m3"] == pytest.approx(1.53125, abs=1e-12)


def test_only_facing_facets_are_exposed(target):
    """A face turned away from the cloud contributes nothing."""
    area, contributions = target.projected_area((0.0, -1.0, 0.0))
    assert area == pytest.approx(target.metadata["area_superior_m2"], abs=1e-12)
    assert contributions["superior"] > 0.0
    assert contributions["inferior"] == 0.0

    area, contributions = target.projected_area((0.0, 1.0, 0.0))
    assert area == pytest.approx(target.metadata["area_inferior_m2"], abs=1e-12)
    assert contributions["inferior"] > 0.0
    assert contributions["superior"] == 0.0

    area, contributions = target.projected_area((1.0, 0.0, 0.0))
    assert area == pytest.approx(target.metadata["area_traseira_m2"], abs=1e-12)
    assert contributions["traseira"] > 0.0

    area, contributions = target.projected_area((0.0, 0.0, -1.0))
    assert area == pytest.approx(3.5 * 0.35, abs=1e-12)
    assert contributions["lateral_direita"] > 0.0
    assert contributions["lateral_esquerda"] == 0.0


def test_head_on_burst_selects_the_nose_zone(target, warhead):
    """Axis pointing straight at the target lands in the 0-15 degree band."""
    burst, damage = evaluate_engagement(FakeTrajectory(), target, warhead)

    assert burst.triggered is True
    assert abs(damage.phi_target_deg) < 1e-9
    assert damage.alpha1_zone_center_deg == pytest.approx(7.5, abs=1e-9)
    assert damage.static_zone == (0.0, 15.0, 10)
    assert damage.n_fragments_model == 2113
    assert damage.expected_fragments > 0.0
    assert damage.p_destruction == pytest.approx(
        1.0 - exp(-damage.expected_fragments), abs=1e-12
    )
    assert 0.0 <= damage.p_destruction <= 1.0
    assert damage.assumptions == {
        "penetracao_total": True,
        "dano_critico_por_fragmento": True,
    }


def test_phi_follows_the_axis_not_the_velocity(target, warhead):
    """The zone is picked by the shell axis even when the velocity is at 90 deg."""
    _, damage = evaluate_engagement(FakeTrajectoryPhiNotAlpha1(), target, warhead)

    assert damage.alpha1_zone_center_deg == pytest.approx(7.5, abs=1e-9)
    assert abs(damage.phi_target_deg) < 1e-9
    assert abs(damage.phi_velocity_deg) < 1e-9
    assert damage.angle_velocity_to_target_deg == pytest.approx(90.0, abs=1e-9)


def test_oblique_burst_selects_the_40_65_zone(target, warhead):
    """A 30 degree offset falls in the swept band of the 40-65 degree zone."""
    _, damage = evaluate_engagement(FakeTrajectoryObliquePhi(), target, warhead)

    assert damage.phi_target_deg == pytest.approx(30.0, abs=1e-9)
    assert damage.phi_velocity_deg == pytest.approx(30.0, abs=1e-9)
    assert damage.static_zone == (40.0, 65.0, 30)
    assert damage.alpha1_zone_center_deg == pytest.approx(52.5, abs=1e-9)


# ----------------------------------------------------------------------
# behaviour of the pieces
# ----------------------------------------------------------------------
def test_destruction_probability_is_monotonic():
    """``p = 1 - exp(-M)`` rises with M, is 0 at 0 and saturates below 1."""
    assert destruction_probability(0.0) == 0.0
    assert destruction_probability(-1.0) == 0.0
    assert destruction_probability(1.0) == pytest.approx(1 - exp(-1.0))
    values = [destruction_probability(m) for m in (0.1, 0.5, 1.0, 5.0, 50.0)]
    assert values == sorted(values)
    assert values[-1] <= 1.0


def test_forward_sweep_compresses_the_pattern(warhead):
    """A faster shell pulls every ejection angle towards the nose."""
    static = radians(90.0)
    at_rest = warhead.alpha2_from_alpha1(0.0, static)
    in_flight = warhead.alpha2_from_alpha1(600.0, static)
    assert at_rest == pytest.approx(static)
    assert in_flight < at_rest


def test_fragment_speed_bounds(warhead):
    """Forward fragments gain the shell speed, rearward ones lose it."""
    forward = warhead.fragment_speed(600.0, 0.0)
    sideways = warhead.fragment_speed(600.0, radians(90.0))
    rearward = warhead.fragment_speed(600.0, radians(180.0))
    assert forward == pytest.approx(1243.6 + 600.0)
    assert rearward == pytest.approx(abs(1243.6 - 600.0))
    assert rearward < sideways < forward


def test_fuze_reports_closest_approach_when_it_never_triggers(target, warhead):
    """A round that stays outside the radius yields a miss distance, not a burst."""

    class Miss:
        t = [0.0, 1.0, 2.0]
        x = [500.0, 400.0, 300.0]
        y = [0.0, 0.0, 0.0]
        z = [0.0, 0.0, 0.0]
        V1 = [-100.0, -100.0, -100.0]
        V2 = [0.0, 0.0, 0.0]
        V3 = [0.0, 0.0, 0.0]
        i1 = [-1.0, -1.0, -1.0]
        i2 = [0.0, 0.0, 0.0]
        i3 = [0.0, 0.0, 0.0]

    burst, damage = evaluate_engagement(Miss(), target, warhead)
    assert burst.triggered is False
    assert burst.source == "menor_distancia_amostrada"
    assert burst.distance_m == pytest.approx(300.0)
    assert damage.fuze_triggered is False


def test_fuze_returns_none_before_arming(target, warhead):
    """No sample past the arming delay means there is nothing to evaluate."""

    class TooEarly:
        t = [0.0, 0.1]
        x = [0.0, 0.0]
        y = [-1.0, -0.5]
        z = [0.0, 0.0]
        V1 = [0.0, 0.0]
        V2 = [1.0, 1.0]
        V3 = [0.0, 0.0]
        i1 = [0.0, 0.0]
        i2 = [1.0, 1.0]
        i3 = [0.0, 0.0]

    burst, damage = evaluate_engagement(TooEarly(), target, warhead)
    assert burst is None and damage is None


def test_burst_on_the_target_centre_is_rejected(target, warhead):
    """A zero burst-to-target distance has no defined direction."""
    model = FragmentDamageModel(target, warhead)
    with pytest.raises(ValueError):
        model.evaluate_state(
            burst_position=target.center,
            projectile_velocity=(0.0, 600.0, 0.0),
            projectile_axis=(0.0, 1.0, 0.0),
        )


# ----------------------------------------------------------------------
# genericity: another shape, another warhead
# ----------------------------------------------------------------------
def test_box_target_projections():
    """A box presents exactly one face to each axis direction."""
    box = box_target("caixa", length=4.0, width=2.0, height=1.0, center=(0.0, 0.0, 0.0))
    top, _ = box.projected_area((0.0, -1.0, 0.0))
    side, _ = box.projected_area((0.0, 0.0, -1.0))
    front, _ = box.projected_area((-1.0, 0.0, 0.0))
    assert top == pytest.approx(8.0)
    assert side == pytest.approx(4.0)
    assert front == pytest.approx(2.0)


def test_model_runs_with_a_different_target_and_warhead():
    """Nothing in the damage chain is specific to the Shahed or the VT round."""
    quadcopter = box_target("quadricoptero", 0.6, 0.6, 0.2, center=(0.0, 0.0, 0.0))
    small_warhead = FragmentationWarhead(
        name="ogiva generica 40 mm",
        fragment_velocity_mps=1100.0,
        polar_zones=[PolarZone(0.0, 90.0, 300), PolarZone(90.0, 180.0, 100)],
        effective_fragments=400,
    )
    burst, damage = evaluate_engagement(
        FakeTrajectory(),
        quadcopter,
        small_warhead,
        fuze=ProximityFuze(target_center=quadcopter.center, radius_m=24.38, arm_time_s=0.5),
    )
    assert burst.triggered is True
    assert damage.static_zone == (0.0, 90.0, 300)
    assert 0.0 <= damage.p_destruction <= 1.0
    assert damage.exposed_area_m2 == pytest.approx(0.36)


def test_custom_prism_dimensions_flow_through():
    """The prism builder is parameterised, not hard-coded to one airframe."""
    wide = triangular_prism_target("asa larga", length=5.0, span=6.0, thickness=0.5,
                                   center=(0.0, 0.0, 0.0))
    assert wide.metadata["area_superior_m2"] == pytest.approx(15.0)
    assert wide.metadata["volume_aproximado_m3"] == pytest.approx(7.5)
    area, _ = wide.projected_area((0.0, -1.0, 0.0))
    assert area == pytest.approx(15.0)


def test_fuze_radius_is_configurable(target, warhead):
    """A tighter fuze does not trigger where the standard one does.

    The fake trajectory's closest sample sits 5 m from the target, so a 3 m
    fuze misses it while the standard 24.38 m fuze bursts at the 24.38 m sample.
    """
    tight = ProximityFuze(target_center=target.center, radius_m=3.0, arm_time_s=0.5)
    burst = tight.find_burst(FakeTrajectory())
    assert burst.triggered is False
    assert burst.distance_m == pytest.approx(5.0)

    standard = ProximityFuze(target_center=target.center, radius_m=24.38, arm_time_s=0.5)
    assert standard.find_burst(FakeTrajectory()).triggered is True
