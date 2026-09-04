"""Console report for an anti-air engagement.

Kept apart from the model so that the numbers can be consumed programmatically
without any printing, and so the report layout can change without touching the
physics.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .damage import DamageAssessment
from .fuze import BurstPoint
from .geometry import Target
from .warhead import FragmentationWarhead


def format_vector(v: Sequence[float]) -> str:
    """``(x, y, z)`` with three decimals -- the layout used in the thesis output."""
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


def print_trajectory_summary(trajectory) -> None:
    """Range, altitude, drift, terminal speed and why the integration stopped."""
    print("\n" + "-" * 80)
    print("RESUMO DA TRAJETORIA")
    print("-" * 80)
    print(f"Tempo de voo simulado : {trajectory.flight_time:.3f} s")
    print(f"Alcance final         : {trajectory.max_range:.3f} m")
    print(f"Altura maxima         : {trajectory.max_altitude:.3f} m")
    print(f"Desvio lateral maximo : {trajectory.max_lateral_drift:.3f} m")
    print(f"Velocidade final      : {trajectory.V_mag[-1]:.3f} m/s")
    print(f"Motivo da parada      : {trajectory.stop_reason}")


def print_engagement_setup(
    target: Target,
    warhead: FragmentationWarhead,
    elevation_deg: float,
    azimuth_deg: float,
    fuze_radius_m: float,
    fuze_arm_time_s: float,
    stop_on_fuze: bool,
    aero_table_path: Optional[str] = None,
) -> None:
    """Echo the scenario before running it, so a saved log is self-describing."""
    print("\n" + "=" * 80)
    print("VERIFICACAO CANONICA: 6-DOF + FUZE + DANO FRAGMENTARIO")
    print("=" * 80)
    if aero_table_path is not None:
        print(f"Coeficientes aerodinamicos: {aero_table_path}")
    print(f"Tiro: elevacao={elevation_deg:.2f} deg | azimute={azimuth_deg:.2f} deg")
    print(f"Alvo {target.name}: centro={format_vector(target.center)} m")
    print(
        f"Orientacao alvo: nariz={format_vector(target.nose_direction)}, "
        f"normal={format_vector(target.vertical_axis)}"
    )
    print(f"Fuze: raio={fuze_radius_m:.2f} m | armado apos {fuze_arm_time_s:.2f} s")
    print(f"Parar simulacao no fuze: {stop_on_fuze}")
    print(f"Ogiva: {warhead.name}")


def print_damage_report(burst: BurstPoint, assessment: DamageAssessment) -> None:
    """Full burst-and-damage breakdown, one quantity per line."""
    print("\n" + "-" * 80)
    print("RESUMO DO FUZE E DANO")
    print("-" * 80)
    print(f"Fuze acionado dentro do raio : {burst.triggered}")
    print(f"Origem do ponto de burst     : {burst.source}")
    print(f"Indice da amostra de burst   : {burst.index}")
    print(f"Tempo do burst               : {burst.time_s:.3f} s")
    print(f"Posicao do burst             : {format_vector(burst.position_m)} m")
    print(f"Distancia burst-alvo         : {assessment.distance_m:.3f} m")
    print(f"Phi alvo (eixo i -> alvo)    : {assessment.phi_target_deg:.3f} deg")
    print(f"Alpha1 zona centro estatica  : {assessment.alpha1_zone_center_deg:.3f} deg")
    print(f"Phi usado na velocidade      : {assessment.phi_velocity_deg:.3f} deg")
    print(f"Angulo velocidade -> alvo    : {assessment.angle_velocity_to_target_deg:.3f} deg")
    print(f"Zona polar NPG estatica      : {assessment.static_zone}")

    dynamic = assessment.dynamic_zone_deg
    hits = assessment.static_zone[2] if assessment.static_zone is not None else "n/a"
    if dynamic is not None:
        print(
            f"Zona polar dinamica alpha2   : ({dynamic[0]:.3f}, {dynamic[1]:.3f}) deg "
            f"| hits NPG={hits}"
        )
    else:
        print("Zona polar dinamica alpha2   : n/a")

    print(f"Obliquidade no alvo          : {assessment.obliquity_deg:.3f} deg")
    print(f"Face dominante do alvo       : {assessment.dominant_facet}")
    print(f"Area exposta do alvo         : {assessment.exposed_area_m2:.6f} m^2")
    print(f"Contribuicoes de area        : {assessment.facet_areas_m2}")
    print(f"Velocidade inicial frag.     : {assessment.fragment_speed_mps:.3f} m/s")
    print(f"Fragmentos totais no modelo  : {assessment.n_fragments_model:.0f}")
    print(f"Fragmentos na zona dinamica  : {assessment.n_fragments_zone:.6f}")
    print(f"Area faixa dinamica          : {assessment.band_area_m2:.6f} m^2")
    print(f"Densidade rho''              : {assessment.density_per_m2:.9f} frag/m^2")
    print(f"M fragmentos esperados       : {assessment.expected_fragments:.6f}")
    print("Hipotese de penetracao       : todo fragmento que cruza a area penetra")
    print("Hipotese de dano             : qualquer penetracao e dano critico")
    print(f"Prob. destruicao Bernoulli   : {assessment.p_destruction:.6%}")


__all__ = [
    "format_vector",
    "print_trajectory_summary",
    "print_engagement_setup",
    "print_damage_report",
]
