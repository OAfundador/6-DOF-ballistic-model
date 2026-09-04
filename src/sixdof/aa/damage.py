"""Fragment damage model: from a burst point to a probability of destruction.

The chain is:

1. the direction from burst to target sets ``phi``, the angle off the shell's
   axis, which picks the polar zone of the fragment pattern;
2. that zone gives an areal fragment density at the burst-to-target distance
   (:meth:`sixdof.aa.warhead.FragmentationWarhead.angular_density`);
3. the target's projected area along the same direction converts the density
   into an expected number of fragments ``M`` crossing the target;
4. treating the crossings as Poisson gives a Bernoulli destruction probability
   ``p = 1 - exp(-M)``.

Step 4 rests on two assumptions, both recorded in the returned assessment so a
reader can see them without reading the code: every fragment that crosses the
projected area is taken to perforate, and any single perforation is taken to be
a kill.  Both are optimistic.  Relaxing them means filtering the fragment
population by mass and residual velocity against a penetration criterion, which
this layer deliberately leaves out -- it is the next thing to add, not something
silently approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .fuze import BurstPoint, ProximityFuze
from .geometry import Target, angle_between, unit_vector
from .warhead import DynamicZone, FragmentationWarhead


def destruction_probability(expected_fragments: float) -> float:
    """``p = 1 - exp(-M)``: at least one hit, for Poisson-distributed crossings."""
    if expected_fragments <= 0.0:
        return 0.0
    return float(np.clip(1.0 - exp(-expected_fragments), 0.0, 1.0))


@dataclass
class DamageAssessment:
    """Everything the damage model computed for one burst.

    The field names double as the report layout, so that a run can be audited
    line by line against the equations in the thesis.
    """

    fuze_triggered: bool
    distance_m: float
    phi_target_deg: float
    phi_velocity_deg: float
    angle_velocity_to_target_deg: float
    alpha1_zone_center_deg: float
    static_zone: Optional[Tuple[float, float, int]]
    dynamic_zone_deg: Optional[Tuple[float, float]]
    obliquity_deg: float
    dominant_facet: str
    facet_areas_m2: Dict[str, float]
    exposed_area_m2: float
    fragment_speed_mps: float
    n_fragments_model: float
    n_fragments_zone: float
    band_area_m2: float
    density_per_m2: float
    expected_fragments: float
    p_destruction: float
    assumptions: Dict[str, bool] = field(
        default_factory=lambda: {
            "penetracao_total": True,
            "dano_critico_por_fragmento": True,
        }
    )
    probabilistic_model: str = "Bernoulli/Poisson: p=1-exp(-M)"

    def to_dict(self) -> Dict[str, object]:
        """Flat dictionary, convenient for building a DataFrame of many bursts."""
        return {
            "fuze_triggered": self.fuze_triggered,
            "distance_m": self.distance_m,
            "phi_target_deg": self.phi_target_deg,
            "phi_velocity_deg": self.phi_velocity_deg,
            "angle_velocity_to_target_deg": self.angle_velocity_to_target_deg,
            "alpha1_zone_center_deg": self.alpha1_zone_center_deg,
            "static_zone": self.static_zone,
            "dynamic_zone_deg": self.dynamic_zone_deg,
            "obliquity_deg": self.obliquity_deg,
            "dominant_facet": self.dominant_facet,
            "exposed_area_m2": self.exposed_area_m2,
            "fragment_speed_mps": self.fragment_speed_mps,
            "n_fragments_model": self.n_fragments_model,
            "n_fragments_zone": self.n_fragments_zone,
            "band_area_m2": self.band_area_m2,
            "density_per_m2": self.density_per_m2,
            "expected_fragments": self.expected_fragments,
            "p_destruction": self.p_destruction,
        }


class FragmentDamageModel:
    """Evaluate fragment lethality for a given target and warhead.

    Parameters
    ----------
    target:
        Faceted target; see :mod:`sixdof.aa.geometry`.
    warhead:
        Fragmenting warhead; see :mod:`sixdof.aa.warhead`.

    Examples
    --------
    >>> model = FragmentDamageModel(target, warhead)          # doctest: +SKIP
    >>> assessment = model.evaluate(burst)                    # doctest: +SKIP
    >>> assessment.p_destruction                              # doctest: +SKIP
    0.87
    """

    def __init__(self, target: Target, warhead: FragmentationWarhead) -> None:
        self.target = target
        self.warhead = warhead

    def evaluate(
        self,
        burst: BurstPoint,
        n_fragments: Optional[float] = None,
    ) -> DamageAssessment:
        """Assess one burst.

        Parameters
        ----------
        burst:
            Where and how fast the shell was when the warhead functioned.
        n_fragments:
            Override the warhead's effective fragment count for this burst.

        Raises
        ------
        ValueError
            If the burst coincides with the target centre, or the shell axis is
            degenerate.
        """
        return self.evaluate_state(
            burst_position=burst.position_m,
            projectile_velocity=burst.velocity_mps,
            projectile_axis=burst.axis_i,
            fuze_triggered=burst.triggered,
            n_fragments=n_fragments,
        )

    def evaluate_state(
        self,
        burst_position: Sequence[float],
        projectile_velocity: Sequence[float],
        projectile_axis: Sequence[float],
        fuze_triggered: bool = True,
        n_fragments: Optional[float] = None,
    ) -> DamageAssessment:
        """Assess a burst given raw vectors instead of a :class:`BurstPoint`."""
        burst_position = np.array(burst_position, dtype=float)
        projectile_velocity = np.array(projectile_velocity, dtype=float)

        to_target = self.target.center - burst_position
        distance_m = float(np.linalg.norm(to_target))
        if distance_m <= 1e-6:
            raise ValueError("burst coincidente com o centro do alvo")

        fragment_direction = to_target / distance_m

        projectile_speed = float(np.linalg.norm(projectile_velocity))
        velocity_direction = projectile_velocity / projectile_speed
        axis_direction = unit_vector(projectile_axis)

        # phi: angle off the shell axis, which selects the polar zone.  For a
        # point target the observed dispersion direction is simply burst->target.
        phi_target = angle_between(axis_direction, fragment_direction)
        angle_velocity_fragment = angle_between(velocity_direction, fragment_direction)

        exposed_area, facet_areas = self.target.projected_area(fragment_direction)
        obliquity, dominant_facet = self.target.effective_obliquity(
            fragment_direction, facet_areas
        )

        fragment_speed = self.warhead.fragment_speed(projectile_speed, phi_target)

        n_fragments_model = (
            self.warhead.effective_fragments if n_fragments is None else n_fragments
        )
        sample = self.warhead.angular_density(
            distance_m=distance_m,
            phi_rad=phi_target,
            projectile_speed_mps=projectile_speed,
            n_fragments=n_fragments_model,
        )

        expected_fragments = max(0.0, sample.density * exposed_area)

        return DamageAssessment(
            fuze_triggered=fuze_triggered,
            distance_m=distance_m,
            phi_target_deg=float(np.degrees(phi_target)),
            phi_velocity_deg=float(np.degrees(phi_target)),
            angle_velocity_to_target_deg=float(np.degrees(angle_velocity_fragment)),
            alpha1_zone_center_deg=_zone_center(sample.zone),
            static_zone=sample.zone.static.as_tuple() if sample.zone else None,
            dynamic_zone_deg=sample.zone.bounds_deg if sample.zone else None,
            obliquity_deg=float(np.degrees(obliquity)),
            dominant_facet=dominant_facet,
            facet_areas_m2=facet_areas,
            exposed_area_m2=exposed_area,
            fragment_speed_mps=fragment_speed,
            n_fragments_model=n_fragments_model,
            n_fragments_zone=sample.zone_fragments,
            band_area_m2=sample.band_area_m2,
            density_per_m2=sample.density,
            expected_fragments=expected_fragments,
            p_destruction=destruction_probability(expected_fragments),
        )


def _zone_center(zone: Optional[DynamicZone]) -> float:
    """Mid-angle of a zone's *static* bounds, or NaN when there is no zone."""
    if zone is None:
        return float("nan")
    return 0.5 * (zone.static.theta_min_deg + zone.static.theta_max_deg)


def evaluate_engagement(
    trajectory,
    target: Target,
    warhead: FragmentationWarhead,
    fuze: Optional[ProximityFuze] = None,
) -> Tuple[Optional[BurstPoint], Optional[DamageAssessment]]:
    """Run the whole chain: find the burst, then score the damage.

    Parameters
    ----------
    trajectory:
        A :class:`~sixdof.trajectory.Trajectory` (or any object with the same
        sample attributes).
    target, warhead:
        What is being shot at, and with what.
    fuze:
        Defaults to a fuze centred on ``target.center`` with the standard
        24.38 m radius and 0.5 s arming delay.

    Returns
    -------
    tuple
        ``(burst, assessment)``, or ``(None, None)`` when the trajectory has no
        sample past the arming time.
    """
    if fuze is None:
        fuze = ProximityFuze(target_center=target.center)

    burst = fuze.find_burst(trajectory)
    if burst is None:
        return None, None

    model = FragmentDamageModel(target, warhead)
    return burst, model.evaluate(burst)


__all__ = [
    "FragmentDamageModel",
    "DamageAssessment",
    "destruction_probability",
    "evaluate_engagement",
]
