"""Fragmenting warhead: static polar distribution and its dynamic correction.

An arena trial reports how many fragments leave the warhead inside each polar
zone measured from the shell's own axis, with the shell at rest.  In flight the
pattern is swept forward by the shell's velocity, so a fragment nominally
leaving at a static angle ``alpha1`` actually departs at

.. math:: \\alpha_2 = \\arctan\\frac{v_0 \\sin\\alpha_1}
                                    {v_0 \\cos\\alpha_1 + v_1}

and its speed becomes the vector sum of the ejection and shell velocities.
Both relations are implemented below; :meth:`FragmentationWarhead.angular_density`
combines them to give the areal fragment density at a point in space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, radians, sin, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PolarZone:
    """Fragment count in one polar band of the static (arena) distribution.

    Attributes
    ----------
    theta_min_deg, theta_max_deg:
        Band limits measured from the nose, in degrees.
    hits:
        Fragments recorded in the band during the trial.  Only the ratios
        between zones matter; the absolute scale comes from
        :attr:`FragmentationWarhead.effective_fragments`.
    """

    theta_min_deg: float
    theta_max_deg: float
    hits: int

    def as_tuple(self) -> Tuple[float, float, int]:
        """``(theta_min_deg, theta_max_deg, hits)``, the reporting form."""
        return (self.theta_min_deg, self.theta_max_deg, self.hits)


@dataclass(frozen=True)
class DynamicZone:
    """A polar zone after the forward sweep correction has been applied."""

    static: PolarZone
    alpha2_min_rad: float
    alpha2_max_rad: float

    @property
    def bounds_deg(self) -> Tuple[float, float]:
        """``(alpha2_min, alpha2_max)`` in degrees."""
        return (
            float(np.degrees(self.alpha2_min_rad)),
            float(np.degrees(self.alpha2_max_rad)),
        )

    def contains(self, phi_rad: float, *, inclusive_upper: bool = False) -> bool:
        """Whether ``phi_rad`` falls inside the swept band."""
        if inclusive_upper:
            return self.alpha2_min_rad <= phi_rad <= self.alpha2_max_rad
        return self.alpha2_min_rad <= phi_rad < self.alpha2_max_rad

    def solid_band_area(self, distance_m: float) -> float:
        """Area of the spherical band this zone sweeps at ``distance_m``.

        ``A = 2 pi r^2 |cos a_min - cos a_max|``.
        """
        return (
            2.0
            * pi
            * distance_m**2
            * abs(cos(self.alpha2_min_rad) - cos(self.alpha2_max_rad))
        )


@dataclass
class FragmentationWarhead:
    """A fragmenting warhead described by its static polar zones.

    Parameters
    ----------
    name:
        Label used in reports.
    fragment_velocity_mps:
        Ejection speed ``v0`` relative to the shell, in m/s.
    polar_zones:
        The static distribution.  Zones should tile ``[0, 180]`` degrees.
    effective_fragments:
        Total number of fragments credited to the warhead.  The zone hit counts
        distribute this total; they do not set its scale.
    metadata:
        Free-form provenance notes.

    Examples
    --------
    >>> w = FragmentationWarhead("test", 1243.6,
    ...                          [PolarZone(0, 90, 1), PolarZone(90, 180, 1)], 100)
    >>> w.total_hits
    2
    """

    name: str
    fragment_velocity_mps: float
    polar_zones: Sequence[PolarZone]
    effective_fragments: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.polar_zones = tuple(
            zone if isinstance(zone, PolarZone) else PolarZone(*zone) for zone in self.polar_zones
        )
        if not self.polar_zones:
            raise ValueError("a warhead needs at least one polar zone")

    @property
    def total_hits(self) -> int:
        """Sum of the hit counts over every static zone."""
        return int(sum(zone.hits for zone in self.polar_zones))

    def zones_as_tuples(self) -> List[Tuple[float, float, int]]:
        """The static distribution in plain-tuple form, for reports and tests."""
        return [zone.as_tuple() for zone in self.polar_zones]

    # ------------------------------------------------------------------
    # dynamic correction
    # ------------------------------------------------------------------
    def alpha2_from_alpha1(self, projectile_speed_mps: float, alpha1_rad: float) -> float:
        """Sweep a static ejection angle forward by the shell's own velocity."""
        v0 = self.fragment_velocity_mps
        v1 = projectile_speed_mps
        return atan2(v0 * sin(alpha1_rad), v0 * cos(alpha1_rad) + v1)

    def fragment_speed(self, projectile_speed_mps: float, phi_rad: float) -> float:
        """Resultant fragment speed, in m/s.

        Law of cosines on the ejection and shell velocity vectors:
        ``sqrt(v1^2 + v0^2 + 2 v1 v0 cos(phi))``.
        """
        v0 = self.fragment_velocity_mps
        v1 = projectile_speed_mps
        return sqrt(v1**2 + v0**2 + 2.0 * v1 * v0 * cos(phi_rad))

    def dynamic_zones(self, projectile_speed_mps: float) -> List[DynamicZone]:
        """Every static zone mapped through :meth:`alpha2_from_alpha1`."""
        zones: List[DynamicZone] = []
        for zone in self.polar_zones:
            a_min = self.alpha2_from_alpha1(projectile_speed_mps, radians(zone.theta_min_deg))
            a_max = self.alpha2_from_alpha1(projectile_speed_mps, radians(zone.theta_max_deg))
            if a_min > a_max:
                a_min, a_max = a_max, a_min
            zones.append(DynamicZone(static=zone, alpha2_min_rad=a_min, alpha2_max_rad=a_max))
        return zones

    def find_zone(
        self, projectile_speed_mps: float, phi_rad: float
    ) -> Optional[DynamicZone]:
        """The swept zone containing ``phi_rad``, or ``None`` if outside them all."""
        zones = self.dynamic_zones(projectile_speed_mps)
        last_index = len(zones) - 1
        for index, zone in enumerate(zones):
            if zone.contains(phi_rad, inclusive_upper=index == last_index):
                return zone
        return None

    # ------------------------------------------------------------------
    # density
    # ------------------------------------------------------------------
    def angular_density(
        self,
        distance_m: float,
        phi_rad: float,
        projectile_speed_mps: float,
        n_fragments: Optional[float] = None,
    ) -> "DensitySample":
        """Areal fragment density at a point, in fragments per m^2.

        Parameters
        ----------
        distance_m:
            Burst-to-target distance, in m.
        phi_rad:
            Angle between the shell axis and the burst-to-target direction.
        projectile_speed_mps:
            Shell speed at burst, used for the forward sweep.
        n_fragments:
            Total fragments to distribute; defaults to
            :attr:`effective_fragments`.

        Returns
        -------
        DensitySample
            Zero density (with ``zone=None``) when the direction falls outside
            every zone or the geometry is degenerate.
        """
        n_fragments = self.effective_fragments if n_fragments is None else n_fragments
        if distance_m <= 0.0 or n_fragments <= 0.0:
            return DensitySample(0.0, None, 0.0, 0.0)

        zone = self.find_zone(projectile_speed_mps, phi_rad)
        if zone is None:
            return DensitySample(0.0, None, 0.0, 0.0)

        zone_fragments = n_fragments * zone.static.hits / self.total_hits
        band_area = zone.solid_band_area(distance_m)
        if band_area <= 0.0:
            return DensitySample(0.0, None, 0.0, 0.0)

        return DensitySample(zone_fragments / band_area, zone, zone_fragments, band_area)


@dataclass(frozen=True)
class DensitySample:
    """Outcome of one density evaluation.

    Attributes
    ----------
    density:
        Fragments per m^2 at the sampled point.
    zone:
        The swept zone used, or ``None`` when the point lies outside them all.
    zone_fragments:
        Fragments credited to that zone.
    band_area_m2:
        Area of the spherical band the zone sweeps at the sampled distance.
    """

    density: float
    zone: Optional[DynamicZone]
    zone_fragments: float
    band_area_m2: float


__all__ = ["PolarZone", "DynamicZone", "FragmentationWarhead", "DensitySample"]
