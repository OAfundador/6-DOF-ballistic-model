"""Target geometry for fragment-lethality calculations.

A target is represented by a list of flat facets, each with an area and an
outward normal.  The area a fragment cloud actually sees is then the sum of the
facet areas weighted by how squarely they face the cloud.  That is enough for a
first-order lethality estimate and it generalises to any convex shape: a box, a
cylinder approximated by panels, or the triangular prism used for the Shahed.

The projection is deliberately simple -- no ray tracing, no self-shadowing
beyond the back-face test -- because the uncertainty in the fragment
distribution dominates the uncertainty in the silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, sqrt
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def unit_vector(v: Sequence[float]) -> np.ndarray:
    """Normalise ``v``; raise if it is (numerically) the zero vector."""
    arr = np.array(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("vetor com norma nula")
    return arr / norm


@dataclass(frozen=True)
class Facet:
    """One flat face of a target.

    Attributes
    ----------
    name:
        Identifier used when reporting which face dominated the exposure.
    area_m2:
        Area of the face, in m^2.
    outward_normal:
        Unit normal pointing away from the target's interior.
    """

    name: str
    area_m2: float
    outward_normal: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "outward_normal", unit_vector(self.outward_normal))

    def visibility(self, arrival_direction: np.ndarray) -> float:
        """Cosine of the incidence angle, clamped at zero for back faces.

        ``arrival_direction`` points from the burst towards the target, so a
        face is lit when its outward normal opposes it.
        """
        return max(0.0, -float(np.dot(arrival_direction, self.outward_normal)))


@dataclass
class Target:
    """A faceted target with a position and a body frame.

    Parameters
    ----------
    name:
        Label used in reports.
    center:
        ``(x, y, z)`` of the target centre, in m, in simulator world axes.
    facets:
        The faces making up the silhouette model.
    geometry_model:
        Short description of the approximation used, carried into reports.
    nose_direction, vertical_axis, lateral_axis:
        Body triad.  Currently fixed, but written out explicitly so a future
        version can rotate the target without changing the projection formula.
    metadata:
        Free-form dictionary for dimensions, volumes and provenance notes.

    Examples
    --------
    >>> target = triangular_prism_target("drone", 3.5, 2.5, 0.35, (0, 200, 0))
    >>> area, _ = target.projected_area((0.0, -1.0, 0.0))
    >>> round(area, 3)
    4.375
    """

    name: str
    center: np.ndarray
    facets: Tuple[Facet, ...]
    geometry_model: str = "faceted"
    nose_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )
    vertical_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float)
    )
    lateral_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = np.array(self.center, dtype=float)
        self.facets = tuple(self.facets)
        self.nose_direction = np.array(self.nose_direction, dtype=float)
        self.vertical_axis = np.array(self.vertical_axis, dtype=float)
        self.lateral_axis = np.array(self.lateral_axis, dtype=float)

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------
    def projected_area(
        self, arrival_direction: Sequence[float]
    ) -> Tuple[float, Dict[str, float]]:
        """Area presented to a fragment cloud arriving along ``arrival_direction``.

        Parameters
        ----------
        arrival_direction:
            Vector from the burst point towards the target; normalised
            internally.

        Returns
        -------
        tuple
            ``(total_area_m2, per_facet_area_m2)``.

        Notes
        -----
        This is the 3-D replacement for the flat ``S sin(phi)`` presented area
        of the closed-form lethality formula: instead of one plane, each facet
        of the body contributes ``A_i max(0, -d . n_i)``.
        """
        d = unit_vector(arrival_direction)
        contributions = {facet.name: facet.area_m2 * facet.visibility(d) for facet in self.facets}
        return sum(contributions.values()), contributions

    def effective_obliquity(
        self,
        arrival_direction: Sequence[float],
        contributions: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, str]:
        """Incidence angle on the facet that contributes most of the exposed area.

        Returns
        -------
        tuple
            ``(obliquity_rad, dominant_facet_name)``.
        """
        d = unit_vector(arrival_direction)
        if contributions is None:
            _, contributions = self.projected_area(d)

        dominant = max(contributions, key=contributions.get)
        normals = {facet.name: facet.outward_normal for facet in self.facets}
        cos_face = max(0.0, -float(np.clip(np.dot(d, normals[dominant]), -1.0, 1.0)))
        return acos(cos_face), dominant

    def total_area(self) -> float:
        """Sum of all facet areas, in m^2 (the full wetted area, not a silhouette)."""
        return float(sum(facet.area_m2 for facet in self.facets))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Target(name={self.name!r}, model={self.geometry_model!r}, "
            f"facets={len(self.facets)})"
        )


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def box_target(
    name: str,
    length: float,
    width: float,
    height: float,
    center: Sequence[float],
    **metadata: float,
) -> Target:
    """A rectangular box, aligned with the world axes.

    Parameters
    ----------
    length, width, height:
        Extents along ``x`` (nose-tail), ``z`` (span) and ``y`` (vertical), in m.
    """
    facets = (
        Facet("superior", length * width, np.array([0.0, 1.0, 0.0])),
        Facet("inferior", length * width, np.array([0.0, -1.0, 0.0])),
        Facet("lateral_direita", length * height, np.array([0.0, 0.0, 1.0])),
        Facet("lateral_esquerda", length * height, np.array([0.0, 0.0, -1.0])),
        Facet("frontal", width * height, np.array([1.0, 0.0, 0.0])),
        Facet("traseira", width * height, np.array([-1.0, 0.0, 0.0])),
    )
    return Target(
        name=name,
        center=np.array(center, dtype=float),
        facets=facets,
        geometry_model="caixa retangular",
        metadata={
            "comprimento_m": length,
            "envergadura_m": width,
            "espessura_m": height,
            "volume_aproximado_m3": length * width * height,
            **metadata,
        },
    )


def triangular_prism_target(
    name: str,
    length: float,
    span: float,
    thickness: float,
    center: Sequence[float],
    **metadata: float,
) -> Target:
    """A delta-planform body: a triangle extruded through ``thickness``.

    This is the "cheese wedge" approximation used for a flying-wing drone: a
    triangular upper and lower surface, two inclined rectangular flanks meeting
    at the nose, and a rectangular trailing face.

    Parameters
    ----------
    length:
        Nose-to-tail length along ``x``, in m.
    span:
        Trailing-edge span along ``z``, in m.
    thickness:
        Vertical extrusion along ``y``, in m.

    Returns
    -------
    Target
        With ``metadata`` carrying every derived area and the volume.
    """
    half_span = 0.5 * span
    side_length = sqrt(length**2 + half_span**2)

    upper_area = 0.5 * span * length
    lower_area = upper_area
    side_area = side_length * thickness
    rear_area = span * thickness
    volume = upper_area * thickness

    right_side_normal = np.array([half_span / side_length, 0.0, length / side_length])
    left_side_normal = np.array([half_span / side_length, 0.0, -length / side_length])

    facets = (
        Facet("superior", upper_area, np.array([0.0, 1.0, 0.0])),
        Facet("inferior", lower_area, np.array([0.0, -1.0, 0.0])),
        Facet("lateral_direita", side_area, right_side_normal),
        Facet("lateral_esquerda", side_area, left_side_normal),
        Facet("traseira", rear_area, np.array([-1.0, 0.0, 0.0])),
    )

    return Target(
        name=name,
        center=np.array(center, dtype=float),
        facets=facets,
        geometry_model="prisma triangular simplificado",
        metadata={
            "comprimento_m": length,
            "envergadura_m": span,
            "espessura_m": thickness,
            "comprimento_lateral_m": side_length,
            "area_superior_m2": upper_area,
            "area_inferior_m2": lower_area,
            "area_lateral_m2": side_area,
            "area_laterais_total_m2": 2.0 * side_area,
            "area_traseira_m2": rear_area,
            "volume_aproximado_m3": volume,
            **metadata,
        },
    )


def angle_between(direction_a: Sequence[float], direction_b: Sequence[float]) -> float:
    """Angle in radians between two unit vectors, robust to rounding at +/-1."""
    cos_angle = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    return acos(cos_angle)


__all__ = [
    "Facet",
    "Target",
    "box_target",
    "triangular_prism_target",
    "unit_vector",
    "angle_between",
]
