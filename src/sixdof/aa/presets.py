"""The specific target and warhead used in the anti-air case study.

Both are thin wrappers over the generic builders, so swapping in a different
drone or a different shell is a matter of calling
:func:`sixdof.aa.geometry.triangular_prism_target` and
:class:`sixdof.aa.warhead.FragmentationWarhead` with other numbers rather than
editing the model.
"""

from __future__ import annotations

from typing import Sequence

from .geometry import Target, triangular_prism_target
from .warhead import FragmentationWarhead, PolarZone

#: Default position of the drone in the reference engagement, in m.
SHAHED_DEFAULT_CENTER = (16673.0, 200.0, 0.7)

#: Static polar distribution of the 5"/38 VT(FCL) Mk 49 warhead.
#:
#: Source: NPG Report No. 1124, PDF pp. 9-10.  Angles are measured from the
#: shell's nose; the counts are the hits recorded per band in the arena trial.
VT_FCL_POLAR_ZONES = (
    PolarZone(0.0, 15.0, 10),
    PolarZone(15.0, 40.0, 0),
    PolarZone(40.0, 65.0, 30),
    PolarZone(65.0, 115.0, 1072),
    PolarZone(115.0, 165.0, 95),
    PolarZone(165.0, 180.0, 10),
)


def shahed_136(center: Sequence[float] = SHAHED_DEFAULT_CENTER) -> Target:
    """Shahed-136 / Geran-2 as a triangular prism.

    Length 3.5 m, span 2.5 m, thickness 0.35 m.  The delta planform is modelled
    as a triangle extruded vertically: triangular top and bottom, two inclined
    flanks and a rectangular trailing face.  See
    ``docs/shahed_target_geometry.pt-BR.md`` for the derivation of each area.
    """
    target = triangular_prism_target(
        name="Shahed-136 / Geran-2",
        length=3.5,
        span=2.5,
        thickness=0.35,
        center=center,
    )
    target.metadata["fonte"] = "geometria de primeira ordem, ver docs/"
    return target


def vt_fcl_mk49() -> FragmentationWarhead:
    """The 5"/38 VT(FCL) Mk 49 Comp A-3 fragmenting warhead.

    2113 effective fragments; ejection speed 1243.6 m/s (4080 ft/s, the median
    fragment velocity measured in the 80-110 degree band over the first 30 ft).
    The polar distribution is :data:`VT_FCL_POLAR_ZONES`.

    Source: NPG Report No. 1124, PDF pp. 5, 8-10 and 22.
    """
    return FragmentationWarhead(
        name='5"/38 VT(FCL) Mk 49 Comp A-3',
        fragment_velocity_mps=1243.6,
        polar_zones=VT_FCL_POLAR_ZONES,
        effective_fragments=2113,
        metadata={
            "fonte": "NPG Report No. 1124",
            "observacao_v0": (
                "velocidade mediana dos fragmentos na zona 80-110 graus, "
                "medida nos primeiros 30 ft"
            ),
        },
    )


__all__ = ["shahed_136", "vt_fcl_mk49", "VT_FCL_POLAR_ZONES", "SHAHED_DEFAULT_CENTER"]
