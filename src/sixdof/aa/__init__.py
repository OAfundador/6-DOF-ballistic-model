"""Anti-air lethality: target geometry, fragmenting warhead, fuze, damage.

Optional layer on top of the 6-DOF engine.  Importing :mod:`sixdof` does not
pull it in; a script that needs it says ``from sixdof import aa`` or imports the
names directly.

A minimal engagement reads:

.. code-block:: python

    from sixdof import BallisticSimulator
    from sixdof.aa import ProximityFuze, evaluate_engagement, shahed_136, vt_fcl_mk49

    target = shahed_136(center=(16673.0, 200.0, 0.7))
    warhead = vt_fcl_mk49()
    fuze = ProximityFuze(target_center=target.center)

    trajectory = simulator.simulate(fuze=fuze)
    burst, damage = evaluate_engagement(trajectory, target, warhead, fuze)

Nothing above is specific to a drone or to a VT round: substitute another
:class:`~sixdof.aa.geometry.Target` and another
:class:`~sixdof.aa.warhead.FragmentationWarhead` and the same chain applies.
"""

from .damage import (
    DamageAssessment,
    FragmentDamageModel,
    destruction_probability,
    evaluate_engagement,
)
from .fuze import BurstPoint, ProximityFuze
from .geometry import (
    Facet,
    Target,
    angle_between,
    box_target,
    triangular_prism_target,
    unit_vector,
)
from .presets import SHAHED_DEFAULT_CENTER, VT_FCL_POLAR_ZONES, shahed_136, vt_fcl_mk49
from .report import (
    format_vector,
    print_damage_report,
    print_engagement_setup,
    print_trajectory_summary,
)
from .warhead import DensitySample, DynamicZone, FragmentationWarhead, PolarZone

__all__ = [
    # geometry
    "Facet",
    "Target",
    "box_target",
    "triangular_prism_target",
    "unit_vector",
    "angle_between",
    # warhead
    "PolarZone",
    "DynamicZone",
    "DensitySample",
    "FragmentationWarhead",
    # fuze
    "ProximityFuze",
    "BurstPoint",
    # damage
    "FragmentDamageModel",
    "DamageAssessment",
    "destruction_probability",
    "evaluate_engagement",
    # presets
    "shahed_136",
    "vt_fcl_mk49",
    "VT_FCL_POLAR_ZONES",
    "SHAHED_DEFAULT_CENTER",
    # reporting
    "format_vector",
    "print_trajectory_summary",
    "print_engagement_setup",
    "print_damage_report",
]
