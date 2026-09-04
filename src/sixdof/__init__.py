"""Six-degree-of-freedom exterior ballistics for spin-stabilised projectiles.

Implements the rigid-body model of McCoy, *Modern Exterior Ballistics* (2nd
ed.): six degrees of freedom, tabulated aerodynamic coefficients as functions of
Mach and total angle of attack, and integration with ``scipy.integrate.solve_ivp``.

A minimal run:

.. code-block:: python

    from sixdof import (
        BallisticSimulator, naval_5in38_coefficients,
        naval_5in38_gun, naval_5in38_projectile, standard_atmosphere,
    )

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(elevation_deg=43.3),
        environment=standard_atmosphere(),
        aero_coeffs=naval_5in38_coefficients(),
    )
    trajectory = simulator.simulate()
    trajectory.print_statistics()

Two optional layers sit on top and are not imported by default:

``sixdof.aa``
    Anti-air lethality -- target geometry, fragmenting warhead, proximity fuze
    and the fragment damage model.

``sixdof.montecarlo``
    The dispersion campaign: angle sweep, aim-point selection, per-point Monte
    Carlo and engagement cost.

Provenance
----------

The engine behind this package was written and verified by Luiz Guilherme de
Padua Sanches for his undergraduate thesis in Applied and Computational
Mathematics at IME-USP, and is preserved unchanged on the ``legacy`` branch.
This package is a later refactor of that engine, written by Claude (Anthropic's
AI assistant) at the author's request and machine-verified to be bit-identical
to it -- see ``scripts/proof_of_equivalence.py`` and ``docs/verification.md``.

Cite the ``legacy`` branch for the thesis artefact; use this one to build on.
See ``README.md`` for the full account.
"""

from .aerodynamics import (
    EQUATION_COEFFICIENTS,
    AerodynamicCoefficients,
    load_coefficients,
)
from .dynamics import SixDofEquations, build_initial_state, six_dof_rhs
from .environment import Environment
from .events import make_ground_event, make_proximity_fuze_event
from .paths import (
    AERO_COEFFICIENTS_5IN38,
    AERO_SOURCE_5IN38,
    AERO_WORKBOOK_5IN38,
    DATA_DIR,
    OPTIMAL_AZIMUTHS,
    PUBLISHED_CAMPAIGN,
    SELECTED_POINTS_100M,
)
from .presets import (
    SURFACE_TARGET_SPECS,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
    surface_target_fleet,
)
from .projectile import Projectile
from .simulator import THESIS_SETTINGS, BallisticSimulator, IntegrationSettings
from .trajectory import SimulationResult, Trajectory
from .vessel import Vessel
from .weapon import Weapon

__version__ = "2.0.0"

__all__ = [
    "__version__",
    # aerodynamics
    "AerodynamicCoefficients",
    "load_coefficients",
    "EQUATION_COEFFICIENTS",
    # physical objects
    "Projectile",
    "Weapon",
    "Vessel",
    "Environment",
    # engine
    "SixDofEquations",
    "six_dof_rhs",
    "build_initial_state",
    "BallisticSimulator",
    "IntegrationSettings",
    "THESIS_SETTINGS",
    "Trajectory",
    "SimulationResult",
    "make_ground_event",
    "make_proximity_fuze_event",
    # presets and data
    "naval_5in38_projectile",
    "naval_5in38_gun",
    "naval_5in38_coefficients",
    "standard_atmosphere",
    "surface_target_fleet",
    "SURFACE_TARGET_SPECS",
    "DATA_DIR",
    "AERO_COEFFICIENTS_5IN38",
    "AERO_SOURCE_5IN38",
    "AERO_WORKBOOK_5IN38",
    "OPTIMAL_AZIMUTHS",
    "SELECTED_POINTS_100M",
    "PUBLISHED_CAMPAIGN",
]


def __getattr__(name: str):
    """Expose ``TrajectoryPlotter``/``Palette`` without importing matplotlib eagerly."""
    if name in ("TrajectoryPlotter", "Palette", "DEFAULT_PALETTE"):
        from . import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
