"""Ready-made configurations for the cases studied in the thesis.

Having the 5"/38 gun and shell in one place keeps every example and test using
exactly the same numbers, instead of repeating the literals in each script the
way the original repository did.

Sources for the projectile data: US Navy ordnance pamphlets for the 5"/38
calibre gun, transcribed in the thesis.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .aerodynamics import AerodynamicCoefficients, load_coefficients
from .environment import Environment
from .paths import AERO_COEFFICIENTS_5IN38
from .projectile import Projectile
from .vessel import Vessel
from .weapon import Weapon


def naval_5in38_coefficients() -> AerodynamicCoefficients:
    """The 5"/38 aerodynamic coefficients, as the seven the equations read.

    Loads ``data/aero_coefficients_5in38_spin73.npz`` — the table the thesis
    campaign was flown on.  See :data:`sixdof.paths.AERO_COEFFICIENTS_5IN38`
    for the two respects in which it departs from the McCoy contract, both
    inherited from the thesis and both kept so its results reproduce.
    """
    return load_coefficients(AERO_COEFFICIENTS_5IN38)


def naval_5in38_projectile(name: str = 'Projétil Naval 5"/38') -> Projectile:
    """The 5"/38 calibre common shell used throughout the thesis.

    68.10 lb, 5.0 in calibre, ``I_P = 240.9`` and ``I_T = 2619.0`` lb.in^2,
    fired from a barrel with one turn in 25 calibres.
    """
    return Projectile.from_imperial(
        name=name,
        mass_lb=68.10,
        diameter_in=5.0,
        I_P_lbin2=240.9,
        I_T_lbin2=2619.0,
        rifling_twist_calibers=25.0,
    )


def naval_5in38_gun(
    elevation_deg: float = 45.0,
    azimuth_deg: float = 0.0,
    *,
    name: str = 'Canhão Naval 5"/38',
    height_m: float = 10.0,
    muzzle_velocity_mps: float = 807.0,
    rate_of_fire_rpm: float = 15.0,
    mounted_on_vessel: Optional[Vessel] = None,
) -> Weapon:
    """The 5"/38 mount, sited ``height_m`` above the water line.

    Muzzle velocity 807 m/s, 15 rounds per minute.
    """
    return Weapon(
        name=name,
        position=(0.0, height_m, 0.0),
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        rate_of_fire_rpm=rate_of_fire_rpm,
        muzzle_velocity_mps=muzzle_velocity_mps,
        mounted_on_vessel=mounted_on_vessel,
    )


def standard_atmosphere() -> Environment:
    """Sea-level ICAO conditions, still air: the thesis reference atmosphere."""
    return Environment(rho=1.225, g=9.81, W1=0.0, W2=0.0, W3=0.0)


#: Hull dimensions (length, width, in m) of the surface targets scored in the
#: Monte Carlo campaign, with a short description of each.
SURFACE_TARGET_SPECS = {
    "Drone_Sea_Baby": {"length": 6.0, "width": 2.0, "description": "Drone naval ucraniano"},
    "IRIS_Paykan": {"length": 56.0, "width": 7.6, "description": "Fast attack craft (Irã)"},
    "Osa_class": {"length": 38.6, "width": 7.6, "description": "Osa-class missile boat (URSS)"},
    "Hayabusa_class": {
        "length": 50.1,
        "width": 8.4,
        "description": "Hayabusa-class torpedo boat (Japão)",
    },
    "SMS_V4": {"length": 72.0, "width": 7.34, "description": "SMS V4 destroyer (Alemanha WWI)"},
    "PT_105": {"length": 24.4, "width": 6.3, "description": "PT-105 patrol torpedo boat (EUA)"},
}


def surface_target_fleet(
    center_position: Sequence[float],
    height: float = 1.0,
    specs: Optional[dict] = None,
) -> dict:
    """Instantiate every hull in :data:`SURFACE_TARGET_SPECS` at one aim point.

    Parameters
    ----------
    center_position:
        ``(x, z)`` where all hulls are centred, in m.
    height:
        Hull height in m.  Irrelevant when impacts are scored with
        ``check_height=False``, which is what the campaign does.
    specs:
        Override the default fleet.

    Returns
    -------
    dict
        Target name -> :class:`~sixdof.vessel.Vessel`.
    """
    specs = SURFACE_TARGET_SPECS if specs is None else specs
    return {
        name: Vessel(
            name=name,
            center_position=center_position,
            length=spec["length"],
            width=spec["width"],
            height=height,
            velocity=(0.0, 0.0),
        )
        for name, spec in specs.items()
    }


__all__ = [
    "naval_5in38_coefficients",
    "naval_5in38_projectile",
    "naval_5in38_gun",
    "standard_atmosphere",
    "surface_target_fleet",
    "SURFACE_TARGET_SPECS",
]
