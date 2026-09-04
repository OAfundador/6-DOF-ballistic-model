"""Rigid-body description of a spin-stabilised projectile."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

LB_TO_KG = 0.453592
IN_TO_M = 0.0254
LBIN2_TO_KGM2 = LB_TO_KG * (IN_TO_M**2)


@dataclass
class Projectile:
    """Mass and inertia properties of an axially symmetric projectile.

    Attributes
    ----------
    name:
        Free-form label used in reports and plots.
    mass:
        Mass in kg.
    diameter:
        Reference diameter (calibre) in m.
    I_P:
        Polar (axial) moment of inertia in kg.m^2.
    I_T:
        Transverse moment of inertia in kg.m^2.
    rifling_twist:
        Barrel twist in calibres per turn.
    S:
        Reference area in m^2, derived from ``diameter``.

    Notes
    -----
    Gyroscopic stability requires ``I_P < I_T`` for a conventional shell; the
    ratio ``I_P / I_T`` drives the spin term in the moment equations.
    """

    name: str = "Projétil Naval"
    mass: float = 0.0
    diameter: float = 0.0
    I_P: float = 0.0
    I_T: float = 0.0
    rifling_twist: float = 25.0
    S: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.S = pi * (self.diameter / 2) ** 2 if self.diameter else 0.0

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_imperial(
        cls,
        name: str,
        mass_lb: float,
        diameter_in: float,
        I_P_lbin2: float,
        I_T_lbin2: float,
        rifling_twist_calibers: float = 25.0,
    ) -> "Projectile":
        """Build a projectile from the imperial units used in ordnance reports.

        Parameters
        ----------
        mass_lb:
            Mass in pounds.
        diameter_in:
            Calibre in inches.
        I_P_lbin2, I_T_lbin2:
            Polar and transverse moments of inertia in lb.in^2.
        """
        return cls(
            name=name,
            mass=mass_lb * LB_TO_KG,
            diameter=diameter_in * IN_TO_M,
            I_P=I_P_lbin2 * LBIN2_TO_KGM2,
            I_T=I_T_lbin2 * LBIN2_TO_KGM2,
            rifling_twist=rifling_twist_calibers,
        )

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def calculate_initial_spin(self, muzzle_velocity: float) -> float:
        """Axial spin imparted by the rifling, in rad/s.

        With a twist of ``n`` calibres per turn the projectile completes one
        revolution every ``n * d`` metres of travel, so
        ``p0 = 2 pi V0 / (n d)``.
        """
        n = self.rifling_twist
        return (2 * np.pi * muzzle_velocity) / (n * self.diameter)

    @property
    def inertia_ratio(self) -> float:
        """``I_P / I_T``, the coefficient of the spin term in ``dh/dt``."""
        return self.I_P / self.I_T

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def get_info(self) -> str:
        """Human-readable summary, in the layout of the original script."""
        info = f"\n{'='*60}\n"
        info += f"PROJÉTIL: {self.name}\n"
        info += f"{'='*60}\n"
        info += f"  Massa: {self.mass:.2f} kg\n"
        info += f"  Diâmetro: {self.diameter*1000:.1f} mm\n"
        info += f"  I_P: {self.I_P:.6f} kg·m²\n"
        info += f"  I_T: {self.I_T:.6f} kg·m²\n"
        info += f"  I_P/I_T: {self.I_P/self.I_T:.6f}\n"
        info += f"  Área de referência: {self.S:.6f} m²\n"
        info += f"  Rifling twist: {self.rifling_twist:.1f} calibres/volta\n"
        return info


__all__ = ["Projectile", "LB_TO_KG", "IN_TO_M", "LBIN2_TO_KGM2"]
