"""Atmosphere and wind seen by the projectile.

The model is deliberately uniform: constant density, constant gravity and a
constant wind vector.  That is what the thesis used, and it keeps the reference
trajectories reproducible.  Altitude-dependent atmospheres can be layered on
later without touching the equations of motion, which only ever read the five
scalars below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Environment:
    """Uniform atmospheric conditions.

    Attributes
    ----------
    rho:
        Air density in kg/m^3.
    g:
        Gravitational acceleration in m/s^2, applied along -y.
    W1, W2, W3:
        Wind components in m/s along x, y and z.  The aerodynamic forces use
        the air-relative velocity ``V - W``.
    sound_speed:
        Speed of sound in m/s, used to form the Mach number.
    """

    rho: float = 1.225
    g: float = 9.81
    W1: float = 0.0
    W2: float = 0.0
    W3: float = 0.0
    sound_speed: float = 340.0

    @property
    def wind(self) -> tuple:
        """Wind vector as ``(W1, W2, W3)``."""
        return (self.W1, self.W2, self.W3)

    def get_info(self) -> str:
        """Human-readable summary."""
        info = f"\n{'='*60}\n"
        info += "AMBIENTE\n"
        info += f"{'='*60}\n"
        info += f"  Densidade do ar: {self.rho:.4f} kg/m³\n"
        info += f"  Gravidade: {self.g:.4f} m/s²\n"
        info += f"  Vento (W1, W2, W3): ({self.W1:.2f}, {self.W2:.2f}, {self.W3:.2f}) m/s\n"
        info += f"  Velocidade do som: {self.sound_speed:.1f} m/s\n"
        return info


__all__ = ["Environment"]
