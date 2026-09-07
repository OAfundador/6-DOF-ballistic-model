"""Atmosphere and wind seen by the projectile.

Two models live here.

:class:`Environment` is uniform -- constant density, gravity and wind.  That is
what the thesis used, and it is the default, because the reference trajectories
are pinned to it.

:class:`LayeredAtmosphere` varies density and speed of sound with altitude,
following the ICAO standard.  The equations reach both through
``density_at(altitude)`` and ``sound_speed_at(altitude)``; for the uniform model
those return the fixed attributes unchanged, so nothing about the default path
moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt


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

    # ------------------------------------------------------------------
    # what the equations of motion actually ask for
    # ------------------------------------------------------------------
    def density_at(self, altitude_m: float) -> float:
        """Air density at an altitude.  Constant here, by construction.

        The equations call this instead of reading :attr:`rho` so that an
        altitude-dependent atmosphere can be dropped in without touching them.
        For this class it returns :attr:`rho` unchanged, so the numbers are
        bit-identical to reading the attribute directly.
        """
        return self.rho

    def sound_speed_at(self, altitude_m: float) -> float:
        """Speed of sound at an altitude.  Constant here; see above."""
        return self.sound_speed

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


__all__ = ["Environment", "LayeredAtmosphere"]


@dataclass
class LayeredAtmosphere(Environment):
    """ICAO / US Standard Atmosphere: density and speed of sound vary with height.

    A shot that climbs to 5 km flies through air roughly 40% thinner than at
    the surface and about 7% slower in sound speed, and neither is a detail:
    a constant atmosphere costs kilometres of range on a high-angle trajectory,
    and holding the speed of sound fixed reads the coefficient table at the
    wrong Mach — which matters most transonically, where the drag coefficient
    changes by a factor of two over a narrow band.

    :attr:`rho` and :attr:`sound_speed` keep their sea-level meaning and are
    what the summaries print; the equations ask for
    :meth:`density_at` and :meth:`sound_speed_at`, which is where the profile
    enters.

    Parameters
    ----------
    sea_level_temperature_k, sea_level_pressure_pa:
        Conditions at the reference altitude.
    lapse_rate_k_per_m:
        Temperature gradient through the troposphere.
    tropopause_m:
        Above this the profile is isothermal.

    Examples
    --------
    >>> air = LayeredAtmosphere()
    >>> round(air.density_at(0.0), 4)
    1.225
    >>> round(air.density_at(5000.0), 4)
    0.7361
    >>> round(air.sound_speed_at(0.0), 2)
    340.29
    >>> round(air.sound_speed_at(5000.0), 2)
    320.53

    Sea level is unchanged going up and then back down:

    >>> air.density_at(0.0) == air.density_at(0.0)
    True
    """

    #: Gas constant of dry air, J/(kg K).
    gas_constant: float = 287.05287
    #: Ratio of specific heats.
    gamma: float = 1.4
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101325.0
    lapse_rate_k_per_m: float = -0.0065
    tropopause_m: float = 11000.0

    def __post_init__(self) -> None:
        self.g = 9.80665
        self.rho = self.density_at(0.0)
        self.sound_speed = self.sound_speed_at(0.0)

    def _temperature_and_pressure(self, altitude_m: float):
        """Temperature in K and pressure in Pa at an altitude."""
        t0 = self.sea_level_temperature_k
        p0 = self.sea_level_pressure_pa
        lapse = self.lapse_rate_k_per_m
        exponent = -9.80665 / (lapse * self.gas_constant)

        if altitude_m <= self.tropopause_m:
            temperature = t0 + lapse * altitude_m
            pressure = p0 * (temperature / t0) ** exponent
            return temperature, pressure

        temperature = t0 + lapse * self.tropopause_m
        pressure_tropopause = p0 * (temperature / t0) ** exponent
        pressure = pressure_tropopause * exp(
            -9.80665 * (altitude_m - self.tropopause_m)
            / (self.gas_constant * temperature)
        )
        return temperature, pressure

    def density_at(self, altitude_m: float) -> float:
        temperature, pressure = self._temperature_and_pressure(max(0.0, altitude_m))
        return pressure / (self.gas_constant * temperature)

    def sound_speed_at(self, altitude_m: float) -> float:
        temperature, _ = self._temperature_and_pressure(max(0.0, altitude_m))
        return sqrt(self.gamma * self.gas_constant * temperature)
