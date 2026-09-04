"""Integrated trajectory: raw state histories, derived quantities, statistics.

:class:`Trajectory` is a plain data object.  It does no plotting -- that lives
in :mod:`sixdof.plotting` -- so it can be used inside a Monte Carlo loop
without importing matplotlib.

Attribute names are English, with the Portuguese names of the original engine
kept as read-only aliases so scripts written against the legacy branch keep
working.
"""

from __future__ import annotations

from math import atan2, sqrt
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .environment import Environment
    from .projectile import Projectile


class Trajectory:
    """Result of one integration.

    Parameters
    ----------
    solution:
        The ``OdeResult`` returned by ``scipy.integrate.solve_ivp``.
    projectile, environment:
        Needed to recover the spin history and the angle of attack.
    stop_reason:
        ``"ground"``, ``"fuze"`` or ``"max_time"``.
    muzzle_velocity:
        Carried so the plots can draw the initial spin reference line.

    Attributes
    ----------
    t:
        Sample times, in s.
    V1, V2, V3:
        Velocity components in world axes, m/s.
    h1, h2, h3:
        Angular momentum components.
    i1, i2, i3:
        Components of the axis of symmetry.
    x, y, z:
        Position, in m: downrange, altitude, lateral.
    V_mag, mach, h_mag, spin_rate, alpha_traj:
        Derived histories (see :meth:`_compute_derived`).
    """

    def __init__(
        self,
        solution: Any,
        projectile: "Projectile",
        environment: "Environment",
        *,
        stop_reason: str = "unknown",
        muzzle_velocity: Optional[float] = None,
    ) -> None:
        self.solution = solution
        self.projectile = projectile
        self.environment = environment
        self.stop_reason = stop_reason
        self.muzzle_velocity = muzzle_velocity

        self.t = solution.t
        self.V1, self.V2, self.V3 = solution.y[0:3]
        self.h1, self.h2, self.h3 = solution.y[3:6]
        self.i1, self.i2, self.i3 = solution.y[6:9]
        self.x, self.y, self.z = solution.y[9:12]

        self._compute_derived()

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def _compute_derived(self) -> None:
        """Speed, Mach, |h|, axial spin, angle of attack and the summary stats.

        The spin history inverts ``omega1 = (I_T / I_P)(h . i')``; the angle of
        attack is the angle between the air-relative velocity and the axis of
        symmetry, taken with ``atan2`` of the perpendicular and parallel
        components so it stays accurate at small angles.
        """
        self.V_mag = np.sqrt(self.V1**2 + self.V2**2 + self.V3**2)
        self.mach = self.V_mag / self.environment.sound_speed
        self.h_mag = np.sqrt(self.h1**2 + self.h2**2 + self.h3**2)

        I_P = self.projectile.I_P
        I_T = self.projectile.I_T

        spin_rate = []
        for idx in range(len(self.t)):
            h_dot_i = (
                self.h1[idx] * self.i1[idx]
                + self.h2[idx] * self.i2[idx]
                + self.h3[idx] * self.i3[idx]
            )
            spin_rate.append((I_T / I_P) * h_dot_i)
        self.spin_rate = np.array(spin_rate)

        alpha_traj = []
        for idx in range(len(self.t)):
            v1 = self.V1[idx] - self.environment.W1
            v2 = self.V2[idx] - self.environment.W2
            v3 = self.V3[idx] - self.environment.W3

            v_along = v1 * self.i1[idx] + v2 * self.i2[idx] + v3 * self.i3[idx]
            v_perp1 = v1 - v_along * self.i1[idx]
            v_perp2 = v2 - v_along * self.i2[idx]
            v_perp3 = v3 - v_along * self.i3[idx]
            v_perp = sqrt(v_perp1**2 + v_perp2**2 + v_perp3**2)

            alpha_traj.append(np.degrees(atan2(v_perp, v_along)))
        self.alpha_traj = np.array(alpha_traj)

        self.max_range = float(self.x[-1])
        self.max_altitude = float(np.max(self.y))
        self.max_lateral_drift = float(np.max(np.abs(self.z)))
        self.flight_time = float(self.t[-1])

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------
    @property
    def impact_point(self) -> np.ndarray:
        """Final ``(x, y, z)`` of the trajectory, in m."""
        return np.array([self.x[-1], self.y[-1], self.z[-1]], dtype=float)

    @property
    def impact_speed(self) -> float:
        """Speed at the final sample, in m/s."""
        return float(self.V_mag[-1])

    def state_at(self, index: int) -> np.ndarray:
        """The full 12-component state at sample ``index``."""
        return self.solution.y[:, index].copy()

    def position_at(self, index: int) -> np.ndarray:
        """Position at sample ``index``, in m."""
        return np.array([self.x[index], self.y[index], self.z[index]], dtype=float)

    def velocity_at(self, index: int) -> np.ndarray:
        """Velocity at sample ``index``, in m/s."""
        return np.array([self.V1[index], self.V2[index], self.V3[index]], dtype=float)

    def axis_at(self, index: int) -> np.ndarray:
        """Axis of symmetry at sample ``index`` (unit vector)."""
        return np.array([self.i1[index], self.i2[index], self.i3[index]], dtype=float)

    def summary(self) -> Dict[str, float]:
        """Summary statistics as a dictionary, handy for building DataFrames."""
        return {
            "max_range_m": self.max_range,
            "max_altitude_m": self.max_altitude,
            "max_lateral_drift_m": self.max_lateral_drift,
            "flight_time_s": self.flight_time,
            "impact_speed_mps": self.impact_speed,
            "alpha_min_deg": float(np.min(self.alpha_traj)),
            "alpha_max_deg": float(np.max(self.alpha_traj)),
            "alpha_mean_deg": float(np.mean(self.alpha_traj)),
        }

    def print_statistics(self) -> None:
        """Print the summary block of the original engine, unchanged."""
        print(f"\n{'='*80}")
        print("ESTATÍSTICAS DA TRAJETÓRIA")
        print(f"{'='*80}")
        print(f"  Alcance: {self.max_range/1000:.2f} km")
        print(f"  Altura máxima: {self.max_altitude/1000:.2f} km")
        print(f"  Desvio lateral: {self.max_lateral_drift:.2f} m")
        print(f"  Tempo de voo: {self.flight_time:.2f} s")
        print("\nÂNGULO DE ATAQUE:")
        print(f"  Mínimo: {np.min(self.alpha_traj):.2f}°")
        print(f"  Máximo: {np.max(self.alpha_traj):.2f}°")
        print(f"  Médio: {np.mean(self.alpha_traj):.2f}°")

    # ------------------------------------------------------------------
    # legacy aliases (names used by the single-file engine)
    # ------------------------------------------------------------------
    @property
    def alcance_max(self) -> float:
        """Alias of :attr:`max_range`."""
        return self.max_range

    @property
    def altura_max(self) -> float:
        """Alias of :attr:`max_altitude`."""
        return self.max_altitude

    @property
    def desvio_lateral_max(self) -> float:
        """Alias of :attr:`max_lateral_drift`."""
        return self.max_lateral_drift

    @property
    def tempo_voo(self) -> float:
        """Alias of :attr:`flight_time`."""
        return self.flight_time

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Trajectory(samples={len(self.t)}, flight_time={self.flight_time:.3f} s, "
            f"max_range={self.max_range:.1f} m, stop_reason={self.stop_reason!r})"
        )


#: Backwards-compatible alias for the class name used in the legacy branch.
SimulationResult = Trajectory

__all__ = ["Trajectory", "SimulationResult"]
