"""Box-shaped surface target, optionally moving at constant velocity."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


class Vessel:
    """A rectangular-box target used for hit/miss scoring.

    The axis convention matches the simulator: ``x`` forward (range), ``y`` up
    (altitude), ``z`` to the right (lateral drift).  The hull sits on the water
    line, so its vertical extent runs from 0 to ``height``.

    Parameters
    ----------
    name:
        Label used in reports.
    center_position:
        ``(x, z)`` of the hull centre at ``t = 0``, in m.
    length, width, height:
        Hull dimensions along ``x``, ``z`` and ``y``, in m.
    velocity:
        Constant ``(vx, vz)`` in m/s.  The default is a stationary target.
    """

    def __init__(
        self,
        name: str = "Embarcação",
        center_position: Sequence[float] = (0.0, 0.0),
        length: float = 100.0,
        width: float = 20.0,
        height: float = 30.0,
        velocity: Sequence[float] = (0.0, 0.0),
    ) -> None:
        self.name = name
        self.center = np.array(center_position, dtype=float)
        self.length = length
        self.width = width
        self.height = height
        self.velocity = np.array(velocity, dtype=float)

    def get_bounds(self, time: float = 0.0) -> Dict[str, float]:
        """Axis-aligned bounding box of the hull at ``time`` seconds."""
        current_center = self.center + self.velocity * time
        return {
            "x_min": current_center[0] - self.length / 2,
            "x_max": current_center[0] + self.length / 2,
            "y_min": 0.0,
            "y_max": self.height,
            "z_min": current_center[1] - self.width / 2,
            "z_max": current_center[1] + self.width / 2,
        }

    def check_impact(
        self,
        projectile_position: Sequence[float],
        time: float = 0.0,
        check_height: bool = True,
    ) -> bool:
        """Whether ``projectile_position`` lies inside the hull box.

        Parameters
        ----------
        check_height:
            When ``False`` the vertical coordinate is ignored, which is how the
            Monte Carlo campaign scores impacts: the trajectory is integrated
            until it reaches the water line, so only the ground footprint of
            the hull matters.
        """
        bounds = self.get_bounds(time)
        x, y, z = projectile_position

        inside_footprint = (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
        )
        if not check_height:
            return bool(inside_footprint)
        return bool(inside_footprint and bounds["y_min"] <= y <= bounds["y_max"])

    def center_at(self, time: float = 0.0) -> Tuple[float, float]:
        """``(x, z)`` of the hull centre at ``time`` seconds."""
        current = self.center + self.velocity * time
        return float(current[0]), float(current[1])

    def get_info(self) -> str:
        """Human-readable summary, in the layout of the original script."""
        info = f"\n{'='*60}\n"
        info += f"EMBARCAÇÃO: {self.name}\n"
        info += f"{'='*60}\n"
        info += f"  Posição do centro (x, z): ({self.center[0]:.1f}, {self.center[1]:.1f}) m\n"
        info += (
            f"  Dimensões (L×W×H): {self.length:.1f} × {self.width:.1f} × "
            f"{self.height:.1f} m\n"
        )
        info += f"  Velocidade (vx, vz): ({self.velocity[0]:.1f}, {self.velocity[1]:.1f}) m/s\n"
        return info

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Vessel(name={self.name!r}, length={self.length}, "
            f"width={self.width}, height={self.height})"
        )


__all__ = ["Vessel"]
