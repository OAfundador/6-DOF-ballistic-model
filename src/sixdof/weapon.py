"""Gun mount: position, laying angles and the platform it rides on."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .vessel import Vessel


class Weapon:
    """A gun, either emplaced ashore or mounted on a :class:`Vessel`.

    Parameters
    ----------
    name:
        Label used in reports.
    position:
        ``(x, y, z)`` in m.  Absolute when the gun is ashore, relative to the
        hull centre when it is mounted on a vessel.
    elevation_deg, azimuth_deg:
        Laying angles in degrees, stored internally in radians.
    rate_of_fire_rpm:
        Rounds per minute; carried for engagement studies, not used by the
        integrator.
    muzzle_velocity_mps:
        Muzzle velocity in m/s.
    mounted_on_vessel:
        Platform the gun rides on; ``None`` means a stationary land mount.
    """

    def __init__(
        self,
        name: str = "Canhão Naval",
        position: Sequence[float] = (0.0, 0.0, 0.0),
        elevation_deg: float = 45.0,
        azimuth_deg: float = 0.0,
        rate_of_fire_rpm: float = 15.0,
        muzzle_velocity_mps: float = 807.0,
        mounted_on_vessel: Optional[Vessel] = None,
    ) -> None:
        self.name = name
        self.position = np.array(position, dtype=float)
        self.elevation = np.radians(elevation_deg)
        self.azimuth = np.radians(azimuth_deg)
        self.rate_of_fire = rate_of_fire_rpm
        self.muzzle_velocity = muzzle_velocity_mps
        self.mounted_on_vessel = mounted_on_vessel

    # ------------------------------------------------------------------
    # laying
    # ------------------------------------------------------------------
    def set_firing_angles(self, elevation_deg: float, azimuth_deg: float) -> None:
        """Re-lay the gun.  Cheap enough to call inside a sweep loop."""
        self.elevation = np.radians(elevation_deg)
        self.azimuth = np.radians(azimuth_deg)

    @property
    def elevation_deg(self) -> float:
        return float(np.degrees(self.elevation))

    @property
    def azimuth_deg(self) -> float:
        return float(np.degrees(self.azimuth))

    def calculate_firing_angles(self) -> Tuple[float, float]:
        """Convert (elevation, azimuth) into the simulator's ``(theta0, phi0)``.

        The state vector is written in a frame where ``phi`` is measured in the
        vertical plane and ``theta`` out of it, so

        ``theta0 = asin(cos E sin A)`` and ``phi0 = asin(sin E / cos theta0)``.

        Returns
        -------
        tuple
            ``(theta0, phi0)`` in radians.
        """
        E = self.elevation
        A = self.azimuth

        theta0 = np.arcsin(np.cos(E) * np.sin(A))
        phi0 = np.arcsin(np.sin(E) / np.cos(theta0)) if np.cos(theta0) != 0 else np.pi / 2

        return theta0, phi0

    # ------------------------------------------------------------------
    # platform coupling
    # ------------------------------------------------------------------
    def get_absolute_position(self, time: float = 0.0) -> np.ndarray:
        """Muzzle position in world coordinates at ``time`` seconds."""
        if self.mounted_on_vessel is None:
            return self.position.copy()

        vessel_bounds = self.mounted_on_vessel.get_bounds(time)
        vessel_center_x = (vessel_bounds["x_min"] + vessel_bounds["x_max"]) / 2
        vessel_center_z = (vessel_bounds["z_min"] + vessel_bounds["z_max"]) / 2

        return np.array(
            [
                vessel_center_x + self.position[0],
                self.position[1],
                vessel_center_z + self.position[2],
            ]
        )

    def get_velocity(self) -> np.ndarray:
        """Platform velocity added to the muzzle velocity, in m/s.

        A land mount contributes nothing; a vessel contributes its horizontal
        velocity, since the hull is assumed not to heave.
        """
        if self.mounted_on_vessel is None:
            return np.array([0.0, 0.0, 0.0])

        return np.array(
            [
                self.mounted_on_vessel.velocity[0],
                0.0,
                self.mounted_on_vessel.velocity[1],
            ]
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def get_info(self) -> str:
        """Human-readable summary, in the layout of the original script."""
        info = f"\n{'='*60}\n"
        info += f"ARMA: {self.name}\n"
        info += f"{'='*60}\n"

        if self.mounted_on_vessel is None:
            info += (
                f"  Posição (x, y, z): ({self.position[0]:.1f}, "
                f"{self.position[1]:.1f}, {self.position[2]:.1f}) m\n"
            )
            info += "  Montada em: Terra (velocidade = 0)\n"
        else:
            abs_pos = self.get_absolute_position()
            vessel_vel = self.get_velocity()
            info += (
                f"  Posição relativa (x, y, z): ({self.position[0]:.1f}, "
                f"{self.position[1]:.1f}, {self.position[2]:.1f}) m\n"
            )
            info += (
                f"  Posição absoluta (x, y, z): ({abs_pos[0]:.1f}, "
                f"{abs_pos[1]:.1f}, {abs_pos[2]:.1f}) m\n"
            )
            info += f"  Montada em: {self.mounted_on_vessel.name}\n"
            info += (
                f"  Velocidade da plataforma: ({vessel_vel[0]:.1f}, "
                f"{vessel_vel[1]:.1f}, {vessel_vel[2]:.1f}) m/s\n"
            )

        info += f"  Elevação: {np.degrees(self.elevation):.1f}°\n"
        info += f"  Azimute: {np.degrees(self.azimuth):.1f}°\n"
        info += f"  Taxa de tiro: {self.rate_of_fire:.1f} tiros/min\n"
        info += f"  Velocidade na boca: {self.muzzle_velocity:.1f} m/s\n"
        return info

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Weapon(name={self.name!r}, elevation_deg={self.elevation_deg:.3f}, "
            f"azimuth_deg={self.azimuth_deg:.3f}, "
            f"muzzle_velocity_mps={self.muzzle_velocity})"
        )


__all__ = ["Weapon"]
