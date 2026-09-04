"""Proximity fuze: where along the trajectory the round bursts.

Two paths reach the same answer, and both are supported:

1. the integration itself was stopped by the fuze event (see
   :func:`sixdof.events.make_proximity_fuze_event`), in which case the last
   sample *is* the burst point;
2. the trajectory was integrated to ground impact, and the burst point has to
   be recovered by scanning the samples for the first one inside the burst
   radius.

If the round never enters the radius, the closest approach is reported instead,
flagged as not triggered.  That case still carries useful information: it is
the miss distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BurstPoint:
    """State of the round at the moment the warhead functions.

    Attributes
    ----------
    triggered:
        ``True`` when the round actually entered the burst radius.
    source:
        How the point was found: ``"evento_integrador"``, ``"amostra_trajetoria"``
        or ``"menor_distancia_amostrada"``.
    index:
        Trajectory sample index.
    time_s:
        Time of burst, in s.
    position_m, velocity_mps, axis_i:
        Position, velocity and axis of symmetry at burst.
    distance_m:
        Distance from the burst point to the target centre, in m.
    """

    triggered: bool
    source: str
    index: int
    time_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    axis_i: np.ndarray
    distance_m: float

    @property
    def speed_mps(self) -> float:
        """Magnitude of the velocity at burst, in m/s."""
        return float(np.linalg.norm(self.velocity_mps))


@dataclass
class ProximityFuze:
    """A proximity fuze with a burst radius and an arming delay.

    Parameters
    ----------
    target_center:
        ``(x, y, z)`` the fuze is looking for, in m.  Also what
        :meth:`sixdof.simulator.BallisticSimulator.simulate` uses as the
        terminal-event centre when this object is passed as ``fuze=``.
    radius_m:
        Burst radius, in m.  Default 24.38 m (80 ft), the figure quoted for the
        5"/38 VT round.
    arm_time_s:
        Safety delay after launch, in s.

    Examples
    --------
    >>> fuze = ProximityFuze(target_center=(16673.0, 200.0, 0.7))
    >>> fuze.radius_m
    24.38
    """

    target_center: Sequence[float]
    radius_m: float = 24.38
    arm_time_s: float = 0.5

    def __post_init__(self) -> None:
        self.target_center = np.array(self.target_center, dtype=float)

    def find_burst(self, trajectory) -> Optional[BurstPoint]:
        """Locate the burst point along ``trajectory``.

        Parameters
        ----------
        trajectory:
            Anything exposing ``t``, ``x``, ``y``, ``z``, ``V1``-``V3``,
            ``i1``-``i3`` as sequences, and optionally ``stop_reason``.

        Returns
        -------
        BurstPoint or None
            ``None`` only when no sample is past the arming time.
        """
        center = self.target_center

        if getattr(trajectory, "stop_reason", None) == "fuze":
            index = len(trajectory.t) - 1
            return self._sample(trajectory, index, center, True, "evento_integrador")

        best_index: Optional[int] = None
        best_distance = inf

        for i in range(len(trajectory.t)):
            if trajectory.t[i] < self.arm_time_s:
                continue

            position = np.array(
                [trajectory.x[i], trajectory.y[i], trajectory.z[i]], dtype=float
            )
            distance = float(np.linalg.norm(position - center))

            if distance < best_distance:
                best_index = i
                best_distance = distance

            if distance <= self.radius_m + 1e-6:
                return self._sample(trajectory, i, center, True, "amostra_trajetoria")

        if best_index is None:
            return None

        return self._sample(
            trajectory, best_index, center, False, "menor_distancia_amostrada"
        )

    @staticmethod
    def _sample(trajectory, index: int, center: np.ndarray, triggered: bool, source: str):
        """Build a :class:`BurstPoint` from one trajectory sample."""
        position = np.array(
            [trajectory.x[index], trajectory.y[index], trajectory.z[index]], dtype=float
        )
        return BurstPoint(
            triggered=triggered,
            source=source,
            index=index,
            time_s=float(trajectory.t[index]),
            position_m=position,
            velocity_mps=np.array(
                [trajectory.V1[index], trajectory.V2[index], trajectory.V3[index]],
                dtype=float,
            ),
            axis_i=np.array(
                [trajectory.i1[index], trajectory.i2[index], trajectory.i3[index]],
                dtype=float,
            ),
            distance_m=float(np.linalg.norm(position - center)),
        )


__all__ = ["ProximityFuze", "BurstPoint"]
