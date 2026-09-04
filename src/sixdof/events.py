"""Terminal events that stop the integration.

``scipy.integrate.solve_ivp`` locates the root of each event function and, when
the event is marked terminal, stops there.  The two events below are the ones
the thesis needs: hitting the water/ground, and a proximity fuze arming and
then closing on a target.

Each event is built by a small factory so that the parameters (target centre,
burst radius, arming delay) stay explicit instead of being captured in an
anonymous closure.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

#: Index of the vertical coordinate inside the state vector.
_Y_INDEX = 10

#: Indices of the position block inside the state vector.
_POSITION_INDICES = (9, 10, 11)


def make_ground_event(ground_level: float = 0.0) -> Callable[[float, np.ndarray], float]:
    """Event that fires when the projectile descends through ``ground_level``.

    ``direction = -1`` restricts the root to downward crossings, so a gun
    emplaced above the water line does not trigger at launch.
    """

    def ground_event(t: float, y: np.ndarray) -> float:
        return y[_Y_INDEX] - ground_level

    ground_event.direction = -1
    ground_event.terminal = True
    return ground_event


def make_proximity_fuze_event(
    target_center: Sequence[float],
    radius_m: float = 24.38,
    arm_time_s: float = 0.5,
) -> Callable[[float, np.ndarray], float]:
    """Event that fires when the round first enters ``radius_m`` of the target.

    Parameters
    ----------
    target_center:
        ``(x, y, z)`` of the target centre, in m.
    radius_m:
        Burst radius of the fuze, in m.  The default 24.38 m is 80 ft, the
        figure used for the 5"/38 VT round.
    arm_time_s:
        Safety delay after launch during which the fuze cannot fire.  Before
        it elapses the event function returns a positive constant, so no root
        can be found.

    Notes
    -----
    The returned function is discontinuous at ``arm_time_s``.  That is
    harmless here because the discontinuity is a step *upwards* away from
    zero: the root finder only ever brackets a genuine inward crossing.
    """
    center = np.array(target_center, dtype=float)

    def fuze_event(t: float, y: np.ndarray) -> float:
        if t < arm_time_s:
            return 1.0
        pos = np.array([y[9], y[10], y[11]], dtype=float)
        return float(np.linalg.norm(pos - center) - radius_m)

    fuze_event.direction = -1
    fuze_event.terminal = True
    return fuze_event


__all__ = ["make_ground_event", "make_proximity_fuze_event"]
