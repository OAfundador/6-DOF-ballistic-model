"""Closest approach of a trajectory to a list of fixed points in space.

The naval campaign asks where a shot *lands*: reduce each trajectory to its
impact point and compare that with the target.  An air target is never landed
on, so the question becomes how near the round *passed*, in three dimensions,
while still in flight.  That is the only thing this module does.

The points are static.  Nothing here knows about a target's motion, and no
sample is matched to a time -- a moving target enters as a list of the
positions it occupies at the successive opportunities, and the choice of which
opportunity a shot serves is made by whoever reads the results.  This keeps the
geometry reusable: a corridor of drone waypoints, a set of aim points, or a
line of range markers are all the same problem to it.

Two pieces, because a sweep needs both:

:func:`closest_approach`
    One trajectory against every point, vectorised -- an ``(n_samples,
    n_points)`` distance matrix reduced along the samples.

:class:`NearestApproach`
    The accumulator over a whole sweep.  Feed it one trajectory at a time and
    it keeps, for each point, the nearest pass seen so far and the label of the
    trajectory that made it.  That is the reduction the anti-air point
    selection is built on, and it is what :meth:`AngleSweep.run
    <sixdof.montecarlo.sweep.AngleSweep.run>` calls through its ``reduce``
    hook.

A "trajectory" here is anything carrying ``x``, ``y``, ``z`` and ``t`` as equal
length sequences -- :class:`~sixdof.trajectory.Trajectory` does, and so does a
three-line stub in a test.

Examples
--------
>>> import numpy as np
>>> class Straight:
...     t = np.linspace(0.0, 1.0, 11)
...     x = np.linspace(0.0, 100.0, 11)
...     y = np.zeros(11)
...     z = np.zeros(11)
>>> tracker = NearestApproach([(50.0, 3.0, 0.0), (100.0, 0.0, 0.0)])
>>> _ = tracker.absorb(Straight(), label="shot A")
>>> [round(float(c.distance_m), 6) for c in tracker.best()]
[3.0, 0.0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ClosestApproach",
    "NearestApproach",
    "closest_approach",
    "trajectory_points",
]


def trajectory_points(trajectory) -> np.ndarray:
    """Stack a trajectory's ``x``/``y``/``z`` into an ``(n_samples, 3)`` array."""
    return np.column_stack(
        [
            np.asarray(trajectory.x),
            np.asarray(trajectory.y),
            np.asarray(trajectory.z),
        ]
    )


def closest_approach(
    trajectory, points: Sequence[Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest sample of one trajectory to each of several fixed points.

    Parameters
    ----------
    trajectory:
        Anything with ``x``, ``y``, ``z`` (and ``t``, for the caller's use).
    points:
        ``(n_points, 3)`` positions, in m, in the same frame as the trajectory.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sample_index, distance_m)``, one entry per point.  ``sample_index``
        indexes the trajectory's own samples, so ``trajectory.t[i]`` is when the
        pass happened.

    Notes
    -----
    Distance is measured between the integrator's *samples*, not along the
    interpolated path, so the resolution of the answer is the step of the
    trajectory -- with ``max_step=0.1`` and a round doing 700 m/s that is 70 m
    between samples near the muzzle and much less near the summit.  For the
    anti-air work this is comfortably finer than the grid of laying angles,
    which is what actually limits the accuracy; if it ever stops being so, the
    fix is a denser output grid on the integrator rather than anything here.

    Examples
    --------
    >>> import numpy as np
    >>> class Straight:
    ...     t = np.linspace(0.0, 1.0, 11)
    ...     x = np.linspace(0.0, 100.0, 11)
    ...     y = np.zeros(11)
    ...     z = np.zeros(11)
    >>> sample, distance = closest_approach(Straight(), [(30.0, 4.0, 0.0)])
    >>> int(sample[0]), round(float(distance[0]), 6)
    (3, 4.0)
    """
    positions = np.asarray(points, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("points must be an (n_points, 3) array of positions")

    samples = trajectory_points(trajectory)
    # (n_samples, n_points) distances in one shot.
    deltas = samples[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    nearest_sample = np.argmin(distances, axis=0)
    nearest_distance = distances[nearest_sample, np.arange(len(positions))]
    return nearest_sample, nearest_distance


@dataclass(frozen=True)
class ClosestApproach:
    """How near one trajectory came to one point.

    Attributes
    ----------
    point:
        Index into the list of points given to :class:`NearestApproach`.
    label:
        Whatever identified the trajectory -- a ``(elevation, azimuth)`` tuple
        in the sweeps, but the accumulator never looks inside it.
    distance_m:
        The miss distance, in m.
    position_m:
        The point on the trajectory where it happened, ``(x, y, z)`` in m.
    time_s:
        Time of flight at that sample, in s.
    sample:
        Index of that sample in the trajectory.
    """

    point: int
    label: Any
    distance_m: float
    position_m: np.ndarray
    time_s: float
    sample: int


class NearestApproach:
    """Keep the nearest pass to each point over a whole sweep.

    One instance is fed every trajectory of a sweep, in any order, and holds
    only the running best -- so a sweep of tens of thousands of trajectories
    costs a few hundred bytes per point rather than storing the flights.

    Parameters
    ----------
    points:
        ``(n_points, 3)`` positions to be approached, in m.

    Examples
    --------
    >>> import numpy as np
    >>> class Straight:
    ...     def __init__(self, offset):
    ...         self.t = np.linspace(0.0, 1.0, 11)
    ...         self.x = np.linspace(0.0, 100.0, 11)
    ...         self.y = np.full(11, offset)
    ...         self.z = np.zeros(11)
    >>> tracker = NearestApproach([(50.0, 5.0, 0.0)])
    >>> _ = tracker.absorb(Straight(0.0), label="low")
    >>> _ = tracker.absorb(Straight(4.0), label="high")
    >>> tracker.best()[0].label, round(float(tracker.best()[0].distance_m), 6)
    ('high', 1.0)
    """

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be an (n_points, 3) array of positions")
        self._best: List[Optional[ClosestApproach]] = [None] * len(self.points)
        self._best_distance = np.full(len(self.points), np.inf)

    def __len__(self) -> int:
        return len(self.points)

    # ------------------------------------------------------------------
    def absorb(self, trajectory, label: Any = None) -> ClosestApproach:
        """Score one trajectory against every point, updating the running best.

        Returns
        -------
        ClosestApproach
            This trajectory's *own* nearest miss -- the point it came closest
            to, whether or not it is the best that point has seen.  That is
            what a sweep reports as "the waypoint this pair served".

        Notes
        -----
        Ties keep the earlier trajectory, so a sweep's answer does not depend
        on the order its grid happens to be walked in when two pairs are
        exactly equidistant.
        """
        samples = trajectory_points(trajectory)
        deltas = samples[:, None, :] - self.points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        nearest_sample = np.argmin(distances, axis=0)
        nearest_distance = distances[nearest_sample, np.arange(len(self.points))]
        times = np.asarray(trajectory.t)

        def record(point: int) -> ClosestApproach:
            index = int(nearest_sample[point])
            return ClosestApproach(
                point=point,
                label=label,
                distance_m=float(nearest_distance[point]),
                position_m=samples[index].copy(),
                time_s=float(times[index]),
                sample=index,
            )

        for point in range(len(self.points)):
            if nearest_distance[point] < self._best_distance[point]:
                self._best_distance[point] = nearest_distance[point]
                self._best[point] = record(point)

        return record(int(np.argmin(nearest_distance)))

    # ------------------------------------------------------------------
    def best(self) -> List[Optional[ClosestApproach]]:
        """The nearest pass to each point so far; ``None`` where nothing flew."""
        return list(self._best)

    def unreached(self) -> List[int]:
        """Points no trajectory has been scored against yet."""
        return [i for i, candidate in enumerate(self._best) if candidate is None]

    def distances(self) -> np.ndarray:
        """The running best distance per point, ``inf`` where nothing flew."""
        return self._best_distance.copy()
