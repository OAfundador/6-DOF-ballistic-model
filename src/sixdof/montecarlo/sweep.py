"""Elevation/azimuth sweep and the zero-drift azimuth it produces.

A spinning shell drifts sideways, and the drift grows with time of flight, so
the azimuth that puts the round on the line of sight is a function of elevation.
The thesis finds it by brute force: integrate the whole elevation x azimuth
grid, then for each elevation keep the azimuth minimising ``|z_impact|``.  The
resulting table is the input to the point selection and the Monte Carlo
campaign.

The sweep proper and what is *made* of it are separate concerns, and worth
keeping so.  :meth:`AngleSweep.run` walks the grid and reduces each flight to
the six numbers of :data:`SWEEP_COLUMNS`, which is all the thesis needs; its
``reduce`` hook hands the whole trajectory to a caller that needs more.  The
anti-air point selection is exactly that case -- the round never lands on an
air target, so the useful reduction is the closest approach in three
dimensions, which lives in :mod:`sixdof.montecarlo.proximity` and is driven
through this same loop rather than a second copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from ..simulator import BallisticSimulator
    from ..trajectory import Trajectory

#: Column names of the raw sweep table.
SWEEP_COLUMNS = (
    "Elevacao_deg",
    "Azimute_deg",
    "Alcance_x_m",
    "Desvio_z_m",
    "Altura_max_m",
    "Tempo_voo_s",
)

#: Column names of the zero-drift azimuth table.
OPTIMAL_COLUMNS = (
    "Elevacao_deg",
    "Azimute_otimo_deg",
    "Desvio_z_resultante_m",
    "Alcance_x_m",
    "Altura_max_m",
    "Tempo_voo_s",
)


def inclusive_range(start: float, stop: float, step: float, decimals: int = 2) -> Iterator[float]:
    """Yield ``start`` to ``stop`` inclusive, rounding each value.

    Rounding at every step is what the original loops did, and it matters: it
    keeps the grid values exactly representable as, say, ``-1.65`` so that later
    equality masks on the azimuth column match.
    """
    if step == 0:
        raise ValueError("step must be non-zero")
    value = round(start, decimals)
    if step > 0:
        while value <= stop:
            yield value
            value = round(value + step, decimals)
    else:
        while value >= stop:
            yield value
            value = round(value + step, decimals)


@dataclass(frozen=True)
class SweepGrid:
    """The elevation x azimuth grid to integrate.

    Attributes
    ----------
    elevation_start, elevation_stop, elevation_step:
        Elevation range in degrees; the default sweeps downwards from 45 deg.
    azimuth_start, azimuth_stop, azimuth_step:
        Azimuth range in degrees.
    """

    elevation_start: float = 45.0
    elevation_stop: float = -15.0
    elevation_step: float = -0.1
    azimuth_start: float = -1.65
    azimuth_stop: float = 0.0
    azimuth_step: float = 0.05

    def elevations(self) -> list:
        return list(inclusive_range(self.elevation_start, self.elevation_stop,
                                    self.elevation_step, decimals=1))

    def azimuths(self) -> list:
        return list(inclusive_range(self.azimuth_start, self.azimuth_stop,
                                    self.azimuth_step, decimals=2))

    def __len__(self) -> int:
        return len(self.elevations()) * len(self.azimuths())


class AngleSweep:
    """Integrate a whole grid of laying angles and tabulate the impacts.

    Parameters
    ----------
    simulator:
        A configured :class:`~sixdof.simulator.BallisticSimulator`.  Its weapon
        is re-laid in place for every grid point.
    grid:
        The angles to cover.
    max_time:
        Integration horizon per shot, in s.
    w_j0, w_k0:
        Initial transverse rates, in rad/s.

    Notes
    -----
    The full default grid is 601 x 34 = 20 434 trajectories, which takes hours.
    Shrink :class:`SweepGrid` for a smoke run.
    """

    def __init__(
        self,
        simulator: "BallisticSimulator",
        grid: SweepGrid = SweepGrid(),
        max_time: float = 100.0,
        w_j0: float = 5.0,
        w_k0: float = 5.0,
    ) -> None:
        self.simulator = simulator
        self.grid = grid
        self.max_time = max_time
        self.w_j0 = w_j0
        self.w_k0 = w_k0

    def run(
        self,
        progress_every: int = 100,
        callback: Optional[Callable[[int, int, float, float, float], None]] = None,
        reduce: Optional[Callable[[float, float, "Trajectory"], None]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Integrate every grid point.

        Parameters
        ----------
        progress_every:
            Print a progress line every N shots; 0 disables it.
        callback:
            Called as ``(count, total, elevation, azimuth, range_m)`` after each
            shot, for custom progress reporting.
        reduce:
            Called as ``(elevation, azimuth, trajectory)`` after each shot, with
            the flight itself rather than a summary of it.  The returned table
            is unchanged; this is for callers whose question the six columns
            cannot answer.  The trajectory is discarded as soon as it returns,
            so the hook must take what it needs -- see
            :class:`~sixdof.montecarlo.proximity.NearestApproach`, which keeps
            only a running best::

                tracker = NearestApproach(waypoints)
                sweep.run(reduce=lambda e, a, traj: tracker.absorb(traj, (e, a)))

        Returns
        -------
        pandas.DataFrame
            One row per shot, columns :data:`SWEEP_COLUMNS`.
        """
        elevations = self.grid.elevations()
        azimuths = self.grid.azimuths()
        total = len(elevations) * len(azimuths)

        records = []
        count = 0

        for azimuth in azimuths:
            if verbose:
                print(f"\n--- Azimute: {azimuth:.2f}° ---")

            for elevation in elevations:
                count += 1
                self.simulator.weapon.set_firing_angles(
                    elevation_deg=elevation, azimuth_deg=azimuth
                )
                trajectory = self.simulator.simulate(
                    max_time=self.max_time,
                    alpha0_deg=0.0,
                    beta0_deg=0.0,
                    w_j0=self.w_j0,
                    w_k0=self.w_k0,
                    verbose=False,
                )

                records.append(
                    {
                        "Elevacao_deg": elevation,
                        "Azimute_deg": azimuth,
                        "Alcance_x_m": trajectory.max_range,
                        "Desvio_z_m": float(trajectory.z[-1]),
                        "Altura_max_m": trajectory.max_altitude,
                        "Tempo_voo_s": trajectory.flight_time,
                    }
                )

                if reduce is not None:
                    reduce(elevation, azimuth, trajectory)

                if callback is not None:
                    callback(count, total, elevation, azimuth, trajectory.max_range)
                elif verbose and progress_every and (count % progress_every == 0 or count == total):
                    print(
                        f"  [{count}/{total}] Elev: {elevation:5.1f}° | "
                        f"Azim: {azimuth:5.2f}° | "
                        f"Alcance: {trajectory.max_range/1000:6.2f} km"
                    )

        return pd.DataFrame.from_records(records, columns=list(SWEEP_COLUMNS))


def optimal_azimuths(sweep: pd.DataFrame) -> pd.DataFrame:
    """For each elevation, the azimuth that minimises the lateral drift.

    Parameters
    ----------
    sweep:
        Output of :meth:`AngleSweep.run`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`OPTIMAL_COLUMNS`, one row per distinct elevation.
    """
    records = []
    for elevation in np.unique(sweep["Elevacao_deg"].values):
        mask = np.abs(sweep["Elevacao_deg"].values - elevation) < 0.001
        block = sweep[mask]
        best = block.iloc[int(np.argmin(np.abs(block["Desvio_z_m"].values)))]
        records.append(
            {
                "Elevacao_deg": elevation,
                "Azimute_otimo_deg": best["Azimute_deg"],
                "Desvio_z_resultante_m": best["Desvio_z_m"],
                "Alcance_x_m": best["Alcance_x_m"],
                "Altura_max_m": best["Altura_max_m"],
                "Tempo_voo_s": best["Tempo_voo_s"],
            }
        )
    return pd.DataFrame.from_records(records, columns=list(OPTIMAL_COLUMNS))


def max_range_shot(sweep: pd.DataFrame) -> Tuple[float, float, float, float]:
    """The longest-range row of a sweep.

    Returns
    -------
    tuple
        ``(elevation_deg, azimuth_deg, range_m, drift_m)``.
    """
    row = sweep.loc[sweep["Alcance_x_m"].idxmax()]
    return (
        float(row["Elevacao_deg"]),
        float(row["Azimute_deg"]),
        float(row["Alcance_x_m"]),
        float(row["Desvio_z_m"]),
    )


__all__ = [
    "AngleSweep",
    "SweepGrid",
    "optimal_azimuths",
    "max_range_shot",
    "inclusive_range",
    "SWEEP_COLUMNS",
    "OPTIMAL_COLUMNS",
]
