"""Pick firing points spaced by a target distance in range.

The Monte Carlo campaign cannot afford one thousand trajectories at every one
of the 600 elevations in the sweep, so it samples the range envelope on a
roughly uniform ladder -- about 100 m between consecutive aim points.  The
spacing is not exact because the available elevations are quantised, so the
selection widens its tolerance in stages rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class SpacingPolicy:
    """Tolerance ladder used when no candidate sits at the exact spacing.

    Attributes
    ----------
    spacing_m:
        Desired range step between consecutive points, in m.
    base_tolerance_m:
        First pass: accept the first candidate within this of the target
        spacing.
    max_tolerance_m:
        Second pass: accept the candidate closest to the target spacing within
        this wider band.
    minimum_gap_m:
        Last resort: accept the first candidate at least this far from the
        previous point, and flag it as out of tolerance.
    """

    spacing_m: float = 100.0
    base_tolerance_m: float = 20.0
    max_tolerance_m: float = 50.0
    minimum_gap_m: float = 50.0


def select_points_by_spacing(
    optimal: pd.DataFrame,
    policy: SpacingPolicy = SpacingPolicy(),
    elevation_max: Optional[float] = 39.6,
    elevation_min: Optional[float] = -15.0,
    range_column: str = "Alcance_x_m",
    elevation_column: str = "Elevacao_deg",
) -> pd.DataFrame:
    """Sample the zero-drift table on a roughly uniform range ladder.

    Parameters
    ----------
    optimal:
        Output of :func:`sixdof.montecarlo.sweep.optimal_azimuths`.
    policy:
        Spacing and tolerance ladder.
    elevation_max, elevation_min:
        Restrict the elevation band before selecting; ``None`` disables either
        bound.  The default upper bound of 39.6 deg is the maximum-range
        elevation of the reference gun.

    Returns
    -------
    pandas.DataFrame
        The selected rows, in decreasing elevation, with two extra columns:
        ``Diferenca_alcance_m`` (gap to the previous point) and
        ``Dentro_tolerancia`` (whether the gap met the base or max tolerance).
    """
    frame = optimal.copy()
    if elevation_max is not None:
        frame = frame[frame[elevation_column] <= elevation_max]
    if elevation_min is not None:
        frame = frame[frame[elevation_column] >= elevation_min]

    frame = frame.sort_values(elevation_column, ascending=False).reset_index(drop=True)
    if frame.empty:
        return frame

    selected: List[dict] = []
    first = frame.iloc[0].to_dict()
    first["Diferenca_alcance_m"] = 0.0
    first["Dentro_tolerancia"] = True
    selected.append(first)

    previous_range = float(frame.iloc[0][range_column])
    search_from = 1

    while search_from < len(frame):
        index, gap, within_tolerance = _next_index(
            frame, search_from, previous_range, policy, range_column
        )
        if index is None:
            break

        row = frame.iloc[index].to_dict()
        row["Diferenca_alcance_m"] = gap
        row["Dentro_tolerancia"] = within_tolerance
        selected.append(row)

        previous_range = float(frame.iloc[index][range_column])
        search_from = index + 1

    return pd.DataFrame(selected)


def _next_index(
    frame: pd.DataFrame,
    start: int,
    previous_range: float,
    policy: SpacingPolicy,
    range_column: str,
):
    """Find the next point, widening the tolerance in three stages."""
    ranges = frame[range_column].values

    # Stage 1: first candidate within the base tolerance of the target spacing.
    for index in range(start, len(frame)):
        gap = abs(previous_range - float(ranges[index]))
        if abs(gap - policy.spacing_m) <= policy.base_tolerance_m:
            return index, gap, True

    # Stage 2: best candidate within the wider tolerance.
    best_index = None
    best_gap = None
    for index in range(start, len(frame)):
        gap = abs(previous_range - float(ranges[index]))
        error = abs(gap - policy.spacing_m)
        if error <= policy.max_tolerance_m:
            if best_index is None or error < abs(best_gap - policy.spacing_m):
                best_index, best_gap = index, gap
    if best_index is not None:
        return best_index, best_gap, True

    # Stage 3: any candidate at least ``minimum_gap_m`` away.
    for index in range(start, len(frame)):
        gap = abs(previous_range - float(ranges[index]))
        if gap >= policy.minimum_gap_m:
            return index, gap, False

    return None, None, False


__all__ = ["select_points_by_spacing", "SpacingPolicy"]
