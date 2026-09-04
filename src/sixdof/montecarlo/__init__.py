"""Monte Carlo campaign: the pipeline that produced the thesis results.

The four stages, in order:

1. :class:`~sixdof.montecarlo.sweep.AngleSweep` integrates the elevation x
   azimuth grid, and :func:`~sixdof.montecarlo.sweep.optimal_azimuths` reduces
   it to the azimuth that zeroes the drift at each elevation;
2. :func:`~sixdof.montecarlo.selection.select_points_by_spacing` samples that
   table onto a ladder of aim points about 100 m apart in range;
3. :class:`~sixdof.montecarlo.dispersion.MonteCarloCampaign` fires a thousand
   perturbed rounds at each aim point and scores hits against a fleet of hulls;
4. :func:`~sixdof.montecarlo.cost.expected_engagement_cost` turns the resulting
   hit rates into an expected engagement cost, with
   :func:`~sixdof.montecarlo.cost.wald_interval` for the error bars.

Each stage writes a table that the next one reads, so a stage can be re-run
from stored intermediate results without repeating the one before it.  Stage 1
is the expensive one -- twenty thousand trajectories -- and its output ships in
``data/``.
"""

from .cost import (
    EngagementCost,
    Z_95,
    expected_engagement_cost,
    margin_of_error,
    wald_interval,
    wilson_interval,
)
from .dispersion import AimPoint, DispersionSettings, MonteCarloCampaign, PointResult
from .selection import SpacingPolicy, select_points_by_spacing
from .sweep import (
    AngleSweep,
    SweepGrid,
    inclusive_range,
    max_range_shot,
    optimal_azimuths,
)

__all__ = [
    # stage 1
    "AngleSweep",
    "SweepGrid",
    "optimal_azimuths",
    "max_range_shot",
    "inclusive_range",
    # stage 2
    "select_points_by_spacing",
    "SpacingPolicy",
    # stage 3
    "MonteCarloCampaign",
    "DispersionSettings",
    "AimPoint",
    "PointResult",
    # stage 4
    "expected_engagement_cost",
    "EngagementCost",
    "wald_interval",
    "wilson_interval",
    "margin_of_error",
    "Z_95",
]
