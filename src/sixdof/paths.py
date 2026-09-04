"""Filesystem locations of the data files shipped with the repository."""

from __future__ import annotations

from pathlib import Path

#: Root of the repository checkout (``src/sixdof/paths.py`` -> two levels up).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

#: Directory holding the tabulated inputs used by the examples.
DATA_DIR = REPO_ROOT / "data"

#: The 5"/38 coefficients as the seven the equations read, on the full
#: (Mach, alpha) grid.  This is the default, and the table the thesis campaign
#: was flown on: loading it reproduces the published trajectories bit for bit.
#:
#: The name says ``spin73`` because this table does **not** meet the McCoy
#: contract that :mod:`sixdof.aerodynamics` documents, in two known ways -- both
#: inherited from the thesis, both kept deliberately so the published results
#: stay reproducible:
#:
#: 1. ``CLP``, ``CMQ``, ``CYP`` and ``CNP`` come from a SPIN-73 tabulation,
#:    nondimensionalised on ``(pd/2V)``.  McCoy's equations want ``(pd/V)``, so
#:    these four are **twice** what the model should read.  Symptom: spin at
#:    impact 94 rev/s where an independent code gives 129.
#: 2. ``CD`` was assembled with the yaw term subtracted rather than added, so
#:    drag falls with angle of attack instead of rising.  Worth about -0.3 % of
#:    range on the thesis trajectory, and more at high yaw.
#:
#: ``docs/table_5in38_provenance.md`` carries the derivation and the numbers.
AERO_COEFFICIENTS_5IN38 = DATA_DIR / "aero_coefficients_5in38_spin73.npz"

#: The same seven, same two caveats, as an editable two-sheet spreadsheet.
AERO_WORKBOOK_5IN38 = DATA_DIR / "aero_coefficients_5in38_spin73_sheets.xlsx"

#: The *source* table those were derived from, in its own tabulation
#: convention.  The package does not read it: converting a source table is the
#: user's job, and ``examples/07_bring_your_own_table.py`` is the worked case.
#: It ships for provenance, and the frozen engine in
#: ``tests/reference/legacy_motor.py`` reads it directly.
#:
#: The file carries an ``.xlsx`` extension because the payload really is an
#: Excel workbook; the upstream repository shipped the same bytes under the
#: name ``Coeficientes que vi 2 casas.csv``, which ``pandas.read_excel`` still
#: parsed correctly.  The path is fixed: that frozen engine hard-codes it.
AERO_SOURCE_5IN38 = DATA_DIR / "aero_coefficients_5in38.xlsx"

#: Elevation/azimuth pairs that zero the lateral drift (angle-sweep output).
OPTIMAL_AZIMUTHS = DATA_DIR / "optimal_azimuths_zero_drift.xlsx"

#: Firing points spaced by roughly 100 m in range (Monte Carlo input).
SELECTED_POINTS_100M = DATA_DIR / "selected_points_100m.xlsx"

#: The Monte Carlo campaign results published in the thesis: 163 aim points,
#: 1000 rounds each, hit counts against six hulls.  Kept so the campaign can be
#: checked point by point without re-running 163 000 trajectories.
PUBLISHED_CAMPAIGN = DATA_DIR / "monte_carlo_campanha_publicada.xlsx"

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "AERO_COEFFICIENTS_5IN38",
    "AERO_WORKBOOK_5IN38",
    "AERO_SOURCE_5IN38",
    "OPTIMAL_AZIMUTHS",
    "SELECTED_POINTS_100M",
    "PUBLISHED_CAMPAIGN",
]
