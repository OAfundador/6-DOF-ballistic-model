"""The aerodynamic coefficients the equations of motion read.

There are seven of them, and the model needs nothing else:

===== ================================ =========================
Name  Term                             Depends on
===== ================================ =========================
CD    drag force, ``C_D``              Mach and angle of attack
CLA   lift force, ``C_Lalpha``         Mach and angle of attack
CNP   Magnus moment, ``C_Mpalpha``     Mach and angle of attack
CYP   Magnus force, ``C_Npalpha``      Mach
CLP   spin damping moment, ``C_lp``    Mach
CMA   overturning moment, ``C_Malpha`` Mach
CMQ   pitch damping moment, ``C_Mq``   Mach
===== ================================ =========================

:class:`AerodynamicCoefficients` holds exactly those, in whatever form you have
them: constants, tables in Mach, grids in ``(Mach, alpha)``, or callables.

The reference at the input is McCoy
===================================

The seven are read as **McCoy, "Modern Exterior Ballistics", 2nd ed., ch. 2**
defines them, and :mod:`sixdof.dynamics` implements that chapter's equations
term for term.  Two things about that reference decide what your numbers have to
mean, and neither is visible from a column name:

**Nondimensionalisation.**  The four coefficients that multiply an angular rate
-- ``CLP``, ``CMQ``, ``CYP``, ``CNP`` -- are nondimensionalised on ``(pd/V)``,
per McCoy eq. (2.4).  McCoy flags the trap himself, in the paragraph after it:
the NACA aeroballistic system uses ``(pd/2V)`` instead, *"which accounts for the
factor of two difference in coefficients that depend on angular velocity"*.
A table in the NACA system therefore carries **twice** the value this model
wants for those four, and nothing in the arithmetic will complain -- the
trajectory simply comes out with the wrong spin decay, Magnus and pitch damping.
Halve them on the way in.  ``CMA`` does not depend on an angular rate and passes
through either system unchanged, which is the signature to check against.

**Axis system and sign.**  ``CD`` and ``CLA`` are wind-axis coefficients: drag
along the velocity vector, lift perpendicular to it.  A table that gives body-axis
axial and normal force instead needs the rotation through the angle of attack
before it gets here, and the sign of the axial coefficient depends on whether the
source counts it positive forward (McCoy's convention, where ``C_X ~ -C_D``) or
positive rearward.  Getting that backwards is silent too.

Where the numbers came from
===========================

Deliberately absent.  Wind-tunnel reports, range firings, CFD and the various
in-house prediction codes each tabulate their own intermediate quantities, and
each needs its own arithmetic to reach the seven above.  That arithmetic is
**yours**, not the model's: convert your source table once, hand over the seven,
and the engine never has to know which report they came out of.
``examples/07_bring_your_own_table.py`` is a worked conversion to copy from.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d, make_interp_spline

PathLike = Union[str, "os.PathLike[str]"]

#: The seven coefficients the equations of motion read, in McCoy's notation.
EQUATION_COEFFICIENTS: Sequence[str] = ("CD", "CLA", "CYP", "CLP", "CMA", "CNP", "CMQ")

#: What each one is, and where it enters the equations.
EQUATION_COEFFICIENT_MEANING = {
    "CD": "drag force (C_D)",
    "CLA": "lift force (C_Lalpha)",
    "CYP": "Magnus force (C_Npalpha)",
    "CLP": "spin damping moment (C_lp)",
    "CMA": "overturning moment (C_Malpha)",
    "CNP": "Magnus moment (C_Mpalpha)",
    "CMQ": "pitch damping moment (C_Mq)",
}

#: Of the seven, these vary with the angle of attack as well as Mach.
YAW_DEPENDENT: Sequence[str] = ("CD", "CLA", "CNP")

#: Internal key each is delivered under, because that is what
#: :func:`sixdof.dynamics.six_dof_rhs` indexes.  An implementation detail of the
#: right-hand side, kept because changing it would perturb nothing but would
#: churn the frozen-reference comparison for no gain.
RHS_KEY = {
    "CD": "CD_total",
    "CLA": "CLA_total",
    "CNP": "CNP_total",
    "CYP": "CYP",
    "CLP": "CLP",
    "CMA": "CMA",
    "CMQ": "CMQ",
}


def _scalar(value) -> float:
    """Unwrap an interpolator result to a Python float.

    ``RectBivariateSpline`` returns a ``(1, 1)`` array for scalar arguments and
    ``interp1d`` returns a 0-d array.  ``float()`` accepted both up to NumPy
    1.x, but since NumPy 2.0 it raises ``TypeError`` on anything that is not
    0-dimensional -- which is why the original single-file engine no longer runs
    on a current SciPy/NumPy stack.  Taking the first element is exactly what
    ``float()`` used to do, so the value is unchanged.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


class AerodynamicCoefficients:
    """The seven coefficients, in whatever form you have them.

    Parameters
    ----------
    mach_grid:
        Mach nodes the tabulated values are given on.  Not needed if every
        coefficient is a scalar or a callable.
    alpha_grid:
        Angle-of-attack nodes in radians, for coefficients given on a 2-D grid.
    CD, CLA, CYP, CLP, CMA, CNP, CMQ:
        Each may be

        * a scalar — constant everywhere;
        * a 1-D array over ``mach_grid`` — a function of Mach only;
        * a 2-D array over ``(mach_grid, alpha_grid)`` — a function of both;
        * a callable ``f(mach, alpha_rad)``.

        Anything omitted is zero, which is how you switch a term off.

    Notes
    -----
    Requests outside the tabulated envelope are clipped to it rather than
    extrapolated or rejected.

    Examples
    --------
    Constant coefficients, which is all some analyses need:

    >>> coeffs = AerodynamicCoefficients(CD=0.3, CLA=1.8, CMA=3.5, CMQ=-9.4)
    >>> round(coeffs.get_coefficients(2.0, 0.05)["CD_total"], 3)
    0.3

    A drag-only sanity case:

    >>> vacuum_but_for_drag = AerodynamicCoefficients(CD=0.25)
    >>> vacuum_but_for_drag.as_equation_names(1.0)["CLA"]
    0.0
    """

    def __init__(
        self,
        mach_grid: Optional[np.ndarray] = None,
        alpha_grid: Optional[np.ndarray] = None,
        **coefficients,
    ) -> None:
        unknown = set(coefficients) - set(EQUATION_COEFFICIENTS)
        if unknown:
            raise ValueError(
                f"unknown coefficient(s): {sorted(unknown)}; "
                f"expected any of {list(EQUATION_COEFFICIENTS)}. "
                "These seven are what the equations read, in McCoy's "
                "definitions; the intermediate quantities of a particular "
                "tabulation convention have to be converted before they get "
                "here. See examples/07_bring_your_own_table.py."
            )

        self.mach_grid = None if mach_grid is None else np.asarray(mach_grid, dtype=float)
        self.alpha_grid = None if alpha_grid is None else np.asarray(alpha_grid, dtype=float)

        self.mach_min = float(self.mach_grid.min()) if self.mach_grid is not None else -np.inf
        self.mach_max = float(self.mach_grid.max()) if self.mach_grid is not None else np.inf

        #: The values exactly as supplied, so :meth:`save` can write them back
        #: without a round trip through the interpolants -- which would perturb
        #: the last bits even at the grid nodes.
        self._raw = {name: coefficients.get(name, 0.0) for name in EQUATION_COEFFICIENTS}

        self._evaluators = {
            name: self._make_evaluator(name, self._raw[name])
            for name in EQUATION_COEFFICIENTS
        }

        # Fast path for the Mach-only tables.  ``interp1d(kind="cubic")`` is
        # ``make_interp_spline(k=3)`` underneath, so stacking those coefficients
        # into one vector-valued spline returns bit-identical values from a
        # single call instead of one wrapped scalar call each.  Coefficient
        # lookup is ~77% of a trajectory's runtime and this is most of it.
        self._stacked_names: Sequence[str] = ()
        self._stacked_spline = None
        self._scalar_names: Sequence[str] = EQUATION_COEFFICIENTS
        self._build_stacked_evaluator()

    # ------------------------------------------------------------------
    def _build_stacked_evaluator(self) -> None:
        """Collect the 1-D Mach tables into a single vector-valued spline.

        Purely a speed change: ``make_interp_spline(x, Y, k=3)`` evaluates the
        same cubic as ``interp1d(x, y, kind="cubic")`` on each column, to the
        last bit.  Anything that is not a plain 1-D table keeps its own
        evaluator.
        """
        stacked_names = []
        columns = []
        for name in EQUATION_COEFFICIENTS:
            raw = self._raw[name]
            if callable(raw) or self.mach_grid is None:
                continue
            array = np.asarray(raw, dtype=float)
            if array.ndim == 1 and len(array) == len(self.mach_grid):
                stacked_names.append(name)
                columns.append(array)

        if len(stacked_names) < 2:
            return  # not worth a second code path

        self._stacked_names = tuple(stacked_names)
        self._stacked_spline = make_interp_spline(
            self.mach_grid, np.column_stack(columns), k=3
        )
        self._scalar_names = tuple(
            name for name in EQUATION_COEFFICIENTS if name not in self._stacked_names
        )

    # ------------------------------------------------------------------
    def _make_evaluator(self, name: str, value):
        """Turn a scalar / 1-D table / 2-D grid / callable into ``f(mach, alpha)``."""
        if callable(value):
            return value

        array = np.asarray(value, dtype=float)

        if array.ndim == 0:
            constant = float(array)
            return lambda mach, alpha: constant

        if array.ndim == 1:
            if self.mach_grid is None:
                raise ValueError(f"{name}: a 1-D table needs mach_grid")
            if len(array) != len(self.mach_grid):
                raise ValueError(
                    f"{name}: 1-D table has {len(array)} values but mach_grid has "
                    f"{len(self.mach_grid)}"
                )
            spline = interp1d(
                self.mach_grid, array, kind="cubic", bounds_error=False,
                fill_value=(array[0], array[-1]),
            )
            return lambda mach, alpha: _scalar(spline(mach))

        if array.ndim == 2:
            if self.mach_grid is None or self.alpha_grid is None:
                raise ValueError(f"{name}: a 2-D grid needs mach_grid and alpha_grid")
            if array.shape != (len(self.mach_grid), len(self.alpha_grid)):
                raise ValueError(
                    f"{name}: 2-D grid is {array.shape}, expected "
                    f"({len(self.mach_grid)}, {len(self.alpha_grid)})"
                )
            spline = RectBivariateSpline(self.mach_grid, self.alpha_grid, array, kx=3, ky=3)
            return lambda mach, alpha: _scalar(spline(mach, alpha))

        raise ValueError(f"{name}: expected a scalar, a 1-D table, a 2-D grid or a callable")

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def get_coefficients(self, mach: float, alpha_rad: float = 0.0) -> Dict[str, float]:
        """Evaluate the seven, keyed as :func:`sixdof.dynamics.six_dof_rhs` expects."""
        if self.mach_grid is not None:
            mach = np.clip(mach, self.mach_min, self.mach_max)
        if self.alpha_grid is not None:
            alpha_rad = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])

        values = {}
        if self._stacked_spline is not None:
            stacked = self._stacked_spline(mach)
            for name, value in zip(self._stacked_names, stacked):
                values[RHS_KEY[name]] = float(value)
        for name in self._scalar_names:
            values[RHS_KEY[name]] = float(self._evaluators[name](mach, alpha_rad))
        return values

    def as_equation_names(self, mach: float, alpha_rad: float = 0.0) -> Dict[str, float]:
        """Same values, keyed by :data:`EQUATION_COEFFICIENTS` instead."""
        values = self.get_coefficients(mach, alpha_rad)
        return {name: values[RHS_KEY[name]] for name in EQUATION_COEFFICIENTS}

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_mach_table(
        cls,
        table: Union[Mapping[str, Iterable[float]], "pd.DataFrame", PathLike],
        mach_column: str = "Mach",
    ) -> "AerodynamicCoefficients":
        """Build from the simplest possible table: Mach plus the seven columns.

        The table may be a workbook path, a DataFrame or a mapping, and needs
        columns ``Mach, CD, CLA, CYP, CLP, CMA, CNP, CMQ``; missing ones are
        zero.

        Warning
        -------
        A Mach-only table has **no yaw dependence**.  For ``CD`` and ``CLA``
        that is a modest approximation.  For ``CNP`` it is not an approximation
        at all: the Magnus moment is odd in the angle of attack, so a single
        Mach-indexed value cannot represent it, and a table sampled at zero yaw
        removes the term rather than approximating it.  Use this where that is
        intended, or where you supply a ``CNP`` that means something else.
        """
        if isinstance(table, pd.DataFrame):
            frame = table
        elif isinstance(table, Mapping):
            frame = pd.DataFrame(dict(table))
        else:
            frame = pd.read_excel(table)

        if mach_column not in frame.columns:
            raise ValueError(f"table must contain a {mach_column!r} column")

        mach_grid = np.asarray(frame[mach_column].values, dtype=float)
        columns = {
            name: np.asarray(frame[name].values, dtype=float)
            for name in EQUATION_COEFFICIENTS
            if name in frame.columns
        }
        return cls(mach_grid=mach_grid, **columns)

    @classmethod
    def from_workbook(
        cls, path: PathLike, alpha_column: str = "Alpha_deg", mach_column: str = "Mach"
    ) -> "AerodynamicCoefficients":
        """Read the two-sheet spreadsheet format written by :meth:`to_workbook`.

        Sheet ``mach_only`` carries ``Mach`` plus the four coefficients that
        depend on Mach alone (``CYP``, ``CLP``, ``CMA``, ``CMQ``).  Sheet
        ``yaw_dependent`` carries ``Mach``, ``Alpha_deg`` and the three that
        also depend on the angle of attack (``CD``, ``CLA``, ``CNP``), one row
        per grid point.

        This is the editable form: the seven, and the yaw dependence kept where
        it is real.
        """
        mach_only = pd.read_excel(path, sheet_name="mach_only")
        yaw = pd.read_excel(path, sheet_name="yaw_dependent")

        mach_grid = np.asarray(mach_only[mach_column].values, dtype=float)
        alpha_grid = np.radians(np.unique(np.asarray(yaw[alpha_column].values, dtype=float)))
        yaw_machs = np.unique(np.asarray(yaw[mach_column].values, dtype=float))

        if not np.array_equal(mach_grid, yaw_machs):
            raise ValueError(
                "the two sheets must use the same Mach nodes: "
                f"mach_only has {len(mach_grid)}, yaw_dependent has {len(yaw_machs)}"
            )

        columns = {
            name: np.asarray(mach_only[name].values, dtype=float)
            for name in EQUATION_COEFFICIENTS
            if name in mach_only.columns
        }
        for name in YAW_DEPENDENT:
            if name in yaw.columns:
                grid = yaw.pivot_table(index=mach_column, columns=alpha_column, values=name)
                columns[name] = np.asarray(grid.values, dtype=float)

        return cls(mach_grid=mach_grid, alpha_grid=alpha_grid, **columns)

    @classmethod
    def load(cls, path: PathLike) -> "AerodynamicCoefficients":
        """Load the tabulated grids written by :meth:`save`, losing nothing."""
        with np.load(str(path)) as data:
            mach_grid = data["mach_grid"]
            alpha_grid = data["alpha_grid"]
            columns = {
                name: data[name] for name in EQUATION_COEFFICIENTS if name in data
            }
        return cls(mach_grid=mach_grid, alpha_grid=alpha_grid, **columns)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def save(self, path: PathLike) -> None:
        """Write the tabulated grids to ``.npz``.

        The round trip through :meth:`load` is exact.  Callables cannot be
        serialised, so they are sampled onto the grid first.
        """
        if self.mach_grid is None:
            raise ValueError("nothing to save: this instance has no mach_grid")

        alpha_grid = self.alpha_grid if self.alpha_grid is not None else np.array([0.0])
        arrays = {"mach_grid": self.mach_grid, "alpha_grid": alpha_grid}
        for name in EQUATION_COEFFICIENTS:
            raw = self._raw[name]
            if callable(raw):
                arrays[name] = np.array(
                    [[raw(m, a) for a in alpha_grid] for m in self.mach_grid], dtype=float
                )
            else:
                # Write what was handed in, untouched.  Re-evaluating an
                # interpolant at its own nodes is not the identity in floating
                # point, and this file is meant to reload exactly.
                arrays[name] = np.asarray(raw, dtype=float)
        np.savez_compressed(str(path), **arrays)

    def to_workbook(
        self, path: PathLike, mach_values=None, alpha_deg_values=None
    ) -> None:
        """Write the editable two-sheet spreadsheet read by :meth:`from_workbook`.

        Parameters
        ----------
        mach_values:
            Mach nodes to tabulate; defaults to this instance's grid.
        alpha_deg_values:
            Yaw nodes in degrees; defaults to this instance's grid in degrees.
            A coarser grid keeps the file readable at some cost in fidelity.
        """
        machs = self.mach_grid if mach_values is None else np.asarray(mach_values, dtype=float)
        if machs is None:
            raise ValueError("provide mach_values for an instance without a mach_grid")

        if alpha_deg_values is None:
            alphas_deg = (
                np.degrees(self.alpha_grid) if self.alpha_grid is not None else np.array([0.0])
            )
        else:
            alphas_deg = np.asarray(alpha_deg_values, dtype=float)

        mach_only_rows = []
        for mach in machs:
            values = self.as_equation_names(float(mach), 0.0)
            mach_only_rows.append(
                {"Mach": float(mach), **{n: values[n] for n in ("CYP", "CLP", "CMA", "CMQ")}}
            )

        yaw_rows = []
        for mach in machs:
            for alpha_deg in alphas_deg:
                values = self.as_equation_names(float(mach), np.radians(float(alpha_deg)))
                yaw_rows.append(
                    {
                        "Mach": float(mach),
                        "Alpha_deg": float(alpha_deg),
                        **{n: values[n] for n in YAW_DEPENDENT},
                    }
                )

        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            pd.DataFrame(mach_only_rows).to_excel(writer, sheet_name="mach_only", index=False)
            pd.DataFrame(yaw_rows).to_excel(writer, sheet_name="yaw_dependent", index=False)

    def to_frame(self, alpha_deg: float = 0.0, mach_values=None) -> "pd.DataFrame":
        """A readable Mach-indexed slice at one angle of attack.

        Handy for inspecting a table, but it is a *slice*: rebuilding from it
        via :meth:`from_mach_table` drops the yaw dependence.
        """
        mach_values = self.mach_grid if mach_values is None else np.asarray(mach_values)
        if mach_values is None:
            raise ValueError("provide mach_values for an instance without a mach_grid")

        alpha = np.radians(alpha_deg)
        records = []
        for mach in mach_values:
            row = {"Mach": float(mach)}
            row.update(self.as_equation_names(float(mach), alpha))
            records.append(row)
        return pd.DataFrame.from_records(records)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        shape = "constant" if self.mach_grid is None else f"mach={len(self.mach_grid)}"
        if self.alpha_grid is not None:
            shape += f" x alpha={len(self.alpha_grid)}"
        return f"AerodynamicCoefficients({shape})"


def load_coefficients(path: PathLike, **kwargs) -> AerodynamicCoefficients:
    """Read a file of the seven coefficients, picking the reader by its shape.

    Recognised:

    * ``.npz`` — the grids written by :meth:`AerodynamicCoefficients.save`.
    * a workbook with sheets ``mach_only`` and ``yaw_dependent`` —
      :meth:`AerodynamicCoefficients.from_workbook`.
    * a workbook or CSV with ``Mach`` plus the seven columns —
      :meth:`AerodynamicCoefficients.from_mach_table`.

    All three carry the seven and nothing else.  A table in some source's own
    tabulation convention is not read here and is not guessed at: convert it
    first, so that the conversion is written down somewhere you can check,
    rather than inferred from column names.

    Raises
    ------
    ValueError
        If the file carries none of the seven.
    """
    path = str(path)

    if path.endswith(".npz"):
        return AerodynamicCoefficients.load(path)

    if path.endswith(".csv"):
        return AerodynamicCoefficients.from_mach_table(pd.read_csv(path), **kwargs)

    sheets = pd.read_excel(path, sheet_name=None)
    if {"mach_only", "yaw_dependent"} <= set(sheets):
        return AerodynamicCoefficients.from_workbook(path, **kwargs)

    frame = next(iter(sheets.values()))
    columns = set(frame.columns)

    if "Mach" in columns and set(EQUATION_COEFFICIENTS) & columns:
        return AerodynamicCoefficients.from_mach_table(frame, **kwargs)

    raise ValueError(
        f"{path}: none of the seven coefficients "
        f"{list(EQUATION_COEFFICIENTS)} are in this table (columns: "
        f"{sorted(map(str, columns))}). If it is in some other tabulation "
        "convention, convert it to the seven first -- "
        "examples/07_bring_your_own_table.py is a worked conversion."
    )


__all__ = [
    "AerodynamicCoefficients",
    "EQUATION_COEFFICIENTS",
    "EQUATION_COEFFICIENT_MEANING",
    "YAW_DEPENDENT",
    "RHS_KEY",
    "load_coefficients",
]
