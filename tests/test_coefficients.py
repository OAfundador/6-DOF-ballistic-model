"""The coefficient interface: the seven, in McCoy's terms, and nothing else.

The model takes seven numbers and knows nothing about where they came from.
This file checks that the boundary really is where the documentation says:

* :class:`AerodynamicCoefficients` accepts the seven in every form the docs
  promise, and rejects anything else — including a source table's intermediate
  columns, which are the user's to convert.
* **No executable code anywhere in** ``src/sixdof/`` **mentions a source
  convention.**  Not the engine, not the loader, not the presets.  The package
  has no adapter to leak one.
* The conversion that produced the shipped 5"/38 table lives in
  ``examples/07_bring_your_own_table.py``, outside the package, and still
  reproduces that table bit for bit — so the provenance stays checkable.

The last two groups measure what the known departures cost: a Mach-only table,
and the two convention mistakes the shipped table inherits from the thesis.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sixdof import (  # noqa: E402
    EQUATION_COEFFICIENTS,
    AerodynamicCoefficients,
    BallisticSimulator,
    load_coefficients,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.aerodynamics import YAW_DEPENDENT  # noqa: E402
from sixdof.paths import (  # noqa: E402
    AERO_COEFFICIENTS_5IN38,
    AERO_SOURCE_5IN38,
    AERO_WORKBOOK_5IN38,
)

#: The seven, as keyed inside the right-hand side.
RHS_KEYS = ("CD_total", "CLA_total", "CNP_total", "CYP", "CLP", "CMA", "CMQ")

#: Column names belonging to the tabulation convention the 5"/38 source is
#: written in.  None of these may appear in executable code inside the package.
SOURCE_CONVENTION_NAMES = (
    "CX0", "CX2", "CNA", "CNPA", "CNPA3", "CNPA5", "Match",
)


def _load_example():
    """Import the worked conversion, which lives outside the package."""
    path = REPO_ROOT / "examples" / "07_bring_your_own_table.py"
    spec = importlib.util.spec_from_file_location("bring_your_own_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


@pytest.fixture(scope="module")
def coefficients():
    """The shipped default: the seven, loaded from ``data/*.npz``."""
    return naval_5in38_coefficients()


@pytest.fixture(scope="module")
def converted(example):
    """The same numbers, rebuilt from the source table by the worked example."""
    return example.convert(AERO_SOURCE_5IN38)


def _run(coeffs, elevation=43.3):
    simulator = BallisticSimulator(
        naval_5in38_projectile(),
        naval_5in38_gun(elevation_deg=elevation),
        standard_atmosphere(),
        coeffs,
    )
    return simulator.simulate(verbose=False)


# ----------------------------------------------------------------------
# the model reads seven coefficients and nothing else
# ----------------------------------------------------------------------
def test_the_seven_are_what_the_rhs_reads():
    """The advertised set matches what ``six_dof_rhs`` actually indexes."""
    source = (REPO_ROOT / "src" / "sixdof" / "dynamics.py").read_text(encoding="utf-8")
    read_in_rhs = {key for key in RHS_KEYS if f'coeffs["{key}"]' in source}
    assert read_in_rhs == set(RHS_KEYS)
    assert len(EQUATION_COEFFICIENTS) == len(RHS_KEYS) == 7


def test_only_the_seven_come_out(coefficients):
    """No bookkeeping columns leak through the interface."""
    assert set(coefficients.get_coefficients(2.0, np.radians(3.0))) == set(RHS_KEYS)
    assert set(coefficients.as_equation_names(2.0, 0.05)) == set(EQUATION_COEFFICIENTS)


def test_source_specific_names_are_rejected():
    """A tabulation convention's intermediates are not model inputs."""
    with pytest.raises(ValueError, match="unknown coefficient"):
        AerodynamicCoefficients(CX0=0.2)
    with pytest.raises(ValueError, match="bring_your_own_table"):
        AerodynamicCoefficients(CNPA3=0.1)


def _code_without_prose(path: Path) -> str:
    """Source with comments and string literals removed.

    Docstrings may *discuss* the source conventions — that is documentation.
    What must not appear is executable code that knows about them.
    """
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


def test_no_module_in_the_package_mentions_a_source_convention():
    """The whole package, not just the engine, is free of the source's names.

    This is the boundary the design rests on.  There is no adapter inside
    ``sixdof`` for a convention to hide in, and this test is what keeps it that
    way: converting a source table is the user's job, done outside the package.
    """
    package = REPO_ROOT / "src" / "sixdof"
    modules = sorted(package.rglob("*.py"))
    assert len(modules) > 5, "expected to be scanning the whole package"

    for module in modules:
        code = _code_without_prose(module)
        for name in SOURCE_CONVENTION_NAMES:
            assert name not in code, f"{name} leaked into {module.relative_to(REPO_ROOT)}"


def test_the_conversion_lives_in_the_example():
    """And it really is written down somewhere — just not in the package."""
    path = REPO_ROOT / "examples" / "07_bring_your_own_table.py"
    code = _code_without_prose(path)
    for name in ("CX0", "CX2", "CNA", "CNPA3", "CNPA5"):
        assert name in code


# ----------------------------------------------------------------------
# supplying the seven
# ----------------------------------------------------------------------
def test_constants():
    """A constant set is one call, and unspecified coefficients are zero."""
    coeffs = AerodynamicCoefficients(CD=0.3, CLA=1.8, CMA=3.5)
    values = coeffs.get_coefficients(2.0, 0.05)
    assert values["CD_total"] == 0.3
    assert values["CLA_total"] == 1.8
    assert values["CMA"] == 3.5
    assert values["CYP"] == 0.0
    assert values["CNP_total"] == 0.0


def test_mach_table_interpolates_and_clamps():
    mach = np.linspace(0.5, 3.0, 6)
    coeffs = AerodynamicCoefficients(mach_grid=mach, CD=np.linspace(0.2, 0.4, 6))
    assert coeffs.get_coefficients(0.5)["CD_total"] == pytest.approx(0.2)
    assert coeffs.get_coefficients(3.0)["CD_total"] == pytest.approx(0.4)
    assert coeffs.get_coefficients(0.1)["CD_total"] == pytest.approx(0.2)
    assert coeffs.get_coefficients(9.0)["CD_total"] == pytest.approx(0.4)


def test_callable_coefficient():
    """A callable is taken as-is, which covers any closed-form model."""
    coeffs = AerodynamicCoefficients(CD=lambda mach, alpha: 0.2 + 0.5 * np.sin(alpha) ** 2)
    assert coeffs.get_coefficients(2.0, 0.0)["CD_total"] == pytest.approx(0.2)
    assert coeffs.get_coefficients(2.0, np.radians(10))["CD_total"] > 0.2


def test_two_dimensional_grid():
    mach = np.linspace(0.5, 3.0, 8)
    alpha = np.radians(np.linspace(-10, 10, 9))
    coeffs = AerodynamicCoefficients(
        mach_grid=mach, alpha_grid=alpha, CD=np.outer(mach, np.cos(alpha))
    )
    assert coeffs.get_coefficients(2.0, 0.0)["CD_total"] == pytest.approx(2.0, abs=1e-9)


def test_shapes_are_checked():
    mach = np.linspace(0.5, 3.0, 8)
    alpha = np.radians(np.linspace(-10, 10, 9))
    with pytest.raises(ValueError, match="2-D grid is"):
        AerodynamicCoefficients(mach_grid=mach, alpha_grid=alpha, CD=np.zeros((8, 5)))
    with pytest.raises(ValueError, match="needs mach_grid"):
        AerodynamicCoefficients(CD=[0.1, 0.2, 0.3])


def test_from_mach_table_reads_a_frame():
    frame = pd.DataFrame(
        {
            "Mach": [0.5, 1.0, 2.0, 3.0],
            "CD": [0.16, 0.30, 0.37, 0.29],
            "CLA": [1.58, 1.80, 2.46, 2.64],
            "CMA": [3.4, 3.7, 3.6, 3.5],
            "CMQ": [-9.4] * 4,
            "CLP": [-0.03] * 4,
            "CYP": [0.1] * 4,
            "CNP": [0.5] * 4,
        }
    )
    coeffs = AerodynamicCoefficients.from_mach_table(frame)
    values = coeffs.get_coefficients(1.0, np.radians(5.0))
    assert values["CD_total"] == pytest.approx(0.30)
    # Mach-only: the same value whatever the yaw.
    assert values["CD_total"] == coeffs.get_coefficients(1.0, 0.0)["CD_total"]


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_npz_round_trip_is_bit_identical(coefficients, tmp_path):
    """Saving and reloading loses nothing, down to the trajectory."""
    path = tmp_path / "coefficients.npz"
    coefficients.save(path)

    reference = _run(coefficients)
    result = _run(AerodynamicCoefficients.load(path))
    assert np.array_equal(reference.solution.y, result.solution.y)


def test_workbook_round_trip(coefficients, tmp_path):
    """Written on its own grid, the workbook reloads to the same values."""
    path = tmp_path / "coefficients.xlsx"
    machs = np.linspace(0.5, 4.0, 12)
    alphas = np.linspace(-10.0, 10.0, 11)
    coefficients.to_workbook(path, mach_values=machs, alpha_deg_values=alphas)

    reloaded = AerodynamicCoefficients.from_workbook(path)
    for mach in machs[::4]:
        for alpha_deg in alphas[::4]:
            a = coefficients.get_coefficients(float(mach), np.radians(float(alpha_deg)))
            b = reloaded.get_coefficients(float(mach), np.radians(float(alpha_deg)))
            for key in RHS_KEYS:
                assert a[key] == pytest.approx(b[key], abs=1e-9), key


def test_shipped_workbook_has_only_the_seven():
    """The spreadsheet carries the seven coefficients and nothing else."""
    mach_only = pd.read_excel(AERO_WORKBOOK_5IN38, sheet_name="mach_only")
    yaw = pd.read_excel(AERO_WORKBOOK_5IN38, sheet_name="yaw_dependent")

    assert set(mach_only.columns) == {"Mach", "CYP", "CLP", "CMA", "CMQ"}
    assert set(yaw.columns) == {"Mach", "Alpha_deg"} | set(YAW_DEPENDENT)
    assert set(mach_only.columns) | set(yaw.columns) == (
        set(EQUATION_COEFFICIENTS) | {"Mach", "Alpha_deg"}
    )


def test_shipped_workbook_tracks_the_default(coefficients):
    """The editable workbook stays within a fraction of a per cent."""
    from_sheets = AerodynamicCoefficients.from_workbook(AERO_WORKBOOK_5IN38)
    for mach in (0.6, 1.3, 2.5, 4.0):
        for alpha_deg in (-6.0, 0.5, 4.0):
            alpha = np.radians(alpha_deg)
            a = from_sheets.get_coefficients(mach, alpha)
            b = coefficients.get_coefficients(mach, alpha)
            for key in RHS_KEYS:
                assert a[key] == pytest.approx(b[key], rel=5e-3, abs=1e-6), key


@pytest.mark.parametrize("path", [AERO_COEFFICIENTS_5IN38, AERO_WORKBOOK_5IN38])
def test_load_coefficients_dispatches_on_content(path):
    """The loader picks a reader from what is in the file, not from its name."""
    loaded = load_coefficients(path)
    values = loaded.get_coefficients(2.0, 0.05)
    assert set(values) == set(RHS_KEYS)
    assert values["CD_total"] > 0.0


def test_load_coefficients_refuses_the_source_table():
    """A table in the source's own convention is not guessed at.

    This is the behaviour change that removing the adapter buys: the package
    will not silently interpret a tabulation it was never told how to read.
    """
    with pytest.raises(ValueError, match="none of the seven"):
        load_coefficients(AERO_SOURCE_5IN38)


def test_load_coefficients_refuses_an_unknown_table(tmp_path):
    """And it says where the conversion belongs."""
    path = tmp_path / "mystery.xlsx"
    pd.DataFrame({"Mach": [1.0, 2.0], "Zeta": [0.1, 0.2]}).to_excel(path, index=False)
    with pytest.raises(ValueError, match="bring_your_own_table"):
        load_coefficients(path)


# ----------------------------------------------------------------------
# provenance: the example still reproduces the shipped table
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mach", [0.3, 0.9, 1.05, 1.8, 2.7, 4.9])
@pytest.mark.parametrize("alpha_deg", [-9.0, -2.0, 0.0, 1.5, 7.5])
def test_example_reproduces_the_shipped_table(converted, coefficients, mach, alpha_deg):
    """Converting the source table gives exactly what ``data/*.npz`` holds."""
    alpha = np.radians(alpha_deg)
    a = converted.get_coefficients(mach, alpha)
    b = coefficients.get_coefficients(mach, alpha)
    for key in RHS_KEYS:
        assert a[key] == b[key], key


def test_example_reproduces_the_shipped_grids_bit_for_bit(converted):
    """Not just at sample points — the stored arrays themselves."""
    with np.load(str(AERO_COEFFICIENTS_5IN38)) as shipped:
        assert np.array_equal(shipped["mach_grid"], converted.mach_grid)
        assert np.array_equal(shipped["alpha_grid"], converted.alpha_grid)
        for name in EQUATION_COEFFICIENTS:
            assert np.array_equal(
                shipped[name], np.asarray(converted._raw[name], dtype=float)
            ), name


@pytest.mark.slow
def test_example_and_default_fly_identically(converted, coefficients):
    """And the whole trajectory is the same, bit for bit."""
    a = _run(converted)
    b = _run(coefficients)
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.solution.y, b.solution.y)


def test_example_rejects_a_table_without_its_mach_column(example):
    with pytest.raises(ValueError, match="Match"):
        example.convert(pd.DataFrame({"Mach": [1.0], "CX0": [0.2]}))


def test_unused_columns_are_really_unused(example, converted):
    """The four columns nothing reads can be deleted without changing anything."""
    dead = set(example.unused_columns(AERO_SOURCE_5IN38))
    assert dead == {"CPN", "CPF1", "CPF5", "CNPA-5"}

    frame = pd.read_excel(AERO_SOURCE_5IN38)
    trimmed = example.convert(frame.drop(columns=list(dead)))

    for mach in (0.5, 1.1, 2.5):
        for alpha in (0.0, 0.05, -0.1):
            a = converted.get_coefficients(mach, alpha)
            b = trimmed.get_coefficients(mach, alpha)
            for key in RHS_KEYS:
                assert a[key] == b[key], key


# ----------------------------------------------------------------------
# the two departures the shipped table carries, measured
# ----------------------------------------------------------------------
def test_shipped_table_has_the_thesis_drag_sign(example, coefficients):
    """Yaw drag is far too small in the shipped table, and negative up high.

    The exact projection is ``CD = CX cos(a) + CNA sin^2(a)``; the thesis engine
    subtracted that term, leaving ``(CX2 - CNA)`` where ``(CX2 + CNA)`` belongs.
    Keeping the shipped table as the thesis ran it is deliberate — it is what
    makes the published results reproduce — so this test pins the departure
    rather than forbidding it.
    """
    corrected = example.convert(AERO_SOURCE_5IN38, yaw_drag_sign="add")

    def yaw_increment(coeffs, mach, alpha_deg=8.0):
        """How much drag grows from zero yaw to ``alpha_deg``."""
        at_zero = coeffs.get_coefficients(mach, 0.0)["CD_total"]
        at_yaw = coeffs.get_coefficients(mach, np.radians(alpha_deg))["CD_total"]
        return at_yaw - at_zero

    # Zero-yaw drag is untouched: the sign only affects the yaw term.
    assert corrected.get_coefficients(2.0, 0.0)["CD_total"] == pytest.approx(
        coefficients.get_coefficients(2.0, 0.0)["CD_total"]
    )

    # At Mach 2 both rise, but the shipped table rises several times too slowly.
    assert yaw_increment(coefficients, 2.0) > 0.0
    assert yaw_increment(corrected, 2.0) > 3.0 * yaw_increment(coefficients, 2.0)

    # By Mach 3.5 the subtracted term has overtaken the axial one and the
    # shipped table's drag *falls* with yaw, which no body does.
    assert yaw_increment(coefficients, 3.5) < 0.0
    assert yaw_increment(corrected, 3.5) > 0.0


def test_the_factor_of_two_touches_only_the_rate_coefficients(example, coefficients):
    """NACA -> McCoy halves four coefficients and leaves the rest alone.

    ``CMA`` carries no angular rate, so it is identical in both systems.  That
    is the cross-check that tells a factor-of-two mismatch from a bad table.
    """
    mccoy = example.convert(
        AERO_SOURCE_5IN38, yaw_drag_sign="add", naca_to_mccoy=True
    )
    only_sign = example.convert(AERO_SOURCE_5IN38, yaw_drag_sign="add")

    a = only_sign.get_coefficients(2.0, np.radians(4.0))
    b = mccoy.get_coefficients(2.0, np.radians(4.0))

    for key in ("CLP", "CMQ", "CYP", "CNP_total"):
        assert b[key] == pytest.approx(a[key] / 2.0, rel=1e-9), key
    for key in ("CMA", "CD_total", "CLA_total"):
        assert b[key] == pytest.approx(a[key], rel=1e-12), key


@pytest.mark.slow
def test_the_drag_sign_is_worth_about_fifty_metres(example, coefficients):
    """Measured, so the size of the known departure is on record.

    Correcting the drag projection shortens the reference shot by roughly 50 m
    in 16.7 km — 0.3 %, and in the direction physics demands, since drag now
    rises with yaw instead of falling.
    """
    reference = _run(coefficients)
    corrected = _run(example.convert(AERO_SOURCE_5IN38, yaw_drag_sign="add"))

    delta = corrected.max_range - reference.max_range
    assert -60.0 < delta < -40.0
    assert abs(corrected.z[-1] - reference.z[-1]) < 5.0  # drift barely moves


# ----------------------------------------------------------------------
# what a Mach-only table costs
# ----------------------------------------------------------------------
def test_magnus_moment_is_odd_in_yaw(coefficients):
    """CNP changes sign with the yaw angle and is zero at exactly zero.

    This is why a single Mach-indexed CNP cannot represent the Magnus moment:
    sampling it at zero yaw does not approximate the term, it removes it.
    """
    positive = coefficients.get_coefficients(2.0, np.radians(3.0))["CNP_total"]
    negative = coefficients.get_coefficients(2.0, np.radians(-3.0))["CNP_total"]
    at_zero = coefficients.get_coefficients(2.0, 0.0)["CNP_total"]

    assert positive == pytest.approx(-negative, rel=1e-6)
    assert abs(at_zero) < 1e-9
    assert abs(positive) > 0.1


@pytest.mark.slow
def test_mach_only_table_costs_less_than_a_tenth_of_a_percent(coefficients):
    """A zero-yaw Mach-only table is a real but small modelling change.

    Measured on the reference shot: the range moves by about 13 m in 16.7 km
    and the drift by about 1 m in 452 m.  Small enough for scoping work, too
    large to call the two tables equivalent.
    """
    reference = _run(coefficients)
    flattened = AerodynamicCoefficients.from_mach_table(
        coefficients.to_frame(alpha_deg=0.0)
    )
    simplified = _run(flattened)

    range_error = abs(simplified.max_range - reference.max_range)
    drift_error = abs(simplified.z[-1] - reference.z[-1])

    assert range_error / reference.max_range < 1e-3
    assert range_error > 1.0  # not equivalent, and should not be sold as such
    assert drift_error / abs(reference.z[-1]) < 1e-2
