"""aeroballistics as a coefficient source (examples/14).

Skipped when ``aeroballistics`` is not installed; the package does not depend on it.
The checks are about the conversion -- sines, directions, the factor of two --
not about whether the library's table is right.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

aeroballistics = pytest.importorskip("aeroballistics")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example():
    path = REPO_ROOT / "examples" / "14_aeroballistics_coefficients.py"
    spec = importlib.util.spec_from_file_location("aeroballistics_coefficients", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


@pytest.fixture(scope="module")
def report(example):
    return example.aerodynamics()


@pytest.fixture(scope="module")
def modern(example):
    return aeroballistics.Aerodinamica(aeroballistics.Projetil(**example.CARD_5IN38), convencao="moderna")


@pytest.fixture(scope="module")
def ours(example, report):
    return example.from_aeroballistics(report)


def _at_nodes(coefficients, alpha_deg):
    """The seven at every Mach node and one yaw node, as arrays."""
    alpha = np.radians(alpha_deg)
    rows = [coefficients.as_equation_names(m, alpha) for m in coefficients.mach_grid]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def test_small_yaw_limit_matches_the_librarys_modern_convention(ours, modern):
    # Two independent statements of the conversion: ours from the report's
    # definitions, the library's convencao="moderna".  A missing or doubled
    # factor of two puts a ratio at 2 or 0.5.
    mach = ours.mach_grid
    zero = _at_nodes(ours, 0.0)
    pairs = {"CD": "CD0", "CLA": "CLa", "CMA": "Cma", "CYP": "CNpa", "CMQ": "Cmq_Cmad", "CLP": "Clp"}
    for name, other in pairs.items():
        np.testing.assert_allclose(zero[name], modern.coeficiente(other, mach),
                                   rtol=1e-9, atol=1e-12, err_msg=f"{name} vs {other}")
    np.testing.assert_allclose(_at_nodes(ours, 1.0)["CNP"], modern.coeficiente("Cmpa", mach),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_at_nodes(ours, 5.0)["CNP"], modern.coeficiente("Cmpa_5graus", mach),
                               rtol=1e-9, atol=1e-12)


def test_yaw_terms_project_body_axes_onto_wind_axes(ours, modern):
    mach = ours.mach_grid
    cx, cna = modern.coeficiente("CD0", mach), modern.coeficiente("CNa", mach)
    cx2 = modern.coeficiente("CDd2", mach) - cna          # the library's CDd2 is CX2 + CNA
    for alpha_deg in (2.0, 6.0, 10.0):
        a = np.radians(alpha_deg)
        s2 = np.sin(a) ** 2
        got = _at_nodes(ours, alpha_deg)
        axial = cx + cx2 * s2
        np.testing.assert_allclose(got["CD"], axial * np.cos(a) + cna * s2, rtol=1e-9)
        np.testing.assert_allclose(got["CLA"], cna * np.cos(a) - axial, rtol=1e-9)
    # Direction: yaw adds drag and takes lift slope away.
    assert np.all(_at_nodes(ours, 5.0)["CD"] > _at_nodes(ours, 0.0)["CD"])
    assert np.all(_at_nodes(ours, 5.0)["CLA"] < _at_nodes(ours, 0.0)["CLA"])


def test_the_table_is_even_in_yaw(ours):
    for alpha_deg in (0.6, 3.0, 8.0):
        plus, minus = _at_nodes(ours, alpha_deg), _at_nodes(ours, -alpha_deg)
        for name in ("CD", "CLA", "CNP"):
            np.testing.assert_allclose(plus[name], minus[name], rtol=1e-12, atol=1e-14)


def test_moments_are_force_times_lever_arm_in_the_source(report):
    # SPIN-73 builds both moments from a force and its centre of pressure:
    # CMA = (VCG - CPN) CNA, CNPA = (VCG - CPF1) CYPA.  The equations put the
    # normal force and the Magnus force at those points with the same levers,
    # so moment and force keep a common sign -- the relation is what ties the
    # Magnus moment's direction to the Magnus force's.
    t = {k: np.asarray(v, float) for k, v in report.tabela.items()}
    vcg = report.projetil.VCG
    np.testing.assert_allclose(t["CMA"], (vcg - t["CPN"]) * t["CNA"], atol=2e-3)
    np.testing.assert_allclose(t["CNPA"], (vcg - t["CPF1"]) * t["CYPA"], atol=1e-9)
    np.testing.assert_allclose(t["CNPA5"], (vcg - t["CPF5"]) * t["CYPA"], atol=1e-9)


def test_the_transcribed_workbook_carries_the_same_lever_relation():
    # The same relation in the table typed from print, which the
    # library had no hand in: Magnus force and moment share one sign
    # convention there too.
    frame = pd.read_excel(REPO_ROOT / "data" / "aero_coefficients_5in38.xlsx")
    lever = (2.710 - frame["CPF1"]) * frame["CYP"]
    assert np.median(np.abs(frame["CNPA"] - lever)) < 0.01


def test_polynomial_magnus_is_even_and_starts_at_the_zero_yaw_slope(example, report):
    mach = np.array([0.3, 1.5, 2.5])
    np.testing.assert_allclose(example.magnus_secant(report, mach, 0.0, "polinomio"),
                               report.coeficiente("CNPA", mach), rtol=1e-12)
    three = np.radians(3.0)
    np.testing.assert_allclose(example.magnus_secant(report, mach, -three, "polinomio"),
                               example.magnus_secant(report, mach, three, "polinomio"))


def test_written_out_secant_is_the_librarys_momento_magnus(example, report):
    mach = np.linspace(0.01, 5.0, 37)[:, None]
    alpha = np.radians(np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0]))[None, :]
    np.testing.assert_array_equal(example.magnus_secant(report, mach, alpha),
                                  report.momento_magnus(mach, alpha))


def test_modern_input_is_refused(example, modern):
    with pytest.raises(ValueError, match="convencao='spin73'"):
        example.from_aeroballistics(modern)


@pytest.mark.slow
def test_flown_it_drifts_right_with_the_nose_right_at_the_summit(example, ours):
    # Right-hand twist and an overturning CMA: gyroscopic drift to the right
    # (+z), with the axis to the right of the velocity near the summit.
    trajectory = example.fly(ours, 43.3)
    assert trajectory.z[-1] > 0
    k = int(np.argmax(trajectory.y))
    v = np.array([trajectory.V1[k], trajectory.V2[k], trajectory.V3[k]])
    axis = np.array([trajectory.i1[k], trajectory.i2[k], trajectory.i3[k]])
    assert (axis - v / np.linalg.norm(v))[2] > 0


@pytest.mark.slow
def test_flown_it_stays_near_the_transcribed_table(example, report):
    # Same cubic Mach interpolation as the transcribed table, so what is left
    # is the difference between the two tables.
    a = example.fly(example._transcribed_table(), 43.3)
    b = example.fly(example.from_aeroballistics(report, mach_interp="cubica"), 43.3)
    assert abs(b.max_range - a.max_range) < 1e-3 * a.max_range
    assert abs(b.z[-1] - a.z[-1]) < 0.01 * abs(a.z[-1])
