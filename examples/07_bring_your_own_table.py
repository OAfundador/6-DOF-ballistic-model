"""Bring your own coefficient table: converting a source tabulation to the seven.

The model reads seven numbers -- ``CD``, ``CLA``, ``CYP``, ``CLP``, ``CMA``,
``CNP``, ``CMQ`` -- in McCoy's definitions, and nothing else.  No source hands
you those seven.  Wind-tunnel reports, range reductions, CFD campaigns and the
various prediction codes each tabulate their own intermediate quantities, in
their own decomposition and their own nondimensionalisation, and getting from
one of those to the seven is arithmetic specific to that source.

That arithmetic is **yours**.  It is not in the package, deliberately: a
conversion buried in a library is a conversion nobody checks, and the two
mistakes below are exactly the kind that produce a plausible trajectory and a
wrong answer.  Write it once, next to your data, where a reader can see it.

This file is a worked example of doing that, using the 5"/38 table the thesis
was built on.  Run it to see the conversion and what each choice costs::

    python examples/07_bring_your_own_table.py

Copy it, change the arithmetic to match your source, and hand the result to
``BallisticSimulator``.

The two traps
-------------

**Nondimensionalisation.**  ``CLP``, ``CMQ``, ``CYP`` and ``CNP`` multiply an
angular rate, so they depend on how that rate was made dimensionless.  McCoy
eq. (2.4) uses ``(pd/V)``; the NACA aeroballistic system -- which SPIN-73 and
most tabulations follow -- uses ``(pd/2V)``, and McCoy flags the consequence
himself: *"a factor of two difference in coefficients that depend on angular
velocity"*.  A NACA-normalised table needs those four halved.  ``CMA`` carries
no rate and is identical in both systems, which is the cross-check.

**Axis system and sign.**  ``CD`` and ``CLA`` are wind-axis.  A source giving
body-axis axial and normal force needs the rotation through the angle of attack::

    D = A cos(a) + N sin(a)          =>   CD  = CX cos(a) + CNA sin^2(a)
    L = N cos(a) - A sin(a)          =>   CLA = CNA cos(a) - CX

with ``CX`` counted positive rearward.  Both signs matter and neither will
raise an exception.

What this script shows
----------------------

Three conversions of the same source table:

``thesis``
    Exactly what the thesis engine did, reproduced bit for bit.  It is the
    shipped default (``data/aero_coefficients_5in38_spin73.npz``) and it carries
    both mistakes above.
``sign``
    The same, with the drag projection corrected.
``mccoy``
    The sign correction plus the factor of two, i.e. the table the McCoy
    equations actually want.

and what each does to a reference trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402

from sixdof import (  # noqa: E402
    AerodynamicCoefficients,
    BallisticSimulator,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.paths import AERO_SOURCE_5IN38  # noqa: E402

# --------------------------------------------------------------------------
# The source convention
# --------------------------------------------------------------------------
# Everything below here is specific to *this* table and would be different for
# yours.  The Mach column is spelled "Match" in the source workbook; the typo is
# preserved so the original file reads without being edited.

MACH_COLUMN = "Match"

#: Every column the source supplies.
SOURCE_COLUMNS = (
    "CX0", "CX2", "CNA", "CMA", "CPN", "CYP", "CNPA",
    "CNPA3", "CNPA5", "CPF1", "CPF5", "CNPA-5", "CMQ", "CLP",
)

#: Of those, the ones any of the seven depend on.  The other four -- ``CPN``,
#: ``CPF1``, ``CPF5``, ``CNPA-5`` -- are carried by the source and read by
#: nothing.  Dropping them changes no result.
SOURCE_COLUMNS_USED = (
    "CX0", "CX2", "CNA", "CNPA", "CNPA3", "CNPA5", "CMA", "CYP", "CMQ", "CLP",
)

#: The four that multiply an angular rate, and so carry the factor of two
#: between the NACA ``(pd/2V)`` system this source uses and McCoy's ``(pd/V)``.
RATE_DEPENDENT = ("CLP", "CMQ", "CYP", "CNP")


def convert(
    source=AERO_SOURCE_5IN38,
    *,
    yaw_drag_sign: str = "subtract",
    naca_to_mccoy: bool = False,
    n_mach: int = 100,
    n_alpha: int = 100,
    alpha_limit_deg: float = 10.0,
) -> AerodynamicCoefficients:
    """Convert this source table into the seven the equations read.

    Parameters
    ----------
    yaw_drag_sign:
        ``"add"`` for the correct projection ``CD = CX cos(a) + CNA sin^2(a)``;
        ``"subtract"`` to reproduce the thesis engine, which subtracts.
    naca_to_mccoy:
        Halve the four rate-dependent coefficients, converting the source's
        ``(pd/2V)`` normalisation to the ``(pd/V)`` the equations want.
    n_mach, n_alpha, alpha_limit_deg:
        Size and extent of the pre-computed grid.  The defaults are the ones the
        thesis used; keeping them is what makes the ``thesis`` conversion
        reproduce bit for bit.
    """
    if yaw_drag_sign not in ("add", "subtract"):
        raise ValueError("yaw_drag_sign must be 'add' or 'subtract'")

    frame = source if isinstance(source, pd.DataFrame) else pd.read_excel(str(source))
    if MACH_COLUMN not in frame.columns:
        raise ValueError(f"table must contain a {MACH_COLUMN!r} column")

    mach_values = frame[MACH_COLUMN].values
    mach_grid = np.linspace(float(mach_values.min()), float(mach_values.max()), n_mach)
    alpha_grid = np.linspace(
        -np.radians(alpha_limit_deg), np.radians(alpha_limit_deg), n_alpha
    )

    # Cubic in Mach, clamped to the table ends outside the tabulated range.
    on_mach = {}
    for column in SOURCE_COLUMNS:
        if column in frame.columns:
            values = frame[column].values
            on_mach[column] = interp1d(
                mach_values, values, kind="cubic",
                bounds_error=False, fill_value=(values[0], values[-1]),
            )(mach_grid)

    mach_mesh, alpha_mesh = np.meshgrid(mach_grid, alpha_grid, indexing="ij")

    def spread(name):
        """Replicate a Mach-only column across the yaw axis."""
        grid = np.zeros_like(mach_mesh)
        if name in on_mach:
            for i in range(len(mach_grid)):
                grid[i, :] = on_mach[name][i]
        return grid

    CX0_grid = spread("CX0")
    CX2_grid = spread("CX2")
    CNA_grid = spread("CNA")
    CNPA_grid = spread("CNPA")
    CNPA3_grid = spread("CNPA3")
    CNPA5_grid = spread("CNPA5")

    sin_alpha_mesh = np.sin(alpha_mesh)
    cos_alpha_mesh = np.cos(alpha_mesh)
    sin_alpha_2_mesh = np.sin(alpha_mesh) ** 2
    sin_alpha_3_mesh = sin_alpha_mesh**3
    sin_alpha_5th_mesh = sin_alpha_mesh**5

    CX_total = CX0_grid + CX2_grid * sin_alpha_2_mesh

    # The projection.  "subtract" is what the thesis engine did; the expression
    # is kept character for character so the grid comes out bit-identical.
    if yaw_drag_sign == "subtract":
        CD = CX_total * cos_alpha_mesh - (CNA_grid * sin_alpha_2_mesh)
    else:
        CD = CX_total * cos_alpha_mesh + (CNA_grid * sin_alpha_2_mesh)

    CLA = CNA_grid * cos_alpha_mesh - CX_total
    CNP = (
        CNPA_grid * np.sign(alpha_mesh)
        + CNPA3_grid * sin_alpha_3_mesh
        + CNPA5_grid * sin_alpha_5th_mesh
    )

    carried = {
        name: on_mach[name] for name in ("CYP", "CLP", "CMA", "CMQ") if name in on_mach
    }
    seven = dict(CD=CD, CLA=CLA, CNP=CNP, **carried)

    if naca_to_mccoy:
        for name in RATE_DEPENDENT:
            if name in seven:
                seven[name] = seven[name] / 2.0

    return AerodynamicCoefficients(mach_grid=mach_grid, alpha_grid=alpha_grid, **seven)


def unused_columns(source=AERO_SOURCE_5IN38):
    """Columns this source carries that none of the seven depend on."""
    frame = source if isinstance(source, pd.DataFrame) else pd.read_excel(str(source))
    return tuple(
        column
        for column in frame.columns
        if column != MACH_COLUMN and column not in SOURCE_COLUMNS_USED
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(AERO_SOURCE_5IN38))
    parser.add_argument("--elevation", type=float, default=43.3, help="check shot [deg]")
    parser.add_argument(
        "--write", metavar="STEM",
        help="also write STEM.npz and STEM_sheets.xlsx from the chosen variant",
    )
    parser.add_argument(
        "--write-variant", choices=("thesis", "sign", "mccoy"), default="thesis",
        help="which conversion --write exports (default: thesis, the shipped table)",
    )
    return parser.parse_args()


def fly(coefficients, elevation):
    return BallisticSimulator(
        naval_5in38_projectile(),
        naval_5in38_gun(elevation_deg=elevation),
        standard_atmosphere(),
        coefficients,
    ).simulate(verbose=False)


def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("CONVERTENDO UMA TABELA DE ORIGEM NOS SETE QUE A EQUAÇÃO LÊ")
    print("=" * 78)
    print()
    print(f"  tabela de origem   : {Path(args.source).name}")
    print(f"  colunas nunca lidas: {', '.join(unused_columns(args.source))}")
    print("  colunas auxiliares : CX0, CX2, CNA, CNPA, CNPA3, CNPA5")
    print("                       (só montam CD, CLA e CNP; não entram na equação)")
    print()

    variants = {
        "thesis": convert(args.source),
        "sign": convert(args.source, yaw_drag_sign="add"),
        "mccoy": convert(args.source, yaw_drag_sign="add", naca_to_mccoy=True),
    }

    print("-" * 78)
    print(f"  O QUE CADA ESCOLHA CUSTA — tiro de referência, elevação {args.elevation}°")
    print("-" * 78)
    print()
    reference = fly(variants["thesis"], args.elevation)
    print(f"  {'variante':46s} {'Δ alcance':>12s} {'Δ deriva':>11s}")
    labels = {
        "thesis": "thesis — como o TCC rodou (default do repo)",
        "sign": "sign — projeção do arrasto corrigida",
        "mccoy": "mccoy — sinal + fator 2 (contrato do modelo)",
    }
    for key, label in labels.items():
        result = fly(variants[key], args.elevation)
        d_range = result.max_range - reference.max_range
        d_drift = result.z[-1] - reference.z[-1]
        if key == "thesis":
            print(f"  {label:46s} {'referência':>12s} {'—':>11s}")
        else:
            print(f"  {label:46s} {d_range:+11.3f} m {d_drift:+10.3f} m")

    print()
    print("  Os dois erros se manifestam em lugares diferentes. O sinal do arrasto")
    print("  tira ~0,3 % do alcance neste tiro e quase nada da deriva, e cresce com")
    print("  o ângulo de ataque. O fator 2 mal toca o alcance, mas domina a deriva:")
    print("  ele multiplica por 2 o Magnus e os dois amortecimentos, que são")
    print("  justamente os termos que decidem o ângulo de repouso e o spin no")
    print("  impacto. Nenhum dos dois levanta exceção.")
    print()

    if args.write:
        stem = Path(args.write)
        chosen = variants[args.write_variant]
        chosen.save(str(stem.with_suffix(".npz")))
        chosen.to_workbook(
            str(stem.parent / (stem.name + "_sheets.xlsx")),
            mach_values=np.linspace(chosen.mach_min, chosen.mach_max, 100),
            alpha_deg_values=np.arange(-10.0, 10.5, 0.5),
        )
        print(f"  gravado ({args.write_variant}): {stem.with_suffix('.npz').name}")
        print(f"  gravado ({args.write_variant}): {stem.name}_sheets.xlsx")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
