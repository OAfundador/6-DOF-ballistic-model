"""The reconstructed SPIN-73 as a coefficient source: its outputs, turned into the seven.

``spin73`` (github.com/OAfundador/aeroballistics) computes a projectile's
aerodynamic table from its geometry card, the way SPIN-73 (Whyte, 1973) did::

    pip install git+https://github.com/OAfundador/aeroballistics

It is a reconstruction, not a reference: much of it was inferred from the
printed tables, and its authors say so.  Use it as a source to fly and compare,
not as truth.

This file does the one thing a source needs before the equations can read it
(see ``examples/07_bring_your_own_table.py`` for why that step lives next to
the data and not in the package): convert its output into the seven
coefficients of ``sixdof.dynamics``.  It takes the output **in the report's own
convention** (``convencao="spin73"``) and does every conversion here, so each
one is written down once; the library's ``convencao="moderna"`` is kept as an
independent check (``tests/test_spin73_coupling.py``).

The mapping, term by term
-------------------------
Source (Whyte, Nomenclature and Appendix B): ``q = rho V^2 / 2``, ``A = pi
d^2/4``, body axes, moments about the CG (``VCG`` calibers from the nose),
rates made dimensionless with ``pd/2V`` and ``qd/2V``, and every derivative
"per sin a" except ``CX2``, per ``sin^2 a``.  Axial force ``CX`` is positive
rearward; normal force ``CN = CNA sin a`` acts in the plane of ``v`` and the
axis, on the side that turns the lift toward the axis.

Target (``sixdof.dynamics``, McCoy): each force or moment is a coefficient
times a geometric vector.  **The sines** come from knowing which vectors
already carry one:

=========  =======================  ================  =====================================
term       vector in the equations  magnitude         so the coefficient must be
=========  =======================  ================  =====================================
drag       ``v``                    ``v``             the whole ``C_D(M, a)``
lift       ``v^2 i - v v cos a``    ``v^2 sin a``     ``C_L / sin a``
overturn   ``v x i``                ``v sin a``       per ``sin a``
Magnus F   ``v x i``                ``v sin a``       per ``sin a``
Magnus M   ``v - v cos a i``        ``v sin a``       per ``sin a`` -- the **secant** slope
=========  =======================  ================  =====================================

which gives

* ``CD  = (CX0 + CX2 sin^2 a) cos a + CNA sin^2 a`` -- ``D = A cos a + N sin a``;
* ``CLA = CNA cos a - (CX0 + CX2 sin^2 a)`` -- ``L = N cos a - A sin a``, over ``sin a``;
* ``CMA = CMA`` -- both per ``sin a``, about the CG, positive overturning;
* ``CYP = CYPA / 2`` -- per ``sin a``; ``pd/2V`` to ``pd/V``;
* ``CNP = (Magnus moment secant slope) / 2``, even in ``a`` -- see below;
* ``CMQ = CMQ / 2`` (it is ``Cmq + Cma_dot``) and ``CLP = CLP / 2``.

**The directions.**  Three are fixed by the geometry and checked in the tests:
``C_D`` grows with yaw and ``C_La`` falls (``CX`` rearward, ``CN`` toward the
nose-up side); ``(VCG - CPN) CNA`` reproduces ``CMA`` with the same sign as the
overturning term ``v x i`` in the equations; and a right-hand-twist shell flown
with the result drifts right, nose right of the velocity at the summit (the yaw
of repose).  The Magnus pair is fixed only relatively: SPIN-73 builds the
moment as ``(VCG - CPF) CYPA``, and the equations, placing the force
``v x i`` at ``CPF``, give the moment along ``v - v cos a i`` with exactly that
lever -- so force and moment keep the same sign, or both flip.  That the force
does not flip is checked against an independent code: the 155 mm M107 flown
through RigidFlightLab (github.com/timeout187/RigidFlightLab), its code
untouched and each input converted for its own equations, agrees with this
conversion to 0.02 % in flight time and parts by 0.5 % with the sign reversed.
The force comes from the yaw of repose, which is horizontal, so it is vertical
and shows in flight time and apogee rather than drift.  The moment has no such
check -- RigidFlightLab's moment term cannot be matched by a coefficient -- and
rests on the lever relation and on convention: SPIN-73 takes its stability
analysis from Murphy (BRL 1216) and Nicolaides, the formulation McCoy's
descends from.

**The Magnus moment.**  The equations want ``C_Mpa(a)``, the moment per
``sin a``, i.e. the secant slope, even in ``a``.  SPIN-73 prints two ways to
get it: secant slopes at 1 deg (``CNPA``) and 5 deg (``CNPA-5``, the library's
``CNPA5``), and a polynomial whose columns are derivatives per ``sin^3`` and
``sin^5`` of the moment, so that the secant is ``CNPA + CNPA3 sin^2 a + CNPA5
sin^4 a`` (even powers -- one ``sin`` less than the moment).  The library
reports the polynomial columns as carrying a defect of the original (``CNPA3 +
0.1 CNPA5 = 3.75`` for any projectile) and gives ``momento_magnus``, linear in
``sin^2 a`` between the two secants and held outside 1-5 deg.  ``magnus=
"secante"`` (default) uses that; ``magnus="polinomio"`` uses the even
polynomial.

Mach: the library interpolates linearly between its 17 nodes, as SPIN-73 is
tabulated (``mach_interp="linear"``, default); example 07 fits a cubic through
the same 17 nodes of the transcribed table (``mach_interp="cubica"``).  The
choice is not neutral: on the 43.3 deg reference shot it is worth 8 m of range.
Sampled on ``n_mach`` points.  Yaw: ``+/-alpha_limit_deg`` on ``n_alpha``
points, symmetric and even, so the grid is smooth through zero; the equations
only ask for ``|a|``.  Beyond the limit the lookup clips, and the simulator
warns (``sixdof.diagnostics``).

Usage::

    python examples/14_spin73_coefficients.py
    python examples/14_spin73_coefficients.py --magnus polinomio
    python examples/14_spin73_coefficients.py --mach cubica
    python examples/14_spin73_coefficients.py --correcao voo_livre --d-mm 127
    python examples/14_spin73_coefficients.py --cartao VL=4.05 VN=1.90 VB=0.40 VCG=2.51 OR=7.9 --write data/meu_projetil
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _bootstrap import configure_stdout, ensure_package_on_path  # noqa: E402

REPO_ROOT = ensure_package_on_path()

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

try:
    import spin73
except ImportError:  # pragma: no cover - only without the optional dependency
    spin73 = None

#: The 5"/38 card as the reconstruction reads p. 53 of the report (calibers).
#: VL is the field the scan leaves open between 4.580 and 4.600; 4.593 is the
#: reconstruction's choice.  A reading, like the rest of the page.
CARD_5IN38 = dict(VL=4.593, VN=2.150, VB=0.350, VCG=2.710, DM=0.100, BD=1.040,
                  OR=5.300, BOOM=0.0)

#: The rate-dependent four: SPIN-73 normalises them by pd/2V or qd/2V, the
#: equations by pd/V or qd/V, so the same moment needs half the coefficient.
NACA_TO_MCCOY = 0.5

MAGNUS_FORMS = ("secante", "polinomio")
MACH_INTERPOLATIONS = ("linear", "cubica")

#: Name the library exposes -> key in its table, where they differ.
_TABLE_KEY = {"CX0": "CX"}


def _require_spin73():
    if spin73 is None:
        raise SystemExit(
            "este exemplo precisa do pacote spin73:\n"
            "    pip install git+https://github.com/OAfundador/aeroballistics"
        )


def aerodynamics(card=None, correcoes=None, d_mm=None):
    """``spin73.Aerodinamica`` for a card, in the report's own convention."""
    _require_spin73()
    projetil = spin73.Projetil(**(CARD_5IN38 if card is None else card))
    return spin73.Aerodinamica(projetil, correcoes=correcoes, d_mm=d_mm, convencao="spin73")


def column(aero, name, mach, mach_interp="linear"):
    """One column of the library's table, in the report's convention, at ``mach``."""
    if mach_interp == "linear":
        if name in aero.nomes:
            return np.asarray(aero.coeficiente(name, mach), float)
        nodes = np.asarray(aero.tabela[_TABLE_KEY.get(name, name)], float)
        return np.interp(mach, aero.GRADE, nodes)
    if mach_interp == "cubica":
        nodes = np.asarray(aero.tabela[_TABLE_KEY.get(name, name)], float)
        return interp1d(aero.GRADE, nodes, kind="cubic", bounds_error=False,
                        fill_value=(nodes[0], nodes[-1]))(mach)
    raise ValueError(f"mach_interp must be one of {MACH_INTERPOLATIONS}")


def magnus_secant(aero, mach, alpha_rad, form="secante", mach_interp="linear"):
    """Magnus moment per sin(a) -- the secant slope -- in the report's pd/2V.

    ``secante`` is ``spin73.Aerodinamica.momento_magnus`` (linear in sin^2 a
    between the 1 deg and 5 deg secants, held outside), written out so the
    Mach interpolation can be chosen; with ``mach_interp="linear"`` the two
    agree to the last bit (tested).
    """
    alpha = np.abs(np.asarray(alpha_rad, dtype=float))
    s2 = np.sin(alpha) ** 2
    if form == "secante":
        c1 = column(aero, "CNPA", mach, mach_interp)
        c5 = column(aero, "CNPA5", mach, mach_interp)
        s1, s5 = np.sin(np.radians(1.0)) ** 2, np.sin(np.radians(5.0)) ** 2
        w = np.clip((s2 - s1) / (s5 - s1), 0.0, 1.0)
        return c1 + w * (c5 - c1)
    if form == "polinomio":
        cnpa = column(aero, "CNPA", mach, mach_interp)
        cubic = column(aero, "CNPA3", mach, mach_interp)
        quintic = column(aero, "CNPA5P", mach, mach_interp)
        return cnpa + cubic * s2 + quintic * s2 * s2
    raise ValueError(f"magnus must be one of {MAGNUS_FORMS}")


def from_spin73(aero, *, magnus="secante", mach_interp="linear", n_mach=100,
                n_alpha=101, alpha_limit_deg=10.0) -> AerodynamicCoefficients:
    """The seven the equations read, from a ``spin73.Aerodinamica``.

    ``aero`` must be in the report's convention (``convencao="spin73"``): the
    conversions are done here, and taking the library's modern output as well
    would halve the rate terms twice.
    """
    if getattr(aero, "convencao", None) != "spin73":
        raise ValueError(
            "from_spin73 converts the report's own convention; build the "
            "Aerodinamica with convencao='spin73' (got "
            f"{getattr(aero, 'convencao', None)!r})"
        )
    grid = aero.GRADE
    mach = np.linspace(float(grid[0]), float(grid[-1]), n_mach)
    alpha = np.radians(np.linspace(-alpha_limit_deg, alpha_limit_deg, n_alpha))
    col = {name: column(aero, name, mach, mach_interp)
           for name in ("CX0", "CX2", "CNA", "CMA", "CYPA", "CMQ", "CLP")}

    M, A = np.meshgrid(mach, alpha, indexing="ij")
    s, c = np.sin(A), np.cos(A)
    axial = col["CX0"][:, None] + col["CX2"][:, None] * s * s     # A / qS, rearward
    CD = axial * c + col["CNA"][:, None] * s * s                  # A cos a + N sin a
    CLA = col["CNA"][:, None] * c - axial                         # (N cos a - A sin a)/sin a
    CNP = NACA_TO_MCCOY * magnus_secant(aero, M, A, magnus, mach_interp)

    return AerodynamicCoefficients(
        mach_grid=mach,
        alpha_grid=alpha,
        CD=CD,
        CLA=CLA,
        CNP=CNP,
        CMA=col["CMA"],
        CYP=NACA_TO_MCCOY * col["CYPA"],
        CMQ=NACA_TO_MCCOY * col["CMQ"],
        CLP=NACA_TO_MCCOY * col["CLP"],
    )


def coefficients(card=None, correcoes=None, d_mm=None, **kwargs):
    """Card to the seven in one call."""
    return from_spin73(aerodynamics(card, correcoes, d_mm), **kwargs)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _transcribed_table():
    """The article's table: the transcribed workbook through example 07."""
    path = HERE / "07_bring_your_own_table.py"
    spec = importlib.util.spec_from_file_location("bring_your_own_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert(pd.read_excel(AERO_SOURCE_5IN38), yaw_drag_sign="add",
                          naca_to_mccoy=True)


def _card(pairs):
    if not pairs:
        return dict(CARD_5IN38)
    card = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        card[key.strip()] = float(value)
    return card


def fly(coefficients, elevation, azimuth=0.0):
    return BallisticSimulator(
        naval_5in38_projectile(),
        naval_5in38_gun(elevation_deg=elevation, azimuth_deg=azimuth),
        standard_atmosphere(),
        coefficients,
    ).simulate(verbose=False)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cartao", nargs="+", metavar="CAMPO=VALOR",
                        help='cartão do SPIN-73 em calibres (padrão: o 5"/38 da p. 53)')
    parser.add_argument("--magnus", choices=MAGNUS_FORMS, default="secante")
    parser.add_argument("--mach", choices=MACH_INTERPOLATIONS, default="linear",
                        help="interpolação em Mach entre os 17 nós (padrão: linear, a da biblioteca)")
    parser.add_argument("--correcao", default=None,
                        help="correção do spin73, por exemplo voo_livre ou voo_livre:CX0")
    parser.add_argument("--d-mm", type=float, default=None,
                        help="diâmetro real em mm (a correção de voo livre depende da escala)")
    parser.add_argument("--elevation", type=float, default=43.3, help="tiro de referência [graus]")
    parser.add_argument("--write", metavar="STEM", help="grava STEM.npz (lido por load_coefficients)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    configure_stdout()
    args = parse_args(argv)
    card = _card(args.cartao)
    aero = aerodynamics(card, args.correcao, args.d_mm)
    ours = from_spin73(aero, magnus=args.magnus, mach_interp=args.mach)

    print("=" * 78)
    print("SPIN-73 RECONSTRUÍDO → OS SETE COEFICIENTES DO MODELO")
    print("=" * 78)
    for line in aero.descrever().splitlines():
        print(f"  {line}")
    print(f"  Magnus: {args.magnus}; Mach: {args.mach}")
    print()
    print(f"  {'Mach':>5s} {'CD(0)':>7s} {'CD(5°)':>7s} {'CLA(0)':>7s} {'CMA':>7s} "
          f"{'CYP':>7s} {'CNP(1°)':>8s} {'CNP(5°)':>8s} {'CMQ':>8s} {'CLP':>8s}")
    for m in (0.3, 0.9, 1.0, 1.5, 2.0, 2.5):
        a0 = ours.as_equation_names(m, 0.0)
        a1 = ours.as_equation_names(m, np.radians(1.0))
        a5 = ours.as_equation_names(m, np.radians(5.0))
        print(f"  {m:5.2f} {a0['CD']:7.3f} {a5['CD']:7.3f} {a0['CLA']:7.3f} {a0['CMA']:7.3f} "
              f"{a0['CYP']:7.3f} {a1['CNP']:8.3f} {a5['CNP']:8.3f} {a0['CMQ']:8.3f} {a0['CLP']:8.4f}")
    print()

    if card == CARD_5IN38:
        a = fly(_transcribed_table(), args.elevation)
        b = fly(ours, args.elevation)
        print("-" * 78)
        print(f"  TIRO DE REFERÊNCIA, elevação {args.elevation}°")
        print("-" * 78)
        print(f"  {'tabela':30s} {'alcance [m]':>12s} {'deriva [m]':>11s} {'α máx [°]':>10s}")
        for label, tr in (("transcrita (a do artigo)", a), ("spin73 acoplada", b)):
            print(f"  {label:30s} {tr.max_range:12.2f} {tr.z[-1]:11.2f} "
                  f"{float(np.max(tr.alpha_traj[tr.t > 1.0])):10.3f}")
        print(f"  {'diferença':30s} {b.max_range - a.max_range:+12.2f} {b.z[-1] - a.z[-1]:+11.2f}")
        print()
        print("  As duas tabelas vêm do mesmo programa por caminhos diferentes (uma")
        print("  digitada do impresso, outra recalculada da geometria por uma reconstrução);")
        print("  a diferença mede o quanto elas discordam, não qual está certa.")
        print()

    if args.write:
        stem = Path(args.write)
        ours.save(str(stem.with_suffix(".npz")))
        print(f"  gravado: {stem.with_suffix('.npz')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
