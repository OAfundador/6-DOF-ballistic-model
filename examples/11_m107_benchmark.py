"""Reproduce a published 155 mm M107 benchmark, as an independent check.

Every other check in this repository compares the package against the engine it
was refactored from.  That proves the refactor faithful; it cannot prove the
physics right, because both sides share the same equations and the same data.

This one is different.  It flies the 155 mm M107 case of

    Khalil, M., Abdalla, H., Kamal, O., "Dispersion Analysis for Spinning
    Artillery Projectile", ASAT-13, Military Technical College, Cairo, 2009

using that paper's Table 1 coefficients, and compares against two things the
package had no hand in.

The first is the paper itself: the quantities it states in its own text.

The second is a second opinion, and it is worth recommending on its own terms:

    RigidFlightLab -- https://github.com/timeout187/RigidFlightLab

an independent open implementation of the same case in a different formulation
-- non-rolling frame, ``[x,y,z,u,v,w,phi,theta,psi,p,q,r]``, RK45 -- where this
package uses the vector form of McCoy.  Two codes sharing no line of source and
no integrator, landing on the same numbers for a third party's projectile, is a
stronger statement than either makes alone, so this benchmark exists because
that project does.  Have a look at it.

The figures quoted here are from running it with the same two conversions
described below, which its own README is candid about having had to assume,
since the paper does not print them.

Two conversions are needed on the way in, and both are the subject of
``docs/table_5in38_provenance.md``:

**The factor of two.**  The paper's coefficients are nondimensionalised on
``(pd/2V)``, the NACA aeroballistic system.  McCoy's equations, which this
package implements, use ``(pd/V)``.  So ``CLP``, ``CMQ``, ``CYP`` and ``CNP``
are halved.  ``CMA`` carries no angular rate and passes through untouched --
the cross-check that tells this apart from a wrong table.

**The atmosphere.**  This shot climbs above 5 km, where the air is some 40%
thinner and the speed of sound 7% lower.  The package's default
:class:`~sixdof.environment.Environment` is uniform, because that is what the
thesis used; here :class:`~sixdof.environment.LayeredAtmosphere` is used
instead, and the difference is reported.

**The axis system.**  Table 1 gives ``C_A`` (total axial force) and ``C_Nalpha``
(normal force derivative), not drag and lift -- the paper's Nomenclature says so
outright, though the table's own header does not.  They need the same rotation
through the angle of attack that the 5"/38 table needs.  It is worth being
explicit about because the difference is not small and not visible in the
usual checks: taken as wind-axis coefficients the flight time, apogee and
initial deceleration all still land within a percent of the paper, while the
drift comes out about 12% high.  Whatever a code does here, it should say so.

    python examples/11_m107_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from sixdof import (  # noqa: E402
    AerodynamicCoefficients,
    BallisticSimulator,
    Environment,
    Projectile,
    Weapon,
)
from sixdof.environment import LayeredAtmosphere  # noqa: E402

# ---------------------------------------------------------------------------
# The published case
# ---------------------------------------------------------------------------
#: 155 mm M107 physical data, as the paper's case is set up.
M107 = dict(
    mass_kg=43.0,
    diameter_m=0.155,
    I_P=0.144,      # axial, kg m^2
    I_T=1.216,      # transverse, kg m^2
)
MUZZLE_VELOCITY = 684.3      # m/s
ELEVATION_DEG = 44.0
MUZZLE_SPIN_RPS = 175.48     # rev/s

#: Table 1 of the paper, verbatim, with the paper's own column names.
#:
#: These are **body-axis** coefficients: ``C_A`` is the total axial force and
#: ``C_Nalpha`` the normal force derivative, per the paper's Nomenclature.  They
#: are not drag and lift, and the projection below is what turns them into the
#: wind-axis pair the equations read.  Taking them as ``CD`` and ``CLA``
#: directly is an easy reading -- the table's header does not say which they are
#: -- and costs about 12% of the drift while leaving flight time, apogee and
#: initial deceleration within a percent of the paper.
MACH = np.array([0.01, 0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00])
C_A = np.array([.144, .144, .146, .167, .221, .327, .383, .381, .370, .353, .338, .314, .294])
C_A_ALPHA2 = np.array([2.343, 2.343, 2.847, 3.372, 3.73, 4.180, 4.691, 5.209, 5.702, 5.130, 4.561, 3.970, 3.460])
#: The paper tabulates the normal force and Magnus force negative; the sign is a
#: convention of its axis system, and the magnitude is what the projection wants
#: -- with the tabulated sign, drag would fall with yaw and the lift would point
#: the wrong way.  Same reconciliation the 5"/38 table needs for ``C_X``.
C_N_ALPHA = np.abs(np.array([-1.763, -1.763, -1.783, -1.827, -2.038, -2.153, -2.207, -2.255, -2.325, -2.442, -2.556, -2.692, -2.747]))
CMAG_F = np.abs(np.array([-0.767, -0.767, -0.767, -0.857, -1.082, -0.992, -0.902, -0.857, -0.767, -0.767, -0.767, -0.767, -0.767]))
CSPIN = np.array([-.023, -.023, -.022, -.021, -.020, -.020, -.020, -.019, -.020, -.020, -.020, -.020, -.021])
CM_ALPHA = np.array([3.355, 3.378, 3.571, 3.957, 3.886, 3.682, 3.415, 3.384, 3.424, 3.278, 3.264, 3.201, 3.013])
CMQ = np.array([-5.1, -5.1, -5.1, -7.4, -9.9, -13.8, -13.3, -14.6, -15.8, -15.6, -15.3, -15.3, -15.3])

#: The Magnus moment is tabulated against total angle of attack as well.
CNPA_ALPHA_DEG = np.array([0.0, 2.0, 5.0, 10.0])
CNPA = np.array([
    [-0.500, 0.005, 0.294, 0.58],
    [-0.500, 0.005, 0.294, 0.58],
    [-0.355, 0.078, 0.366, 0.65],
    [-0.112, 0.172, 0.415, 0.86],
    [0.085, 0.292, 0.500, 1.12],
    [0.198, 0.388, 0.482, 0.72],
    [0.293, 0.430, 0.465, 0.55],
    [0.334, 0.432, 0.456, 0.54],
    [0.352, 0.424, 0.438, 0.51],
    [0.366, 0.424, 0.438, 0.51],
    [0.373, 0.424, 0.438, 0.51],
    [0.381, 0.431, 0.438, 0.51],
    [0.388, 0.431, 0.438, 0.51],
])

#: The four that multiply an angular rate, and so carry the NACA factor of two.
RATE_DEPENDENT = ("CLP", "CMQ", "CYP", "CNP")

#: What the paper states in its text, and what the independent code reports.
PUBLISHED = {
    "tempo de voo (s)": (66.67, 66.4),
    "tempo ao apogeu (s)": (31.0, 30.5),
    "desaceleração axial inicial (g)": (-4.45, -4.47),
    "alpha máximo (graus)": (1.29, 1.30),   # paper Fig. 10
    "apogeu (m)": (5600.0, 5647.0),         # paper Fig. 4
}


def build_coefficients(naca_to_mccoy: bool = True, n_alpha: int = 81, alpha_limit_deg: float = 12.0):
    """Table 1 as the seven the equations read."""
    alpha_deg = np.linspace(-alpha_limit_deg, alpha_limit_deg, n_alpha)
    alpha_rad = np.radians(alpha_deg)
    mach_mesh, alpha_mesh = np.meshgrid(MACH, alpha_rad, indexing="ij")
    sin2 = np.sin(alpha_mesh) ** 2

    # Body axes to wind axes.  The source gives axial and normal force, so the
    # rotation through the angle of attack has to happen here:
    #     C_D   = C_A cos(a) + C_Na sin^2(a)
    #     C_La  = C_Na cos(a) - C_A
    # Identical in form to the 5"/38 conversion in
    # examples/07_bring_your_own_table.py, because both sources come from a
    # spinner code and share the convention.
    cos_alpha = np.cos(alpha_mesh)
    C_A_total = C_A[:, None] + C_A_ALPHA2[:, None] * sin2
    CD = C_A_total * cos_alpha + C_N_ALPHA[:, None] * sin2
    CLA = C_N_ALPHA[:, None] * cos_alpha - C_A_total

    # Magnus moment: bilinear in (Mach, |alpha|), mirrored so the stored grid
    # spans the same range as the others.  The equations only ever ask for a
    # non-negative total angle of attack.
    CNP = np.empty_like(CD)
    for i in range(len(MACH)):
        CNP[i] = np.interp(np.abs(alpha_deg), CNPA_ALPHA_DEG, CNPA[i])

    seven = dict(
        CD=CD,
        CLA=CLA,
        CNP=CNP,
        CYP=CMAG_F.copy(),
        CLP=CSPIN.copy(),
        CMA=CM_ALPHA.copy(),
        CMQ=CMQ.copy(),
    )
    if naca_to_mccoy:
        for name in RATE_DEPENDENT:
            seven[name] = seven[name] / 2.0

    return AerodynamicCoefficients(mach_grid=MACH, alpha_grid=alpha_rad, **seven)


def fly(coefficients, environment, max_time=120.0):
    # The paper gives muzzle spin directly; the model derives it from the
    # rifling, so invert spin = 2 pi V / (n d) for the twist n that reproduces
    # 175.48 rev/s at the muzzle.
    twist_calibres = MUZZLE_VELOCITY / (MUZZLE_SPIN_RPS * M107["diameter_m"])
    projectile = Projectile(
        name="155 mm M107",
        mass=M107["mass_kg"],
        diameter=M107["diameter_m"],
        I_P=M107["I_P"],
        I_T=M107["I_T"],
        rifling_twist=twist_calibres,
    )
    weapon = Weapon(
        elevation_deg=ELEVATION_DEG,
        azimuth_deg=0.0,
        muzzle_velocity_mps=MUZZLE_VELOCITY,
    )
    simulator = BallisticSimulator(projectile, weapon, environment, coefficients)
    # The thesis case starts with 5 rad/s of transverse rate as a simplification;
    # this benchmark does not, and carrying it over would inject an initial yaw
    # transient of several degrees that the published case never had.
    return simulator.simulate(
        verbose=False, max_time=max_time, w_j0=0.0, w_k0=0.0
    )


def summarise(trajectory, environment):
    """The quantities the published sources report."""
    speed = trajectory.V_mag
    apogee_index = int(np.argmax(trajectory.y))
    initial_deceleration = float(
        (speed[1] - speed[0]) / (trajectory.t[1] - trajectory.t[0]) / 9.80665
    )
    return {
        "tempo de voo (s)": float(trajectory.t[-1]),
        "alcance (m)": float(trajectory.max_range),
        "apogeu (m)": float(trajectory.max_altitude),
        "tempo ao apogeu (s)": float(trajectory.t[apogee_index]),
        "velocidade no impacto (m/s)": float(speed[-1]),
        "desaceleração axial inicial (g)": initial_deceleration,
        "alpha máximo (graus)": float(np.max(np.abs(trajectory.alpha_traj))),
        "spin no impacto (rev/s)": float(trajectory.spin_rate[-1] / (2 * np.pi)),
        "deriva lateral (m)": float(trajectory.z[-1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-time", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("BENCHMARK 155 mm M107 — Khalil, Abdalla & Kamal, ASAT-13 (2009)")
    print("=" * 78)
    print()
    print("  Verificação independente: outra formulação, outro integrador,")
    print("  outro projétil, e uma tabela de coeficientes que este pacote")
    print("  nunca viu antes.")
    print()

    cases = {
        "atmosfera constante, sem fator 2": (
            build_coefficients(naca_to_mccoy=False),
            Environment(rho=1.225, g=9.80665, sound_speed=340.0),
        ),
        "atmosfera ICAO, sem fator 2": (
            build_coefficients(naca_to_mccoy=False),
            LayeredAtmosphere(),
        ),
        "atmosfera ICAO, com fator 2": (
            build_coefficients(naca_to_mccoy=True),
            LayeredAtmosphere(),
        ),
    }

    results = {}
    for label, (coefficients, environment) in cases.items():
        results[label] = summarise(fly(coefficients, environment, args.max_time), environment)

    keys = list(next(iter(results.values())))
    width = max(len(k) for k in keys)
    print(f"  {'grandeza':{width}s} " + " ".join(f"{lbl[:26]:>26s}" for lbl in results))
    print("  " + "-" * (width + 27 * len(results)))
    for key in keys:
        row = " ".join(f"{results[lbl][key]:26.3f}" for lbl in results)
        print(f"  {key:{width}s} {row}")

    print()
    print("  " + "-" * 74)
    print("  CONTRA O QUE ESTÁ PUBLICADO")
    print("  " + "-" * 74)
    print(f"  {'grandeza':{width}s} {'artigo':>12s} {'código indep.':>15s} {'este pacote':>14s}")
    final = results["atmosfera ICAO, com fator 2"]
    for key, (paper, other) in PUBLISHED.items():
        ours = final[key]
        print(f"  {key:{width}s} {paper:12.2f} {other:15.2f} {ours:14.3f}"
              f"   ({100*(ours-paper)/abs(paper):+.2f}% vs artigo)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
