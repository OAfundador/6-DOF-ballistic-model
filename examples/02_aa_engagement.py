"""Anti-air engagement: 6-DOF trajectory, proximity fuze and fragment damage.

The canonical verification case of the anti-air study.  A 5"/38 VT(FCL) round is
fired at a Shahed-136 loitering at 200 m; the integration stops at the burst
point found by the fuze, and the fragment model scores the damage::

    python examples/02_aa_engagement.py
    python examples/02_aa_engagement.py --target-x 12000 --target-y 300
    python examples/02_aa_engagement.py --no-fuze-stop     # fly on to the water

Nothing in the chain is hard-wired to this drone or this shell: swap the
``shahed_136``/``vt_fcl_mk49`` presets for any other
:class:`~sixdof.aa.geometry.Target` and
:class:`~sixdof.aa.warhead.FragmentationWarhead`.  ``--target-shape box`` does
exactly that, using the generic box builder instead of the delta planform.
"""

from __future__ import annotations

import argparse

from _bootstrap import configure_stdout, ensure_package_on_path

ensure_package_on_path()
configure_stdout()

from sixdof import (  # noqa: E402
    BallisticSimulator,
    load_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.aa import (  # noqa: E402
    box_target,
    evaluate_engagement,
    print_damage_report,
    print_engagement_setup,
    print_trajectory_summary,
    ProximityFuze,
    shahed_136,
    vt_fcl_mk49,
)
from sixdof.paths import AERO_COEFFICIENTS_5IN38  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elevation", type=float, default=39.6, help="elevation [deg]")
    parser.add_argument("--azimuth", type=float, default=-1.35, help="azimuth [deg]")
    parser.add_argument("--target-x", type=float, default=16673.0, help="target range [m]")
    parser.add_argument("--target-y", type=float, default=200.0, help="target altitude [m]")
    parser.add_argument("--target-z", type=float, default=0.7, help="target lateral [m]")
    parser.add_argument(
        "--target-shape", choices=("prism", "box"), default="prism",
        help="prism = Shahed delta planform; box = generic rectangular body",
    )
    parser.add_argument("--fuze-radius", type=float, default=24.38, help="burst radius [m]")
    parser.add_argument("--fuze-arm-time", type=float, default=0.5, help="arming delay [s]")
    parser.add_argument(
        "--no-fuze-stop", action="store_true",
        help="integrate to ground impact and recover the burst point afterwards",
    )
    parser.add_argument("--max-time", type=float, default=100.0, help="horizon [s]")
    parser.add_argument("--coefficients", default=str(AERO_COEFFICIENTS_5IN38))
    return parser.parse_args()


def build_target(args):
    """Either the Shahed preset or a generic box of the same overall size."""
    center = (args.target_x, args.target_y, args.target_z)
    if args.target_shape == "prism":
        return shahed_136(center=center)
    return box_target("alvo generico", length=3.5, width=2.5, height=0.35, center=center)


def main() -> int:
    args = parse_args()

    target = build_target(args)
    warhead = vt_fcl_mk49()
    fuze = ProximityFuze(
        target_center=target.center,
        radius_m=args.fuze_radius,
        arm_time_s=args.fuze_arm_time,
    )
    stop_on_fuze = not args.no_fuze_stop

    print_engagement_setup(
        target=target,
        warhead=warhead,
        elevation_deg=args.elevation,
        azimuth_deg=args.azimuth,
        fuze_radius_m=fuze.radius_m,
        fuze_arm_time_s=fuze.arm_time_s,
        stop_on_fuze=stop_on_fuze,
        aero_table_path=args.coefficients,
    )

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(name='Projetil Naval 5"/38 AA VT(FCL)'),
        weapon=naval_5in38_gun(elevation_deg=args.elevation, azimuth_deg=args.azimuth),
        environment=standard_atmosphere(),
        aero_coeffs=load_coefficients(args.coefficients),
    )

    trajectory = simulator.simulate(
        max_time=args.max_time,
        fuze=fuze if stop_on_fuze else None,
    )

    print_trajectory_summary(trajectory)

    burst, damage = evaluate_engagement(trajectory, target, warhead, fuze)
    if burst is None or damage is None:
        print("\nFuze/dano: sem amostras validas para avaliacao.")
        return 1

    print_damage_report(burst, damage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
