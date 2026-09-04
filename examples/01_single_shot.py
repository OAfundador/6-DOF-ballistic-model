"""Single shot with the full figure set -- the reference scenario.

This is the example that used to have to be pasted into the bottom of
``Motor.py``: a 5"/38 gun ashore at 43.3 degrees, no wind, no initial yaw.  It
now runs on its own::

    python examples/01_single_shot.py
    python examples/01_single_shot.py --elevation 39.6 --azimuth -1.35
    python examples/01_single_shot.py --no-plots

The trajectory it produces is bit-identical to the one the original engine
produced; ``tests/test_regression_vs_original.py`` is what checks that.
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
from sixdof.paths import AERO_COEFFICIENTS_5IN38  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elevation", type=float, default=43.3, help="elevation [deg]")
    parser.add_argument("--azimuth", type=float, default=0.0, help="azimuth [deg]")
    parser.add_argument("--muzzle-velocity", type=float, default=807.0, help="[m/s]")
    parser.add_argument("--height", type=float, default=10.0, help="mount height [m]")
    parser.add_argument("--max-time", type=float, default=100.0, help="horizon [s]")
    parser.add_argument("--alpha0", type=float, default=0.0, help="initial pitch [deg]")
    parser.add_argument("--beta0", type=float, default=0.0, help="initial yaw [deg]")
    parser.add_argument("--w-j0", type=float, default=5.0, help="initial rate about j' [rad/s]")
    parser.add_argument("--w-k0", type=float, default=5.0, help="initial rate about k' [rad/s]")
    parser.add_argument(
        "--coefficients", default=str(AERO_COEFFICIENTS_5IN38), help="coefficient table"
    )
    parser.add_argument("--output-dir", default=".", help="where the PNGs are written")
    parser.add_argument("--no-plots", action="store_true", help="statistics only")
    parser.add_argument(
        "--no-show", action="store_true", help="save the figures without opening windows"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print("TIRO ÚNICO - SIMULADOR BALÍSTICO 6-DOF")
    print(f"Elevação {args.elevation:.2f}° | Azimute {args.azimuth:.2f}°")
    print("=" * 80)

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(
            elevation_deg=args.elevation,
            azimuth_deg=args.azimuth,
            height_m=args.height,
            muzzle_velocity_mps=args.muzzle_velocity,
        ),
        environment=standard_atmosphere(),
        aero_coeffs=load_coefficients(args.coefficients),
    )

    print(simulator.projectile.get_info())
    print(simulator.weapon.get_info())

    trajectory = simulator.simulate(
        max_time=args.max_time,
        alpha0_deg=args.alpha0,
        beta0_deg=args.beta0,
        w_j0=args.w_j0,
        w_k0=args.w_k0,
    )
    trajectory.print_statistics()

    if not args.no_plots:
        from sixdof.plotting import TrajectoryPlotter

        TrajectoryPlotter(
            trajectory, output_dir=args.output_dir, show=not args.no_show
        ).plot_all()

    print("\n" + "=" * 80)
    print("CONCLUÍDO")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
