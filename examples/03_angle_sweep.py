"""Stage 1 of the Monte Carlo pipeline: sweep elevation x azimuth.

A spinning shell drifts sideways, and the drift grows with time of flight, so
the azimuth that puts the round on the line of sight depends on the elevation.
This script integrates the whole grid and reduces it to that relationship::

    python examples/03_angle_sweep.py --quick          # small grid, minutes
    python examples/03_angle_sweep.py                  # thesis grid, hours

Two tables are written: the raw sweep, and the zero-drift azimuth per elevation.
The second is the input to ``04_select_points.py``.

The full grid is 601 elevations x 34 azimuths = 20 434 trajectories.  Its output
already ships as ``data/optimal_azimuths_zero_drift.xlsx``, so run this only to
regenerate it or to sweep a different envelope.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _bootstrap import configure_stdout, ensure_package_on_path

ensure_package_on_path()
configure_stdout()

import numpy as np  # noqa: E402

from sixdof import (  # noqa: E402
    BallisticSimulator,
    load_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
)
from sixdof.montecarlo import AngleSweep, SweepGrid, max_range_shot, optimal_azimuths  # noqa: E402
from sixdof.paths import AERO_COEFFICIENTS_5IN38  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elevation-start", type=float, default=45.0)
    parser.add_argument("--elevation-stop", type=float, default=-15.0)
    parser.add_argument("--elevation-step", type=float, default=-0.1)
    parser.add_argument("--azimuth-start", type=float, default=-1.65)
    parser.add_argument("--azimuth-stop", type=float, default=0.0)
    parser.add_argument("--azimuth-step", type=float, default=0.05)
    parser.add_argument(
        "--quick", action="store_true",
        help="coarse grid (1 deg elevation, 0.25 deg azimuth) for a smoke run",
    )
    parser.add_argument("--output-dir", default="output", help="where the tables go")
    parser.add_argument("--coefficients", default=str(AERO_COEFFICIENTS_5IN38))
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> SweepGrid:
    if args.quick:
        return SweepGrid(
            elevation_start=45.0, elevation_stop=-15.0, elevation_step=-1.0,
            azimuth_start=-1.5, azimuth_stop=0.0, azimuth_step=0.25,
        )
    return SweepGrid(
        elevation_start=args.elevation_start,
        elevation_stop=args.elevation_stop,
        elevation_step=args.elevation_step,
        azimuth_start=args.azimuth_start,
        azimuth_stop=args.azimuth_stop,
        azimuth_step=args.azimuth_step,
    )


def main() -> int:
    args = parse_args()
    grid = build_grid(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VARREDURA DE ELEVAÇÃO E AZIMUTE")
    print("=" * 80)
    print(f"  Elevação: {grid.elevation_start}° até {grid.elevation_stop}° "
          f"(passo {grid.elevation_step}°)")
    print(f"  Azimute:  {grid.azimuth_start}° até {grid.azimuth_stop}° "
          f"(passo {grid.azimuth_step}°)")
    print(f"  Total de simulações: {len(grid):,}")

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(elevation_deg=45.0, azimuth_deg=-1.65),
        environment=standard_atmosphere(),
        aero_coeffs=load_coefficients(args.coefficients),
    )

    started = time.time()
    sweep = AngleSweep(simulator, grid).run()
    elapsed = time.time() - started
    print(f"\n✓ Varredura concluída em {elapsed/60:.1f} minutos")

    elevation, azimuth, range_m, drift_m = max_range_shot(sweep)
    print("\n" + "=" * 80)
    print("TIRO DE MAIOR ALCANCE")
    print("=" * 80)
    print(f"  Elevação: {elevation:.1f}°")
    print(f"  Azimute: {azimuth:.2f}°")
    print(f"  Alcance: {range_m/1000:.3f} km ({range_m:.1f} m)")
    print(f"  Desvio lateral: {drift_m:.2f} m")

    optimal = optimal_azimuths(sweep)

    sweep_path = output_dir / "resultados_varredura_completa.csv"
    optimal_xlsx = output_dir / "azimutes_otimos_deriva_zero.xlsx"
    optimal_csv = output_dir / "azimutes_otimos_deriva_zero.csv"

    sweep.to_csv(sweep_path, index=False)
    optimal.to_excel(optimal_xlsx, index=False, engine="openpyxl")
    optimal.to_csv(optimal_csv, index=False)

    print("\n" + "=" * 80)
    print("ARQUIVOS GERADOS")
    print("=" * 80)
    print(f"  {sweep_path} ({len(sweep):,} linhas)")
    print(f"  {optimal_xlsx} ({len(optimal)} linhas)")
    print(f"  {optimal_csv}")

    residual = np.abs(optimal["Desvio_z_resultante_m"].values)
    print("\n  AZIMUTES ÓTIMOS (deriva ~ 0):")
    print(f"    Azimute médio: {optimal['Azimute_otimo_deg'].mean():.3f}°")
    print(f"    Azimute mínimo: {optimal['Azimute_otimo_deg'].min():.3f}°")
    print(f"    Azimute máximo: {optimal['Azimute_otimo_deg'].max():.3f}°")
    print(f"    |Desvio Z| residual médio: {residual.mean():.2f} m")
    print(f"    |Desvio Z| residual máximo: {residual.max():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
