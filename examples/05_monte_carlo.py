"""Stage 3 of the Monte Carlo pipeline: dispersion and hit rates.

At each aim point the gun fires ``--shots`` rounds whose elevation and azimuth
carry independent zero-mean laying errors.  Every impact is scored against a
fleet of hulls centred on the nominal impact point, giving a hit rate per hull
and the CEP of the scatter::

    python examples/05_monte_carlo.py --shots 20 --max-points 3   # smoke run
    python examples/05_monte_carlo.py                             # full campaign

The full campaign is 163 points x 1000 shots = 163 000 trajectories and takes
many hours.  Results are checkpointed after every aim point, so an interrupted
run leaves a usable partial table behind.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _bootstrap import configure_stdout, ensure_package_on_path

ensure_package_on_path()
configure_stdout()

import pandas as pd  # noqa: E402

from sixdof import (  # noqa: E402
    BallisticSimulator,
    load_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
    surface_target_fleet,
)
from sixdof.montecarlo import AimPoint, DispersionSettings, MonteCarloCampaign  # noqa: E402
from sixdof.paths import AERO_COEFFICIENTS_5IN38, SELECTED_POINTS_100M  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", default=str(SELECTED_POINTS_100M), help="aim-point table from stage 2"
    )
    parser.add_argument("--shots", type=int, default=1000, help="rounds per aim point")
    parser.add_argument("--sigma-elevation", type=float, default=0.1, help="[deg]")
    parser.add_argument("--sigma-azimuth", type=float, default=0.05, help="[deg]")
    parser.add_argument("--seed", type=int, default=16184331)
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="run only the first N aim points, still drawing the perturbations "
             "of the full campaign so the results match it",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--coefficients", default=str(AERO_COEFFICIENTS_5IN38))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_table = pd.read_excel(args.input)
    campaign_size = len(full_table)
    points_table = (
        full_table if args.max_points is None else full_table.head(args.max_points)
    )

    aim_points = [AimPoint.from_row(row) for _, row in points_table.iterrows()]

    settings = DispersionSettings(
        n_shots=args.shots,
        sigma_elevation_deg=args.sigma_elevation,
        sigma_azimuth_deg=args.sigma_azimuth,
        seed=args.seed,
    )

    print("=" * 80)
    print("SIMULAÇÃO MONTE CARLO - DISPERSÃO E TAXA DE ACERTO")
    print("=" * 80)
    print(f"  Pontos de mira: {len(aim_points)} de {campaign_size} da campanha completa")
    print(f"  Disparos por ponto: {settings.n_shots}")
    print(f"  Total de simulações: {len(aim_points) * settings.n_shots:,}")
    print(f"  α = {settings.alpha0_deg}° e β = {settings.beta0_deg}° (fixos)")
    print(f"  Perturbação de elevação: N(0, {settings.sigma_elevation_deg}°)")
    print(f"  Perturbação de azimute:  N(0, {settings.sigma_azimuth_deg}°)")
    print(f"  Seed: {settings.seed}")

    fleet = surface_target_fleet((0.0, 0.0))
    print(f"\n  Embarcações alvo: {len(fleet)}")
    for name, vessel in fleet.items():
        print(f"    • {name}: {vessel.length} m × {vessel.width} m")

    simulator = BallisticSimulator(
        projectile=naval_5in38_projectile(),
        weapon=naval_5in38_gun(),
        environment=standard_atmosphere(),
        aero_coeffs=load_coefficients(args.coefficients),
    )
    campaign = MonteCarloCampaign(simulator, surface_target_fleet, settings)

    checkpoint = output_dir / "monte_carlo_resultados_backup.xlsx"
    collected: list = []

    def save_checkpoint(point_number: int, result) -> None:
        collected.append(result)
        MonteCarloCampaign.to_frame(collected).to_excel(
            checkpoint, index=False, engine="openpyxl"
        )

    started = time.time()
    results = campaign.run(
        aim_points, on_point_complete=save_checkpoint, campaign_size=campaign_size
    )
    elapsed = time.time() - started

    frame = MonteCarloCampaign.to_frame(results)
    final_xlsx = output_dir / "monte_carlo_resultados.xlsx"
    final_csv = output_dir / "monte_carlo_resultados.csv"
    frame.to_excel(final_xlsx, index=False, engine="openpyxl")
    frame.to_csv(final_csv, index=False)

    print("\n" + "=" * 80)
    print("CAMPANHA CONCLUÍDA")
    print("=" * 80)
    print(f"  Tempo total: {elapsed/60:.1f} minutos")
    print(f"  Pontos simulados: {len(results)}")
    print(f"  Arquivos: {final_xlsx}, {final_csv}")

    print("\n" + "=" * 80)
    print("TAXA DE ACERTO GLOBAL POR EMBARCAÇÃO")
    print("=" * 80)
    for name in fleet:
        column = f"Taxa_acerto_{name}_pct"
        if column in frame.columns:
            print(
                f"  {name:20s}: média {frame[column].mean():6.2f}% | "
                f"máx {frame[column].max():6.2f}% | mín {frame[column].min():6.2f}%"
            )

    print("\n  DISPERSÃO:")
    print(f"    CEP50 médio: {frame['CEP50_m'].mean():.2f} m")
    print(f"    CEP90 médio: {frame['CEP90_m'].mean():.2f} m")
    print(f"    Erro X (σ) médio: {frame['Erro_X_std_m'].mean():.2f} m")
    print(f"    Erro Z (σ) médio: {frame['Erro_Z_std_m'].mean():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
