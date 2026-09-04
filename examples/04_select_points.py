"""Stage 2 of the Monte Carlo pipeline: pick aim points ~100 m apart.

The campaign cannot afford a thousand trajectories at every elevation in the
sweep, so it samples the range envelope on a roughly uniform ladder::

    python examples/04_select_points.py
    python examples/04_select_points.py --spacing 250 --elevation-min -15

Reads the zero-drift table (stage 1) and writes the aim-point table that
``05_monte_carlo.py`` consumes.  Run against the shipped data it reproduces the
163 aim points used in the thesis exactly -- that equality is asserted in
``tests/test_montecarlo.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import configure_stdout, ensure_package_on_path

ensure_package_on_path()
configure_stdout()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sixdof.montecarlo import SpacingPolicy, select_points_by_spacing  # noqa: E402
from sixdof.paths import OPTIMAL_AZIMUTHS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", default=str(OPTIMAL_AZIMUTHS), help="zero-drift table from stage 1"
    )
    parser.add_argument("--spacing", type=float, default=100.0, help="range step [m]")
    parser.add_argument("--base-tolerance", type=float, default=20.0, help="[m]")
    parser.add_argument("--max-tolerance", type=float, default=50.0, help="[m]")
    parser.add_argument("--minimum-gap", type=float, default=50.0, help="[m]")
    parser.add_argument("--elevation-max", type=float, default=39.6, help="[deg]")
    parser.add_argument("--elevation-min", type=float, default=-1.5, help="[deg]")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SELEÇÃO DE PONTOS COM ESPAÇAMENTO DE ~{args.spacing:.0f} m")
    print("=" * 80)

    optimal = pd.read_excel(args.input)
    print(f"  Arquivo de entrada: {args.input}")
    print(f"  Elevações disponíveis: {len(optimal)}")

    policy = SpacingPolicy(
        spacing_m=args.spacing,
        base_tolerance_m=args.base_tolerance,
        max_tolerance_m=args.max_tolerance,
        minimum_gap_m=args.minimum_gap,
    )
    selected = select_points_by_spacing(
        optimal,
        policy=policy,
        elevation_max=args.elevation_max,
        elevation_min=args.elevation_min,
    )

    if selected.empty:
        print("\n✗ Nenhum ponto selecionado — verifique a faixa de elevação.")
        return 1

    gaps = selected["Diferenca_alcance_m"].values[1:]
    print("\n" + "=" * 80)
    print("RESULTADO DA SELEÇÃO")
    print("=" * 80)
    print(f"  Total de pontos: {len(selected)}")
    print(
        f"  Faixa de elevação: {selected['Elevacao_deg'].max():.1f}° a "
        f"{selected['Elevacao_deg'].min():.1f}°"
    )
    print(
        f"  Faixa de alcance: {selected['Alcance_x_m'].max():.1f} m a "
        f"{selected['Alcance_x_m'].min():.1f} m"
    )
    if len(gaps):
        print("\n  ESPAÇAMENTO ENTRE PONTOS:")
        print(f"    Média: {np.mean(gaps):.1f} m")
        print(f"    Mínimo: {np.min(gaps):.1f} m")
        print(f"    Máximo: {np.max(gaps):.1f} m")
        print(f"    Desvio padrão: {np.std(gaps):.1f} m")
    out_of_tolerance = int((~selected["Dentro_tolerancia"]).sum())
    if out_of_tolerance:
        print(f"    ⚠ Pontos fora da tolerância: {out_of_tolerance}")

    xlsx_path = output_dir / "pontos_selecionados_100m.xlsx"
    csv_path = output_dir / "pontos_selecionados_100m.csv"
    selected.to_excel(xlsx_path, index=False, engine="openpyxl")
    selected.to_csv(csv_path, index=False)

    print("\n  Arquivos gerados:")
    print(f"    {xlsx_path}")
    print(f"    {csv_path}")

    print("\n" + "=" * 80)
    print("PRIMEIROS PONTOS")
    print("=" * 80)
    print(f"  {'#':>3s} | {'Elevação':>9s} | {'Azimute':>10s} | {'Alcance X':>11s} | {'Δ':>9s}")
    print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*10}-+-{'-'*11}-+-{'-'*9}")
    for i in range(min(10, len(selected))):
        row = selected.iloc[i]
        print(
            f"  {i+1:3d} | {row['Elevacao_deg']:8.1f}° | "
            f"{row['Azimute_otimo_deg']:9.2f}° | {row['Alcance_x_m']:10.1f} m | "
            f"{row['Diferenca_alcance_m']:7.1f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
