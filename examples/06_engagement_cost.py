"""Stage 4 of the Monte Carlo pipeline: expected cost of an engagement.

Rounds are fired one at a time until the target is destroyed or the allocation
runs out.  With per-round kill probabilities from the Monte Carlo campaign, the
expected cost is the ammunition actually spent plus the penalty for a target
that survives::

    python examples/06_engagement_cost.py --input output/monte_carlo_resultados.xlsx
    python examples/06_engagement_cost.py --round-cost 2000 --target-value 289000000

Also prints 95% confidence intervals on the hit rates, using the conservative
fixed-variance interval the thesis quotes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import configure_stdout, ensure_package_on_path

ensure_package_on_path()
configure_stdout()

import pandas as pd  # noqa: E402

from sixdof.montecarlo import (  # noqa: E402
    expected_engagement_cost,
    margin_of_error,
    wald_interval,
    wilson_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Monte Carlo results table")
    parser.add_argument(
        "--target", default="Drone_Sea_Baby", help="which hull's hit rates to use"
    )
    parser.add_argument("--round-cost", type=float, default=2000.0, help="cost per round")
    parser.add_argument(
        "--target-value", type=float, default=289_000_000.0, help="penalty if the target survives"
    )
    parser.add_argument(
        "--trials", type=int, default=1000, help="shots per point, for the intervals"
    )
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_excel(args.input)
    column = f"Taxa_acerto_{args.target}_pct"
    if column not in frame.columns:
        available = [c for c in frame.columns if c.startswith("Taxa_acerto_")]
        print(f"✗ Coluna '{column}' não encontrada. Disponíveis: {available}")
        return 1

    frame = frame.sort_values("Alcance_m", ascending=False).reset_index(drop=True)
    probabilities = (frame[column] / 100.0).values

    print("=" * 80)
    print("VALOR ESPERADO DO ENGAJAMENTO")
    print("=" * 80)
    print(f"  Alvo: {args.target}")
    print(f"  Disparos disponíveis: {len(probabilities)}")
    print(f"  Custo por munição: {args.round_cost:,.2f}")
    print(f"  Penalidade por falha total: {args.target_value:,.2f}")
    print(f"  Taxa de acerto: mín {frame[column].min():.4f}% | "
          f"máx {frame[column].max():.4f}% | média {frame[column].mean():.4f}%")

    result = expected_engagement_cost(
        probabilities, round_cost=args.round_cost, target_value=args.target_value
    )

    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"  Número esperado de disparos : {result.expected_rounds:.4f}")
    print(f"  Custo esperado de munição   : {result.ammunition_cost:,.2f}")
    print(f"  Probabilidade de falha total: {result.total_failure_probability:.6e}")
    print(f"  Custo esperado da falha     : {result.failure_cost:,.2f}")
    print(f"  CUSTO TOTAL ESPERADO        : {result.total_expected_cost:,.2f}")
    print(f"  Probabilidade de sucesso    : {result.success_probability:.6%}")

    print("\n" + "=" * 80)
    print("CUSTO ESPERADO vs NÚMERO DE DISPAROS ALOCADOS")
    print("=" * 80)
    records = []
    for n in range(1, len(probabilities) + 1):
        partial = expected_engagement_cost(
            probabilities[:n], round_cost=args.round_cost, target_value=args.target_value
        )
        records.append({"n_disparos": n, **partial.to_dict()})
    curve = pd.DataFrame(records)

    milestones = sorted({n for n in (1, 2, 5, 10, 20, 50, len(probabilities))
                         if 1 <= n <= len(probabilities)})
    for n in milestones:
        row = curve[curve["n_disparos"] == n].iloc[0]
        print(
            f"  n={n:4d} | E[custo]={row['custo_total_esperado']:16,.2f} | "
            f"P(sucesso)={row['prob_sucesso']:.6%}"
        )

    print("\n" + "=" * 80)
    print(f"INTERVALOS DE CONFIANÇA 95% (K={args.trials})")
    print("=" * 80)
    print(f"  Margem de erro (variância 1/4): ±{100*margin_of_error(args.trials):.2f} p.p.")
    interval_records = []
    for _, row in frame.iterrows():
        p_hat = row[column] / 100.0
        wald_low, wald_high = wald_interval(p_hat, args.trials)
        wilson_low, wilson_high = wilson_interval(p_hat, args.trials)
        interval_records.append(
            {
                "Alcance_m": row["Alcance_m"],
                "Elevacao_deg": row["Elevacao_deg"],
                "p_hat": p_hat,
                "Wald_inf": wald_low,
                "Wald_sup": wald_high,
                "Wilson_inf": wilson_low,
                "Wilson_sup": wilson_high,
            }
        )
    intervals = pd.DataFrame(interval_records)

    curve_path = output_dir / "custo_esperado_por_n_disparos.csv"
    intervals_path = output_dir / "intervalos_confianca.csv"
    curve.to_csv(curve_path, index=False)
    intervals.to_csv(intervals_path, index=False)
    print(f"\n  Arquivos gerados: {curve_path}, {intervals_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
