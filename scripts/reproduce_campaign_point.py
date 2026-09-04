"""Reproduce a published Monte Carlo aim point of the naval-drone campaign.

This is the end-to-end check of the thesis result itself: for a chosen aim
point, fire the same 1000 perturbed rounds the campaign fired, score them
against the same six hulls, and compare every published number.

    python scripts/reproduce_campaign_point.py                 # cheap points
    python scripts/reproduce_campaign_point.py --point 1       # ~1 h, max range
    python scripts/reproduce_campaign_point.py --point 1 160 163 --report out.md

What to expect, and why
-----------------------

**Hit counts reproduce exactly.** They are integers decided by whether an
impact falls inside a hull footprint, and the impacts land in the same place.

**Continuous statistics agree to about 1e-6 relative, never more than a
millimetre.** They do not match to the last bit, and that is *not* the
refactor's doing: ``tests/test_naval_pipeline.py`` shows the frozen original
engine and this package agreeing with each other exactly while both differ from
the published workbook by the same amount.  The workbook was produced years ago
on Windows with older SciPy/NumPy; ``solve_ivp`` is adaptive, so a last-bit
difference in the platform's ``sin``/``cos`` sends it down a different but
equally valid step sequence.  The integrator's own tolerance is ``rtol=1e-7``,
which is the scale of what is seen.

Cost note: aim point 1 is at maximum range, a 62 s flight, so 1000 rounds take
about an hour.  The low-elevation points at the end of the ladder fly for well
under a second and run in seconds -- which is why they are the default.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sixdof import (  # noqa: E402
    BallisticSimulator,
    naval_5in38_coefficients,
    naval_5in38_gun,
    naval_5in38_projectile,
    standard_atmosphere,
    surface_target_fleet,
)
from sixdof.montecarlo import AimPoint, DispersionSettings, MonteCarloCampaign  # noqa: E402
from sixdof.paths import (  # noqa: E402
    PUBLISHED_CAMPAIGN,
    SELECTED_POINTS_100M,
)

#: Continuous statistics compared between the run and the published workbook.
STATISTIC_COLUMNS = (
    "Erro_X_medio_m",
    "Erro_X_std_m",
    "Erro_X_min_m",
    "Erro_X_max_m",
    "Erro_Z_medio_m",
    "Erro_Z_std_m",
    "Erro_Z_min_m",
    "Erro_Z_max_m",
    "CEP50_m",
    "CEP90_m",
    "CEP95_m",
    "Tempo_voo_medio_s",
)

#: Bound on the relative disagreement of the continuous statistics.  Set an
#: order of magnitude above what is observed; a modelling change would blow
#: past it, floating-point drift will not.
STATISTIC_RTOL = 1e-4

#: Points that finish quickly: short range, sub-second flights.
DEFAULT_POINTS = (162, 163)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--point", type=int, nargs="+", default=list(DEFAULT_POINTS),
        help="1-based aim point number(s) from the published campaign",
    )
    parser.add_argument("--shots", type=int, default=1000, help="rounds per aim point")
    parser.add_argument("--seed", type=int, default=16184331)
    parser.add_argument("--report", default=None, help="write the output here too")
    return parser.parse_args()


def compare_point(campaign, aim_table, published, point_number, out) -> bool:
    """Re-fire one aim point and compare it with the published row."""
    index = point_number - 1
    aim_point = AimPoint.from_row(aim_table.iloc[index])
    published_row = published[published["Ponto_numero"] == point_number].iloc[0]

    published_shots = int(published_row["N_simulacoes"])
    if campaign.settings.n_shots != published_shots:
        out(f"  PONTO {point_number}: comparação impossível — a campanha publicada usou "
            f"{published_shots} tiros e esta execução usa {campaign.settings.n_shots}.")
        out("  Amostras de tamanhos diferentes produzem estatísticas diferentes por")
        out(f"  construção. Rode com --shots {published_shots}.")
        out("")
        return False

    out(f"  PONTO {point_number} — elevação {aim_point.elevation_deg:.1f}°, "
        f"azimute {aim_point.azimuth_deg:.2f}°, alcance nominal "
        f"{aim_point.nominal_range_m:.3f} m")
    out("")

    started = time.time()
    results = campaign.run(
        [aim_point], verbose=False,
        campaign_size=len(aim_table), first_point_index=index,
    )
    elapsed = time.time() - started
    row = MonteCarloCampaign.to_frame(results).iloc[0]

    out(f"    {campaign.settings.n_shots} tiros em {elapsed:.1f} s")
    out("")
    out(f"    {'contagem de acertos':30s} {'publicado':>12s} {'reproduzido':>12s}   igual")

    hits_ok = True
    for column in [c for c in published_row.index if c.startswith("Acertos_")]:
        a, b = int(published_row[column]), int(row[column])
        hits_ok &= a == b
        out(f"    {column:30s} {a:12d} {b:12d}   {'SIM' if a == b else 'NAO'}")

    out("")
    out(f"    {'estatística contínua':30s} {'publicado':>14s} {'reproduzido':>14s} {'Δ rel':>10s}")
    relative = []
    for column in STATISTIC_COLUMNS:
        a, b = float(published_row[column]), float(row[column])
        rel = abs(b - a) / max(abs(a), 1e-12)
        relative.append(rel)
        out(f"    {column:30s} {a:14.9f} {b:14.9f} {rel:10.2e}")

    worst = max(relative)
    worst_abs = max(
        abs(float(row[c]) - float(published_row[c])) for c in STATISTIC_COLUMNS
    )
    stats_ok = worst < STATISTIC_RTOL

    out("")
    out(f"    acertos idênticos          : {'SIM' if hits_ok else 'NAO'}")
    out(f"    Δ relativa máxima (stats)  : {worst:.2e}  (limite {STATISTIC_RTOL:.0e})")
    out(f"    Δ absoluta máxima (stats)  : {worst_abs*1000:.3f} mm")
    out("")
    return hits_ok and stats_ok


def main() -> int:
    args = parse_args()
    lines: List[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    aim_table = pd.read_excel(SELECTED_POINTS_100M)
    published = pd.read_excel(PUBLISHED_CAMPAIGN)

    out("=" * 78)
    out("REPRODUÇÃO DA CAMPANHA MONTE CARLO DO TCC (tiro contra drone naval)")
    out("=" * 78)
    out("")
    out(f"  data (UTC)        : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"  numpy / scipy     : {np.__version__} / "
        f"{__import__('scipy').__version__}")
    out(f"  campanha publicada: {PUBLISHED_CAMPAIGN.name} ({len(published)} pontos)")
    out(f"  pontos de mira    : {SELECTED_POINTS_100M.name} ({len(aim_table)} pontos)")
    out(f"  disparos por ponto: {args.shots}")
    out(f"  seed              : {args.seed}")
    out("  perturbações      : elevação N(0, 0.1°), azimute N(0, 0.05°)")
    out("")
    out("  As perturbações são sorteadas para a campanha inteira (163 pontos) e")
    out("  fatiadas por ponto, porque o gerador legado sorteia todas as elevações")
    out("  antes de todos os azimutes — sortear só para um ponto daria azimutes")
    out("  diferentes.")
    out("")

    simulator = BallisticSimulator(
        naval_5in38_projectile(), naval_5in38_gun(), standard_atmosphere(),
        naval_5in38_coefficients(),
    )
    campaign = MonteCarloCampaign(
        simulator, surface_target_fleet,
        DispersionSettings(n_shots=args.shots, seed=args.seed,
                           sigma_elevation_deg=0.1, sigma_azimuth_deg=0.05),
    )

    everything = True
    for point_number in args.point:
        if not 1 <= point_number <= len(aim_table):
            out(f"  ponto {point_number} fora da faixa 1..{len(aim_table)}")
            everything = False
            continue
        out("-" * 78)
        everything &= compare_point(campaign, aim_table, published, point_number, out)

    out("=" * 78)
    out("VEREDITO")
    out("=" * 78)
    out("")
    out("  " + ("Contagens de acertos idênticas e estatísticas dentro do limite."
                if everything else "ALGUM PONTO DIVERGIU."))
    out("")

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Reprodução da campanha Monte Carlo do TCC\n\n"
            "Saída literal de `scripts/reproduce_campaign_point.py`.\n\n"
            "```text\n" + "\n".join(lines) + "\n```\n",
            encoding="utf-8",
        )
        print(f"Relatório gravado em {path}")

    return 0 if everything else 1


if __name__ == "__main__":
    raise SystemExit(main())
