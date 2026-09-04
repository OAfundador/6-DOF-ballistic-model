# Examples

Every example runs straight from a fresh clone — no installation step, no
copy-and-paste into the engine:

```bash
python examples/01_single_shot.py
```

All of them accept `--help`.

| Script | What it does | Runtime |
| --- | --- | --- |
| `01_single_shot.py` | One trajectory plus the eighteen standard figures. The reference scenario: 5"/38 ashore at 43.3°. | seconds |
| `02_aa_engagement.py` | Anti-air engagement — 6-DOF flight, proximity fuze, fragment damage against a Shahed-136. | seconds |
| `03_angle_sweep.py` | Stage 1: sweep elevation × azimuth and extract the zero-drift azimuth per elevation. | hours (`--quick` for minutes) |
| `04_select_points.py` | Stage 2: sample the range envelope onto aim points ~100 m apart. | seconds |
| `05_monte_carlo.py` | Stage 3: perturbed firing, hit rates per hull and CEP. | hours (`--shots`/`--max-points` to shrink) |
| `06_engagement_cost.py` | Stage 4: expected engagement cost and 95% confidence intervals. | seconds |
| `07_bring_your_own_table.py` | Convert a source coefficient table into the seven the equations read, and measure what each convention choice costs. | seconds |

Script 7 stands apart from the campaign. Converting a source tabulation into the
seven is deliberately *not* in the package — see
[the coefficients section of the README](../README.md#coefficients-the-seven-the-equations-read)
— so this is the worked example to copy when you bring your own table. It also
documents the two respects in which the shipped 5"/38 table departs from the
model's McCoy contract, and lets you fly the corrected version.

## The pipeline

Scripts 3 to 6 are the stages of the thesis campaign, and each one reads the
table the previous one wrote:

```
03_angle_sweep  ──▶ azimutes_otimos_deriva_zero.xlsx
04_select_points ─▶ pontos_selecionados_100m.xlsx
05_monte_carlo  ──▶ monte_carlo_resultados.xlsx
06_engagement_cost
```

Stage 1 is the expensive one — 20 434 trajectories. Its published output ships
in `data/`, so stages 2 to 4 run against the thesis data without regenerating
it. The defaults of every script point at those files, which is why
`04_select_points.py` works immediately after cloning.

## Reproducing the thesis

```bash
# stage 2 against the published sweep -- reproduces the 163 aim points exactly
python examples/04_select_points.py

# stage 3, full campaign (163 000 trajectories, many hours)
python examples/05_monte_carlo.py

# stage 4
python examples/06_engagement_cost.py --input output/monte_carlo_resultados.xlsx
```

Stage 3 checkpoints after every aim point, so an interrupted run still leaves a
usable table in `output/monte_carlo_resultados_backup.xlsx`.

## Smoke runs

To exercise the whole pipeline in a couple of minutes:

```bash
python examples/03_angle_sweep.py --quick --output-dir output
python examples/04_select_points.py --input output/azimutes_otimos_deriva_zero.xlsx \
                                    --spacing 500 --output-dir output
python examples/05_monte_carlo.py --input output/pontos_selecionados_100m.xlsx \
                                  --shots 20 --max-points 3 --output-dir output
python examples/06_engagement_cost.py --input output/monte_carlo_resultados.xlsx \
                                      --trials 20
```

## Headless machines

`01_single_shot.py` opens a window per figure. On a server, or when scripting:

```bash
python examples/01_single_shot.py --no-show --output-dir figures
python examples/01_single_shot.py --no-plots        # statistics only
```
