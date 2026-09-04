# The shipped 5"/38 table: where it came from, and how it differs from McCoy

`data/aero_coefficients_5in38_spin73.npz` is the default coefficient set, and it
is the table the thesis campaign was flown on. Loading it reproduces the
published trajectories bit for bit, which is the whole reason it ships in this
form.

It does **not** meet the input contract that `sixdof.aerodynamics` documents. It
departs from it in two known, measured ways. Both are inherited from the thesis
and both are kept deliberately: correcting them here would mean the repository
no longer reproduces the work it accompanies. This page is the declaration.

## Provenance

| | |
| --- | --- |
| Source table | `data/aero_coefficients_5in38.xlsx` |
| Generator | Whyte, R. H., *SPIN-73: An Updated Version of the Spinner Computer Program*, 1973 (AD09156281) |
| Nature of the data | empirical, least-squares fits to BRL and AEDC range firings; Whyte states no wind-tunnel data entered the data bank |
| Conversion | `examples/07_bring_your_own_table.py`, `convert()` with its defaults |
| Grid | 100 Mach × 100 yaw nodes, yaw ±10° |

The conversion is verified in both directions:
`tests/test_coefficients.py::test_example_reproduces_the_shipped_grids_bit_for_bit`
rebuilds the shipped arrays from the source table and compares them element by
element, and `tests/reference/` holds the pre-refactor engine that reads the same
source workbook directly.

## Departure 1 — the four rate coefficients are twice what McCoy wants

`CLP`, `CMQ`, `CYP` and `CNP` multiply an angular rate, so their numerical value
depends on how that rate was made dimensionless.

- **McCoy**, eq. (2.4), uses `(pd/V)`. The equations in `sixdof.dynamics` are
  his, term for term.
- **SPIN-73** defines its coefficients on `(pd/2V)` — the NACA aeroballistic
  system. Whyte's NOMENCLATURE gives `Cp = Mp/qAd(pd/2V)`, `Cmq =
  Mmq/qAd(qd/2V)`, `Cnpa = Mnp/qAd(pd/2V)`, `Cypa = Fyp/qA(pd/2V)`.

Both describe the same physical moment, so `C_NACA·(pd/2V) = C_McCoy·(pd/V)`,
i.e. **`C_McCoy = C_NACA / 2`**. McCoy flags exactly this trap in the paragraph
after eq. (2.4): the NACA system *"uses (pd/2V) instead of (pd/V), which accounts
for the factor of two difference in coefficients that depend on angular
velocity."*

The shipped table carries the SPIN-73 values undivided.

**Evidence.** Spin at impact comes out at 94.43 rev/s against 128.78 rev/s from
an independent 6-DOF code reading the same table in the `(pd/2V)` convention.
The ratio of the decay logarithms is
`ln(94.43/175.48) / ln(128.78/175.48) = 2.003`. Halving the four gives
128.74 rev/s — 0.03 % from a code of entirely different formulation.

`CMA` carries no angular rate and is identical in both systems. That it passes
through unaffected is the signature that distinguishes this from a bad table.

**What it moves.** Drift, angle of repose, spin at impact, and anything derived
from them. Range, apogee and time of flight move by less than 0.1 %.

## Departure 2 — the drag projection subtracts the yaw term

The exact body-to-wind rotation is

```
D = A cos(a) + N sin(a)      =>   C_D  = C_X cos(a) + C_Na sin^2(a)
L = N cos(a) - A sin(a)      =>   C_La = C_Na cos(a) - C_X
```

with `C_X` counted positive rearward, which is how the source tabulates it
(`CX0 = +0.42`). The thesis engine assembled the first line with a **minus**:

```python
CD  = CX_total * cos_alpha - (CNA * sin_alpha_2)     # shipped
CLA = CNA * cos_alpha - CX_total                     # correct
```

So the yaw-drag coefficient is `(CX2 - CNA)` where `(CX2 + CNA)` belongs.

**The lift coefficient is right**, and it is worth saying why, because it looks
wrong next to McCoy's printed equation. McCoy (2.13) reads
`C_La = C_Na cos(a) + C_X` — with a plus. But his `C_X` is positive *forward*
(he gives `C_X ≈ -C_D` in (2.14)), while the table's is positive rearward.
Substituting `C_X → -CX_table` turns his plus into the code's minus. The two
agree. Reading (2.13) literally with this table would give `C_La = 3.03` at
Mach 1.5 instead of `2.19` — 38 % too much lift.

**Independent confirmation of the drag sign**, from the generator of the table
rather than from McCoy. Whyte, under "Yaw Axial Force Coefficient": *"The Yaw
Drag coefficient may be computed by **adding** Cx_α2 and CN_α."*

**Size of the error**, at Mach 1.52:

| α | −0.5° | 1.3° | 2.9° | 5.0° | 7.4° | 10° |
|---|---|---|---|---|---|---|
| error in `C_D` | −0.10 % | −0.65 % | −3.1 % | −8.2 % | −16.0 % | −24.7 % |

and across Mach, for the yaw-drag coefficient itself:

| Mach | `CX2+CNA` (correct) | `CX2−CNA` (shipped) | ratio |
|---|---|---|---|
| 0.60 | 4.44 | 0.84 | 5.3 |
| 1.20 | 8.63 | 3.93 | 2.2 |
| 1.50 | 7.70 | 2.49 | 3.1 |
| 2.00 | 6.77 | 1.11 | 6.1 |
| 2.50 | 6.17 | 0.27 | 23.1 |
| 3.00 | 5.60 | **−0.26** | — |
| 5.00 | 4.44 | **−1.02** | — |

Above roughly Mach 3 the shipped coefficient goes negative: drag *falls* with
angle of attack, which no body does.
`tests/test_coefficients.py::test_shipped_table_has_the_thesis_drag_sign` pins
that behaviour so it cannot change unnoticed.

**What it moves.** Range, downwards, since drag should rise with yaw and does
not. Measured on the reference shot (43.3°, constant atmosphere, this table):

| elevation | α max | fraction of flight above 3° | Δ range | Δ drift |
|---|---|---|---|---|
| 43.3° | 7.44° | 2.7 % | −50.5 m (−0.30 %) | −0.48 m |
| 39.5° | 7.44° | 2.9 % | −50.3 m (−0.30 %) | −0.47 m |
| 20.0° | 7.46° | 3.5 % | −48.5 m (−0.35 %) | −0.34 m |
| 60.0° | 7.42° | 2.4 % | −64.8 m (−0.47 %) | −1.16 m |
| 70.0° | 9.89° | 34.2 % | −181.6 m (−1.77 %) | −14.79 m |

The whole term carries `sin²α`, and on the thesis trajectory the yaw transient
is brief — α exceeds 3° for under 3 % of the flight — which is why 0.3 % is all
it costs there. It grows with sustained yaw.

## What this means for the thesis

Neither departure changes a conclusion of the published work. Range moves 0.3 %,
the aim-point spacing and the hit probabilities are unaffected (the drift the
factor of two changes is absorbed by the correcting azimuth, which is solved for
each aim point). They are recorded because they are real, because a reader
reusing this table for a different regime — sustained high yaw, or anything that
depends on spin decay — needs to know, and because the printed eq. (3.1) does not
correspond to what ran.

## Running the corrected physics

`examples/07_bring_your_own_table.py` converts the same source table three ways
and reports what each costs:

```bash
python examples/07_bring_your_own_table.py
```

| variant | what it is |
| --- | --- |
| `thesis` | as the thesis ran; reproduces the shipped `.npz` bit for bit |
| `sign` | drag projection corrected |
| `mccoy` | sign correction plus the factor of two — the table the equations actually want |

To fly the corrected physics, convert and pass the result straight to the
simulator:

```python
from bring_your_own_table import convert          # or copy the function

coefficients = convert(yaw_drag_sign="add", naca_to_mccoy=True)
```

To make it the default instead, write it out and point
`sixdof.paths.AERO_COEFFICIENTS_5IN38` at the new file:

```bash
python examples/07_bring_your_own_table.py \
    --write data/aero_coefficients_5in38_mccoy --write-variant mccoy
```

Doing that will make `scripts/proof_of_equivalence.py` fail, which is correct
behaviour: the equivalence proof asserts that this package reproduces the thesis
engine, and a corrected table no longer does. Fly it as a second configuration
rather than replacing the first.

## Sources

- McCoy, R. L., *Modern Exterior Ballistics*, 2nd ed., ch. 2, pp. 34–35 —
  eqs. (2.4-a)/(2.4-b) in `(pd/V)`; the note on the NACA system's `(pd/2V)` and
  the resulting factor of two; eqs. (2.12)–(2.15), including `C_X ≈ −C_D`.
- Whyte, R. H., *SPIN-73* (1973), AD09156281 — NOMENCLATURE p. 7, defining the
  four rate coefficients on `(pd/2V)`; the "Yaw Axial Force Coefficient" section
  on adding `Cx_α2` and `CN_α`.
- Carlucci, D. E. & Jacobson, S. S., *Ballistics: Theory and Design of Guns and
  Ammunition*, ch. 6, eqs. (6.6), (6.13), (6.14) — same `(pd/V)` convention as
  McCoy.
- The thesis, eqs. (2.31)–(2.32) and (3.1), and `Motor.py` on the `legacy`
  branch.
