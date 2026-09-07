# 6-DOF Ballistic Model

A six-degree-of-freedom exterior ballistics model for spin-stabilised
projectiles, following R. McCoy, *Modern Exterior Ballistics: The Launch and
Flight Dynamics of Symmetric Projectiles* (2nd ed.).

Developed for an undergraduate thesis in Applied and Computational Mathematics
at **IME-USP** (Instituto de Matemática e Estatística, Universidade de São
Paulo), with parameters tuned for a naval artillery case: the 5"/38 calibre gun.

*(Leia em [português](README.pt-BR.md).)*

---

## Which version you are looking at

> **This branch is a refactor, not the thesis artefact.**
>
> The code the author wrote, verified and used in the thesis is the single-file
> engine on the [`legacy`](../../tree/legacy) branch. That is the version behind
> every number in the written work.
>
> This `main` branch is a restructuring of that engine into a package. It was
> written by **Claude** (Anthropic's AI assistant) at the author's request, and
> the test suite that checks it was written and run by Claude as well. The
> author reviewed and approved the result, but has not independently re-derived
> the thesis figures from it.
>
> The refactor is **bit-identical** to the original, and that is demonstrated
> rather than asserted — see [Verification](#verification) and
> [`docs/verification.md`](docs/verification.md), which is the literal output of
> a script you can re-run yourself. Even so:
>
> - **Citing or auditing the thesis?** Use `legacy`. It is the artefact the
>   author verified.
> - **Building on the model?** Use `main`. Same physics, same numbers, a shape
>   you can actually extend.

---

## What is here

Three layers, each usable on its own:

| Layer | Module | What it does |
| --- | --- | --- |
| **Core** | `sixdof` | Integrates the 6-DOF equations of motion for one shot: drag, lift, Magnus, overturning and damping terms, with coefficients interpolated in Mach and total angle of attack. |
| **Anti-air** | `sixdof.aa` | Proximity fuze, faceted target geometry, fragmenting warhead with a polar hit distribution, and a fragment damage model that returns a probability of destruction. |
| **Monte Carlo** | `sixdof.montecarlo` | The full dispersion campaign of the thesis: angle sweep, aim-point selection, perturbed firing with hit scoring, and expected engagement cost. Also the closest-approach reduction, for the question the impact point cannot answer. |

The anti-air and Monte Carlo layers are optional. Importing `sixdof` does not
pull them in.

---

## Quick start

```bash
git clone https://github.com/OAfundador/6-DOF-ballistic-model.git
cd 6-DOF-ballistic-model
pip install -r requirements.txt

python examples/01_single_shot.py          # one trajectory + 18 figures
python examples/02_aa_engagement.py        # fuze + fragment damage
```

Or install the package:

```bash
pip install -e ".[dev]"
pytest
```

Python 3.9 or newer. The only hard dependencies are NumPy, SciPy, pandas and
openpyxl; matplotlib is needed only for the figures.

---

## Using it as a library

```python
from sixdof import (
    BallisticSimulator, naval_5in38_coefficients,
    naval_5in38_gun, naval_5in38_projectile, standard_atmosphere,
)

simulator = BallisticSimulator(
    projectile=naval_5in38_projectile(),
    weapon=naval_5in38_gun(elevation_deg=43.3, azimuth_deg=0.0),
    environment=standard_atmosphere(),
    aero_coeffs=naval_5in38_coefficients(),
)

trajectory = simulator.simulate(verbose=False)
print(trajectory.max_range, trajectory.max_altitude, trajectory.flight_time)
```

Anything can be replaced piece by piece — a different shell, a moving platform,
a wind field, another coefficient table:

```python
from sixdof import Environment, Projectile, Vessel, Weapon

destroyer = Vessel("DD", center_position=(0, 0), length=115, width=12,
                   height=10, velocity=(12.0, 0.0))
weapon = Weapon(position=(20, 8, 0), elevation_deg=30, azimuth_deg=-1.2,
                mounted_on_vessel=destroyer)
environment = Environment(rho=1.18, W1=-8.0, W3=3.0)
```

### The atmosphere: uniform by default, layered on request

`Environment` holds one density and one speed of sound for the whole flight.
That is what the thesis assumed, and it stays the default so that every number
this repository reproduces stays reproduced. It is also a real assumption, and
worth knowing when it starts to cost you: a naval shot peaking below 3 km is
barely touched, while a howitzer reaching 5.6 km flies through air some 40%
thinner and 7% slower in sound, which moves the range by about 12%.

For shots with a real ceiling, `LayeredAtmosphere` is the ICAO model — a
troposphere falling 6.5 K/km to 11 km, isothermal above:

```python
from sixdof.environment import LayeredAtmosphere

environment = LayeredAtmosphere()          # 1.225 kg/m^3, 340.29 m/s at sea level
environment.density_at(5000.0)             # 0.7361
environment.sound_speed_at(5000.0)         # 320.53
```

The engine asks the environment for `density_at(h)` and `sound_speed_at(h)`
rather than reading its attributes, and the base class answers both with the
constants it already had. So swapping the class is the whole change, a custom
profile is two methods, and the uniform case is untouched — the bit-exact proof
of equivalence covers it.

The speed of sound matters more than it looks, because it sets the Mach number
that indexes the coefficient table. Held at a sea-level 340 m/s, a high shot
reads its coefficients up to 13% off in Mach right through the transonic region,
where drag varies by a factor of 2.5.

### Anti-air engagement

```python
from sixdof.aa import ProximityFuze, evaluate_engagement, shahed_136, vt_fcl_mk49

target = shahed_136(center=(16673.0, 200.0, 0.7))
warhead = vt_fcl_mk49()
fuze = ProximityFuze(target_center=target.center, radius_m=24.38, arm_time_s=0.5)

trajectory = simulator.simulate(fuze=fuze)      # integration stops at burst
burst, damage = evaluate_engagement(trajectory, target, warhead, fuze)

print(damage.expected_fragments, damage.p_destruction)
```

Nothing in that chain is specific to a Shahed or to a VT round. A target is a
list of facets with areas and outward normals; a warhead is an ejection speed
plus a table of polar zones. Both have generic builders:

```python
from sixdof.aa import FragmentationWarhead, PolarZone, box_target, triangular_prism_target

target = box_target("quadcopter", length=0.6, width=0.6, height=0.2,
                    center=(4000.0, 150.0, 0.0))
warhead = FragmentationWarhead(
    name="40 mm proximity round",
    fragment_velocity_mps=1100.0,
    polar_zones=[PolarZone(0, 90, 300), PolarZone(90, 180, 100)],
    effective_fragments=400,
)
```

### Coefficients: the seven the equations read

The model reads **seven** numbers, and nothing else:

| Input | Term in the equations | Depends on |
| --- | --- | --- |
| `CD` | drag force, `C_D` | Mach and yaw |
| `CLA` | lift force, `C_Lα` | Mach and yaw |
| `CNP` | Magnus moment, `C_Mpα` | Mach and yaw |
| `CYP` | Magnus force, `C_Npα` | Mach |
| `CLP` | spin damping moment, `C_lp` | Mach |
| `CMA` | overturning moment, `C_Mα` | Mach |
| `CMQ` | pitch damping moment, `C_Mq` | Mach |

`AerodynamicCoefficients` holds exactly those, in whatever form you have them —
a constant, a table in Mach, a grid in `(Mach, yaw)`, or a callable. Anything
omitted is zero, which is how you switch a term off:

```python
from sixdof import AerodynamicCoefficients, load_coefficients

AerodynamicCoefficients(CD=0.3, CLA=1.8, CMA=3.5, CMQ=-9.4, CLP=-0.03)
AerodynamicCoefficients(mach_grid=machs, CD=cd_values, CLA=cla_values, ...)
AerodynamicCoefficients(CD=lambda mach, alpha: 0.2 + 0.5 * np.sin(alpha) ** 2)

load_coefficients("my_shell.npz")     # or .xlsx, or .csv — it reads the content
```

#### The reference at the input is McCoy

The seven are read as **McCoy, *Modern Exterior Ballistics*, 2nd ed., ch. 2**
defines them, and `sixdof.dynamics` implements that chapter term for term. Two
properties of that reference decide what your numbers have to mean, and neither
is visible from a column name:

**Nondimensionalisation.** The four that multiply an angular rate — `CLP`,
`CMQ`, `CYP`, `CNP` — are nondimensionalised on `(pd/V)`, per McCoy eq. (2.4).
The NACA aeroballistic system uses `(pd/2V)` instead, and McCoy flags the
consequence himself: *"a factor of two difference in coefficients that depend on
angular velocity."* A NACA-normalised table needs those four **halved** on the
way in. `CMA` carries no rate and is identical in both systems, which is the
cross-check.

**Axis system and sign.** `CD` and `CLA` are wind-axis. A source giving
body-axis axial and normal force needs the rotation through the angle of attack
first, and the sign of the axial term depends on whether that source counts it
positive forward (McCoy's convention, `C_X ≈ −C_D`) or positive rearward.

Neither mistake raises an exception. Both produce a plausible trajectory and a
wrong answer.

#### Converting a source table is yours to do

Wind-tunnel reports, range reductions, CFD and the various prediction codes each
tabulate their own intermediate quantities — axial force split into `CX0` and
`CX2`, normal force as `CNA`, Magnus as a series in `CNPA`/`CNPA3` — and each
needs its own arithmetic to reach the seven. **That arithmetic is not in the
package.** A conversion buried in a library is a conversion nobody checks, and
the two traps above are exactly the kind that survive review. Convert your table
once, next to your data, where a reader can see it, and hand over the seven.

`tests/test_coefficients.py` holds the package to that: it tokenises every
module under `src/sixdof/`, strips comments and strings, and asserts that no
source convention's column names appear in executable code anywhere.
`load_coefficients` reads the seven — from `.npz`, from a two-sheet workbook, or
from a flat `Mach` + seven-column table — and refuses anything else rather than
guessing at it.

`examples/07_bring_your_own_table.py` is a worked conversion to copy from. It
takes the 5"/38 source table apart and measures what each choice costs.

#### The shipped 5"/38 table

- `data/aero_coefficients_5in38_spin73.npz` — the seven, on the full `(Mach,
  yaw)` grid. The default `naval_5in38_coefficients()` loads, and the table the
  thesis campaign was flown on.
- `data/aero_coefficients_5in38_spin73_sheets.xlsx` — the same seven as two
  editable sheets (`mach_only` and `yaw_dependent`).
- `data/aero_coefficients_5in38.xlsx` — the source table, kept for provenance.
  The package does not read it; the worked example and the frozen tests do.

The name says `spin73` because **this table does not meet the McCoy contract
above**, in two known ways — both inherited from the thesis, both kept
deliberately so the published results stay reproducible:

1. `CLP`, `CMQ`, `CYP` and `CNP` come from a SPIN-73 tabulation, normalised on
   `(pd/2V)`, so they are twice what the equations want. Symptom: spin at impact
   94 rev/s where an independent code gives 129.
2. `CD` was assembled with the yaw term subtracted rather than added, so drag
   falls with angle of attack instead of rising. Worth about −0.3 % of range on
   the thesis trajectory, and more at high yaw.

[`docs/table_5in38_provenance.md`](docs/table_5in38_provenance.md) has the
derivation, the primary sources, the measurements, and how to run the corrected
physics instead.

**One caution before flattening further.** It is tempting to drop the yaw axis
and keep a plain `Mach, CD, CLA, CYP, CLP, CMA, CNP, CMQ` table. `CD` and `CLA`
survive that reasonably — they move a few per cent over the yaw angles a stable
shell actually flies. `CNP` does not: the Magnus moment is **odd** in the angle
of attack, so it is exactly zero at zero yaw and no single Mach-indexed value
can stand for it. Sampling at zero yaw does not approximate the term, it deletes
it. Measured on the reference shot, a zero-yaw Mach-only table moves the range
by 12.6 m in 16.7 km and the drift by 0.9 m in 452 m — small, but not nothing,
and `from_mach_table` says so in its docstring.

### Monte Carlo campaign

```python
from sixdof import surface_target_fleet
from sixdof.montecarlo import AimPoint, DispersionSettings, MonteCarloCampaign

campaign = MonteCarloCampaign(
    simulator, surface_target_fleet,
    DispersionSettings(n_shots=1000, sigma_elevation_deg=0.1, sigma_azimuth_deg=0.05),
)
results = campaign.run([AimPoint(39.6, -1.35, 16796.8, 4.26)])
table = MonteCarloCampaign.to_frame(results)
```

### How near did it pass?

A sweep reduces every flight to a few numbers, and which few is the whole
question. The thesis wants the impact point, because it is shooting at
something floating. A shot at an air target is never landed on, so the useful
reduction is the closest approach in three dimensions to a list of fixed points
in space:

```python
from sixdof.montecarlo import AngleSweep, NearestApproach, SweepGrid

points = [(16673.7, 200.0, 0.0), (16468.7, 197.5, 0.0)]   # (x, y, z) in m
tracker = NearestApproach(points)

sweep = AngleSweep(simulator, SweepGrid())
table = sweep.run(reduce=lambda elev, azim, traj: tracker.absorb(traj, (elev, azim)))

for approach in tracker.best():
    print(approach.label, approach.distance_m, approach.time_s)
```

One grid walk answers both questions: `table` is the ordinary sweep, and
`tracker` holds the pair that came nearest to each point, when it did, and how
near. The accumulator keeps only a running best, so a sweep of tens of thousands
of flights costs a few hundred bytes per point rather than storing anything.

The points are static and nothing is matched to a time. That is the point of
the abstraction: it does not know what the points *are*. A corridor of drone
waypoints, a ladder of aim points and a line of range markers are the same
problem to it, and the anti-air study is one caller among the possible ones.

---

## Layout

```
src/sixdof/
  aerodynamics.py   the seven coefficients the equations read, and the loader
  projectile.py     mass, inertia, calibre, rifling
  weapon.py         mount, laying angles, platform coupling
  vessel.py         box-shaped surface target
  environment.py    density, gravity, wind, speed of sound
  dynamics.py       equations of motion and the initial state
  events.py         ground impact and proximity fuze stop conditions
  simulator.py      integration driver
  trajectory.py     state histories, derived quantities, statistics
  plotting.py       the eighteen standard figures
  presets.py        the 5"/38 configuration used throughout the thesis
  aa/               target geometry, warhead, fuze, damage model
  montecarlo/       sweep, closest approach, point selection, dispersion, cost

examples/           runnable scripts -- see examples/README.md
scripts/            proof_of_equivalence.py, reproduce_campaign_point.py
tests/              the suite, including the bit-exact regression
tests/reference/    frozen copies of the pre-refactor code (never imported)
data/               coefficient table and the published intermediate results
docs/               architecture, table provenance, target geometry, verification
```

---

## Verification

The engine this repository grew out of was a single 1600-line file
(`Motor.py`), whose results are quoted in the thesis. The refactor is
**bit-identical** to it, and that is demonstrated, not asserted.

`tests/reference/` holds unmodified copies of the pre-refactor code — the
single-file engine, the anti-air engine and the damage appendix — with their
MD5 sums documented in [`tests/reference/README.md`](tests/reference/README.md)
so you can confirm they are the original bytes. Nothing in `src/` imports them;
they exist only to be compared against.

### Run the proof yourself

```bash
python scripts/proof_of_equivalence.py
```

It loads the frozen code and the package, runs them side by side, and exits 1
if anything differs. The stored output is
[`docs/verification.md`](docs/verification.md), regenerated with
`--report docs/verification.md`.

It reports, for each of eight scenarios, the sample count, the maximum absolute
difference over the whole `12 × N` state history, and a **SHA-256 of each
engine's raw bytes**. Matching digests mean no bit differs anywhere in roughly
19 000 doubles per trajectory:

```text
  Exemplo.py de referência (43.3°)
    amostras                    : 1671
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    SHA-256 pacote refatorado   : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    idêntico                    : SIM
```

#### What "identical" is a claim about

Both engines are handed **the same coefficient arrays** — the frozen engine
builds its 100×100 grid from `data/aero_coefficients_5in38.xlsx`, and the
package is given those very arrays rather than the `.npz` bake of them that
ships in `data/`. So what the equality demonstrates is that the two bodies of
code agree, which is the claim; it is not accidentally also a claim about the
machine you are on.

That distinction matters, and it is not pedantic. Three of the seven
coefficients — `CD`, `CLA`, `CNP` — are built with `sin` and `cos` of the yaw
mesh, and libm implementations disagree in the last place across platforms
(glibc and the MSVC runtime differ by up to one ULP). The `.npz` was baked once,
on one machine. Compare a bake against a rebuild somewhere else and one ULP is
enough for `solve_ivp` to choose a different, equally valid step sequence — 1594
samples instead of 1592 — which looks like a physics failure and is not one.

Nor is trigonometry the only culprit: the cubic interpolation of the source
columns solves a tridiagonal system, and LAPACK builds are no more bit-portable
than libm, so even the four coefficients with no `sin` or `cos` in them drift by
a last bit between platforms. The bake is therefore checked in **ULPs** — a
budget of 8, against the ~1e12 a stale or wrong bake would be off by. That keeps
the check pointed at the failure it is for.

The `.npz` is a convenience — it saves rebuilding the grid on every import, and
it is what `naval_5in38_coefficients()` returns for ordinary use. Nothing about
the physics depends on which of the two you load. `tests/matched_coefficients.py`
carries the full account.

### What is checked

| Check | Against what | Result |
| --- | --- | --- |
| Coefficient grids and point lookups | frozen `motor_original.py` | identical |
| Launch state and right-hand side | frozen `motor_original.py` | identical |
| Whole trajectories, 8 scenarios (5 elevations, wind, moving platform) | frozen `motor_original.py` | identical, matching SHA-256 |
| Derived histories — speed, Mach, `\|h\|`, spin, angle of attack | frozen `motor_original.py` | max \|Δ\| = 0 |
| The four headline statistics of the thesis | frozen `motor_original.py` | equal as floats |
| Anti-air console report | frozen `legacy_motor.py` | character for character |
| Damage model, 3 burst geometries | frozen `apendice_dano.py` | every quantity equal |

### An outside check: a published M107 case

Everything above compares the package against the engine it was refactored
from. That proves the refactor faithful. It cannot prove the physics right,
because both sides share the equations, the data and any mistake in either.

So one example flies somebody else's projectile with somebody else's
coefficients:

```bash
python examples/11_m107_benchmark.py
```

It takes the 155 mm M107 case and the Table 1 coefficients of

> Khalil, M., Abdalla, H., Kamal, O., "Dispersion Analysis for Spinning
> Artillery Projectile", ASAT-13, Military Technical College, Cairo, 2009

and compares against the numbers the paper states in its own text. As a second opinion it also quotes
**[RigidFlightLab](https://github.com/timeout187/RigidFlightLab)** — worth a
look in its own right, and recommended: an independent open implementation of
the same case in a different formulation, non-rolling frame,
`[x,y,z,u,v,w,φ,θ,ψ,p,q,r]`, RK45, where this package uses McCoy's vector form.
Two codes that share no line of source and no integrator, agreeing on a third
party's projectile, is a stronger statement than either makes alone — so this
benchmark exists because that project does.

| Quantity | Paper | Independent code | This package |
| --- | --- | --- | --- |
| Time of flight (s) | 66.67 | 66.40 | 66.137 |
| Time to summit (s) | 31.00 | 30.50 | 30.386 |
| Initial axial deceleration (g) | −4.45 | −4.47 | −4.468 |
| Maximum angle of attack (°) | 1.29 | 1.30 | 1.287 |
| Apogee (m) | ~5600 | 5647 | 5635.7 |
| Spin at impact (rev/s) | — | 128.8 | 128.853 |
| Drift (m) | — | 483 | 482.821 |

Within 0.8% of the paper on everything it publishes as text, and within 0.5% of
the independent code. Both codes sit about 1.7% below the paper on apogee, most
likely because the paper uses an unspecified atmosphere of its own and includes
Coriolis terms neither code implements.

Getting there needs two conversions on the way in, and the example is written to
make both visible rather than quietly right. Table 1 is nondimensionalised on
`(pd/2V)` while McCoy's equations use `(pd/V)`, so the four rate-dependent
coefficients are halved — `CMA` carries no rate and passes through untouched,
which is the cross-check that tells a convention mismatch apart from a wrong
table. And Table 1 gives body-axis coefficients, `C_A` and `C_Nα`, per the
paper's own Nomenclature, so they need the rotation through the angle of attack
described under [the shipped 5"/38 table](#the-shipped-538-table). The script
prints the run with and without each assumption, so their cost is measured
rather than asserted — the factor of two alone is worth 61 m of drift and 34
rev/s of spin.

### The naval-drone campaign — the thesis case proper

The anti-air layer above is newer work. What the thesis is actually about is
surface fire against naval drones, and that runs through the four Monte Carlo
stages. Each is checked against what the thesis published:

| Stage | Check | Result |
| --- | --- | --- |
| 1 — angle sweep | Published sweep rows re-integrated | Agrees to ~1e-9 relative, always sub-millimetre — see the caveat below |
| 2 — aim-point selection | The 163 published aim points | Reproduced cell for cell |
| 3 — dispersion campaign | Published hit counts at aim points 1, 160, 162 and 163 (1000 rounds each) | **Every hit count exact**; dispersion statistics to ~1e-6 relative, worst 0.4 mm |
| 4 — engagement cost | Frozen `Custo.py` on the published hit rates | Every field equal, and the whole E[cost] curve |

Aim point 1 is the maximum-range solution the thesis leads with. Firing its
1000 rounds again reproduced all six hit counts exactly — 63, 227, 227, 249,
219, 182 — and therefore all six published hit rates:

```text
    Acertos_Drone_Sea_Baby                   63           63   SIM
    Acertos_IRIS_Paykan                     227          227   SIM
    Acertos_Osa_class                       227          227   SIM
    Acertos_Hayabusa_class                  249          249   SIM
    Acertos_SMS_V4                          219          219   SIM
    Acertos_PT_105                          182          182   SIM
```

Reproduce it yourself:

```bash
python scripts/reproduce_campaign_point.py              # points 162 and 163, minutes
python scripts/reproduce_campaign_point.py --point 1    # maximum range, ~1 hour
```

The stored output is [`docs/verification_campaign.md`](docs/verification_campaign.md),
covering points 162 and 163; points 1 and 160 were verified the same way and are
reported in the table above.

### A caveat about the published tables

The published workbooks are **not** bit-reproducible on a different machine,
and that is a property of the original code, not of this refactor. The frozen
engine and this package agree with each other exactly while **both** differ
from the workbook by the same tiny amount — typically 1e-9 relative, worst case
about 1e-7, never more than a millimetre of range.

The cause is ordinary: `solve_ivp` is adaptive, the workbooks were produced on
Windows with an older SciPy/NumPy, and a last-bit difference in the platform's
`sin`/`cos` sends the integrator down a different but equally valid step
sequence. The scale of the disagreement is the integrator's own `rtol=1e-7`.

This matters for how you read the numbers. Hit counts and hit rates are
integers and reproduce exactly. Continuous statistics — CEP, error standard
deviations — should be quoted as agreeing to about six significant figures, not
to the last digit. `tests/test_naval_pipeline.py` is what pins the blame on the
platform rather than the refactor.

### Running the checks

```bash
pytest                                          # 181 tests, ~3 min
pytest tests/test_regression_vs_original.py -v  # the bit-exact suite
pytest tests/test_naval_pipeline.py -v          # the thesis pipeline
pytest -m "not slow"                            # skip the full integrations
```

---

## What changed from the previous version

The previous layout lives on the [`legacy`](../../tree/legacy) branch and still
runs. It was one file plus example snippets that had to be **pasted into the
bottom of that file** to run at all. This branch keeps the physics and changes
the packaging:

- **The engine is a package.** Ten focused modules instead of one file, with the
  plotting split off so a Monte Carlo loop never imports matplotlib.
- **Examples are scripts.** `python examples/01_single_shot.py`, with
  command-line arguments, instead of copy-and-paste.
- **No hard-coded paths.** The old `C:\Users\DELL\Downloads\...` defaults are
  gone; data files resolve relative to the repository.
- **The anti-air appendix is a module**, and a generic one: any faceted target,
  any polar fragment distribution.
- **The thesis campaign is reproducible.** Sweep, selection, dispersion and cost
  are library code with a script per stage, and the published intermediate
  tables ship in `data/`.
- **A test suite and runnable proofs** — 181 tests plus the scripts in
  `scripts/`, which are what back the equivalence claims.
- **Coefficients you supply directly** — `AerodynamicCoefficients` takes the
  seven the equations read, in McCoy's definitions, instead of fourteen columns
  of which four are dead. Converting a source tabulation is deliberately outside
  the package: `examples/07_bring_your_own_table.py` is the worked case, and
  `docs/table_5in38_provenance.md` declares the two respects in which the
  shipped 5"/38 table departs from that contract.
- **One real bug fixed.** The original unwrapped interpolator results with
  `float(x)` on a 1×1 array, which NumPy 2.0 rejects — so `Motor.py` no longer
  runs on a current SciPy/NumPy stack. The fix takes the first element, which is
  exactly what `float()` used to do, so no number changes.
- **Identifiers and docstrings are in English**, matching the class names the
  original already used, so the code reads for an international audience.
  Console output stays in Portuguese, unchanged.

---

## Provenance

Who wrote and checked what, in order:

**The original engine (`legacy` branch).** Written by Luiz Guilherme de Padua
Sanches with the help of large language models (Claude and ChatGPT), **reviewed
and verified by the author**, and used in his undergraduate thesis in Applied
and Computational Mathematics at IME-USP. Every figure and table in the written
work comes from this code. It is the verified artefact.

**This refactor (`main` branch).** Written by **Claude** (Anthropic's AI
assistant) at the author's request, in a single working session, along with its
test suite and the equivalence proof. The author set the requirements, chose the
trade-offs, and reviewed and approved the result — but the thesis figures have
not been independently re-derived from this branch, and it was not part of the
work submitted to the university. It is new.

What that means in practice: the equivalence to `legacy` is machine-checked and
reproducible by anyone (`scripts/proof_of_equivalence.py`), so the numbers are
not in question. What has not happened is a second human review of the
restructured code itself. Treat `legacy` as the citable artefact and `main` as
the maintained one.

**Sources for the physical data.** The aerodynamic coefficients and the fragment
distribution come from published ordnance reports; the specific sources are
cited in the modules that use them (`sixdof/aa/presets.py` for the warhead,
`docs/shahed_target_geometry.pt-BR.md` for the target geometry).

## Citation

If you use this in academic work, please cite the thesis alongside the
software — see [`CITATION.cff`](CITATION.cff). State which branch you used.

## License

MIT — see [`LICENSE`](LICENSE). You may use, modify and redistribute this
freely, including commercially, as long as the copyright notice and the licence
text travel with it. For academic use, please also cite the thesis.

## Reference

McCoy, R. L. *Modern Exterior Ballistics: The Launch and Flight Dynamics of
Symmetric Projectiles.* 2nd ed. Schiffer Publishing, 2012.
