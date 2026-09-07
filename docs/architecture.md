# Architecture

Why the code is split the way it is, and where to add things.

## Layering

```
                    ┌─────────────────────────────┐
   optional  ───▶   │  sixdof.aa                  │   fuze, target, warhead,
                    │                             │   fragment damage
                    ├─────────────────────────────┤
   optional  ───▶   │  sixdof.montecarlo          │   sweep, selection,
                    │                             │   dispersion, cost
                    ├─────────────────────────────┤
   core      ───▶   │  sixdof                     │   equations of motion,
                    │                             │   integration, trajectory
                    └─────────────────────────────┘
```

The dependency arrows point downwards only. `sixdof.aa` and
`sixdof.montecarlo` import from the core; the core imports from neither, and
they do not import each other. Importing `sixdof` loads neither optional layer.

## Core, module by module

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `paths` | Where the shipped data files live | — |
| `environment` | Density, gravity, wind, speed of sound, as functions of altitude | — |
| `projectile` | Mass, inertia, calibre, rifling, unit conversion | — |
| `vessel` | Box target: bounds and hit test | — |
| `weapon` | Mount, laying angles, platform coupling | `vessel` |
| `aerodynamics` | The seven coefficients the equations read, in any form, and the loader | — |
| `dynamics` | Equations of motion, initial state | `projectile`, `environment` |
| `events` | Terminal stop conditions | — |
| `simulator` | Integration driver | `dynamics`, `events`, `trajectory` |
| `trajectory` | State histories, derived quantities, statistics | — |
| `plotting` | The eighteen standard figures | `trajectory` |
| `presets` | The 5"/38 configuration used in the thesis | `projectile`, `weapon`, `environment`, `vessel` |

Two boundaries carry weight:

**`trajectory` does not import matplotlib.** A Monte Carlo campaign builds
163 000 `Trajectory` objects; none of them should drag a plotting backend into
the process. `plotting.TrajectoryPlotter` takes a trajectory as an argument
instead, and even then imports `pyplot` lazily inside the methods.

**The package contains no coefficient adapter at all.** `aerodynamics` defines
a seven-value contract, in McCoy's definitions and his `(pd/V)`
nondimensionalisation, and a loader that reads those seven from a file. It does
not convert. Every tabulation convention — axial/normal force decompositions,
Magnus series, NACA-normalised rate coefficients — is converted *outside* the
package, by whoever brings the table;
`examples/07_bring_your_own_table.py` is the worked case.

This started life as an adapter module and was deliberately removed. The reason
is that the two ways a coefficient table goes wrong — a factor of two from the
`(pd/2V)` convention, and the sign of the yaw-drag term — are both silent: they
produce a plausible trajectory and a wrong answer. A conversion inside the
library is one nobody reads; a conversion next to the data is one that gets
reviewed. `tests/test_coefficients.py` enforces this by tokenising **every**
module under `src/sixdof/`, discarding comments and strings, and asserting that
no source convention's column names appear in executable code anywhere. The
loader refuses a table it cannot read as the seven rather than guessing at it.

The shipped 5"/38 table carries both of those errors, inherited from the thesis
and kept so the published results reproduce; `docs/table_5in38_provenance.md`
declares them and measures them.

**The atmosphere is asked, not read.** `six_dof_rhs` calls
`environment.density_at(h)` and `environment.sound_speed_at(h)` rather than
reading `rho` and `sound_speed` off the dataclass. On the base `Environment`
both return the constants it already holds, so the uniform atmosphere the
thesis assumed is unchanged and the proof of equivalence still passes bit for
bit; `LayeredAtmosphere` overrides them with the ICAO profile. Two methods is
the whole extension point, and the default stays the thesis's assumption rather
than quietly improving on it — a model that changes published numbers when you
upgrade it is worse than one that makes you ask.

**`dynamics` is a free function, not a method.** `six_dof_rhs` takes the
projectile, the environment and the coefficient source explicitly. That keeps
the physics testable without constructing a simulator, and it is why
`tests/test_regression_vs_original.py` can compare the right-hand side directly
against the original engine.

## Where to extend

| You want to… | Do this |
| --- | --- |
| Model a different shell | `Projectile.from_imperial(...)` or `Projectile(...)`; nothing else changes. |
| Supply your own coefficients | `AerodynamicCoefficients(CD=..., CLA=..., ...)` — each a constant, a Mach table, a `(Mach, yaw)` grid or a callable. |
| Read a table in someone else's convention | Copy `examples/07_bring_your_own_table.py`, change the arithmetic to match your source, and pass the resulting `AerodynamicCoefficients` to the simulator. Keep it with your data, not in the package. |
| Use an altitude-dependent atmosphere | `LayeredAtmosphere()` for ICAO; for anything else, subclass `Environment` and override `density_at` / `sound_speed_at`. |
| Add an aerodynamic term | Edit `six_dof_rhs` — but expect `test_regression_vs_original.py` to fail, which is the point: it is telling you the trajectories moved. |
| Stop the integration on a new condition | Add an event factory in `events.py` and pass it through `simulate(extra_events=...)`. |
| Model another air target | `box_target`, `triangular_prism_target`, or a `Target` built from your own `Facet` list. |
| Model another warhead | `FragmentationWarhead` with your own `PolarZone` table. |
| Score damage differently | Subclass or replace `FragmentDamageModel`; it reads only `Target` and `FragmentationWarhead`. |
| Change the dispersion study | `DispersionSettings` for the perturbations, `build_targets` for what counts as a hit. |

## The frozen references

`tests/reference/` holds unmodified copies of the code this package was
refactored from: the single-file engine `motor_original.py`, the anti-air engine
`legacy_motor.py`, the procedural `apendice_dano.py`, and the cost script
`custo_original.py.txt`. They are test fixtures, never imported by the package,
and must not be edited — their whole value is that they are the pre-refactor
bytes. `tests/reference/README.md` documents each file's provenance and MD5.

`tests/reference/compat.py` is the one concession: it reinstates NumPy 1.x
behaviour for `float()` on a 1×1 array so the frozen engine can run at all on a
current stack. It changes how a value is unwrapped, never the value.

## Numerical fidelity

The refactor is bit-identical to the original engine, which constrains how the
core may be written:

- the expressions in `six_dof_rhs` and `build_initial_state` are kept in their
  original associativity — re-grouping floating-point operations moves the last
  digits;
- the coefficient grid is built in the same order with the same interpolants;
- `solve_ivp` is called with the same method, tolerances and step ceiling;
- the derived histories in `Trajectory._compute_derived` keep the original
  scalar loops rather than being vectorised, because `math.atan2` and
  `numpy.arctan2` are not guaranteed to agree to the last bit.

If a change here is worth making anyway, make it deliberately and update the
regression test in the same commit, with the reason in the message.
