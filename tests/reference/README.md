# Frozen reference implementations

These files are **not part of the package**. They are the pre-refactor code,
kept byte-for-byte so the refactored engine can be checked against it. Nothing
in `src/` imports them; only the tests and
`scripts/proof_of_equivalence.py` do.

**Do not edit them.** Their entire value is that they are the original bytes.
If one of them needs to change, the change belongs in `src/`, and the
regression tests exist to tell you the results moved.

| File | Provenance | MD5 |
| --- | --- | --- |
| `motor_original.py` | `Motor.py` from the `legacy` branch (commit `257a024`), unmodified. The 6-DOF engine whose results the thesis quotes. | `9d1909cead60ace2eaf963b4b6493505` |
| `apendice_dano.py` | The procedural damage appendix from the author's working tree, unmodified. Generalised into `src/sixdof/aa/`. | `bff9266b9846ff6b5f6d75bf2fcece32` |
| `legacy_motor.py` | `Motor.py` plus the proximity-fuze event and the article's verification `main`, from the author's working tree, unmodified. Produced the reference anti-air report. | `9fd38885ba132057cb408fa6057080d1` |
| `custo_original.py.txt` | `TCC/Custo/Custo.py`, unmodified — an IDLE session paste, hence `.txt`: it carries the interpreter banner and `... ` continuation prompts and is not importable as-is. | `8fd29d317fcfccdabeb1c6aa7e928166` |
| `compat.py` | **Not frozen.** Helpers written for this refactor — a NumPy 2.x shim and a de-REPL loader; see below. | — |

Verify them yourself:

```bash
md5sum tests/reference/*.py
git show legacy:Motor.py | md5sum          # must match motor_original.py
```

## Why `compat.py` exists

The frozen files unwrap interpolator results with `float(x)` where `x` is the
`(1, 1)` array `RectBivariateSpline` returns. NumPy accepted that for size-1
arrays up to 1.x and rejects it since 2.0:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

So on a current SciPy/NumPy stack the original engine does not run at all.
Rather than editing a frozen file, `compat.patch_legacy_coefficients()`
reinstates the old behaviour at import time, replacing
`get_coefficients` with a version whose **only** difference is how the 1×1
array is unwrapped. `float(a)` on a size-1 array returned `a.flat[0]`, and the
replacement returns exactly that. The shim changes plumbing, never arithmetic.

On NumPy 1.x it detects that no patch is needed and does nothing.

`src/sixdof/aerodynamics.py` carries the same fix permanently, in `_scalar()`.
That is the one behavioural difference between the frozen code and the package,
and it changes no value — which is what the bit-exact regression tests
demonstrate.

## Why `custo_original.py.txt` is a `.txt`

It is what the author pasted out of an IDLE window, so almost every line begins
with a `... ` continuation prompt and the first two lines are the interpreter
banner. Editing it into valid Python would mean touching a frozen artefact, so
instead `compat.strip_repl_prompts()` removes the banner and the prompts
mechanically — nothing else — and `compat.load_repl_function()` compiles the one
function the tests need out of the recovered source. The function's own bytes
are the author's, character for character.
