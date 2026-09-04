"""Helpers that make the frozen reference artefacts runnable, without editing them.

Two of them:

* :func:`patch_legacy_coefficients` -- a NumPy 2.x shim for the frozen engines.
* :func:`load_repl_function` -- extracts a function from ``custo_original.py.txt``,
  which is an IDLE session paste rather than a source file.

Neither changes any arithmetic.

NumPy 2.x compatibility shim for the frozen reference engine.

``tests/reference/motor_original.py`` is kept byte-for-byte identical to the
``Motor.py`` of the ``legacy`` branch, so it can serve as the ground truth for
the regression tests.  That file unwraps interpolator results with ``float(x)``,
which NumPy accepted for size-1 arrays up to 1.x and rejects since 2.0::

    TypeError: only 0-dimensional arrays can be converted to Python scalars

:func:`patch_legacy_coefficients` reinstates the old behaviour by replacing
``RealAerodynamicCoefficients.get_coefficients`` with a version whose only
difference is how the 1x1 array is unwrapped.  ``float(a)`` on a size-1 array
returned ``a.flat[0]``, and that is precisely what the replacement does, so the
values are bit-identical -- the shim changes plumbing, never arithmetic.

On NumPy 1.x the shim is a no-op: the original method already works and is left
in place.
"""

from __future__ import annotations

import numpy as np


def _needs_patch() -> bool:
    """Whether this NumPy rejects ``float()`` on a 1x1 array."""
    try:
        float(np.zeros((1, 1)))
    except TypeError:
        return True
    return False


def _scalar(value) -> float:
    """``float(value)`` as NumPy 1.x defined it for size-1 arrays."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


def patch_legacy_coefficients(module) -> bool:
    """Make ``module.RealAerodynamicCoefficients`` runnable on NumPy 2.x.

    Parameters
    ----------
    module:
        The imported ``motor_original`` module.

    Returns
    -------
    bool
        ``True`` if the patch was applied, ``False`` if it was unnecessary.
    """
    if not _needs_patch():
        return False

    def get_coefficients(self, mach, alpha_rad=0.0):
        mach = np.clip(mach, self.mach_min, self.mach_max)
        alpha_rad = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])

        coeffs = {}
        coeffs["CD_total"] = _scalar(self.interp_2d["CD_total"](mach, alpha_rad))
        coeffs["CNP_total"] = _scalar(self.interp_2d["CNP_total"](mach, alpha_rad))
        coeffs["CLA_total"] = _scalar(self.interp_2d["CLA_total"](mach, alpha_rad))

        for name in ["CNA", "CMA", "CMQ", "CLP", "CYP"]:
            if name in self.interp_2d:
                coeffs[name] = _scalar(self.interp_2d[name](mach))

        return coeffs

    module.RealAerodynamicCoefficients.get_coefficients = get_coefficients
    return True


def strip_repl_prompts(text: str) -> str:
    """Turn an IDLE session paste back into plain source.

    ``custo_original.py.txt`` is what the author pasted out of an IDLE window:
    two banner lines, then the real source with ``... `` continuation prompts on
    most lines.  It is therefore not importable as-is.  This removes the banner
    and the prompts and nothing else -- no reformatting, no edits -- so the
    recovered source is the author's, character for character.
    """
    lines = text.replace("\r\n", "\n").split("\n")

    # Drop the interpreter banner, if present.
    if lines and lines[0].startswith("Python ") and " on " in lines[0]:
        lines = lines[2:]

    recovered = []
    for line in lines:
        if line.startswith("... ") or line.startswith(">>> "):
            recovered.append(line[4:])
        elif line in ("...", ">>>"):
            recovered.append("")
        else:
            recovered.append(line)
    return "\n".join(recovered)


def load_repl_function(path, name: str, namespace: dict = None):
    """Compile one named function out of a REPL paste and return it.

    Only the requested ``def`` is compiled -- the script's module-level body
    reads hard-coded Windows paths and draws plots, neither of which belongs in
    a test.  The function's own bytes are untouched.

    Parameters
    ----------
    path:
        The frozen ``.py.txt`` paste.
    name:
        Name of the function to recover.
    namespace:
        Globals the function needs (e.g. ``{"np": numpy}``).
    """
    import ast
    from pathlib import Path

    source = strip_repl_prompts(Path(path).read_text(encoding="utf-8", errors="replace"))
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            scope = dict(namespace or {})
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
            return scope[name]

    raise LookupError(f"function {name!r} not found in {path}")


__all__ = ["patch_legacy_coefficients", "strip_repl_prompts", "load_repl_function"]
