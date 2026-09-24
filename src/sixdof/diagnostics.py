"""Warnings raised after a trajectory is integrated.

Two conditions on the total angle of attack ``alpha_t`` -- the angle between the
axis of symmetry and the air-relative velocity -- are worth flagging, because
the integration completes and returns a plausible trajectory in both:

``alpha_t`` above :data:`HIGH_ALPHA_DEG` (10 degrees)
    The coefficients are small-yaw quantities: derivatives per ``sin(alpha)``,
    a yaw-drag term in ``sin^2(alpha)``, a Magnus polynomial fitted at a few
    degrees.  The shipped tables are tabulated to +/-10 degrees and clipped
    beyond it (``docs/table_5in38_provenance.md``).  Past that the forces and
    moments follow from the formulation, not from the air.  Flat fire never
    gets there; high-angle fire does, near the summit, where the yaw of repose
    grows as ``1/V^3``.

``alpha_t`` near 180 degrees, above :data:`SINGULAR_ALPHA_DEG`
    The equations take ``alpha_t = acos(v . i')``, whose sensitivity grows as
    ``1/sin(alpha_t)``, and every yaw-dependent force and moment acts along a
    vector whose direction is undefined when the axis and the velocity are
    antiparallel.  Near there, small integration errors in ``v`` or ``i'`` turn
    into large errors in the direction of those terms.

Neither changes the trajectory.  Each is a :class:`UserWarning` subclass, so
it can be silenced or turned into an error with the standard ``warnings``
filters.  The message text is fixed, so Python's default filter prints it once
per call site rather than once per shot in a Monte Carlo run; the numbers are
in :attr:`sixdof.trajectory.Trajectory.diagnostics`.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np

#: Above this total angle of attack, in degrees, the coefficient formulation is
#: outside the small-yaw range it was built for.
HIGH_ALPHA_DEG = 10.0

#: Above this total angle of attack, in degrees, the geometry of the equations
#: approaches the singularity at 180 degrees.
SINGULAR_ALPHA_DEG = 175.0


class HighAngleOfAttackWarning(UserWarning):
    """``alpha_t`` exceeded the range the aerodynamic coefficients describe."""


class AngleOfAttackSingularityWarning(UserWarning):
    """``alpha_t`` came close to 180 degrees, where the equations are singular."""


def _excursions(t: np.ndarray, alpha_deg: np.ndarray, limit_deg: float) -> Dict[str, float]:
    above = alpha_deg > limit_deg
    k = np.flatnonzero(above)
    i_max = int(np.argmax(alpha_deg))
    dt = np.diff(t)
    return dict(
        limite_deg=float(limit_deg),
        alpha_max_deg=float(alpha_deg[i_max]),
        t_alpha_max_s=float(t[i_max]),
        t_inicio_s=float(t[k[0]]),
        t_fim_s=float(t[k[-1]]),
        tempo_acima_s=float(np.sum(dt[above[:-1]])) if len(dt) else 0.0,
    )


def check_angle_of_attack(
    trajectory,
    *,
    high_deg: float = HIGH_ALPHA_DEG,
    singular_deg: float = SINGULAR_ALPHA_DEG,
    stacklevel: int = 2,
) -> List[Dict[str, float]]:
    """Warn if the trajectory's ``alpha_t`` crossed either limit.

    Returns the findings, one dict per condition met, with the limit, the
    maximum ``alpha_t`` and when it happened, and the time spent above the
    limit.  An empty list means neither limit was crossed.
    """
    t = np.asarray(trajectory.t, dtype=float)
    alpha = np.asarray(trajectory.alpha_traj, dtype=float)
    findings: List[Dict[str, float]] = []
    if alpha.size == 0:
        return findings

    if np.any(alpha > high_deg):
        found = dict(tipo="angulo_de_ataque_alto", **_excursions(t, alpha, high_deg))
        findings.append(found)
        warnings.warn(
            HighAngleOfAttackWarning(
                f"ângulo de ataque total alto (α_t > {high_deg:g}°): possibilidade de "
                "instabilidade por formulação dos coeficientes aerodinâmicos "
                "(detalhes em Trajectory.diagnostics)"
            ),
            stacklevel=stacklevel + 1,
        )

    if np.any(alpha > singular_deg):
        found = dict(tipo="singularidade_180", **_excursions(t, alpha, singular_deg))
        findings.append(found)
        warnings.warn(
            AngleOfAttackSingularityWarning(
                f"α_t perto de 180° (α_t > {singular_deg:g}°): possibilidade de erro "
                "numérico por singularidade (detalhes em Trajectory.diagnostics)"
            ),
            stacklevel=stacklevel + 1,
        )

    return findings


def describe(findings: List[Dict[str, float]]) -> List[str]:
    """One line per finding, for the verbose report."""
    lines = []
    for f in findings:
        label = ("ângulo de ataque alto" if f["tipo"] == "angulo_de_ataque_alto"
                 else "α_t perto de 180°")
        lines.append(
            f"⚠ {label}: α_t máx {f['alpha_max_deg']:.2f}° em t = {f['t_alpha_max_s']:.2f} s; "
            f"acima de {f['limite_deg']:g}° por {f['tempo_acima_s']:.2f} s "
            f"(de t = {f['t_inicio_s']:.2f} a {f['t_fim_s']:.2f} s)"
        )
    return lines


__all__ = [
    "HIGH_ALPHA_DEG",
    "SINGULAR_ALPHA_DEG",
    "HighAngleOfAttackWarning",
    "AngleOfAttackSingularityWarning",
    "check_angle_of_attack",
    "describe",
]
