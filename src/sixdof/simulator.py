"""Driver that integrates the 6-DOF equations for one shot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import SixDofEquations, build_initial_state
from .events import make_ground_event, make_proximity_fuze_event
from .trajectory import Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from .environment import Environment
    from .projectile import Projectile
    from .weapon import Weapon


class FuzeSpec(Protocol):
    """Minimal interface a fuze must expose to be usable as a stop condition."""

    target_center: Sequence[float]
    radius_m: float
    arm_time_s: float


@dataclass(frozen=True)
class IntegrationSettings:
    """Numerical settings handed to ``scipy.integrate.solve_ivp``.

    The defaults are the ones used throughout the thesis.  ``DOP853`` is an
    explicit 8th-order Runge-Kutta pair; the tight tolerances and the 0.1 s
    step ceiling keep the fast epicyclic motion resolved, which matters because
    the yaw cycle is much shorter than the flight time.

    Attributes
    ----------
    method:
        ``solve_ivp`` integrator name.
    rtol, atol:
        Relative and absolute tolerances.
    max_step:
        Upper bound on the step size, in s.
    """

    method: str = "DOP853"
    rtol: float = 1e-7
    atol: float = 1e-8
    max_step: float = 0.1


#: The settings used for every reference run in the thesis.
THESIS_SETTINGS = IntegrationSettings()


class BallisticSimulator:
    """Integrate one trajectory for a given projectile, gun and atmosphere.

    Parameters
    ----------
    projectile, weapon, environment:
        Physical configuration.
    aero_coeffs:
        Anything exposing ``get_coefficients(mach, alpha_rad)``.
    settings:
        Integrator settings; defaults to :data:`THESIS_SETTINGS`.

    Examples
    --------
    >>> sim = BallisticSimulator(projectile, weapon, environment, coeffs)
    >>> trajectory = sim.simulate(verbose=False)
    >>> round(trajectory.max_range)          # doctest: +SKIP
    16673
    """

    def __init__(
        self,
        projectile: "Projectile",
        weapon: "Weapon",
        environment: "Environment",
        aero_coeffs,
        settings: IntegrationSettings = THESIS_SETTINGS,
    ) -> None:
        self.projectile = projectile
        self.weapon = weapon
        self.environment = environment
        self.aero_coeffs = aero_coeffs
        self.settings = settings

        self.equations = SixDofEquations(projectile, environment, aero_coeffs)
        self.result: Optional[Trajectory] = None
        self.stop_reason: str = "unknown"

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def build_initial_conditions(
        self,
        alpha0_deg: float = 0.0,
        beta0_deg: float = 0.0,
        w_j0: float = 5.0,
        w_k0: float = 5.0,
    ) -> np.ndarray:
        """State vector at muzzle exit; see :func:`sixdof.dynamics.build_initial_state`."""
        return build_initial_state(
            self.projectile, self.weapon, alpha0_deg, beta0_deg, w_j0, w_k0
        )

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the equations of motion (see :mod:`sixdof.dynamics`)."""
        return self.equations(t, y)

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def simulate(
        self,
        max_time: float = 100.0,
        alpha0_deg: float = 0.0,
        beta0_deg: float = 0.0,
        w_j0: float = 5.0,
        w_k0: float = 5.0,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        *,
        max_step: Optional[float] = None,
        method: Optional[str] = None,
        ground_level: float = 0.0,
        fuze: Optional[FuzeSpec] = None,
        extra_events: Optional[Sequence[Callable]] = None,
        verbose: bool = True,
    ) -> Trajectory:
        """Integrate one shot and return the resulting :class:`Trajectory`.

        Parameters
        ----------
        max_time:
            Integration horizon, in s.  Reached only if no event fires.
        alpha0_deg, beta0_deg:
            Initial pitch and yaw of the projectile axis, in degrees.
        w_j0, w_k0:
            Initial transverse angular rates, in rad/s.
        rtol, atol, max_step, method:
            Override the corresponding fields of :attr:`settings` for this run.
        ground_level:
            Altitude of the terminating surface, in m.
        fuze:
            Optional proximity fuze.  When given, the integration also stops the
            first time the round comes within ``fuze.radius_m`` of
            ``fuze.target_center``, after ``fuze.arm_time_s``.
        extra_events:
            Further event callables appended after the built-in ones.
        verbose:
            Print the progress banner of the original engine.

        Returns
        -------
        Trajectory
            Carries ``stop_reason`` in ``{"ground", "fuze", "max_time"}``.

        Notes
        -----
        With ``fuze=None`` and default settings this reproduces the original
        single-file engine bit for bit; see
        ``tests/test_regression_vs_original.py``.
        """
        settings = self.settings
        rtol = settings.rtol if rtol is None else rtol
        atol = settings.atol if atol is None else atol
        max_step = settings.max_step if max_step is None else max_step
        method = settings.method if method is None else method

        if verbose:
            print("\n" + "=" * 80)
            print("INICIANDO SIMULAÇÃO")
            print("=" * 80)

        y0 = self.build_initial_conditions(alpha0_deg, beta0_deg, w_j0, w_k0)

        events: List[Callable] = [make_ground_event(ground_level)]
        fuze_index = None
        if fuze is not None:
            fuze_index = len(events)
            events.append(
                make_proximity_fuze_event(
                    target_center=fuze.target_center,
                    radius_m=fuze.radius_m,
                    arm_time_s=fuze.arm_time_s,
                )
            )
        if extra_events:
            events.extend(extra_events)

        if verbose:
            print("\nIntegrando trajetória...")

        sol = solve_ivp(
            self.equations,
            (0.0, max_time),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            events=events,
            max_step=max_step,
        )

        self.stop_reason = self._classify_stop(sol, fuze_index)

        if verbose:
            if sol.success:
                print("✓ Integração bem-sucedida!")
                print(f"  Tempo de voo: {sol.t[-1]:.2f} s")
                if fuze is not None:
                    print(f"  Motivo da parada: {self.stop_reason}")
            else:
                print(f"✗ Erro na integração: {sol.message}")

        self.result = Trajectory(
            sol,
            self.projectile,
            self.environment,
            stop_reason=self.stop_reason,
            muzzle_velocity=self.weapon.muzzle_velocity,
        )
        return self.result

    @staticmethod
    def _classify_stop(sol, fuze_index: Optional[int]) -> str:
        """Name the event that ended the integration."""
        reason = "max_time"
        if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
            reason = "ground"
        if fuze_index is not None and len(sol.t_events) > fuze_index:
            if len(sol.t_events[fuze_index]) > 0:
                reason = "fuze"
        return reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BallisticSimulator(projectile={self.projectile.name!r}, "
            f"weapon={self.weapon.name!r})"
        )


__all__ = ["BallisticSimulator", "IntegrationSettings", "THESIS_SETTINGS", "FuzeSpec"]
