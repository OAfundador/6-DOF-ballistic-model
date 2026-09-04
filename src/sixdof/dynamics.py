"""Six-degree-of-freedom equations of motion and their initial conditions.

The formulation follows McCoy, *Modern Exterior Ballistics* (2nd ed.), chapter
on the six-degree-of-freedom rigid-body model.  The state carried by the
integrator is

.. code-block:: text

    y = [V1, V2, V3, h1, h2, h3, i1, i2, i3, x, y, z]

where ``V`` is the velocity in world axes, ``i'`` is the unit vector along the
projectile's axis of symmetry, ``h`` is the (mass-normalised) angular momentum
used by McCoy, and ``(x, y, z)`` is the position.  Axes are ``x`` downrange,
``y`` up and ``z`` to the right.

The axial spin follows from the state as ``omega1 = (I_T / I_P) (h . i')``,
which is why ``h`` rather than the body rates is integrated: the orientation
equation ``di'/dt = h x i'`` is then linear and keeps ``|i'|`` bounded without
a quaternion renormalisation step.

Every expression in :func:`six_dof_rhs` is written exactly as in the original
single-file engine.  Re-associating the floating-point operations would perturb
the trajectory in the last digits, and the regression test in
``tests/test_regression_vs_original.py`` compares the two engines bit for bit.
"""

from __future__ import annotations

from math import acos, cos, sin, sqrt
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .environment import Environment
    from .projectile import Projectile
    from .weapon import Weapon

#: Index of each block inside the state vector.
IDX_VELOCITY = slice(0, 3)
IDX_ANGULAR_MOMENTUM = slice(3, 6)
IDX_AXIS = slice(6, 9)
IDX_POSITION = slice(9, 12)

#: Length of the state vector.
STATE_SIZE = 12

#: Names of the state components, in order.
STATE_NAMES: Tuple[str, ...] = (
    "V1",
    "V2",
    "V3",
    "h1",
    "h2",
    "h3",
    "i1",
    "i2",
    "i3",
    "x",
    "y",
    "z",
)


def build_initial_state(
    projectile: "Projectile",
    weapon: "Weapon",
    alpha0_deg: float = 0.0,
    beta0_deg: float = 0.0,
    w_j0: float = 5.0,
    w_k0: float = 5.0,
) -> np.ndarray:
    """Assemble the state vector at muzzle exit.

    Parameters
    ----------
    projectile, weapon:
        Provide the inertia ratio, the muzzle velocity, the laying angles and
        the launch point.
    alpha0_deg:
        Initial pitch of the projectile axis relative to the line of fire, in
        degrees (an initial yaw-of-repose / tip-off angle).
    beta0_deg:
        Initial yaw of the projectile axis, in degrees.
    w_j0, w_k0:
        Initial transverse angular rates about the ``j'`` and ``k'`` axes, in
        rad/s.  These seed the epicyclic (nutation/precession) motion.

    Returns
    -------
    numpy.ndarray
        The 12-component state vector.

    Notes
    -----
    When the gun rides on a moving platform, the platform velocity is added to
    the muzzle velocity, so the state is always expressed in world axes.
    """
    theta0, phi0 = weapon.calculate_firing_angles()
    alpha0 = np.radians(alpha0_deg)
    beta0 = np.radians(beta0_deg)

    # Muzzle velocity resolved along the line of fire.
    V0 = weapon.muzzle_velocity
    V1_rel = V0 * cos(theta0) * cos(phi0)
    V2_rel = V0 * cos(theta0) * sin(phi0)
    V3_rel = V0 * sin(theta0)

    # Absolute velocity = relative + platform.
    platform_velocity = weapon.get_velocity()
    V1_0 = V1_rel + platform_velocity[0]
    V2_0 = V2_rel + platform_velocity[1]
    V3_0 = V3_rel + platform_velocity[2]

    w_i0 = projectile.calculate_initial_spin(V0)

    # Axis of symmetry i', offset from the line of fire by (alpha0, beta0).
    phi_eff = phi0 + alpha0
    theta_eff = theta0 + beta0

    i1_0 = cos(phi_eff) * cos(theta_eff)
    i2_0 = cos(theta_eff) * sin(phi_eff)
    i3_0 = sin(theta_eff)

    # Companion axes j' and k' completing the body triad.
    Q = sin(theta_eff) ** 2 + cos(theta_eff) ** 2 * cos(phi_eff) ** 2
    sqrt_Q = sqrt(Q)

    # di'/dt implied by the initial transverse rates.
    di1_dt = (
        w_j0 * sin(theta_eff) - w_k0 * cos(theta_eff) ** 2 * sin(phi_eff) * cos(phi_eff)
    ) / sqrt_Q

    di2_dt = (w_k0 / sqrt_Q) * (cos(theta_eff) ** 2 * cos(phi_eff) ** 2 + sin(theta_eff) ** 2)

    di3_dt = (
        -w_j0 * cos(theta_eff) * cos(phi_eff)
        - w_k0 * sin(phi_eff) * cos(theta_eff) * sin(theta_eff)
    ) / sqrt_Q

    # h = (I_P / I_T) omega1 i' + i' x di'/dt
    omega1_inertial = w_i0
    I_P = projectile.I_P
    I_T = projectile.I_T

    term1_h1 = (I_P / I_T) * omega1_inertial * i1_0
    term1_h2 = (I_P / I_T) * omega1_inertial * i2_0
    term1_h3 = (I_P / I_T) * omega1_inertial * i3_0

    term2_h1 = i2_0 * di3_dt - i3_0 * di2_dt
    term2_h2 = i3_0 * di1_dt - i1_0 * di3_dt
    term2_h3 = i1_0 * di2_dt - i2_0 * di1_dt

    h1_0 = term1_h1 + term2_h1
    h2_0 = term1_h2 + term2_h2
    h3_0 = term1_h3 + term2_h3

    x0, y0, z0 = weapon.get_absolute_position()

    return np.array(
        [V1_0, V2_0, V3_0, h1_0, h2_0, h3_0, i1_0, i2_0, i3_0, x0, y0, z0],
        dtype=float,
    )


def six_dof_rhs(
    t: float,
    y: np.ndarray,
    projectile: "Projectile",
    environment: "Environment",
    aero_coeffs,
) -> np.ndarray:
    """Time derivative of the 12-component state.

    Parameters
    ----------
    t:
        Time in s.  The right-hand side is autonomous; the argument exists for
        the ``solve_ivp`` signature.
    y:
        Current state vector.
    projectile, environment:
        Physical parameters.
    aero_coeffs:
        Anything exposing ``get_coefficients(mach, alpha_rad)``.

    Returns
    -------
    numpy.ndarray
        ``dy/dt``.

    Notes
    -----
    Forces included: drag, lift, Magnus and pitch damping.  Moments included:
    spin damping, overturning (static) moment, Magnus moment and pitch damping.
    The control terms ``C_l_delta`` and ``delta_F`` are present but zeroed, as
    are ``C_Nq``, ``C_Nalpha_dot`` and ``C_Malpha_dot``, whose tabulated values
    are not available for this projectile.
    """
    V1, V2, V3, h1, h2, h3, i1, i2, i3, x, ypos, z = y

    # Air-relative velocity.
    v1 = V1 - environment.W1
    v2 = V2 - environment.W2
    v3 = V3 - environment.W3
    v = sqrt(v1 * v1 + v2 * v2 + v3 * v3)

    mach = v / environment.sound_speed

    # Total angle of attack between the velocity and the axis of symmetry.
    cos_alpha_t = (v1 * i1 + v2 * i2 + v3 * i3) / v
    cos_alpha_t = np.clip(cos_alpha_t, -1.0, 1.0)
    alpha_rad = acos(cos_alpha_t)

    coeffs = aero_coeffs.get_coefficients(mach, alpha_rad)

    C_D = coeffs["CD_total"]  # drag force
    C_Lalpha = coeffs["CLA_total"]  # lift force
    C_Npalpha = coeffs["CYP"]  # Magnus force
    C_Nq = 0  # pitch damping force
    C_Nalpha_dot = 0  # pitch damping force (second component)
    C_lp = coeffs["CLP"]  # spin damping moment
    C_Malpha = coeffs["CMA"]  # overturning moment
    C_Mpalpha = coeffs["CNP_total"]  # Magnus moment
    C_Mq = coeffs["CMQ"]  # pitch damping moment
    C_Malpha_dot = 0

    C_l_delta = 0.0
    delta_F = 0.0

    m = projectile.mass
    S = projectile.S
    d = projectile.diameter
    I_P = projectile.I_P
    I_T = projectile.I_T
    rho = environment.rho
    g = environment.g

    # Axial spin recovered from the angular momentum: omega1 = (I_T/I_P)(h . i')
    h_dot_i = h1 * i1 + h2 * i2 + h3 * i3
    omega1 = (I_T / I_P) * h_dot_i

    # ---- force equations, dV/dt -------------------------------------------
    dV1 = (
        -(rho * v * S * C_D) / (2 * m) * v1
        + (rho * S * C_Lalpha) / (2 * m) * ((v * v) * i1 - v * v1 * cos_alpha_t)
        - (rho * S * d * C_Npalpha * omega1) / (2 * m) * (v3 * i2 - v2 * i3)
        + (rho * v * S * d * (C_Nq + C_Nalpha_dot)) / (2 * m) * (h2 * i3 - h3 * i2)
    )
    dV2 = (
        -(rho * v * S * C_D) / (2 * m) * v2
        + (rho * S * C_Lalpha) / (2 * m) * ((v * v) * i2 - v * v2 * cos_alpha_t)
        - (rho * S * d * C_Npalpha * omega1) / (2 * m) * (v1 * i3 - v3 * i1)
        + (rho * v * S * d * (C_Nq + C_Nalpha_dot)) / (2 * m) * (h3 * i1 - h1 * i3)
        - g
    )
    dV3 = (
        -(rho * v * S * C_D) / (2 * m) * v3
        + (rho * S * C_Lalpha) / (2 * m) * ((v * v) * i3 - v * v3 * cos_alpha_t)
        - (rho * S * d * C_Npalpha * omega1) / (2 * m) * (v2 * i1 - v1 * i2)
        + (rho * v * S * d * (C_Nq + C_Nalpha_dot)) / (2 * m) * (h1 * i2 - h2 * i1)
    )

    # ---- moment equations, dh/dt ------------------------------------------
    dh1 = (
        (rho * v * S * d**2 * C_lp * omega1) / (2 * I_T) * i1
        + (rho * v**2 * S * d * delta_F * C_l_delta) / (2 * I_T) * i1
        + (rho * v * S * d * C_Malpha) / (2 * I_T) * (v2 * i3 - v3 * i2)
        + (rho * S * d**2 * C_Mpalpha * omega1) / (2 * I_T) * (v1 - v * i1 * cos_alpha_t)
        + (rho * v * S * d**2 * (C_Mq + C_Malpha_dot))
        / (2 * I_T)
        * (h1 - ((I_P / I_T) * omega1) * i1)
    )
    dh2 = (
        (rho * v * S * d**2 * C_lp * omega1) / (2 * I_T) * i2
        + (rho * v**2 * S * d * delta_F * C_l_delta) / (2 * I_T) * i2
        + (rho * v * S * d * C_Malpha) / (2 * I_T) * (v3 * i1 - v1 * i3)
        + (rho * S * d**2 * C_Mpalpha * omega1) / (2 * I_T) * (v2 - v * i2 * cos_alpha_t)
        + (rho * v * S * d**2 * (C_Mq + C_Malpha_dot))
        / (2 * I_T)
        * (h2 - ((I_P / I_T) * omega1) * i2)
    )
    dh3 = (
        (rho * v * S * d**2 * C_lp * omega1) / (2 * I_T) * i3
        + (rho * v**2 * S * d * delta_F * C_l_delta) / (2 * I_T) * i3
        + (rho * v * S * d * C_Malpha) / (2 * I_T) * (v1 * i2 - v2 * i1)
        + (rho * S * d**2 * C_Mpalpha * omega1) / (2 * I_T) * (v3 - v * i3 * cos_alpha_t)
        + (rho * v * S * d**2 * (C_Mq + C_Malpha_dot))
        / (2 * I_T)
        * (h3 - ((I_P / I_T) * omega1) * i3)
    )

    # ---- orientation, di'/dt = h x i' -------------------------------------
    di1 = h2 * i3 - h3 * i2
    di2 = h3 * i1 - h1 * i3
    di3 = h1 * i2 - h2 * i1

    # ---- kinematics -------------------------------------------------------
    dx, dy, dz = V1, V2, V3

    return np.array(
        [dV1, dV2, dV3, dh1, dh2, dh3, di1, di2, di3, dx, dy, dz],
        dtype=float,
    )


class SixDofEquations:
    """Callable wrapper binding the physical parameters to :func:`six_dof_rhs`.

    Instances are what gets handed to ``scipy.integrate.solve_ivp``.  Keeping
    the parameters on the instance rather than in a closure makes them
    inspectable, which helps when a run has to be reconstructed from a saved
    configuration.
    """

    def __init__(self, projectile: "Projectile", environment: "Environment", aero_coeffs) -> None:
        self.projectile = projectile
        self.environment = environment
        self.aero_coeffs = aero_coeffs

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return six_dof_rhs(t, y, self.projectile, self.environment, self.aero_coeffs)

    def angle_of_attack(self, y: np.ndarray) -> float:
        """Total angle of attack, in radians, for a given state."""
        V1, V2, V3 = y[IDX_VELOCITY]
        i1, i2, i3 = y[IDX_AXIS]
        v1 = V1 - self.environment.W1
        v2 = V2 - self.environment.W2
        v3 = V3 - self.environment.W3
        v = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
        cos_alpha_t = np.clip((v1 * i1 + v2 * i2 + v3 * i3) / v, -1.0, 1.0)
        return float(acos(cos_alpha_t))

    def axial_spin(self, y: np.ndarray) -> float:
        """Axial spin ``omega1`` in rad/s for a given state."""
        h1, h2, h3 = y[IDX_ANGULAR_MOMENTUM]
        i1, i2, i3 = y[IDX_AXIS]
        h_dot_i = h1 * i1 + h2 * i2 + h3 * i3
        return float((self.projectile.I_T / self.projectile.I_P) * h_dot_i)


__all__ = [
    "SixDofEquations",
    "six_dof_rhs",
    "build_initial_state",
    "STATE_NAMES",
    "STATE_SIZE",
    "IDX_VELOCITY",
    "IDX_ANGULAR_MOMENTUM",
    "IDX_AXIS",
    "IDX_POSITION",
]
