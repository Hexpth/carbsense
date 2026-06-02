"""
Minimal glucose-insulin model for the meal-carb inverse problem.

Design philosophy: smallest model that
(a) is stable at fasting under basal insulin (T1D context),
(b) responds plausibly to meals and boluses,
(c) is differentiable end-to-end (works under jax.grad and jax.jit),
(d) inverts in ~10% MAE on its own synthetic data.

Structure: Bergman minimal model (Bergman 1979) + 2-compartment gut
absorption (Dalla Man 2007, simplified). 5 ODE states.

This is NOT calibrated against SimGlucose at this stage — that comes
after we verify the inverse-problem pipeline end-to-end.

Refs:
- Bergman RN et al. (1979) Am J Physiol 236:E667.
- Dalla Man C et al. (2007) IEEE TBME 54:1740.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

# --------------------------------------------------------------------------- #
# Parameter containers
# --------------------------------------------------------------------------- #


class PatientParams(NamedTuple):
    """Per-patient physiological parameters (T1D adult, ~70 kg)."""

    # Body
    BW: jax.Array = jnp.asarray(70.0)  # kg

    # Glucose subsystem (Bergman minimal)
    Gb: jax.Array = jnp.asarray(120.0)  # mg/dL, basal/target glucose
    p1: jax.Array = jnp.asarray(0.025)  # 1/min, glucose effectiveness
    SI: jax.Array = jnp.asarray(0.0005)  # 1/min per pmol/L, insulin sensitivity

    # Remote insulin action
    p2: jax.Array = jnp.asarray(0.025)  # 1/min
    p3: jax.Array = jnp.asarray(1.3e-5)  # min^-2 per pmol/L

    # Insulin pharmacokinetics
    Ib: jax.Array = jnp.asarray(60.0)  # pmol/L, basal plasma insulin
    V_i: jax.Array = jnp.asarray(3.5)  # L, insulin distribution volume
    k_a: jax.Array = jnp.asarray(0.018)  # 1/min, sc -> plasma rate

    # Meal absorption (2-compartment cascade)
    k_meal: jax.Array = jnp.asarray(0.05)  # 1/min, transit rate
    f_carb: jax.Array = jnp.asarray(0.9)  # bioavailability
    V_g: jax.Array = jnp.asarray(1.88)  # dL/kg, glucose distribution volume

    # CGM measurement model
    tau_cgm: jax.Array = jnp.asarray(7.0)  # min, sensor lag
    sigma_cgm: jax.Array = jnp.asarray(2.0)  # mg/dL, measurement noise std


class MealInput(NamedTuple):
    """Meal events. Both fields may be scalars or 1-D arrays."""

    t_meal: jax.Array  # min, meal onset
    D: jax.Array  # grams of carbs


class InsulinInput(NamedTuple):
    """Insulin boluses / square-wave infusions (above basal).

    Fields may be scalars or 1-D arrays. duration=0 means instantaneous bolus.
    """

    t: jax.Array  # min
    u: jax.Array  # pmol/min equivalent rate
    duration: jax.Array  # min, 0 for boluses


# --------------------------------------------------------------------------- #
# Forcing functions (smooth approximations of impulses)
# --------------------------------------------------------------------------- #


def _smooth_pulse(t: jax.Array, t0: jax.Array, width: float = 1.0) -> jax.Array:
    """Gaussian bump of unit area centered at t0. Smooth -> autodiff-friendly."""
    return jnp.exp(-0.5 * ((t - t0) / width) ** 2) / (width * jnp.sqrt(2 * jnp.pi))


def carb_input_rate(t: jax.Array, meals: MealInput) -> jax.Array:
    """Returns g/min of carbs entering Q1 at time t."""
    D = jnp.atleast_1d(meals.D)
    t_meal = jnp.atleast_1d(meals.t_meal)
    return jnp.sum(D * _smooth_pulse(t, t_meal, width=1.0))


def insulin_input_rate(t: jax.Array, infusions: InsulinInput) -> jax.Array:
    """Returns pmol/min of insulin entering the sc depot (above basal)."""
    u = jnp.atleast_1d(infusions.u)
    t_inf = jnp.atleast_1d(infusions.t)
    dur = jnp.atleast_1d(infusions.duration)

    bolus_mask = dur < 0.5
    bolus_rate = u * _smooth_pulse(t, t_inf, width=1.0)
    # Smooth rectangular pulse for square-wave / temp-basal
    basal_rate = u * jax.nn.sigmoid(2.0 * (t - t_inf)) * jax.nn.sigmoid(2.0 * (t_inf + dur - t))
    rate = jnp.where(bolus_mask, bolus_rate, basal_rate)
    return jnp.sum(rate)


# --------------------------------------------------------------------------- #
# ODE right-hand side
# --------------------------------------------------------------------------- #


def vector_field(
    t: jax.Array,
    y: jax.Array,
    args: tuple[PatientParams, MealInput, InsulinInput, jax.Array],
) -> jax.Array:
    """RHS of the 5-state system.

    States:
        y[0] = Q1     (g)        carbs in gut compartment 1
        y[1] = Q2     (g)        carbs in gut compartment 2
        y[2] = G      (mg/dL)    plasma glucose
        y[3] = X      (1/min)    remote insulin action
        y[4] = I_sc   (pmol)     subcutaneous insulin depot
    """
    p, meals, infusions, u_basal = args
    Q1, Q2, G, X, I_sc = y

    D_in = carb_input_rate(t, meals)  # g/min
    U_in = insulin_input_rate(t, infusions)  # pmol/min (above basal)

    # Gut absorption cascade
    dQ1 = D_in - p.k_meal * Q1  # g/min
    dQ2 = p.k_meal * Q1 - p.k_meal * Q2  # g/min

    # Glucose appearance in plasma (mg/dL/min)
    # k_meal*Q2 [g/min] * 1000 [mg/g] / (V_g [dL/kg] * BW [kg])
    Ra = p.f_carb * p.k_meal * Q2 * 1000.0 / (p.V_g * p.BW)

    # Plasma insulin concentration
    I_p = I_sc / p.V_i  # pmol/L

    # Glucose dynamics (Bergman minimal)
    # Term -p1*(G-Gb) lumps basal EGP suppression at steady state.
    dG = -p.p1 * (G - p.Gb) - p.SI * X * G + Ra

    # Remote insulin action
    dX = -p.p2 * X + p.p3 * (I_p - p.Ib)

    # Subcutaneous insulin depot (basal + extra infusion)
    dI_sc = -p.k_a * I_sc + U_in + u_basal

    return jnp.array([dQ1, dQ2, dG, dX, dI_sc])


# --------------------------------------------------------------------------- #
# Forward simulator
# --------------------------------------------------------------------------- #


def simulate(
    params: PatientParams,
    meals: MealInput,
    infusions: InsulinInput,
    t_eval: jax.Array,
    G0: float | jax.Array | None = None,
) -> jax.Array:
    """Forward-simulate plasma glucose over t_eval.

    Returns G_p (mg/dL), shape == t_eval.shape.
    """
    if G0 is None:
        G0 = params.Gb

    # Steady-state basal insulin: I_sc_ss = Ib * V_i, so u_basal = k_a * I_sc_ss
    I_sc_0 = params.Ib * params.V_i
    u_basal = params.k_a * I_sc_0

    y0 = jnp.array(
        [
            0.0,  # Q1
            0.0,  # Q2
            jnp.asarray(G0, jnp.float32),  # G
            0.0,  # X
            I_sc_0,  # I_sc
        ]
    )

    term = ODETerm(vector_field)
    solver = Tsit5()
    saveat = SaveAt(ts=t_eval)
    # dtmax=1.0 ensures the solver doesn't step over our 1-min wide pulses,
    # which would silently zero out meals/boluses in autodiff.
    stepsize_controller = PIDController(rtol=1e-4, atol=1e-4, dtmax=1.0)

    sol = diffeqsolve(
        term,
        solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=0.5,
        y0=y0,
        args=(params, meals, infusions, u_basal),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=20_000,
    )

    return sol.ys[:, 2]  # G column


def cgm_observation(
    G_p: jax.Array,
    params: PatientParams,
    key: jax.Array | None = None,
) -> jax.Array:
    """First-order lag + optional Gaussian noise. Assumes 5-min sampling."""
    alpha = jnp.exp(-5.0 / params.tau_cgm)

    def step(prev: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        new = alpha * prev + (1.0 - alpha) * x
        return new, new

    _, G_m = jax.lax.scan(step, G_p[0], G_p)

    if key is not None:
        G_m = G_m + params.sigma_cgm * jax.random.normal(key, G_m.shape)

    return G_m
