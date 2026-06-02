"""
Sanity tests for the forward simulator (Bergman minimal + meal compartments).

These verify QUALITATIVE behaviour, not quantitative agreement with SimGlucose
(that comes later, in test_against_simglucose.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carbsense.physiology.uvapadova_min import (
    InsulinInput,
    MealInput,
    PatientParams,
    cgm_observation,
    simulate,
)


@pytest.fixture
def params() -> PatientParams:
    return PatientParams()


@pytest.fixture
def t_eval() -> jax.Array:
    # 0..240 min, every 5 min (CGM sampling rate)
    return jnp.arange(0, 241, 5, dtype=jnp.float32)


@pytest.fixture
def no_meal() -> MealInput:
    return MealInput(t_meal=jnp.array([0.0]), D=jnp.array([0.0]))


@pytest.fixture
def no_extra_insulin() -> InsulinInput:
    # u=0 means no boluses above basal. Basal is supplied automatically
    # inside simulate() to keep the patient at steady state.
    return InsulinInput(
        t=jnp.array([0.0]),
        u=jnp.array([0.0]),
        duration=jnp.array([0.0]),
    )


def test_fasting_stable_under_basal(
    params: PatientParams,
    t_eval: jax.Array,
    no_meal: MealInput,
    no_extra_insulin: InsulinInput,
) -> None:
    """At fasting with steady-state basal, glucose drifts < 15 mg/dL."""
    G = simulate(params, no_meal, no_extra_insulin, t_eval, G0=float(params.Gb))
    assert jnp.all(jnp.isfinite(G))
    drift = float(jnp.max(jnp.abs(G - params.Gb)))
    assert drift < 15.0, f"Drift = {drift:.1f} mg/dL, expected < 15"


def test_meal_raises_glucose(
    params: PatientParams,
    t_eval: jax.Array,
    no_extra_insulin: InsulinInput,
) -> None:
    """A 50g meal at t=30 raises glucose by at least 20 mg/dL by t=90."""
    meal = MealInput(t_meal=jnp.array([30.0]), D=jnp.array([50.0]))
    G = simulate(params, meal, no_extra_insulin, t_eval, G0=120.0)

    idx_30 = int(jnp.argmin(jnp.abs(t_eval - 30)))
    idx_90 = int(jnp.argmin(jnp.abs(t_eval - 90)))
    rise = float(G[idx_90] - G[idx_30])
    assert rise > 20.0, f"Glucose should rise >20 mg/dL, got {rise:.1f}"


def test_insulin_lowers_glucose(
    params: PatientParams,
    t_eval: jax.Array,
    no_meal: MealInput,
) -> None:
    """A 5U bolus on hyperglycaemia (G=200) lowers glucose by t=180."""
    # 5U = 30000 pmol
    bolus = InsulinInput(
        t=jnp.array([30.0]),
        u=jnp.array([30000.0]),
        duration=jnp.array([0.0]),
    )
    G = simulate(params, no_meal, bolus, t_eval, G0=200.0)

    idx_30 = int(jnp.argmin(jnp.abs(t_eval - 30)))
    idx_180 = int(jnp.argmin(jnp.abs(t_eval - 180)))
    drop = float(G[idx_30] - G[idx_180])
    assert drop > 10.0, f"Glucose should drop >10 mg/dL after 5U bolus, got {drop:.1f}"


def test_gradient_wrt_meal_size_is_positive(
    params: PatientParams,
    t_eval: jax.Array,
    no_extra_insulin: InsulinInput,
) -> None:
    """KEY INVARIANT for inverse problem: dG_p(90)/dD > 0.

    Uses jnp.atleast_1d to keep the scalar-tracer graph connected (jnp.array
    of a list containing a tracer silently breaks autodiff).
    """

    def G_at_90(D_val: jax.Array) -> jax.Array:
        meal = MealInput(
            t_meal=jnp.array([30.0]),
            D=jnp.atleast_1d(D_val),
        )
        G = simulate(params, meal, no_extra_insulin, t_eval, G0=120.0)
        idx_90 = int(jnp.argmin(jnp.abs(t_eval - 90)))
        return G[idx_90]

    grad = float(jax.grad(G_at_90)(jnp.asarray(50.0)))
    assert grad > 0.0, f"dG/dD should be > 0, got {grad}"
    # Sanity bounds: 1g of carbs raises glucose by ~0.1..5 mg/dL eventually.
    assert 0.01 < grad < 10.0, f"dG/dD = {grad}, outside plausible range"


def test_cgm_lag_smooths_signal(
    params: PatientParams,
    t_eval: jax.Array,
    no_extra_insulin: InsulinInput,
) -> None:
    """CGM observation peaks no earlier than plasma glucose."""
    meal = MealInput(t_meal=jnp.array([30.0]), D=jnp.array([60.0]))
    G_p = simulate(params, meal, no_extra_insulin, t_eval, G0=120.0)
    G_cgm = cgm_observation(G_p, params, key=None)

    peak_plasma = int(jnp.argmax(G_p))
    peak_cgm = int(jnp.argmax(G_cgm))
    assert peak_cgm >= peak_plasma, (
        f"CGM peak (idx {peak_cgm}) should lag plasma peak (idx {peak_plasma})"
    )
