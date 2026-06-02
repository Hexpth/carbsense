"""
Inverse-problem sanity tests: synthetic-data-in, parameter-out.

If the estimator can't recover a meal size from its OWN forward-model output,
nothing downstream will work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carbsense.inverse.map_estimator import estimate_meal
from carbsense.physiology.uvapadova_min import (
    InsulinInput,
    MealInput,
    PatientParams,
    cgm_observation,
    simulate,
)


@pytest.fixture
def setup() -> tuple[PatientParams, InsulinInput, jax.Array]:
    params = PatientParams()
    no_extra_insulin = InsulinInput(
        t=jnp.array([0.0]),
        u=jnp.array([0.0]),
        duration=jnp.array([0.0]),
    )
    t_eval = jnp.arange(0, 181, 5, dtype=jnp.float32)
    return params, no_extra_insulin, t_eval


@pytest.mark.parametrize("D_true", [30.0, 50.0, 80.0, 120.0])
def test_recovers_carb_amount_from_synthetic(
    setup: tuple[PatientParams, InsulinInput, jax.Array],
    D_true: float,
) -> None:
    """Noise-free synthetic data: estimator must recover D within 20%."""
    params, no_extra_insulin, t_eval = setup
    t_meal_true = 30.0

    meal_true = MealInput(t_meal=jnp.array([t_meal_true]), D=jnp.array([D_true]))
    G_p = simulate(params, meal_true, no_extra_insulin, t_eval, G0=120.0)
    y_obs = cgm_observation(G_p, params, key=None)

    result = estimate_meal(
        y_obs=y_obs,
        t_eval=t_eval,
        params=params,
        infusions=no_extra_insulin,
        G0=120.0,
        D_init=40.0,
        t_meal_init=20.0,
        n_iter=300,
        lr=0.1,
    )

    rel_err = abs(result.D_est - D_true) / D_true
    assert rel_err < 0.20, (
        f"D_true={D_true:.1f}, D_est={result.D_est:.1f}, "
        f"rel_err={rel_err:.2%}, loss={result.final_loss:.4f}"
    )


def test_recovers_with_realistic_noise(
    setup: tuple[PatientParams, InsulinInput, jax.Array],
) -> None:
    """With sigma=2 mg/dL Gaussian CGM noise, recovery within 30%."""
    params, no_extra_insulin, t_eval = setup
    D_true = 60.0

    meal_true = MealInput(t_meal=jnp.array([30.0]), D=jnp.array([D_true]))
    G_p = simulate(params, meal_true, no_extra_insulin, t_eval, G0=120.0)

    key = jax.random.PRNGKey(0)
    y_obs = cgm_observation(G_p, params, key=key)

    result = estimate_meal(
        y_obs=y_obs,
        t_eval=t_eval,
        params=params,
        infusions=no_extra_insulin,
        G0=120.0,
        D_init=40.0,
        t_meal_init=20.0,
        n_iter=300,
    )

    rel_err = abs(result.D_est - D_true) / D_true
    assert rel_err < 0.30, (
        f"With noise: D_true={D_true:.1f}, D_est={result.D_est:.1f}, rel_err={rel_err:.2%}"
    )
