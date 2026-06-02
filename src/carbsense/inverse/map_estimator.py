"""
MAP estimator for meal carbohydrates: gradient-based inversion of the
forward physiological model.

Theta = [log_D, t_meal]. We parameterize D in log-space to enforce positivity.
The optimization minimizes a reconstruction-error loss against observed CGM,
regularized by a broad log-normal prior on D.

This is the *physics-only baseline*. Any PINN we add later must clearly beat
it on identical splits.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from carbsense.physiology.uvapadova_min import (
    InsulinInput,
    MealInput,
    PatientParams,
    cgm_observation,
    simulate,
)


class MAPResult(NamedTuple):
    D_est: float
    t_meal_est: float
    final_loss: float
    n_iter: int
    converged: bool


def _make_loss(
    params: PatientParams,
    infusions: InsulinInput,
    t_eval: jax.Array,
    y_obs: jax.Array,
    G0: float,
    D_prior_mean: float = 50.0,
    D_prior_std: float = 30.0,
):
    """Build a JIT-compiled loss function loss(theta) where theta = [log_D, t_meal]."""

    def loss_fn(theta: jax.Array) -> jax.Array:
        log_D = theta[0]
        t_meal = theta[1]
        D = jnp.exp(log_D)

        # CRITICAL: use jnp.atleast_1d on tracer scalars, NOT jnp.array([scalar])
        # which silently disconnects the autodiff graph.
        meals = MealInput(
            t_meal=jnp.atleast_1d(t_meal),
            D=jnp.atleast_1d(D),
        )
        G_p = simulate(params, meals, infusions, t_eval, G0=G0)
        y_hat = cgm_observation(G_p, params, key=None)

        # Data fit: per-sample MSE weighted by inverse CGM noise variance
        data_term = jnp.mean((y_hat - y_obs) ** 2) / (params.sigma_cgm**2)

        # Log-normal prior on D
        log_prior_mean = jnp.log(jnp.asarray(D_prior_mean))
        log_prior_std = jnp.asarray(D_prior_std / D_prior_mean)
        prior_term = 0.5 * ((log_D - log_prior_mean) / log_prior_std) ** 2

        return data_term + 0.1 * prior_term

    return jax.jit(loss_fn)


def estimate_meal(
    y_obs: jax.Array,
    t_eval: jax.Array,
    params: PatientParams,
    infusions: InsulinInput,
    G0: float,
    D_init: float = 40.0,
    t_meal_init: float = 0.0,
    n_iter: int = 200,
    lr: float = 0.05,
    tol: float = 1e-5,
) -> MAPResult:
    """Run MAP optimization for a single meal event."""
    loss_fn = _make_loss(params, infusions, t_eval, y_obs, G0)
    grad_fn = jax.jit(jax.grad(loss_fn))

    theta = jnp.array([jnp.log(jnp.asarray(D_init)), jnp.asarray(t_meal_init)], dtype=jnp.float32)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)

    prev_loss = jnp.inf
    converged = False
    final_loss = jnp.inf
    n_done = 0

    for step in range(n_iter):
        g = grad_fn(theta)
        updates, opt_state = optimizer.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)
        final_loss = loss_fn(theta)
        n_done = step + 1

        if jnp.abs(prev_loss - final_loss) < tol:
            converged = True
            break
        prev_loss = final_loss

    return MAPResult(
        D_est=float(jnp.exp(theta[0])),
        t_meal_est=float(theta[1]),
        final_loss=float(final_loss),
        n_iter=n_done,
        converged=converged,
    )
