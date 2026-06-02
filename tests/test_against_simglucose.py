"""
Light integration test: one SimGlucose episode end-to-end through the MAP
estimator.

This test only checks that the pipeline *runs* and produces a finite,
plausible estimate. It does NOT enforce a tight error bound — the actual
benchmark numbers come from ``scripts/run_simglucose_benchmark.py``, which
is slow (~5-10 s per episode) and intentionally not in pytest.

Skipped automatically if SimGlucose is not installed.
"""

from __future__ import annotations

import math

import pytest

simglucose = pytest.importorskip("simglucose")  # noqa: F841

from carbsense.data.simglucose_bridge import (  # noqa: E402
    basal_only_infusions,
    generate_meal_episode,
)
from carbsense.inverse.map_estimator import estimate_meal  # noqa: E402
from carbsense.physiology.uvapadova_min import PatientParams  # noqa: E402


@pytest.mark.slow
def test_one_simglucose_episode_runs_end_to_end() -> None:
    """Single adult patient, 50 g meal: estimate must be finite and plausible."""
    D_true = 50.0
    episode = generate_meal_episode(
        patient_name="adult#001",
        D_true=D_true,
        t_meal_minutes=30.0,
        duration_minutes=180,
        seed=0,
    )

    assert episode.cgm_mgdl.shape == (37,)  # 0..180 step 5
    assert 60.0 < episode.G0 < 250.0, f"Implausible initial CGM: {episode.G0}"

    result = estimate_meal(
        y_obs=episode.cgm_mgdl,
        t_eval=episode.t_eval,
        params=PatientParams(),
        infusions=basal_only_infusions(),
        G0=episode.G0,
        D_init=40.0,
        t_meal_init=20.0,
        n_iter=300,
        lr=0.1,
    )

    assert math.isfinite(result.D_est)
    assert math.isfinite(result.final_loss)
    # Very loose plausibility: estimate is somewhere in the right ballpark.
    # Tight MAE bounds belong in the benchmark script, not in pytest.
    assert 5.0 < result.D_est < 250.0, f"Implausible D_est: {result.D_est}"
