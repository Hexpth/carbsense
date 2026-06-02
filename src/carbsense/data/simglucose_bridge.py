"""
Bridge between SimGlucose (the FDA-validated UVA/Padova reference simulator)
and our CarbSense data structures.

We generate single-meal episodes using SimGlucose: a virtual patient sits at
basal for some time, eats a known meal of D grams, and we record the resulting
CGM trace. The MAP estimator then tries to recover D from the trace alone.

The resulting MAE on these traces is the first *honest* benchmark of our
inverse pipeline against an independent forward model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MealEpisode:
    """A single-meal scenario suitable for the CarbSense MAP estimator.

    Attributes
    ----------
    patient_name : str
        SimGlucose patient identifier (e.g. ``"adult#001"``).
    D_true : float
        True carbohydrate amount, grams.
    t_meal_true : float
        True meal onset, minutes from start of the recording.
    t_eval : jnp.ndarray
        Sample times, minutes. Uniform 5-min grid by convention.
    cgm_mgdl : jnp.ndarray
        Observed CGM trace at ``t_eval``, mg/dL.
    G0 : float
        First CGM value, mg/dL (used as initial condition for the inverse
        problem).
    """

    patient_name: str
    D_true: float
    t_meal_true: float
    t_eval: jnp.ndarray
    cgm_mgdl: jnp.ndarray
    G0: float


def generate_meal_episode(
    patient_name: str,
    D_true: float,
    t_meal_minutes: float = 30.0,
    duration_minutes: int = 180,
    seed: int = 0,
) -> MealEpisode:
    """Run a single-meal SimGlucose simulation and return a MealEpisode.

    Parameters
    ----------
    patient_name : str
        E.g. ``"adult#001"`` ... ``"adult#010"``, ``"adolescent#001"`` etc.
    D_true : float
        Carbs in grams.
    t_meal_minutes : float
        When the meal is consumed, minutes into the simulation.
    duration_minutes : int
        Total simulation length. Must be larger than ``t_meal_minutes``.
    seed : int
        Sensor / pump RNG seed for reproducibility.

    Notes
    -----
    The simulator delivers basal insulin automatically based on the patient's
    profile. No correction or meal bolus is given — this isolates the
    glucose-rise signal from insulin confounders, which is what we need for
    a clean benchmark of the inverse problem.
    """
    # Import locally so importing this module without simglucose installed
    # still works (e.g. CI without the [sim] extras).
    from datetime import datetime, timedelta

    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.base import Action, Controller
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import CustomScenario
    from simglucose.simulation.sim_engine import SimObj, sim

    start_time = datetime(2026, 1, 1, 0, 0, 0)
    meal_time = start_time + timedelta(minutes=t_meal_minutes)

    # Single meal of D_true grams at meal_time, no other events.
    scenario = CustomScenario(start_time=start_time, scenario=[(meal_time, D_true)])

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Trivial controller — only basal insulin, no correction or meal bolus.
    class BasalOnlyController(Controller):
        def policy(
            self,
            observation: Any,
            reward: Any,
            done: Any,
            **info: Any,
        ) -> Any:
            return Action(basal=0, bolus=0)

        def reset(self) -> None:
            pass

    controller = BasalOnlyController(0)

    end_time = start_time + timedelta(minutes=duration_minutes)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sim_obj = SimObj(
            env,
            controller,
            end_time - start_time,
            animate=False,
            path=tmp,
        )
        sim(sim_obj)
    results = sim_obj.results()

    # results is a pandas DataFrame indexed by datetime, with column "CGM".
    times_dt = results.index.to_pydatetime()
    minutes = np.array(
        [(t - start_time).total_seconds() / 60.0 for t in times_dt], dtype=np.float32
    )
    cgm = results["CGM"].to_numpy(dtype=np.float32)

    # Resample to uniform 5-min grid via nearest-neighbor (SimGlucose
    # natively samples every 3 minutes — we conform to CGM convention).
    grid = np.arange(0, duration_minutes + 1, 5, dtype=np.float32)
    idx = np.searchsorted(minutes, grid, side="left").clip(0, len(minutes) - 1)
    cgm_resampled = cgm[idx]

    return MealEpisode(
        patient_name=patient_name,
        D_true=float(D_true),
        t_meal_true=float(t_meal_minutes),
        t_eval=jnp.asarray(grid),
        cgm_mgdl=jnp.asarray(cgm_resampled),
        G0=float(cgm_resampled[0]),
    )


def basal_only_infusions() -> InsulinInput:
    """Return an empty ``InsulinInput`` (no extra bolus / temp basal).
    ...
    """
    from carbsense.physiology.uvapadova_min import InsulinInput

    return InsulinInput(
        t=jnp.array([0.0]),
        u=jnp.array([0.0]),
        duration=jnp.array([0.0]),
    )
