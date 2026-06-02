# Research Gap Statement

**Draft (review and sharpen monthly):**

> While Physics-Informed Neural Networks (PINNs) have recently been applied to
> glucose-insulin dynamics for *forward* problems (predicting future glucose
> levels [De Carli et al. 2025]) and for *parameter estimation* of physiological
> models (e.g., insulin sensitivity from IVGTT data [Multerer et al. 2025;
> Evidential PINN 2025]), no published work applies the PINN framework to the
> **inverse problem of quantifying ingested carbohydrates** from postprandial
> CGM signals under free-living conditions. Existing meal-size estimation
> methods rely either on classical Kalman-filter / fuzzy-logic approaches
> [Dassau 2008; Turksoy 2017; Mahmoudi 2019], which lack flexibility to adapt
> to individual physiology, or on purely data-driven deep learning
> [Daniels 2022; Zheng 2022], which lack mechanistic interpretability and
> generalize poorly to unseen patients.
>
> We propose **CarbSense**, a physics-informed inverse modeling framework
> grounded in the UVA/Padova S2017 simulator, that combines (i) a differentiable
> ODE-based forward operator, (ii) a Transformer-based amortized posterior
> over meal parameters (C, τ), and (iii) an evidential uncertainty layer.
> The framework is designed to provide calibrated point estimates and
> credible intervals of ingested carbohydrates within 30 minutes of meal
> onset, evaluated on both in-silico UVA/Padova cohorts and real-world
> CGM data from the OhioT1DM and OpenAPS Data Commons datasets.

## Three claims of novelty (must defend each in paper)

### Claim 1: Methodological
First PINN-based **inverse** solver for meal-carbohydrate quantification.
Existing PINN-in-glucose work is forward-prediction or parameter ID, never
inverse problem on the meal disturbance term.

**Defense:** systematic literature search (Scopus + WoS + arXiv + bioRxiv,
done at M1) yields 0 prior works combining all three keywords.

### Claim 2: Empirical
First method to report **simultaneous** improvement on three orthogonal axes:
(a) MAE on carb amount, (b) detection latency, (c) FPR under free-living noise.
Prior work optimizes one at the expense of others.

**Defense:** head-to-head benchmark on identical splits of OhioT1DM and
OpenAPS Data Commons against published baselines (Mahmoudi 2019, Turksoy
2017, Daniels 2022 reimplemented).

### Claim 3: Practical
First open-source implementation deployable on consumer mobile hardware
(<50 MB ONNX model, <100 ms inference on mid-range Android).

**Defense:** publish benchmark + reproducible Docker + Android demo APK.

## Risk to gap (mitigation)

| Risk | Mitigation |
|---|---|
| Cambridge/Hovorka group publishes their meal-detection algorithm during 18 months | Maintain methodological distinction (their approach is MPC-based, ours is PINN). Reposition as comparative study. |
| Multerer/De Carli group extends their PINN to inverse problem | Reach out for collaboration at M3 rather than compete. Joint publication is acceptable outcome. |
| Commercial closed-source release (Beta Bionics, Tandem) | Doesn't affect academic novelty (closed-source ≠ published). |
