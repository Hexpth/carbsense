# Literature Review: Carbohydrate Estimation & Meal Detection from CGM

**Статус:** seed (12 ключевых статей). Целевой объём: 40–60 статей к концу M1.

## Метод работы

Каждая статья читается **активно** с фиксацией в этой таблице. После прочтения — BibTeX-запись в `references.bib` и одна параграф-аннотация в стиле "TL;DR + что украсть + где слабость".

### Шаблон записи

```
### Author Year — Short title

**Citation:** Author A, Author B (Year). Full Title. *Journal*; vol(issue):pages.
**DOI:** 10.xxxx/yyyy
**Code:** github.com/... (или "not available")
**Dataset:** OpenAPS / UVA-Padova in silico / OhioT1DM / proprietary
**Method:** одна фраза.
**Metric:** MAE = X g, sensitivity = Y%, FPR = Z%, latency = N min.
**TL;DR:** 2–3 предложения.
**Strengths:** ...
**Weaknesses:** ...
**Что украсть:** ...
```

---

## Ключевая литература (по приоритету чтения)

### Tier 1: Прямо релевантные — meal detection / carb estimation

#### Dassau 2008 — Detection of a Meal Using CGM

**Citation:** Dassau E, Bequette BW, Buckingham BA, Doyle FJ III (2008). Detection of a Meal Using Continuous Glucose Monitoring. *Diabetes Care*; 31(2):295–300.
**DOI:** 10.2337/dc07-1293
**Method:** Voting ensemble из 4 классических детекторов: backward difference, Kalman filter, BD+Kalman, second derivative.
**Metric:** detection time 29–35 min, >90% meals detected before glucose rises 40 mg/dL.
**TL;DR:** Первая seminal работа. Сейчас — baseline. Чисто детекция, без оценки carbs.
**Что украсть:** voting scheme для robustness.

#### Mahmoudi 2019 — Sensor-based detection and estimation of meal carbohydrates

**Citation:** Mahmoudi Z, Cameron F, Poulsen NK, Madsen H, Bequette BW, Jørgensen JB (2019). Sensor-based detection and estimation of meal carbohydrates in people with diabetes. *Biomedical Signal Processing and Control*; 48:12–25.
**DOI:** 10.1016/j.bspc.2018.09.012
**Method:** Augmented state-space с white-noise double integrator для meal disturbance + Kalman filter + CUSUM + smoother.
**Metric:** 9 virtual T1D, sensitivity 93%, median detection delay 40 min, MAE ±20 g. TIR с автоматическим bolus: 50% → 79%.
**TL;DR:** Сильнейший классический baseline. Без ML, чистая control theory. **Главный конкурент**, относительно которого мы должны показать улучшение.
**Что украсть:** структуру augmented state-space, CUSUM-criterion.
**Слабость:** только in silico (9 пациентов), без real-world данных, без персонализации.

#### Turksoy 2017 — Meal Detection and Carbohydrate Estimation Using CGM

**Citation:** Turksoy K, Samadi S, Feng J, et al (2017). Meal Detection and Carbohydrate Estimation Using Continuous Glucose Sensor Data. *IEEE J Biomed Health Inform*; 21(2):444–451.
**DOI:** 10.1109/JBHI.2015.2496257
**Method:** Fuzzy logic для оценки CHO + unscented Kalman filter на Bergman minimal model.
**Metric:** 30 in silico UVA/Padova subjects: sensitivity 91.3%, FPR 9.3%, MAE 23.1%, TIR 76.8%.
**TL;DR:** Часто цитируемый baseline. Fuzzy подход — устаревший с точки зрения 2026.
**Что украсть:** интеграция с bolus calculator.

#### Daniels 2022 — Multitask learning for personalised glucose prediction

**Citation:** Daniels J, Herrero P, Georgiou P (2022). A Multitask Learning Approach to Personalised Blood Glucose Prediction. *IEEE J Biomed Health Inform*; 26(4):1612–1622.
**DOI:** 10.1109/JBHI.2021.3115083
**Method:** Multi-task transformer для одновременного предсказания glucose + meal detection.
**Metric:** OhioT1DM, RMSE 18 mg/dL на 30-min horizon, F1 meal-detection ~75%.
**TL;DR:** Современный ML-подход, не PINN. Хорошее свидетельство, что **personalised** обходит generic.
**Что украсть:** multitask архитектура, fine-tuning per-patient.

#### Zheng 2022 — Ensemble meal detection (OhioT1DM)

**Citation:** ищется уточнённо (вероятно Zheng et al, 2022, Diabetes Technol Ther).
**Method:** Heterogeneous ensemble (ANN + RF + LR) на OhioT1DM. Three voting configurations.
**Metric:** на OhioT1DM: sensitivity 61%, precision 72%, F1 66%, FPR 0.17/day. TIR +10.5%, TAR −5.2%.
**TL;DR:** Показывает разрыв silico vs real-world (94% vs 61% sensitivity).
**Что украсть:** ensemble configurations, real-world валидация.
**Слабость:** падение точности на real data — открытая проблема, на которой можно сыграть.

---

### Tier 2: Physics-Informed подходы (наш методологический угол)

#### De Carli 2025 — Biological-Informed RNN (BI-RNN)

**Citation:** De Carli S, et al (2025). Integrating Biological-Informed Recurrent Neural Networks for Glucose-Insulin Dynamics Modeling. *Engineering Diabetes Technologies (EDT 2025)*.
**arXiv:** 2503.19158
**Method:** GRU + physics-informed loss, валидация на UVA/Padova.
**Metric:** улучшение vs linear baseline на forward prediction.
**TL;DR:** Прямой методологический предшественник. **НО**: они делают forward prediction (BG в будущем), а не inverse (carbs в прошлом).
**Что украсть:** loss design, GRU architecture.
**Зазор для нас:** их подход не применён к inverse problem. Это **наша ниша**.

#### Multerer 2025 — PINN for hidden insulin dynamics

**Citation:** Multerer L, Acquistapace M, Forgione M, Azzimonti L (2025). Physics-Informed Neural Networks for Hidden Insulin Dynamics Estimation from Glucose Data. *AIME 2025*, Springer LNCS.
**Method:** PINN на Bergman minimal model, IVGTT-сценарий, оценка параметров и латентной X(t).
**Metric:** simulation only, демонстрация.
**TL;DR:** Ближе всего к нашему методу. Но: (a) Bergman, не UVA/Padova; (b) IVGTT, не free-living; (c) оценка параметров, не carbs.
**Что украсть:** PINN-loss с plausible parameter ranges, latent variable reconstruction.

#### E-PINN 2025 — Evidential PINN

**Citation:** [arXiv:2509.14568]
**Method:** PINN с EDL (evidential deep learning) для uncertainty quantification, применение к Bergman модели для IVGTT.
**TL;DR:** Uncertainty quantification в PINN — ровно то, что нам нужно (мы должны давать σ_C, а не только C̃).
**Что украсть:** EDL loss term, calibration уpper bound.

---

### Tier 3: Background — UVA/Padova и физиология

#### Dalla Man 2007 — Meal Simulation Model

**Citation:** Dalla Man C, Rizza RA, Cobelli C (2007). Meal Simulation Model of the Glucose-Insulin System. *IEEE Trans Biomed Eng*; 54(10):1740–1749.
**TL;DR:** Оригинальная UVA/Padova. Read first.

#### Visentin 2018 — UVA/Padova goes from meal to day

**Citation:** Visentin R, et al (2018). The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day. *J Diabetes Sci Technol*; 12(2):273–281.
**TL;DR:** Обновление модели S2013 → S2017 с поддержкой суточной симуляции, цикличной insulin sensitivity, физической активности.

#### Hovorka 2004 — Hovorka model

**Citation:** Hovorka R, et al (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes. *Physiol Meas*; 25:905–920.
**TL;DR:** Альтернатива UVA/Padova. Используется в CamAPS. Знать, потому что rev2 нашей статьи 100% спросит "почему не Hovorka".

---

### Tier 4: ML на time series (методологический фон)

- Раффаэллo et al — Transformer for blood glucose prediction
- Munoz-Organero — LSTM for hypoglycemia prediction
- Li et al — TCN vs LSTM на OhioT1DM

(заполнить при поиске)

---

## Открытые датасеты — статус доступа

| Dataset | Patients | Duration | Access | Status |
|---|---|---|---|---|
| OpenAPS Data Commons | ~200 | 1–5 лет | request via openaps.org | TODO check from RU |
| OhioT1DM | 12 | 8 weeks | sign DUA | open for academic |
| D1NAMO | 9 | 4 days | Zenodo public | ✅ open |
| ShanghaiT1DM | 12 | 3–14 days | Scientific Data 2023 | ✅ open |
| ABC4D (Imperial) | 10 | 6 weeks | request | restricted |
| UVA/Padova in silico | 30 | infinite | with simglucose | ✅ open |

---

## Что нужно сделать после первичного литобзора

1. **Сформулировать research gap в одну фразу.**
   *Draft:* "While both PINN approaches for glucose-insulin parameter estimation and ML-based meal detection algorithms have been developed independently, no published work applies physics-informed neural networks to the **inverse problem of carbohydrate quantification** from postprandial CGM signals in free-living conditions."

2. **Подтвердить gap.** Сделать систематический поиск в Scopus + Web of Science по ключам `("physics-informed" OR "biological-informed") AND ("carbohydrate estimation" OR "meal size")`. Если 0 hits — gap подтверждён.

3. **Связаться с авторами Multerer/De Carli.** Они близки методологически. Возможно, заинтересуются совместной работой или дадут feedback на наш draft. SUPSI (Lugano) — open для коллаборации.

---

## BibTeX

Файл `references.bib` ведётся параллельно. Сейчас содержит 12 записей, целевой объём — 60+ к моменту подачи preprint.

```bibtex
@article{mahmoudi2019sensor,
  title={Sensor-based detection and estimation of meal carbohydrates for people with diabetes},
  author={Mahmoudi, Zeinab and Cameron, Faye and Poulsen, Niels Kj{\o}lstad and Madsen, Henrik and Bequette, B Wayne and J{\o}rgensen, John Bagterp},
  journal={Biomedical Signal Processing and Control},
  volume={48},
  pages={12--25},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.bspc.2018.09.012}
}

@article{dassau2008detection,
  title={Detection of a meal using continuous glucose monitoring: implications for an artificial $\beta$-cell},
  author={Dassau, Eyal and Bequette, B Wayne and Buckingham, Bruce A and Doyle III, Francis J},
  journal={Diabetes Care},
  volume={31},
  number={2},
  pages={295--300},
  year={2008},
  doi={10.2337/dc07-1293}
}

@inproceedings{decarli2025birnn,
  title={Integrating Biological-Informed Recurrent Neural Networks for Glucose-Insulin Dynamics Modeling},
  author={De Carli, Stefano and others},
  booktitle={Engineering Diabetes Technologies (EDT)},
  year={2025},
  eprint={2503.19158},
  archivePrefix={arXiv}
}

@inproceedings{multerer2025pinn,
  title={Physics-Informed Neural Networks for Hidden Insulin Dynamics Estimation from Glucose Data},
  author={Multerer, Lea and Acquistapace, Manuel and Forgione, Marco and Azzimonti, Laura},
  booktitle={Artificial Intelligence in Medicine (AIME)},
  publisher={Springer},
  year={2025}
}
% + ещё 8 записей, добавляются по мере чтения
```
