# VitalDB Monitoring & Early Warning Forecasting

## Problem
Intraoperative and ICU monitoring systems generate high-frequency physiologic telemetry (ECG, ABP, SpO₂), but naïve threshold-based alarms produce excessive false alerts and limited early warning.

This project investigates **early event forecasting from continuous physiologic time series**, with an emphasis on:

> Detecting impending adverse events early, while controlling alert burden and uncertainty.

The goal is not just prediction accuracy, but **operationally meaningful alerting**: fewer false alarms, sufficient lead time, and stable behavior under deployment-like constraints.

---

## Data
- **Dataset**: VitalDB (PhysioNet)
- **Signals used**:
  - ECG
  - Arterial Blood Pressure (ABP)
  - SpO₂
- **Sampling**: high-frequency waveforms aggregated into fixed windows
- **Unit of observation**: rolling time windows per patient case
- **Labeling**: adverse clinical events defined relative to future time horizons

VitalDB provides realistic, noisy physiologic signals representative of real operating room and ICU monitoring environments.

---

## Feature Construction
Two parallel feature representations are supported:

### 1. Time-Series Windows (Neural Models)
- Fixed-length rolling windows over raw waveforms
- Multi-channel inputs (3–6 channels depending on experiment)
- Minimal preprocessing to preserve temporal structure

### 2. Engineered Tabular Features (GBM Baseline)
- Summary statistics per window (e.g., moments, ranges)
- Explicit feature specification via schema files
- Designed to mirror features available in real-time monitoring systems

All windowing logic is centralized and reproducible via the `features/` module.

---

## Modeling Approach
Modeling focused on **early warning performance under alert constraints**, not just AUROC.

Models implemented and evaluated:
1. **LightGBM (tabular baseline)**
2. **Temporal CNN** over waveform windows
3. **Hybrid CNN + tabular variants** (explored, not selected)

Although hybrid architectures were prototyped, they **did not outperform the tabular LightGBM baseline** in operationally relevant metrics and were not used in the final system.

### Final Emphasis
- **LightGBM** provided the best tradeoff between performance, stability, and simplicity
- CNN models were retained for comparison and ablation analysis

---

## Uncertainty-Aware Inference
Neural models incorporate **Monte Carlo Dropout** at inference time to estimate predictive uncertainty.

For each window:
- Mean predicted risk (`p_mean`)
- Predictive uncertainty (`p_std`)

This enables downstream alerting policies that consider **confidence**, not just raw risk.

---

## Alerting Policy
P(event | telemetry) ≥ τ
AND
predictive uncertainty ≤ σ_max


This policy explicitly trades off:
- Sensitivity
- Alert volume
- Stability of predictions

The design reflects real-world monitoring requirements where false alarms are costly.

---

## Evaluation
Evaluation emphasizes **deployment-relevant behavior**, not offline accuracy alone.

Metrics examined include:
- AUROC / AUPRC
- Alerts per hour
- Precision–recall at fixed alert budgets
- Lead time before event onset

Model comparisons and policy sweeps are implemented in `src/analysis/`, with scripts to:
- Compare alerting regimes
- Quantify alert burden vs. recall
- Visualize tradeoffs across thresholds

---

## System Architecture
The project includes an **online inference service** for real-time deployment scenarios.

Telemetry Stream
↓
Rolling Window Builder
↓
Feature Extraction
↓
Model Inference
↓
Uncertainty Estimation
↓
Alert Policy
↓
API Response + Metrics


---

## API
A FastAPI service supports online inference and monitoring.

### Endpoints
- `POST /predict`
  - Input: recent telemetry window
  - Output: `{ p_mean, p_std, alert }`
- `GET /metrics`
  - Prometheus-compatible metrics
- `GET /health`
  - Liveness check

The API layer is intentionally minimal to reflect production monitoring constraints.

---

## Monitoring & Drift Detection
The system includes hooks for **post-deployment monitoring**:

- **Data drift**:
  - KS tests on raw physiologic signals (HR, MAP, SpO₂)
- **Performance drift**:
  - Rolling AUROC / AUPRC degradation
- **Retraining policy**:
  - Triggered only after sustained drift across multiple windows

This avoids reactive retraining on transient noise.

---

## Key Takeaways
- Simple tabular models can outperform more complex neural architectures in operational settings
- Alert quality depends on policy design, not just model accuracy
- Uncertainty estimation is critical for safe deployment
- Evaluation must reflect alert burden and lead time, not just AUROC
- The system is designed to resemble real monitoring pipelines, not Kaggle-style benchmarks

---

## Scope Notes
- This repository is a **research and prototyping system**, not a production monitor
- Models are trained offline and loaded for inference
- Emphasis is on transparency, reproducibility, and honest comparison

---

## Future Extensions
- Formal calibration analysis across patient subgroups
- Online learning under distribution shift
- Integration with hospital alarm management systems
- Extension to additional VitalDB signal modalities


| Category                  | Result / Value                                 | What It Represents                                     | How It Was Used / Why It Matters                         |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| **Data**                  | High-frequency ECG, ABP, SpO₂                  | Continuous perioperative waveform signals from VitalDB | Core input signals for early-warning modeling            |
| **Prediction Horizon**    | ~5 minutes (forward-looking)                   | Time window defining “imminent adverse outcome”        | Defines the early-warning task (not long-term prognosis) |
| **Windowing**             | Fixed-length sliding windows                   | Temporal segmentation of waveform streams              | Converts continuous signals into supervised samples      |
| **Baseline Models**       | CNN, TCN (PyTorch)                             | End-to-end deep learning baselines                     | Used for comparison, not final deployment                |
| **Final Model**           | LightGBM (binary classifier)                   | Tree-based gradient boosting on time-windowed features | Selected for deployment-relevant behavior                |
| **AUROC (GBM)**           | **0.84**                                       | Discrimination ability across all thresholds           | Confirms strong signal separation                        |
| **AUPRC (GBM)**           | **0.39**                                       | Precision–recall performance under class imbalance     | More informative than AUROC for rare events              |
| **CNN / TCN AUROC**       | Lower than 0.84 (exact values model-dependent) | End-to-end DL discrimination                           | Underperformed GBM at usable operating points            |
| **Operating Constraint**  | Alert-rate–constrained thresholds              | Thresholds chosen by alerts/hour, not probability      | Aligns evaluation with deployment reality                |
| **Alert Rate (GBM)**      | **≈3 alerts/hour**                             | Operational false-alert burden                         | Within realistic monitoring tolerances                   |
| **Alert Rate (CNN)**      | **≈8 alerts/hour**                             | False-alert burden at comparable sensitivity           | Too noisy for practical use                              |
| **False Alert Reduction** | **~3× fewer alerts (GBM vs CNN)**              | Relative operational improvement                       | Primary deployment win                                   |
| **Median Lead Time**      | **~95 minutes**                                | Time between alert and adverse outcome                 | Demonstrates clinically meaningful advance warning       |
| **Lead Time Constraint**  | ~1–1.5 hours (implicit)                        | Minimum useful early-warning window                    | Alerts closer to outcome deemed less actionable          |
| **Hybrid Model Result**   | Did **not** outperform GBM                     | CNN/TCN embeddings → LightGBM                          | Negative result informed final model choice              |
| **Evaluation Focus**      | Alerts/hr + lead time (not just AUROC)         | Decision-quality metrics                               | Reflects real monitoring constraints 
