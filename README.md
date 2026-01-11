# vitaldb-monitoring-forecasting
forecasting using vitaldb database

Uncertainty-aware alerting.
We augment the temporal CNN with Monte Carlo Dropout to estimate predictive uncertainty. Alerts are triggered only when the predicted event probability exceeds τ = 0.48 and the predictive standard deviation falls below the 70th percentile of observed uncertainty. This reduces spurious alerts from ambiguous physiological states while preserving early warning capability.

## Online Inference & Monitoring

The system exposes a FastAPI service that:
- Ingests recent telemetry windows for a given case
- Produces probabilistic risk estimates (mean + uncertainty)
- Applies a policy layer to issue alerts
- Exports operational metrics for monitoring

### Endpoints
- POST /predict → { p_mean, p_std, alert }
- GET /metrics → Prometheus counters/histograms
- GET /health → liveness check

### Alert Policy
An alert is raised when:
P(event | telemetry) ≥ τ AND predictive uncertainty ≤ σ_max

### Monitoring & Retraining
- Data drift: KS tests on raw telemetry features (HR, MAP, SpO₂)
- Performance drift: rolling AUPRC/AUROC degradation
- Retraining triggered only after sustained drift across windows
