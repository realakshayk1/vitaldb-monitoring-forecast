# vitaldb-monitoring-forecasting
forecasting using vitaldb database

Uncertainty-aware alerting.
We augment the temporal CNN with Monte Carlo Dropout to estimate predictive uncertainty. Alerts are triggered only when the predicted event probability exceeds τ = 0.48 and the predictive standard deviation falls below the 70th percentile of observed uncertainty. This reduces spurious alerts from ambiguous physiological states while preserving early warning capability.