# run_all.py
from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional


# -----------------------------
# Utilities
# -----------------------------
def banner(msg: str) -> None:
    line = "=" * 88
    print("\n" + line)
    print(msg)
    print(line)

def run_step(name: str, fn: Callable[[], None], *, continue_on_error: bool = True) -> bool:
    banner(f"STEP: {name}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"\n✅ DONE: {name} ({dt:.1f}s)")
        return True
    except Exception as e:
        dt = time.time() - t0
        print(f"\n❌ FAILED: {name} ({dt:.1f}s) -> {type(e).__name__}: {e}")
        traceback.print_exc()
        if continue_on_error:
            print("↪ Continuing to next step...\n")
            return False
        raise


# -----------------------------
# Main runner
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run VitalDB pipeline end-to-end.")
    parser.add_argument("--max-cases", type=int, default=200, help="Max cohort size for build_case_cohort.")
    parser.add_argument("--skip-cohort", action="store_true", help="Skip cohort build (expects data/interim/caseids_v1.txt).")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion (expects data/interim/cases/*.parquet).")
    parser.add_argument("--skip-labels", action="store_true", help="Skip label build (expects data/processed/labeled_cases/*.parquet).")
    parser.add_argument("--skip-windows", action="store_true", help="Skip window feature build (expects data/processed/windows.parquet).")
    parser.add_argument("--skip-tabular", action="store_true", help="Skip logistic + GBM training.")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training + uncertainty inference.")
    parser.add_argument("--skip-policy", action="store_true", help="Skip policy simulations.")
    parser.add_argument("--skip-drift", action="store_true", help="Skip drift report.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on any failure.")
    args = parser.parse_args()

    continue_on_error = not args.stop_on_error

    # Import modules lazily inside closures so partial failures don't break imports globally.
    def step_build_cohort() -> None:
        from src.split.build_case_cohort import main as cohort_main
        cohort_main(max_cases=args.max_cases)

    def step_ingest() -> None:
        from src.vital.ingest_cases import main as ingest_main
        ingest_main()

    def step_labels() -> None:
        from src.labels.build_labels import main as labels_main
        labels_main()

    def step_windows() -> None:
        from src.features.build_windows import main as windows_main
        windows_main()

    def step_train_logreg() -> None:
        from src.models.train_logistic import main as logreg_main
        logreg_main()

    def step_train_gbm() -> None:
        from src.models.train_gbm import main as gbm_main
        gbm_main()

    def step_train_cnn() -> None:
        from src.nn.train_cnn import main as cnn_main
        cnn_main()

    def step_infer_uncertainty() -> None:
        from src.nn.infer_uncertainty import main as unc_main
        unc_main()

    def step_policy_gbm() -> None:
        from src.policy.simulate_alerts_gbm import main as gbm_policy_main
        gbm_policy_main()

    def step_policy_cnn() -> None:
        from src.policy.simulate_alerts_cnn import main as cnn_policy_main
        cnn_policy_main()

    def step_policy_cnn_uncertainty() -> None:
        from src.policy.simulate_alerts_cnn_uncertainty import main as cnn_u_policy_main
        cnn_u_policy_main()

    def step_drift() -> None:
        from src.monitoring.drift_report import main as drift_main
        drift_main()

    def step_compare() -> None:
        from src.analysis.compare_models import main as compare_main
        compare_main()

    banner("RUN START: VitalDB end-to-end pipeline")

    # ---- Cohort -> Ingest -> Labels -> Windows ----
    if not args.skip_cohort:
        run_step("Build case cohort (writes data/interim/caseids_v1.txt)", step_build_cohort, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping cohort build. Expecting data/interim/caseids_v1.txt to already exist.")

    if not args.skip_ingest:
        run_step("Ingest cases (writes data/interim/cases/*.parquet)", step_ingest, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping ingestion. Expecting data/interim/cases/*.parquet to already exist.")

    if not args.skip_labels:
        run_step("Build labels (writes data/processed/labeled_cases/*.parquet)", step_labels, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping labels. Expecting data/processed/labeled_cases/*.parquet to already exist.")

    if not args.skip_windows:
        run_step("Build window features (writes data/processed/windows.parquet)", step_windows, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping windows. Expecting data/processed/windows.parquet to already exist.")

    # ---- Tabular models ----
    if not args.skip_tabular:
        run_step("Train Logistic Regression (prints AUROC/AUPRC)", step_train_logreg, continue_on_error=continue_on_error)
        run_step("Train LightGBM (prints AUROC/AUPRC, saves models/gbm.txt)", step_train_gbm, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping tabular training.")

    # ---- CNN + uncertainty ----
    if not args.skip_cnn:
        run_step("Train Temporal CNN (prints val/test AUROC/AUPRC, saves models/nn/cnn_best.pt)", step_train_cnn, continue_on_error=continue_on_error)
        run_step("Infer CNN uncertainty (writes reports/policy/cnn_test_uncertainty.parquet)", step_infer_uncertainty, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping CNN training/inference.")

    # ---- Policy simulations ----
    if not args.skip_policy:
        # GBM policy uses models/gbm.txt and labeled_cases
        run_step("Simulate GBM alert policy (writes reports/policy/*)", step_policy_gbm, continue_on_error=continue_on_error)

        # CNN policy + CNN uncertainty policy depend on CNN inference outputs
        run_step("Simulate CNN alert policy (writes reports/policy/*)", step_policy_cnn, continue_on_error=continue_on_error)
        run_step("Simulate CNN uncertainty-aware policy (writes reports/policy/*)", step_policy_cnn_uncertainty, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping policy simulations.")

    # ---- Drift report ----
    if not args.skip_drift:
        run_step("Drift report (KS tests + retrain trigger CSV)", step_drift, continue_on_error=continue_on_error)
    else:
        print("\n⚠️ Skipping drift report.")

    # ---- Model comparison ----
    run_step("Compare models (policy + plots)", step_compare, continue_on_error=continue_on_error)

    banner("RUN COMPLETE")


if __name__ == "__main__":
    main()
