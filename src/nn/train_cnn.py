from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN


OUT_DIR = Path("models/nn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("reports/nn")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic can be slower; you can turn this off if needed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Helpers: predict + metrics
# -----------------------------
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            p = torch.sigmoid(logits)

            ys.append(y.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())

    y = np.concatenate(ys).astype(np.float32)
    p = np.concatenate(ps).astype(np.float32)
    return y, p


def auc_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    # guard: if a split has only one class, roc_auc_score errors
    out: Dict[str, float] = {}
    out["auprc"] = float(average_precision_score(y, p))
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auroc"] = float("nan")
    return out


def threshold_for_target_recall(y: np.ndarray, p: np.ndarray, target_recall: float = 0.90) -> float:
    """
    Choose the highest threshold that achieves recall >= target_recall on validation.
    This produces fewer alerts for the same recall constraint (good for applied alerting).
    """
    # sort by probability desc
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]

    total_pos = float(np.sum(y_sorted))
    if total_pos <= 0:
        return 1.0  # no positives, arbitrary "never alert"

    tp = 0.0
    best_tau = 0.0
    for i in range(len(p_sorted)):
        if y_sorted[i] > 0.5:
            tp += 1.0
        recall = tp / total_pos
        if recall >= target_recall:
            best_tau = float(p_sorted[i])
            break

    # if never hits target (rare), alert everything
    if best_tau == 0.0:
        best_tau = 0.0
    return best_tau


def policy_metrics(y: np.ndarray, p: np.ndarray, tau: float) -> Dict[str, float]:
    """
    Simple policy metrics from window-level predictions:
    - precision, recall at tau
    - alert_rate = fraction of windows alerted
    NOTE: Your repo has richer event/cooldown policy scripts; this is a fast “resume-ready” summary.
    """
    pred = (p >= tau).astype(np.float32)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    alert_rate = float(np.mean(pred))

    return {
        "tau": float(tau),
        "precision_at_tau": float(precision),
        "recall_at_tau": float(recall),
        "alert_rate_at_tau": float(alert_rate),
    }


# -----------------------------
# Training
# -----------------------------
@dataclass
class RunConfig:
    seeds: List[int] = None
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    target_recall: float = 0.90


def compute_pos_weight(train_ds: VitalSeqDataset, device: str) -> torch.Tensor:
    yvec = np.asarray(train_ds.idx["y"]).astype(np.float32)
    pos = float(np.sum(yvec))
    neg = float(len(yvec) - pos)
    pos_rate = pos / max(len(yvec), 1)
    w = neg / max(pos, 1e-6)
    print(f"train positives: {int(pos)} | negatives: {int(neg)} | pos_rate: {pos_rate:.4f} | pos_weight: {w:.2f}")
    return torch.tensor([w], dtype=torch.float32, device=device)


def train_one_seed(seed: int, cfg: RunConfig) -> Dict[str, float]:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 88)
    print(f"SEED {seed} | device: {device}")
    print("=" * 88)

    train_ds = VitalSeqDataset("train")
    val_ds = VitalSeqDataset("val")
    test_ds = VitalSeqDataset("test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight(train_ds, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = TemporalCNN(in_channels=6, dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )


    best_val_auprc = -1.0
    bad = 0
    best_path = OUT_DIR / f"cnn_best_seed{seed}.pt"
    canonical_path = OUT_DIR / "cnn_best.pt"



    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * len(y)

        total_loss /= max(len(train_ds), 1)

        # validation
        yv, pv = predict_probs(model, val_loader, device)
        val_metrics = auc_metrics(yv, pv)
        val_auc = val_metrics["auroc"]
        val_auprc = val_metrics["auprc"]

        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | loss {total_loss:.4f} | lr {lr_now:.2e} | val AUROC {val_auc:.3f} | val AUPRC {val_auprc:.3f}")

        scheduler.step(val_auprc)

        # early stop on val AUPRC
        if val_auprc > best_val_auprc + 1e-4:
            best_val_auprc = val_auprc
            bad = 0
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), canonical_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping (no val AUPRC improvement for {cfg.patience} epochs)")
                break

    # load best and evaluate test
    model.load_state_dict(torch.load(best_path, map_location=device))

    yt, pt = predict_probs(model, test_loader, device)
    test_metrics = auc_metrics(yt, pt)

    # choose threshold on VAL for target recall, then compute policy metrics on TEST
    tau = threshold_for_target_recall(yv, pv, target_recall=cfg.target_recall)
    pol = policy_metrics(yt, pt, tau=tau)

    print(f"\n✅ SEED {seed} TEST: AUROC {test_metrics['auroc']:.3f} | AUPRC {test_metrics['auprc']:.3f}")
    print(f"✅ Policy @ recall~{cfg.target_recall:.2f} (val-chosen tau={tau:.3f}): "
          f"precision {pol['precision_at_tau']:.3f} | recall {pol['recall_at_tau']:.3f} | alert_rate {pol['alert_rate_at_tau']:.3f}")

    return {
        "seed": float(seed),
        "best_val_auprc": float(best_val_auprc),
        "test_auroc": float(test_metrics["auroc"]),
        "test_auprc": float(test_metrics["auprc"]),
        **pol,
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    cfg = RunConfig(seeds=[7, 13, 29])  # change seeds as you like

    all_runs: List[Dict[str, float]] = []
    for s in cfg.seeds:
        try:
            run = train_one_seed(s, cfg)
            all_runs.append(run)
        except Exception as e:
            # keep going like you requested
            print(f"\n❌ Seed {s} failed: {type(e).__name__}: {e}")
            continue

    if not all_runs:
        raise RuntimeError("All seeds failed; no results to report.")

    aurocs = [r["test_auroc"] for r in all_runs if not np.isnan(r["test_auroc"])]
    auprcs = [r["test_auprc"] for r in all_runs]
    precs = [r["precision_at_tau"] for r in all_runs]
    recs = [r["recall_at_tau"] for r in all_runs]
    alert_rates = [r["alert_rate_at_tau"] for r in all_runs]

    m_auroc, s_auroc = mean_std(aurocs) if aurocs else (float("nan"), float("nan"))
    m_auprc, s_auprc = mean_std(auprcs)
    m_prec, s_prec = mean_std(precs)
    m_rec, s_rec = mean_std(recs)
    m_ar, s_ar = mean_std(alert_rates)

    print("\n" + "=" * 88)
    print("AGGREGATE (across seeds)")
    print("=" * 88)
    print(f"TEST AUROC: {m_auroc:.3f} ± {s_auroc:.3f}")
    print(f"TEST AUPRC: {m_auprc:.3f} ± {s_auprc:.3f}")
    print(f"Policy precision: {m_prec:.3f} ± {s_prec:.3f}")
    print(f"Policy recall: {m_rec:.3f} ± {s_rec:.3f}")
    print(f"Policy alert_rate: {m_ar:.3f} ± {s_ar:.3f}")

    payload = {
        "config": asdict(cfg),
        "runs": all_runs,
        "aggregate": {
            "test_auroc_mean": m_auroc, "test_auroc_std": s_auroc,
            "test_auprc_mean": m_auprc, "test_auprc_std": s_auprc,
            "precision_mean": m_prec, "precision_std": s_prec,
            "recall_mean": m_rec, "recall_std": s_rec,
            "alert_rate_mean": m_ar, "alert_rate_std": s_ar,
        },
    }

    out_path = REPORT_DIR / "cnn_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n📄 Saved: {out_path}")


if __name__ == "__main__":
    main()
