from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN

OUT_DIR = Path("models/nn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            p = torch.sigmoid(logits)

            ys.append(y.cpu().numpy())
            ps.append(p.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds = VitalSeqDataset("train")
    val_ds = VitalSeqDataset("val")
    test_ds = VitalSeqDataset("test")

    # Class imbalance handling via pos_weight
    # pos_weight ~ (#neg / #pos) computed from train labels
    y_train = np.array([train_ds.idx["y"].mean()], dtype=float)[0]
    pos_rate = float(y_train)
    neg_rate = 1.0 - pos_rate
    pos_weight = torch.tensor([neg_rate / max(pos_rate, 1e-6)], dtype=torch.float32).to(device)
    print("train pos_rate:", pos_rate, "pos_weight:", float(pos_weight))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TemporalCNN(in_channels=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc = -1.0
    patience = 3
    bad = 0

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(y)

        total_loss /= len(train_ds)

        val_auc, val_auprc = eval_model(model, val_loader, device)
        print(f"epoch {epoch:02d} | loss {total_loss:.4f} | val AUROC {val_auc:.3f} | val AUPRC {val_auprc:.3f}")

        # early stop on AUPRC
        if val_auprc > best_val_auprc + 1e-4:
            best_val_auprc = val_auprc
            bad = 0
            torch.save(model.state_dict(), OUT_DIR / "cnn_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    # Final test eval using best checkpoint
    model.load_state_dict(torch.load(OUT_DIR / "cnn_best.pt", map_location=device))
    test_auc, test_auprc = eval_model(model, test_loader, device)
    print(f"\n✅ CNN test: AUROC {test_auc:.3f} | AUPRC {test_auprc:.3f}")

if __name__ == "__main__":
    main()
