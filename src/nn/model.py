from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    Input:  (B, C, T)  where C includes values + masks (e.g., 6 channels)
    Output: logits shape (B,)
    """
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


    def __init__(self, in_channels: int = 6, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(dropout),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveMaxPool1d(1),  # (B, 64, 1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),             # (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)          # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        logits = self.head(z).squeeze(-1)
        return logits

