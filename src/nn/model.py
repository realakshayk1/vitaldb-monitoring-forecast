from __future__ import annotations
import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),             # (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)          # logits
        )

    def forward(self, x):
        z = self.net(x)
        logits = self.head(z).squeeze(-1)
        return logits
