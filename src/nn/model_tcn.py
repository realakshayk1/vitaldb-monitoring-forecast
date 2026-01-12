from __future__ import annotations
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation  # for kernel=3, keeps length
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalTCN(nn.Module):
    """
    Drop-in replacement for TemporalCNN:
    - input: (B, C, T)
    - output: logits (B,)
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        dropout: float = 0.2,
        dilations=(1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[TCNBlock(hidden, d, dropout) for d in dilations])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.blocks(z)
        z = self.pool(z).squeeze(-1)  # (B, hidden)
        logits = self.head(z).squeeze(-1)
        return logits
