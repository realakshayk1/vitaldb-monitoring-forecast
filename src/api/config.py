# src/api/config.py

TAU = 0.48
SIGMA_MAX = 0.03229
MC_SAMPLES = 20

# Set these to match CNN training setup.
SEQ_LEN = 60          # number of time steps in the sequence
STRIDE_SEC = 10       # time between points in the sequence (seconds)

# These must match CNN input channels (in_channels=6).
# Put the exact feature columns used when training the CNN.
CHANNEL_COLS = [
    "hr_mean", "hr_std", "hr_slope",
    "map_mean", "map_std", "map_slope",
    # If CNN used spo2 too, swap one in; otherwise keep 6 total.
]
