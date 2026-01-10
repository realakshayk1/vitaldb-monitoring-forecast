# Lookback window
LOOKBACK_SEC = 300   # 5 minutes

# Prediction horizon (already defined in labels)
HORIZON_SEC = 300

# Step size between windows
STRIDE_SEC = 10      # every 10 seconds

# Signals
SIGNALS = ["hr", "spo2", "map"]
