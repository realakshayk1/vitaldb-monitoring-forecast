import sys

import numpy as np
import pandas as pd
import vitaldb

import os
print("FILE:", os.path.abspath(__file__), flush=True)
print("CWD :", os.getcwd(), flush=True)
print("EXE :", sys.executable, flush=True)

def main() -> None:
    print("Python:", sys.version)
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("vitaldb:", getattr(vitaldb, "__version__", "unknown"))
    print("✅ Imports successful.")

if __name__ == "__main__":
    main()
