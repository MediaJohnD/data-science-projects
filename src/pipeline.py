"""Entry point for running the Prefect flow."""

import sys
from orchestrator import main_flow


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    main_flow(path)
