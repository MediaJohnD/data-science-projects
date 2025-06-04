import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import run_pipeline


def test_run_pipeline():
    # Ensure pipeline runs without raising errors
    run_pipeline()
