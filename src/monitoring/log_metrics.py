"""Monitoring utilities."""

from __future__ import annotations


def run(metric: str, value: float) -> None:
    """Log a metric value."""

    print(f"{metric}: {value:.4f}")
