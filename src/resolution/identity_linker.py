"""Simple identity resolution utilities."""

from __future__ import annotations

import hashlib


def run(device_id: str) -> str:
    """Return a hashed identifier for a device."""

    return hashlib.sha256(device_id.encode()).hexdigest()
