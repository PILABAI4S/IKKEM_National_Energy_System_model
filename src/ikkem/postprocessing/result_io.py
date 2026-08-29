"""Generic result readers extracted from the internal analysis workflow.

No paper-specific metric selection, validation thresholds, or figure logic is
included here.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path!s}, got {type(value).__name__}")
    return value


def load_pickle(path: str | Path) -> Any:
    """Read a trusted local pickle artifact.

    Pickle can execute arbitrary code while loading. Never use this function on
    untrusted or externally supplied files.
    """
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def numeric_sum(value: Any) -> float:
    """Recursively sum finite numeric content from common result containers."""
    if isinstance(value, dict):
        return float(sum(numeric_sum(item) for item in value.values()))
    if isinstance(value, (list, tuple, set)):
        return float(sum(numeric_sum(item) for item in value))
    if isinstance(value, np.ndarray):
        return float(np.nansum(value.astype(float)))
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else 0.0
    return 0.0


def numeric_max(value: Any) -> float:
    """Return the maximum absolute finite value in a nested result container."""
    if isinstance(value, dict):
        candidates = [numeric_max(item) for item in value.values()]
    elif isinstance(value, (list, tuple, set)):
        candidates = [numeric_max(item) for item in value]
    elif isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
        return float(np.nanmax(np.abs(array))) if array.size else 0.0
    elif isinstance(value, (int, float, np.number)):
        return abs(float(value)) if np.isfinite(value) else 0.0
    else:
        return 0.0
    return max(candidates, default=0.0)
