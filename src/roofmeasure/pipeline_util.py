"""Small numpy helpers shared by the raster stages."""

from __future__ import annotations

import numpy as np


def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over a (2k+1)^2 window via integral image (edge-clamped)."""
    h, w = a.shape
    p = np.pad(a.astype(np.float64), k + 1, mode="edge")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    n = 2 * k + 1
    return (ii[n + 1:n + 1 + h, n + 1:n + 1 + w] - ii[1:1 + h, n + 1:n + 1 + w]
            - ii[n + 1:n + 1 + h, 1:1 + w] + ii[1:1 + h, 1:1 + w])
