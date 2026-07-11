"""Vectorize facet rasters into clean straight-line polygons (EagleView-style).

Roofs are combinations of straight-edged geometric shapes, so diagrams must be
drawn with straight lines only. Pipeline per facet:

  1. Moore boundary tracing -> ordered pixel chain
  2. Douglas-Peucker simplification -> straight segments
  3. Angle regularization: snap segment directions to the roof's dominant
     axes (0/45/90/135 degrees relative to the primary orientation), then
     rebuild vertices as intersections of consecutive snapped lines
  4. Classify each polygon edge (ridge/hip/valley/eave/rake) by matching to
     the nearest classified boundary cluster from the measurement engine
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import MeasureDetail

DP_EPS_PX = 7.0           # Douglas-Peucker tolerance (pixels, ~0.35 m)
SNAP_TOL_DEG = 22.5       # snap segment angle when within this of a canonical axis
MIN_SEG_PX = 6.0          # drop degenerate segments
MATCH_DIST_PX = 8.0       # max distance to inherit a classified edge kind


def trace_boundary(mask: np.ndarray) -> np.ndarray:
    """Moore-neighbor boundary trace of the largest blob in a boolean mask.
    Returns ordered Nx2 array of (y, x) boundary pixel coordinates."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2))
    # start: topmost-leftmost pixel
    i = np.lexsort((xs, ys))[0]
    start = (int(ys[i]), int(xs[i]))
    # Moore neighborhood, clockwise starting from W
    nbrs = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    h, w = mask.shape

    def at(p):
        y, x = p
        return 0 <= y < h and 0 <= x < w and mask[y, x]

    contour = [start]
    prev_dir = 0
    cur = start
    for _ in range(len(ys) * 4 + 10):
        found = False
        for k in range(8):
            d = (prev_dir + 6 + k) % 8  # backtrack rule
            cand = (cur[0] + nbrs[d][0], cur[1] + nbrs[d][1])
            if at(cand):
                if cand == start and len(contour) > 2:
                    return np.array(contour, dtype=float)
                contour.append(cand)
                cur = cand
                prev_dir = d
                found = True
                break
        if not found:  # isolated pixel
            break
    return np.array(contour, dtype=float)