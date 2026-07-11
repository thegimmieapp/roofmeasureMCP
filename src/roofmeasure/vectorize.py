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


# ------------------------------------------------------------- boundary trace

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


# ------------------------------------------------------------- simplification

def douglas_peucker(pts: np.ndarray, eps: float) -> np.ndarray:
    """Iterative Douglas-Peucker on an open polyline of (y, x) points."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        d = p1 - p0
        norm = math.hypot(d[0], d[1])
        if norm < 1e-9:
            dists = np.hypot(pts[i0 + 1:i1, 0] - p0[0], pts[i0 + 1:i1, 1] - p0[1])
        else:
            dists = np.abs(d[1] * (pts[i0 + 1:i1, 0] - p0[0]) - d[0] * (pts[i0 + 1:i1, 1] - p0[1])) / norm
        k = int(np.argmax(dists))
        if dists[k] > eps:
            mid = i0 + 1 + k
            keep[mid] = True
            stack.append((i0, mid))
            stack.append((mid, i1))
    return pts[keep]


# ------------------------------------------------------- angle regularization

def _dominant_axis(polys: dict) -> float:
    """Primary roof orientation (deg, 0-45) from length-weighted segment angles."""
    votes = np.zeros(90)
    for poly in polys.values():
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            v = b - a
            L = math.hypot(v[0], v[1])
            if L < MIN_SEG_PX:
                continue
            ang = math.degrees(math.atan2(v[0], v[1])) % 90.0
            votes[int(ang) % 90] += L
    # smooth votes circularly over 90 deg, fold onto 45
    best = int(np.argmax(votes))
    return best % 45.0


def snap_polygon(poly: np.ndarray, axis0: float) -> np.ndarray:
    """Snap each segment's direction to the nearest canonical axis
    (axis0 + k*45 deg), then rebuild vertices as line intersections."""
    n = len(poly)
    if n < 3:
        return poly
    lines = []  # (point, direction unit vector)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        v = b - a
        L = math.hypot(v[0], v[1])
        if L < 1e-9:
            continue
        ang = math.degrees(math.atan2(v[0], v[1]))
        # canonical axes: axis0 + k*45
        rel = (ang - axis0) % 45.0
        delta = rel if rel <= 22.5 else rel - 45.0
        if abs(delta) <= SNAP_TOL_DEG:
            ang = ang - delta
        rad = math.radians(ang)
        d = np.array([math.sin(rad), math.cos(rad)])
        mid = (a + b) / 2.0
        # merge with previous line if nearly parallel and collinear
        if lines:
            p_prev, d_prev, w_prev = lines[-1]
            cross = abs(d_prev[0] * d[1] - d_prev[1] * d[0])
            if cross < math.sin(math.radians(10.0)):
                # weighted merge of the two parallel lines
                w = L / (L + w_prev)
                lines[-1] = (p_prev * (1 - w) + mid * w, d_prev, w_prev + L)
                continue
        lines.append((mid, d, L))
    if len(lines) < 3:
        return poly
    # also merge first/last if parallel
    p0, d0, w0 = lines[0]
    p1, d1, w1 = lines[-1]
    if abs(d0[0] * d1[1] - d0[1] * d1[0]) < math.sin(math.radians(10.0)):
        w = w0 / (w0 + w1)
        lines[0] = (p1 * (1 - w) + p0 * w, d0, w0 + w1)
        lines.pop()
    # rebuild vertices as consecutive line intersections
    verts = []
    m = len(lines)
    for i in range(m):
        p1, d1, _ = lines[i]
        p2, d2, _ = lines[(i + 1) % m]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            verts.append((p1 + p2) / 2.0)
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        verts.append(p1 + t * d1)
    return np.array(verts)


# ---------------------------------------------------------------- public API

def facet_polygons(detail: MeasureDetail) -> dict:
    """letter -> Nx2 (y, x) simplified, angle-snapped polygon."""
    labels = detail.labels
    raw = {}
    for letter, fid in detail.facet_ids.items():
        chain = trace_boundary(labels == fid)
        if len(chain) < 8:
            continue
        # close-aware simplification: rotate chain so a corner-ish point is first
        simp = douglas_peucker(chain, DP_EPS_PX)
        if len(simp) >= 2 and np.allclose(simp[0], simp[-1]):
            simp = simp[:-1]
        if len(simp) >= 3:
            raw[letter] = simp
    axis0 = _dominant_axis(raw) if raw else 0.0
    out = {}
    for letter, poly in raw.items():
        snapped = snap_polygon(poly, axis0)
        # sanity: reject snapped polygons that exploded (bad intersections)
        if len(snapped) >= 3:
            span_raw = float(np.ptp(poly, axis=0).max())
            span_new = float(np.ptp(snapped, axis=0).max())
            if span_new < span_raw * 1.6:
                out[letter] = snapped
                continue
        out[letter] = poly
    return out


def classify_polygon_edges(polys: dict, detail: MeasureDetail) -> list[dict]:
    """Split every polygon edge and tag it with kind + true length in ft.

    Returns [{letter, p0, p1, kind, length_ft}], p0/p1 as (y, x). kind may be
    None for unmatched (interior artifacts).
    """
    # build lookup arrays of classified pixels
    kinds = []
    pix_arrays = []
    factors = []  # slope-correction factor per cluster
    for seg in detail.edges:
        arr = np.asarray(seg.pixels, dtype=float)
        kinds.append(seg.kind)
        pix_arrays.append(arr)
        plan = seg.plan_length_ft or seg.length_ft
        factors.append(seg.length_ft / plan if plan > 0 else 1.0)

    px = (detail.px_x + detail.px_y) / 2.0
    ft_per_px = px * 3.280839895
    out = []
    for letter, poly in polys.items():
        n = len(poly)
        for i in range(n):
            p0, p1 = poly[i], poly[(i + 1) % n]
            seg_len_px = math.hypot(*(p1 - p0))
            if seg_len_px < MIN_SEG_PX:
                continue
            # sample 3 points along the edge, find nearest classified cluster
            samples = [p0 + (p1 - p0) * t for t in (0.25, 0.5, 0.75)]
            votes = {}
            for s in samples:
                best_d, best_k = 1e9, None
                for k, arr in enumerate(pix_arrays):
                    d = np.min(np.hypot(arr[:, 0] - s[0], arr[:, 1] - s[1]))
                    if d < best_d:
                        best_d, best_k = d, k
                if best_k is not None and best_d <= MATCH_DIST_PX:
                    votes[best_k] = votes.get(best_k, 0) + 1
            if votes:
                k = max(votes, key=votes.get)
                kind = kinds[k]
                length_ft = seg_len_px * ft_per_px * factors[k]
            else:
                kind, length_ft = None, seg_len_px * ft_per_px
            out.append({"letter": letter, "p0": tuple(p0), "p1": tuple(p1),
                        "kind": kind, "length_ft": round(length_ft, 1)})
    return out


# ------------------------------------------------ line-network representation

def outline_polygon(detail: MeasureDetail) -> np.ndarray:
    """Single clean outer outline of the whole roof (Nx2 (y,x))."""
    mask = detail.labels > 0
    chain = trace_boundary(mask)
    if len(chain) < 8:
        return chain
    simp = douglas_peucker(chain, DP_EPS_PX)
    if len(simp) >= 2 and np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    axis0 = _dominant_axis({"outline": simp})
    snapped = snap_polygon(simp, axis0)
    if len(snapped) >= 3:
        span_raw = float(np.ptp(simp, axis=0).max())
        if float(np.ptp(snapped, axis=0).max()) < span_raw * 1.6:
            return snapped
    return simp


def internal_lines(detail: MeasureDetail) -> list[dict]:
    """Each internal ridge/hip/valley cluster as ONE straight line (PCA endpoints).

    Returns [{p0, p1, kind, length_ft}] in (y, x) raster coords.
    """
    out = []
    for seg in detail.edges:
        if seg.kind not in ("ridge", "hip", "valley"):
            continue
        pts = np.asarray(seg.pixels, dtype=float)
        if len(pts) < 2:
            continue
        c = pts.mean(axis=0)
        centered = pts - c
        cov = centered.T @ centered / len(pts)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        proj = centered @ axis
        p0 = c + axis * float(proj.min())
        p1 = c + axis * float(proj.max())
        out.append({"p0": tuple(p0), "p1": tuple(p1), "kind": seg.kind,
                    "length_ft": seg.length_ft})
    return out


def classify_outline_edges(outline: np.ndarray, detail: MeasureDetail) -> list[dict]:
    """Tag each outline segment as eave/rake by nearest classified cluster."""
    kinds, pix_arrays, factors = [], [], []
    for seg in detail.edges:
        if seg.kind not in ("eave", "rake"):
            continue
        kinds.append(seg.kind)
        pix_arrays.append(np.asarray(seg.pixels, dtype=float))
        plan = seg.plan_length_ft or seg.length_ft
        factors.append(seg.length_ft / plan if plan > 0 else 1.0)
    px = (detail.px_x + detail.px_y) / 2.0
    ft_per_px = px * 3.280839895
    out = []
    n = len(outline)
    for i in range(n):
        p0, p1 = outline[i], outline[(i + 1) % n]
        seg_len_px = math.hypot(*(p1 - p0))
        if seg_len_px < MIN_SEG_PX:
            continue
        samples = [p0 + (p1 - p0) * t for t in (0.25, 0.5, 0.75)]
        votes = {}
        for s in samples:
            best_d, best_k = 1e9, None
            for k, arr in enumerate(pix_arrays):
                d = np.min(np.hypot(arr[:, 0] - s[0], arr[:, 1] - s[1]))
                if d < best_d:
                    best_d, best_k = d, k
            if best_k is not None and best_d <= MATCH_DIST_PX * 1.5:
                votes[best_k] = votes.get(best_k, 0) + 1
        if votes:
            k = max(votes, key=votes.get)
            kind = kinds[k]
            length_ft = seg_len_px * ft_per_px * factors[k]
        else:
            kind, length_ft = "eave", seg_len_px * ft_per_px
        out.append({"p0": tuple(p0), "p1": tuple(p1), "kind": kind,
                    "length_ft": round(length_ft, 1)})
    return out
