"""Synthetic roof DSM builders with analytically known measurements.

Every builder returns (z, mask, px, truth) where truth is a dict of expected
feet values: ridges, hips, valleys, rakes, eaves, surface_sqft, facets.
The surface is the lower envelope of eave planes (a straight-skeleton roof),
so the DSM is exactly what a perfect sensor would see.
"""

from __future__ import annotations

import math

import numpy as np

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671


def _seg_dist(px_, py_, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = ((px_ - ax) * vx + (py_ - ay) * vy) / L2
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px_ - (ax + t * vx), py_ - (ay + t * vy))


def _inside(px_, py_, poly):
    """Even-odd point-in-polygon on arrays."""
    inside = np.zeros(px_.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cond = (y0 > py_) != (y1 > py_)
        xint = (x1 - x0) * (py_ - y0) / ((y1 - y0) + 1e-12) + x0
        inside ^= cond & (px_ < xint)
    return inside


def build_polygon_roof(poly_m, kinds, pitch_ratio=8 / 12, px=0.1, pad=10, base=3.0,
                       noise=0.0, seed=0):
    """poly_m: [(x, y)] meters, CCW or CW; kinds: 'eave' | 'rake' per edge i->i+1.
    Roof height = base + pitch * (distance to nearest EAVE segment)."""
    xs = [p[0] for p in poly_m]
    ys = [p[1] for p in poly_m]
    W = int((max(ys) - min(ys)) / px) + 2 * pad
    L = int((max(xs) - min(xs)) / px) + 2 * pad
    jj, ii = np.meshgrid(np.arange(L), np.arange(W))
    X = (jj - pad) * px + min(xs)
    Y = (ii - pad) * px + min(ys)
    mask = _inside(X + 1e-6, Y + 1e-6, poly_m)
    d = np.full(X.shape, np.inf)
    n = len(poly_m)
    for i in range(n):
        if kinds[i] != "eave":
            continue
        ax, ay = poly_m[i]
        bx, by = poly_m[(i + 1) % n]
        d = np.minimum(d, _seg_dist(X, Y, ax, ay, bx, by))
    z = np.where(mask, base + d * pitch_ratio, 0.0).astype(np.float32)
    if noise:
        z = z + np.random.default_rng(seed).normal(0, noise, z.shape).astype(np.float32)
    return z, mask, px


def gable_roof(length_m=16.0, width_m=10.0, pitch=8 / 12, **kw):
    poly = [(0, 0), (length_m, 0), (length_m, width_m), (0, width_m)]
    kinds = ["eave", "rake", "eave", "rake"]
    z, mask, px = build_polygon_roof(poly, kinds, pitch, **kw)
    half = width_m / 2
    rake = math.hypot(half, half * pitch)
    truth = dict(
        ridges=length_m * M_TO_FT, hips=0.0, valleys=0.0,
        rakes=4 * rake * M_TO_FT, eaves=2 * length_m * M_TO_FT,
        surface_sqft=length_m * width_m * SQM_TO_SQFT / math.cos(math.atan(pitch)),
        facets=2,
    )
    return z, mask, px, truth


def hip_roof(length_m=16.0, width_m=10.0, pitch=8 / 12, **kw):
    poly = [(0, 0), (length_m, 0), (length_m, width_m), (0, width_m)]
    kinds = ["eave"] * 4
    z, mask, px = build_polygon_roof(poly, kinds, pitch, **kw)
    half = width_m / 2
    hip = math.hypot(half * math.sqrt(2), half * pitch)
    truth = dict(
        ridges=(length_m - width_m) * M_TO_FT, hips=4 * hip * M_TO_FT, valleys=0.0,
        rakes=0.0, eaves=2 * (length_m + width_m) * M_TO_FT,
        surface_sqft=length_m * width_m * SQM_TO_SQFT / math.cos(math.atan(pitch)),
        facets=4,
    )
    return z, mask, px, truth


def l_hip_roof(pitch=8 / 12, **kw):
    """L-shaped hip roof: 20 x 10 main body plus an 8-wide hip wing projecting
    6 m from the back, spanning x in [6, 14]. Wing ridge (x=10, 4p high) runs
    from its hip apex (10,12) to where it dies into the main back plane at
    (10,6); two valleys from (6,10) and (14,10) rise to that point.
    """
    px = kw.get("px", 0.1)
    pad = kw.get("pad", 10)
    base = 3.0
    W = int(16 / px) + 2 * pad
    L = int(20 / px) + 2 * pad
    jj, ii = np.meshgrid(np.arange(L), np.arange(W))
    X = (jj - pad) * px
    Y = (ii - pad) * px
    main = (X >= 0) & (X <= 20) & (Y >= 0) & (Y <= 10)
    wing = (X >= 6) & (X <= 14) & (Y >= 0) & (Y <= 16)
    z_main = np.where(main, base + pitch * np.minimum.reduce([X, 20 - X, Y, 10 - Y]), -np.inf)
    z_wing = np.where(wing, base + pitch * np.minimum.reduce([X - 6, 14 - X, 16 - Y]), -np.inf)
    mask = main | wing
    z = np.where(mask, np.maximum(z_main, z_wing), 0.0).astype(np.float32)
    if kw.get("noise"):
        z = z + np.random.default_rng(kw.get("seed", 0)).normal(0, kw["noise"], z.shape).astype(np.float32)
    # The wing runs the full depth, so its side planes poke through the
    # front eave as a gable (two rakes) and the wing ridge is exposed on both
    # sides of the main ridge (back 6 m, front 4 m).
    s = math.sqrt(2)
    hip_main = math.hypot(5 * s, 5 * pitch)
    hip_wing = math.hypot(4 * s, 4 * pitch)
    valley = math.hypot(4 * s, 4 * pitch)
    rake = math.hypot(4, 4 * pitch)
    truth = dict(
        ridges=(10 + 6 + 4) * M_TO_FT,
        hips=(4 * hip_main + 2 * hip_wing) * M_TO_FT,
        valleys=4 * valley * M_TO_FT,
        rakes=2 * rake * M_TO_FT,
        eaves=(12 + 10 + 6 + 6 + 8 + 6 + 6 + 10) * M_TO_FT,
        surface_sqft=(200 + 48) * SQM_TO_SQFT / math.cos(math.atan(pitch)),
        facets=9,
    )
    return z, mask, px, truth


def t_gable_roof(pitch=8 / 12, **kw):
    """Cross-gable: 16 x 10 body (gable ends on the short sides) with a 6-wide
    gable wing projecting 5 m from the front, centered.
        (0,0)-(5,0)-(5,-5)-(11,-5)-(11,0)-(16,0)-(16,10)-(0,10)
    Wing ridge at x=8 from y=-5 (rake) to the main ridge y=5 (meets at a T);
    two valleys from (5,0) and (11,0) at 45 deg up to (8,3).
    """
    poly = [(0, 0), (5, 0), (5, -5), (11, -5), (11, 0), (16, 0), (16, 10), (0, 10)]
    px = kw.get("px", 0.1)
    pad = kw.get("pad", 10)
    base = 3.0
    W = int(15 / px) + 2 * pad
    L = int(16 / px) + 2 * pad
    jj, ii = np.meshgrid(np.arange(L), np.arange(W))
    X = (jj - pad) * px
    Y = (ii - pad) * px - 5.0
    main = (X >= 0) & (X <= 16) & (Y >= 0) & (Y <= 10)
    wing = (X >= 5) & (X <= 11) & (Y >= -5) & (Y <= 10)
    z_main = np.where(main, base + pitch * np.minimum(Y, 10 - Y), -np.inf)
    z_wing = np.where(wing, base + pitch * np.minimum(X - 5, 11 - X), -np.inf)
    mask = main | wing
    z = np.where(mask, np.maximum(z_main, z_wing), 0.0).astype(np.float32)
    if kw.get("noise"):
        z = z + np.random.default_rng(kw.get("seed", 0)).normal(0, kw["noise"], z.shape).astype(np.float32)
    valley_plan = 3 * math.sqrt(2)
    valley = math.hypot(valley_plan, 3 * pitch)
    rake_main = math.hypot(5, 5 * pitch)
    rake_wing = math.hypot(3, 3 * pitch)
    # Wing runs through the full body depth: its ridge spans y=-5..10 (the
    # main ridge is higher only between y=2..8, so 9 m of it is exposed) and
    # it pokes out of the back eave as a second gable (2 rakes, 4 valleys).
    truth = dict(
        ridges=(16 + 9) * M_TO_FT,
        hips=0.0,
        valleys=4 * valley * M_TO_FT,
        rakes=(4 * rake_main + 4 * rake_wing) * M_TO_FT,
        eaves=(5 + 5 + 5 + 5 + 10) * M_TO_FT,
        surface_sqft=(16 * 10 + 6 * 5) * SQM_TO_SQFT / math.cos(math.atan(pitch)),
        facets=6,
    )
    return z, mask, px, truth
