"""Roof wireframe from the plane graph (EagleView-style line network).

Every facet is a fitted plane. Every ridge / hip / valley is the analytic
intersection line of two adjacent planes, clipped to the extent of their
shared raster boundary, with its true 3D length. The outline is the
regularized footprint polygon; each outline edge is an eave (level along the
edge) or a rake (sloped along the edge), split where the facet behind it
changes (a gable end = two rakes). Endpoints are snapped to shared nodes.

This replaces the straight-skeleton model: it needs no assumption about how
wings meet the main roof, because the DSM already shows which plane is
where.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671

MIN_CREASE_PIX = 6          # boundary pixels needed to form a crease
MIN_CREASE_M = 0.45         # drop creases shorter than this (plan)
NODE_SNAP_M = 0.6           # merge nodes within this distance
OUTLINE_SNAP_M = 0.7        # crease ends within this of the outline snap to it
RIDGE_MAX_SLOPE_DEG = 5.0   # convex crease flatter than this = ridge, else hip
EAVE_MAX_SLOPE_DEG = 7.0    # outline edge with along-edge slope below = eave


@dataclass
class WEdge:
    kind: str          # ridge | hip | valley | eave | rake | flashing
    p0: tuple          # (y, x) raster px
    p1: tuple
    length_ft: float   # 3D length
    plan_ft: float
    facets: tuple = ()  # facet ids involved


@dataclass
class Wireframe:
    outline: np.ndarray
    edges: list = field(default_factory=list)
    topology: dict = field(default_factory=dict)   # framing-rule repairs applied
    face_pitch: dict = field(default_factory=dict)
    face_area_sqft: dict = field(default_factory=dict)
    face_plan_sqft: dict = field(default_factory=dict)
    face_centroids: dict = field(default_factory=dict)
    face_downslope: dict = field(default_factory=dict)
    face_slope_deg: dict = field(default_factory=dict)
    face_azimuth: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.edges) > 0


# ------------------------------------------------------------ outline

def dominant_axis_from_planes(labels: np.ndarray, planes: dict) -> float | None:
    """Roof axis (deg, 0-90, raster frame) from facet plane gradients: eaves
    run perpendicular to a facet's downslope, so the area-weighted circular
    mean of 4x the gradient angle gives the building's grid very robustly."""
    if not planes:
        return None
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    cnt = dict(zip(ids.tolist(), counts.tolist()))
    sx = sy = 0.0
    for fid, (a, b, _c) in planes.items():
        g = math.hypot(a, b)
        if g < 0.08:        # flat: no direction information
            continue
        th = math.atan2(b, a)      # gradient direction in (X, Y)
        wgt = cnt.get(fid, 0) * g
        sx += wgt * math.cos(4 * th)
        sy += wgt * math.sin(4 * th)
    if sx == 0 and sy == 0:
        return None
    # atan2(b, a) and seg_dirs' atan2(dy, dx) share the same convention
    return math.degrees(math.atan2(sy, sx) / 4.0) % 90.0


def regularize_outline(chain: np.ndarray, px_m: float, eps_m: float = 0.25,
                       min_edge_m: float = 0.35, axis_hint: float | None = None) -> np.ndarray:
    """Rectilinear-first regularization of a traced boundary (Nx2 (y, x) px).

    1. Douglas-Peucker at eps_m.
    2. Dominant axis from length-weighted angles mod 90.
    3. Each segment snaps to axis0 + k*90; a segment may keep a 45-degree
       direction only if it is long (>= 2.5 m) and within 12 deg of it.
    4. Consecutive same-direction segments merge; parallel offset neighbors
       get a perpendicular connector (a real jog); vertices are rebuilt from
       line intersections; edges shorter than min_edge_m are absorbed.
    """
    from .vectorize import douglas_peucker

    if len(chain) < 4:
        return chain
    simp = douglas_peucker(np.vstack([chain, chain[:1]]), eps_m / px_m)
    if len(simp) >= 2 and np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    if len(simp) < 3:
        return simp

    def seg_dirs(poly):
        out = []
        n = len(poly)
        for i in range(n):
            v = poly[(i + 1) % n] - poly[i]
            out.append((math.degrees(math.atan2(v[0], v[1])), math.hypot(v[0], v[1])))
        return out

    # dominant axis (0..90)
    votes = np.zeros(180)
    for ang, L in seg_dirs(simp):
        a = ang % 90.0
        votes[int(round(a * 2)) % 180] += L
    # circular smoothing over +-2 deg
    sm = votes.copy()
    for s in (-4, -3, -2, -1, 1, 2, 3, 4):
        sm += np.roll(votes, s) * (1.0 - abs(s) / 5.0)
    axis0 = (int(np.argmax(sm)) / 2.0) % 90.0
    if axis_hint is not None:
        axis0 = axis_hint % 90.0

    poly = simp
    for _ in range(3):
        # ---- snap each segment direction ----
        lines = []  # (class_angle, point, dir, length)
        for i, (ang, L) in enumerate(seg_dirs(poly)):
            rel = (ang - axis0) % 90.0
            d90 = min(rel, 90.0 - rel)
            rel45 = (ang - axis0 - 45.0) % 90.0
            d45 = min(rel45, 90.0 - rel45)
            if L * px_m >= 2.5 and d45 <= 12.0 and d45 < d90:
                snapped = ang - (rel45 if rel45 <= 45 else rel45 - 90.0)
            else:
                snapped = ang - (rel if rel <= 45 else rel - 90.0)
            snapped = snapped % 360.0
            rad = math.radians(snapped)
            u = np.array([math.sin(rad), math.cos(rad)])
            mid = (poly[i] + poly[(i + 1) % len(poly)]) / 2.0
            lines.append([round(snapped % 180.0, 3), mid, u, L])
        # ---- merge consecutive same-direction, nearly collinear segments ----
        merged = []
        for cls, mid, u, L in lines:
            if merged and merged[-1][0] == cls:
                c0, m0, u0, L0 = merged[-1]
                nrm = np.array([-u0[1], u0[0]])
                off = abs((mid - m0) @ nrm)
                if off <= max(0.25 / px_m, 2.0):
                    w = L / (L + L0)
                    merged[-1] = [c0, m0 * (1 - w) + mid * w, u0, L0 + L]
                    continue
            merged.append([cls, mid, u, L])
        if len(merged) > 1 and merged[0][0] == merged[-1][0]:
            c0, m0, u0, L0 = merged[0]
            c1, m1, u1, L1 = merged[-1]
            nrm = np.array([-u0[1], u0[0]])
            if abs((m1 - m0) @ nrm) <= max(0.25 / px_m, 2.0):
                w = L1 / (L0 + L1)
                merged[0] = [c0, m0 * (1 - w) + m1 * w, u0, L0 + L1]
                merged.pop()
        # ---- insert perpendicular connectors between parallel offset lines ----
        final = []
        m = len(merged)
        for i in range(m):
            cls, mid, u, L = merged[i]
            final.append([cls, mid, u, L])
            cls2, mid2, u2, L2 = merged[(i + 1) % m]
            if cls == cls2:
                # perpendicular connector through the junction of the two spans
                nrm = np.array([-u[1], u[0]])
                # junction x along u: end of this span / start of next
                end_a = mid + u * (L / 2.0)
                start_b = mid2 - u2 * (L2 / 2.0) if (u2 @ u) > 0 else mid2 + u2 * (L2 / 2.0)
                j = (end_a + start_b) / 2.0
                final.append([round((cls + 90.0) % 180.0, 3), j, nrm, 1.0])
        if len(final) < 3:
            return poly
        # ---- rebuild vertices ----
        verts = []
        m = len(final)
        for i in range(m):
            _, p1, d1, _ = final[i]
            _, p2, d2, _ = final[(i + 1) % m]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                verts.append((p1 + p2) / 2.0)
                continue
            t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
            verts.append(p1 + t * d1)
        verts = np.array(verts)
        # ---- absorb sub-threshold edges ----
        n = len(verts)
        lens = np.array([math.hypot(*(verts[(i + 1) % n] - verts[i])) for i in range(n)])
        short = lens * px_m < min_edge_m
        if not short.any() or n <= 4:
            poly = verts
            break
        keep = []
        i = 0
        while i < n:
            if short[i] and n - len([k for k in range(i) if short[k]]) > 4:
                keep.append((verts[i] + verts[(i + 1) % n]) / 2.0)
                i += 2
            else:
                keep.append(verts[i])
                i += 1
        poly = np.array(keep)
    return poly


def complete_occluded_corners(poly: np.ndarray, dsm: np.ndarray, ground: np.ndarray,
                              px_m: float, max_area_m2: float = 20.0,
                              short_m: float = 2.5, long_m: float = 3.5) -> tuple[np.ndarray, int]:
    """Square off convex corners that tree canopy hides.

    Satellite DSMs lose the roof under canopy, so an occluded corner comes
    out as a staircase of short jogs (or a diagonal) between two long,
    perpendicular edges. When that staircase turns the convex way (an
    outside corner, not an L-notch), encloses a modest area, and the DSM
    there is elevated (canopy, not ground), the corner is restored at the
    intersection of the two long edges. Returns (poly, corners_restored).
    """
    from matplotlib.path import Path as MplPath

    n = len(poly)
    if n < 5:
        return poly, 0
    h, w = dsm.shape
    pts = [poly[k].astype(float) for k in range(n)]
    lens = [math.hypot(*(pts[(k + 1) % n] - pts[k])) * px_m for k in range(n)]
    dirs = []
    for k in range(n):
        v = pts[(k + 1) % n] - pts[k]
        L = math.hypot(*v)
        dirs.append(v / L if L > 1e-9 else np.array([0.0, 0.0]))
    orient = float(np.sum(poly[:, 1] * np.roll(poly[:, 0], -1) - np.roll(poly[:, 1], -1) * poly[:, 0]))
    sign = 1.0 if orient > 0 else -1.0
    is_long = [L >= long_m for L in lens]
    if sum(is_long) < 2:
        return poly, 0

    restored = 0
    k = 0
    guard = 0
    while guard < 3 * n:
        guard += 1
        n = len(pts)
        if n < 5:
            break
        if not is_long[k]:
            k = (k + 1) % len(pts)
            if guard > 3 * n:
                break
            continue
        # staircase run after long edge k
        run = []
        j = (k + 1) % n
        while not is_long[j] and len(run) < n - 2:
            run.append(j)
            j = (j + 1) % n
        if not run or j == k or any(lens[r] > short_m for r in run):
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        d1, d2 = dirs[k], dirs[j]
        if abs(d1 @ d2) > 0.3:             # need perpendicular long edges
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        cross = d1[1] * d2[0] - d1[0] * d2[1]   # (x, y) cross product
        if cross * sign <= 0:                # reflex turn = inside notch: skip
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        a = pts[run[0]]                      # end of long edge k
        b = pts[j]                           # start of long edge j
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            k = (k + 1) % n
            continue
        t = ((b[0] - a[0]) * d2[1] - (b[1] - a[1]) * d2[0]) / den
        corner = a + d1 * t
        region = np.array([pts[r] for r in run] + [b, corner])
        rx, ry = region[:, 1], region[:, 0]
        area_m2 = abs(float(np.sum(rx * np.roll(ry, -1) - np.roll(rx, -1) * ry))) / 2.0 * px_m * px_m
        if area_m2 < 0.5 or area_m2 > max_area_m2:
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        # canopy test over the restored region
        y0, y1 = int(max(region[:, 0].min(), 0)), int(min(region[:, 0].max() + 1, h))
        x0, x1 = int(max(region[:, 1].min(), 0)), int(min(region[:, 1].max() + 1, w))
        if y1 - y0 < 2 or x1 - x0 < 2:
            k = (k + 1) % n
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = MplPath(region[:, ::-1]).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
        if inside.sum() < 4:
            k = (k + 1) % n
            continue
        nd = (dsm[y0:y1, x0:x1] - ground[y0:y1, x0:x1])[inside]
        if float(np.median(nd)) <= 2.0:
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        # restore: replace staircase vertices by the corner
        new_pts = []
        for idx in range(n):
            if idx in run:
                continue
            if idx == j:
                new_pts.append(corner)
            new_pts.append(pts[idx])

        def _area(P):
            P = np.asarray(P)
            return abs(float(np.sum(P[:, 1] * np.roll(P[:, 0], -1) - np.roll(P[:, 1], -1) * P[:, 0]))) / 2.0

        gain = (_area(new_pts) - _area(pts)) * px_m * px_m
        if not (0.3 * area_m2 <= gain <= 1.05 * area_m2):
            # the replacement did not simply add the corner region: skip it
            k = (k + 1) % n
            if k == 0 and guard > n:
                break
            continue
        pts = new_pts
        n = len(pts)
        lens = [math.hypot(*(pts[(q + 1) % n] - pts[q])) * px_m for q in range(n)]
        dirs = []
        for q in range(n):
            v = pts[(q + 1) % n] - pts[q]
            L = math.hypot(*v)
            dirs.append(v / L if L > 1e-9 else np.array([0.0, 0.0]))
        is_long = [L >= long_m for L in lens]
        restored += 1
        k = 0
    return np.array(pts), restored


def extend_labels_to_outline(labels: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Give unlabeled pixels inside the outline polygon (restored corners)
    the label of the nearest labeled pixel."""
    from matplotlib.path import Path as MplPath

    h, w = labels.shape
    y0, y1 = int(max(outline[:, 0].min() - 1, 0)), int(min(outline[:, 0].max() + 2, h))
    x0, x1 = int(max(outline[:, 1].min() - 1, 0)), int(min(outline[:, 1].max() + 2, w))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = MplPath(outline[:, ::-1]).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
    out = labels.copy()
    todo = np.zeros((h, w), dtype=bool)
    todo[y0:y1, x0:x1] = inside & (labels[y0:y1, x0:x1] == 0)
    for _ in range(400):
        if not todo.any():
            break
        grown = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(out, dy, 0), dx, 1)
            fill = todo & (grown == 0) & (nb > 0)
            grown[fill] = nb[fill]
        if np.array_equal(grown, out):
            break
        out = grown
        todo &= out == 0
    return out


# ------------------------------------------------------------ helpers

def _clusters(pixels):
    pset = set(pixels)
    seen = set()
    out = []
    for p in pixels:
        if p in seen:
            continue
        q = deque([p]); seen.add(p); cl = []
        while q:
            y, x = q.popleft(); cl.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    r = (y + dy, x + dx)
                    if r in pset and r not in seen:
                        seen.add(r); q.append(r)
        out.append(cl)
    return out


def _plane_z(p, X, Y):
    return p[0] * X + p[1] * Y + p[2]


def _side_sign(labels, i, j, cluster, nhat, px_x, px_y) -> float:
    """Positive if facet i lies on the +nhat side of its boundary with j.

    Votes over 4-adjacent (i, j) pixel pairs along the boundary cluster: the
    vector from the i pixel to the j pixel points out of facet i, so its
    projection on the normal is negative when i is on the + side."""
    h, w = labels.shape
    out = 0.0
    for (y, x) in cluster:
        li = int(labels[y, x])
        if li not in (i, j):
            continue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            ln = int(labels[ny, nx])
            if ln == li or ln not in (i, j):
                continue
            v = (dx * px_x, dy * px_y)                 # from (y, x) toward the neighbour
            proj = v[0] * nhat[0] + v[1] * nhat[1]
            out += -proj if li == i else proj          # i -> j negative when i on + side
    return out


def _point_segment_dist(p, a, b):
    v = b - a
    L2 = float(v @ v)
    if L2 < 1e-12:
        return float(np.hypot(*(p - a))), a
    t = max(0.0, min(1.0, float((p - a) @ v) / L2))
    q = a + v * t
    return float(np.hypot(*(p - q))), q


# ------------------------------------------------------------ main

def build_wireframe(labels: np.ndarray, planes: dict, outline: np.ndarray,
                    px_x: float, px_y: float) -> Wireframe:
    h, w = labels.shape
    px = (px_x + px_y) / 2.0
    ft = px * M_TO_FT
    wf = Wireframe(outline=outline)
    if len(outline) < 3 or not planes:
        return wf

    # ---- shared boundary pixels per facet pair, plus which side is which ----
    pair_pix: dict[tuple[int, int], list] = defaultdict(list)
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: h - dy or None, : w - dx or None]
        b = labels[dy:, dx:]
        mm = (a > 0) & (b > 0) & (a != b)
        ys, xs = np.where(mm)
        for y, x in zip(ys.tolist(), xs.tolist()):
            i, j = int(a[y, x]), int(b[y, x])
            pair_pix[(min(i, j), max(i, j))].append((y, x))

    creases = []
    for (i, j), pix in pair_pix.items():
        if i not in planes or j not in planes:
            continue
        pi, pj = planes[i], planes[j]
        na = (pi[0] - pj[0], pi[1] - pj[1])
        nn = math.hypot(*na)
        for cl in _clusters(pix):
            if len(cl) < MIN_CREASE_PIX:
                continue
            P = np.array(cl, dtype=float)                 # (y, x) px
            Xm, Ym = P[:, 1] * px_x, P[:, 0] * px_y       # metres
            # A crease has both planes at the same height along the boundary;
            # a wall (step) has a consistent height gap between them there.
            zi_b = _plane_z(pi, Xm, Ym)
            zj_b = _plane_z(pj, Xm, Ym)
            gaps = np.abs(zi_b - zj_b)
            gap = float(np.median(gaps))
            asp_i = math.atan2(-pi[0], pi[1]); asp_j = math.atan2(-pj[0], pj[1])
            dasp = abs((math.degrees(asp_i - asp_j) + 180.0) % 360.0 - 180.0)
            same_aspect = dasp < 35.0
            # a wall between two sections facing the same way tapers from the
            # ridge (no gap) to the eave (full gap): judge it by its high end
            step_line = nn < 0.02 or gap > 0.45 or (same_aspect and float(np.percentile(gaps, 90)) > 0.6)
            if step_line:
                # A step between two planes is a wall. EagleView convention:
                #   upper roof's edge along the wall  -> rake (if the upper
                #     plane slopes along the wall) or eave (if it slopes
                #     toward the wall); these count in rakes / eaves.
                #   lower roof against the wall        -> step flashing (if
                #     the lower plane slopes along the wall) or flashing
                #     (headwall: lower plane slopes away from the wall).
                c = np.column_stack([Ym, Xm]).mean(axis=0)
                cc = np.column_stack([Ym, Xm]) - c
                ev, evec = np.linalg.eigh(cc.T @ cc / len(cc))
                u_yx = evec[:, int(np.argmax(ev))]
                proj = cc @ u_yx
                p0m = c + u_yx * proj.min(); p1m = c + u_yx * proj.max()
                plan_m = float(proj.max() - proj.min())
                if plan_m < MIN_CREASE_M:
                    continue
                ux_, uy_ = float(u_yx[1]), float(u_yx[0])          # along-wall (X, Y)
                zi_c = _plane_z(pi, c[1], c[0]); zj_c = _plane_z(pj, c[1], c[0])
                hi_p, lo_p = (pi, pj) if zi_c >= zj_c else (pj, pi)
                hi_f, lo_f = (i, j) if zi_c >= zj_c else (j, i)
                step_m = abs(zi_c - zj_c)
                if step_m < 0.5:
                    kind_hi = "flashing"          # minor offset: just a flashing line
                    kind_lo = None
                else:
                    g_hi = hi_p[0] * ux_ + hi_p[1] * uy_
                    along_hi = math.degrees(math.atan(abs(g_hi)))
                    kind_hi = "rake" if along_hi > EAVE_MAX_SLOPE_DEG else "eave"
                    g_lo = lo_p[0] * ux_ + lo_p[1] * uy_
                    along_lo = math.degrees(math.atan(abs(g_lo)))
                    kind_lo = "step_flashing" if along_lo > EAVE_MAX_SLOPE_DEG else "flashing"
                p0 = (p0m[0] / px_y, p0m[1] / px_x); p1 = (p1m[0] / px_y, p1m[1] / px_x)
                len_hi = plan_m * math.sqrt(1.0 + (hi_p[0] * ux_ + hi_p[1] * uy_) ** 2) if kind_hi == "rake" else plan_m
                if kind_hi:
                    creases.append(dict(kind=kind_hi, p0=p0, p1=p1, plan_m=plan_m, len_m=len_hi,
                                        facets=(hi_f, lo_f), planes=(hi_p, lo_p), wall=True))
                if kind_lo:
                    len_lo = plan_m * math.sqrt(1.0 + (lo_p[0] * ux_ + lo_p[1] * uy_) ** 2) if kind_lo == "step_flashing" else plan_m
                    creases.append(dict(kind=kind_lo, p0=p0, p1=p1, plan_m=plan_m, len_m=len_lo,
                                        facets=(lo_f, hi_f), planes=(lo_p, hi_p), wall=True))
                continue
            # intersection line in plan: na . (X, Y) + (ci - cj) = 0
            nhat = np.array([na[0], na[1]]) / nn           # (X, Y) components
            u = np.array([-nhat[1], nhat[0]])              # along-line (X, Y)
            # point on line nearest the cluster centroid
            cx, cy = float(Xm.mean()), float(Ym.mean())
            d0 = (na[0] * cx + na[1] * cy + (pi[2] - pj[2])) / nn
            qx, qy = cx - d0 * nhat[0], cy - d0 * nhat[1]
            t = (Xm - qx) * u[0] + (Ym - qy) * u[1]
            tmin, tmax = float(t.min()), float(t.max())
            plan_m = tmax - tmin
            if plan_m < MIN_CREASE_M:
                continue
            e0 = (qx + u[0] * tmin, qy + u[1] * tmin)
            e1 = (qx + u[0] * tmax, qy + u[1] * tmax)
            z0 = _plane_z(pi, *e0); z1 = _plane_z(pi, *e1)
            len_m = math.hypot(plan_m, z1 - z0)
            # which side of the line is facet i? Across every boundary pixel
            # pair the vector from the i pixel to the j pixel points away
            # from facet i; its projection on the line normal votes.
            s = _side_sign(labels, i, j, cl, nhat, px_x, px_y)
            # convex (ridge/hip) if, 1 m off the intersection line on facet i's
            # side, the other plane continued lies ABOVE the roof (plane i);
            # concave (valley) if it lies below.
            step = nhat * (1.0 if s >= 0 else -1.0)
            q = (qx + step[0], qy + step[1])
            zi, zj = _plane_z(pi, *q), _plane_z(pj, *q)
            convex = zj > zi
            along = abs(pi[0] * u[0] + pi[1] * u[1])
            slope_deg = math.degrees(math.atan(along))
            # aspects: a crease needs the two slopes to face different ways;
            # a pitch change on the same slope is a step / flashing line
            if same_aspect:
                kind = "flashing"
            elif not convex:
                kind = "valley"
            elif slope_deg <= RIDGE_MAX_SLOPE_DEG:
                kind = "ridge"
            else:
                kind = "hip"
            creases.append(dict(kind=kind, p0=(e0[1] / px_y, e0[0] / px_x),
                                p1=(e1[1] / px_y, e1[0] / px_x), plan_m=plan_m,
                                len_m=len_m, facets=(i, j), planes=(pi, pj)))

    # ---- snap crease endpoints: to the outline, then to each other ----
    n_out = len(outline)
    snap_out = OUTLINE_SNAP_M / px
    for c in creases:
        for key in ("p0", "p1"):
            p = np.asarray(c[key], dtype=float)
            dv = np.hypot(outline[:, 0] - p[0], outline[:, 1] - p[1])
            k = int(np.argmin(dv))
            if dv[k] <= snap_out:
                c[key] = tuple(outline[k].astype(float)); continue
            best = (snap_out + 1, None)
            for e in range(n_out):
                d, q = _point_segment_dist(p, outline[e].astype(float), outline[(e + 1) % n_out].astype(float))
                if d < best[0]:
                    best = (d, q)
            if best[1] is not None and best[0] <= snap_out:
                c[key] = tuple(best[1])
    pts = [np.asarray(c[k], dtype=float) for c in creases for k in ("p0", "p1")]
    snap = NODE_SNAP_M / px
    nodes: list[np.ndarray] = []
    members: list[list[int]] = []
    assign = []
    for idx, p in enumerate(pts):
        hit = None
        for ni, q in enumerate(nodes):
            if math.hypot(*(p - q)) <= snap:
                hit = ni; break
        if hit is None:
            nodes.append(p.copy()); members.append([idx]); assign.append(len(nodes) - 1)
        else:
            members[hit].append(idx); assign.append(hit)
            nodes[hit] = np.mean([pts[m] for m in members[hit]], axis=0)
    # outline vertices are fixed nodes: a cluster touching one takes it
    for ni, q in enumerate(nodes):
        dv = np.hypot(outline[:, 0] - q[0], outline[:, 1] - q[1])
        k = int(np.argmin(dv))
        if dv[k] <= snap:
            nodes[ni] = outline[k].astype(float)
    for ci, c in enumerate(creases):
        c["p0"] = tuple(nodes[assign[2 * ci]]); c["p1"] = tuple(nodes[assign[2 * ci + 1]])

    # ---- framing topology: every line meets another line at a point ----
    creases, wf.topology = enforce_topology(creases, outline, labels, px_x, px_y)

    # ---- build crease edges (recompute 3D length after snapping) ----
    for c in creases:
        p0, p1 = np.asarray(c["p0"]), np.asarray(c["p1"])
        plan_m = math.hypot((p1[0] - p0[0]) * px_y, (p1[1] - p0[1]) * px_x)
        if plan_m < MIN_CREASE_M:
            continue
        if c["kind"] in ("hip", "valley"):
            pi = c["planes"][0]
            z0 = _plane_z(pi, p0[1] * px_x, p0[0] * px_y)
            z1 = _plane_z(pi, p1[1] * px_x, p1[0] * px_y)
            len_m = math.hypot(plan_m, z1 - z0)
        elif c.get("wall") and c["plan_m"] > 0:
            len_m = plan_m * (c["len_m"] / c["plan_m"])
        else:
            len_m = plan_m
        wf.edges.append(WEdge(kind=c["kind"], p0=tuple(p0), p1=tuple(p1),
                              length_ft=round(len_m * M_TO_FT, 1),
                              plan_ft=round(plan_m * M_TO_FT, 1), facets=c["facets"]))

    # ---- outline edges: eave / rake, split where the facet behind changes ----
    # inward normal sign, determined empirically against the label raster
    flip = 1.0
    votes = 0.0
    for e in range(n_out):
        a, b = outline[e].astype(float), outline[(e + 1) % n_out].astype(float)
        v = b - a
        L = math.hypot(*v)
        if L < 4:
            continue
        u = v / L
        cand = np.array([-u[1], u[0]])
        mid = (a + b) / 2.0
        for depth in (4.0, 8.0):
            for sgn in (1.0, -1.0):
                p = mid + cand * depth * sgn
                y, x = int(round(p[0])), int(round(p[1]))
                if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                    votes += sgn * L
    if votes < 0:
        flip = -1.0
    for e in range(n_out):
        a, b = outline[e].astype(float), outline[(e + 1) % n_out].astype(float)
        v = b - a
        L = math.hypot(*v)
        if L < 1e-9:
            continue
        u = v / L
        cand = np.array([-u[1], u[0]])
        # per-edge side test: whichever side has more labeled pixels is inside
        score = 0.0
        for t in (0.25, 0.5, 0.75):
            for depth in (3.0, 6.0, 10.0):
                for sgn in (1.0, -1.0):
                    p = a + v * t + cand * depth * sgn
                    y, x = int(round(p[0])), int(round(p[1]))
                    if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                        score += sgn
        nrm = cand * (flip if score == 0 else (1.0 if score > 0 else -1.0))
        # sample facet labels just inside the edge
        nsamp = max(int(L / 3), 4)
        ts = (np.arange(nsamp) + 0.5) / nsamp
        labs = []
        for t in ts:
            lab = 0
            for depth in (3, 6, 10, 15):
                p = a + v * t + nrm * depth
                y, x = int(round(p[0])), int(round(p[1]))
                if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                    lab = int(labels[y, x]); break
            labs.append(lab)
        # classify every sample by the along-edge gradient of its facet plane
        ux_m, uy_m = u[1] * px_x, u[0] * px_y
        um = math.hypot(ux_m, uy_m)
        ux_m, uy_m = ux_m / um, uy_m / um
        kinds_s, grads = [], []
        for lab in labs:
            if lab in planes:
                pa = planes[lab]
                g = pa[0] * ux_m + pa[1] * uy_m
                kinds_s.append("eave" if math.degrees(math.atan(abs(g))) <= EAVE_MAX_SLOPE_DEG else "rake")
                grads.append(g)
            else:
                kinds_s.append("eave"); grads.append(0.0)
        # runs of identical kind -> sub-edges; rakes additionally split where
        # the gradient sign flips (gable peak)
        runs = []
        start = 0
        for k in range(1, nsamp + 1):
            flip_ = (k < nsamp and kinds_s[k] == "rake" and kinds_s[start] == "rake"
                     and grads[k] * grads[start] < 0)
            if k == nsamp or kinds_s[k] != kinds_s[start] or flip_:
                runs.append((start, k, kinds_s[start], float(np.median(grads[start:k])), labs[start]))
                start = k
        # absorb runs shorter than 0.9 m into the larger neighbor
        min_run = 0.9 / (L * px / nsamp)
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for idx, (s0, s1, kind, g, lab) in enumerate(runs):
                if s1 - s0 < min_run:
                    nb = idx - 1 if idx > 0 else idx + 1
                    if idx > 0 and idx + 1 < len(runs):
                        nb = idx - 1 if (runs[idx - 1][1] - runs[idx - 1][0]) >= (runs[idx + 1][1] - runs[idx + 1][0]) else idx + 1
                    ns0, ns1, nk, ng, nl = runs[nb]
                    merged_run = (min(s0, ns0), max(s1, ns1), nk, ng, nl)
                    lo, hi = sorted((idx, nb))
                    runs = runs[:lo] + [merged_run] + runs[hi + 1:]
                    changed = True
                    break
        sub = []
        for s0, s1, kind, g, lab in runs:
            t0 = 0.0 if s0 == 0 else s0 / nsamp
            t1 = 1.0 if s1 == nsamp else s1 / nsamp
            sub.append((t0, t1, kind, g if kind == "rake" else 0.0, lab))
        # eaves are not split by facet changes (EagleView counts per outline
        # segment); rakes are (gable end = 2 rakes)
        merged = []
        for t0, t1, kind, g, lab in sub:
            if merged and kind == "eave" and merged[-1][2] == "eave":
                merged[-1] = (merged[-1][0], t1, "eave", 0.0, lab)
            else:
                merged.append((t0, t1, kind, g, lab))
        for t0, t1, kind, g, lab in merged:
            p0, p1 = a + v * t0, a + v * t1
            plan_m = math.hypot((p1[0] - p0[0]) * px_y, (p1[1] - p0[1]) * px_x)
            if plan_m < 0.3:
                continue
            len_m = plan_m * math.sqrt(1.0 + g * g) if kind == "rake" else plan_m
            wf.edges.append(WEdge(kind=kind, p0=tuple(p0), p1=tuple(p1),
                                  length_ft=round(len_m * M_TO_FT, 1),
                                  plan_ft=round(plan_m * M_TO_FT, 1), facets=(lab,)))

    # ---- faces ----
    px_area = px_x * px_y
    for fid, (a, b, c) in planes.items():
        sel = labels == fid
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        slope = math.degrees(math.atan(math.hypot(a, b)))
        plan_sqft = cnt * px_area * SQM_TO_SQFT
        wf.face_plan_sqft[fid] = plan_sqft
        wf.face_area_sqft[fid] = plan_sqft / max(math.cos(math.radians(slope)), 0.2)
        wf.face_slope_deg[fid] = slope
        rise = round(math.tan(math.radians(slope)) * 12)
        wf.face_pitch[fid] = f"{rise}/12"
        ys, xs = np.where(sel)
        wf.face_centroids[fid] = (float(ys.mean()), float(xs.mean()))
        g = math.hypot(a, b)
        if g > 1e-6:
            wf.face_downslope[fid] = (-b / g, -a / g)       # (drow, dcol): +row = south
            wf.face_azimuth[fid] = (math.degrees(math.atan2(-a, b)) + 360.0) % 360.0
        else:
            wf.face_downslope[fid] = (0.0, 0.0)
            wf.face_azimuth[fid] = 0.0

    tot = defaultdict(float)
    cnts = defaultdict(int)
    for e in wf.edges:
        tot[e.kind] += e.length_ft
        cnts[e.kind] += 1
    wf.totals = {
        "ridges_ft": round(tot["ridge"]), "hips_ft": round(tot["hip"]),
        "valleys_ft": round(tot["valley"]), "eaves_ft": round(tot["eave"]),
        "rakes_ft": round(tot["rake"]), "flashing_ft": round(tot["flashing"]),
        "step_flashing_ft": round(tot["step_flashing"]),
        "ridge_count": cnts["ridge"], "hip_count": cnts["hip"],
        "valley_count": cnts["valley"], "eave_count": cnts["eave"],
        "rake_count": cnts["rake"], "flashing_count": cnts["flashing"],
        "step_flashing_count": cnts["step_flashing"],
        "plan_area_sqft": round(sum(wf.face_plan_sqft.values())),
        "surface_area_sqft": round(sum(wf.face_area_sqft.values())),
    }
    return wf


# ------------------------------------------------------------ topology rules

MAX_EXTEND_M = 3.0          # a dangling crease end may be extended this far to meet a line
VALLEY_TO_EAVE_M = 4.5      # a valley's low end may be extended this far to reach the eave
DROP_DANGLING_M = 2.0       # an unrepairable dangling crease shorter than this is dropped


def _seg_intersect(p, d, a, b):
    """Parameter t >= 0 along ray p + t*d where it crosses segment a-b, or None."""
    e = b - a
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    w = a - p
    t = (w[0] * e[1] - w[1] * e[0]) / den
    u = (w[0] * d[1] - w[1] * d[0]) / den
    if t < 1e-6 or u < -1e-6 or u > 1 + 1e-6:
        return None
    return t


def enforce_topology(creases: list, outline: np.ndarray, labels: np.ndarray,
                     px_x: float, px_y: float) -> tuple[list, dict]:
    """Repair the crease graph to the rules of conventional roof framing:

      * every ridge, hip and valley ends where it meets another line: a hip
        starts at an outline corner and ends at the point that begins a
        ridge; a valley starts at the eave and ends at a ridge/hip junction;
        a ridge ends at a hip pair, a valley pair or a gable end (outline);
      * a line that touches nothing at either end does not exist on a roof.

    A crease end that dangles inside the roof is extended along its own
    line until it meets another crease or the outline (within a few feet,
    and only through roof pixels). A valley's low end is pushed to the eave.
    A short crease that still cannot be connected is dropped. Changes are
    counted in the returned info dict for the report notes."""
    px = (px_x + px_y) / 2.0
    h, w = labels.shape
    n_out = len(outline)
    out_pts = outline.astype(float)
    info = dict(extended=0, dropped=0, extended_ft=0.0, dropped_ft=0.0)
    if not creases:
        return creases, info

    def on_outline(p):
        p = np.asarray(p, float)
        if np.min(np.hypot(out_pts[:, 0] - p[0], out_pts[:, 1] - p[1])) <= 1.5:
            return True
        for e in range(n_out):
            d, _q = _point_segment_dist(p, out_pts[e], out_pts[(e + 1) % n_out])
            if d <= 1.5:
                return True
        return False

    def inside(p):
        y, x = int(round(p[0])), int(round(p[1]))
        y0, y1 = max(y - 2, 0), min(y + 3, h); x0, x1 = max(x - 2, 0), min(x + 3, w)
        return bool((labels[y0:y1, x0:x1] > 0).any())

    def degree(p, skip):
        p = np.asarray(p, float)
        n = 0
        for k, c in enumerate(creases):
            if k == skip or c.get("_drop"):
                continue
            for key in ("p0", "p1"):
                if math.hypot(*(np.asarray(c[key], float) - p)) <= 1.0:
                    n += 1
                    break
        return n

    def plane_z_at(c, p):
        return _plane_z(c["planes"][0], p[1] * px_x, p[0] * px_y)

    for _pass in range(2):
        for ci, c in enumerate(creases):
            if c.get("_drop") or c["kind"] not in ("ridge", "hip", "valley"):
                continue
            P = {k: np.asarray(c[k], float) for k in ("p0", "p1")}
            dang = {k: (not on_outline(P[k])) and degree(P[k], ci) == 0 for k in P}
            plan_m = math.hypot((P["p1"][0] - P["p0"][0]) * px_y, (P["p1"][1] - P["p0"][1]) * px_x)
            if dang["p0"] and dang["p1"] and plan_m < DROP_DANGLING_M:
                c["_drop"] = True                      # short isolated line: not a roof line
                info["dropped"] += 1; info["dropped_ft"] += plan_m * M_TO_FT
                continue
            for key in ("p0", "p1"):
                if not dang[key]:
                    continue
                other = "p1" if key == "p0" else "p0"
                d = P[key] - P[other]
                L = math.hypot(*d)
                if L < 1e-6:
                    continue
                d = d / L
                limit_m = MAX_EXTEND_M
                if c["kind"] == "valley" and plane_z_at(c, P[key]) < plane_z_at(c, P[other]):
                    limit_m = VALLEY_TO_EAVE_M        # low end of a valley must reach the eave
                limit = limit_m / px
                best = None
                for e in range(n_out):               # the outline
                    t = _seg_intersect(P[key], d, out_pts[e], out_pts[(e + 1) % n_out])
                    if t is not None and t <= limit and (best is None or t < best[0]):
                        best = (t, "outline")
                for cj, o in enumerate(creases):      # other creases
                    if cj == ci or o.get("_drop"):
                        continue
                    t = _seg_intersect(P[key], d, np.asarray(o["p0"], float), np.asarray(o["p1"], float))
                    if t is not None and t <= limit and (best is None or t < best[0]):
                        best = (t, cj)
                if best is not None:
                    q = P[key] + d * best[0]
                    mid = P[key] + d * best[0] / 2.0
                    if inside(mid) and inside(q):
                        c[key] = tuple(q); P[key] = q
                        info["extended"] += 1; info["extended_ft"] += float(best[0] * px * M_TO_FT)
                        continue
                if plan_m < DROP_DANGLING_M:
                    c["_drop"] = True
                    info["dropped"] += 1; info["dropped_ft"] += plan_m * M_TO_FT
                    break
    kept = [c for c in creases if not c.get("_drop")]
    info["unresolved"] = sum(
        1 for c in kept if c["kind"] in ("ridge", "hip", "valley")
        and any((not on_outline(c[k])) and degree(c[k], creases.index(c)) == 0 for k in ("p0", "p1")))
    info["extended_ft"] = round(info["extended_ft"], 1)
    info["dropped_ft"] = round(info["dropped_ft"], 1)
    return kept, info


# ------------------------------------------------------------ facet polygons

def facet_polygons(labels: np.ndarray, wf: Wireframe, px_m: float) -> dict:
    """fid -> Nx2 (y, x) polygon for each facet, vertices snapped to the
    wireframe nodes / lines so facets tile the diagram cleanly."""
    from .vectorize import douglas_peucker, trace_boundary

    nodes = [np.asarray(p, dtype=float) for e in wf.edges for p in (e.p0, e.p1)]
    segs = [(np.asarray(e.p0, dtype=float), np.asarray(e.p1, dtype=float)) for e in wf.edges]
    node_snap = 0.9 / px_m
    seg_snap = 0.5 / px_m
    out = {}
    for fid in wf.face_area_sqft:
        chain = trace_boundary(labels == fid)
        if len(chain) < 6:
            continue
        simp = douglas_peucker(np.vstack([chain, chain[:1]]), 0.3 / px_m)
        if len(simp) >= 2 and np.allclose(simp[0], simp[-1]):
            simp = simp[:-1]
        if len(simp) < 3:
            continue
        snapped = []
        for p in simp:
            best = None
            if nodes:
                d = [math.hypot(*(p - q)) for q in nodes]
                k = int(np.argmin(d))
                if d[k] <= node_snap:
                    best = nodes[k]
            if best is None:
                bd, bq = seg_snap + 1, None
                for a, b in segs:
                    dd, q = _point_segment_dist(p, a, b)
                    if dd < bd:
                        bd, bq = dd, q
                best = bq if bq is not None and bd <= seg_snap else p
            if not snapped or math.hypot(*(best - snapped[-1])) > 1.5:
                snapped.append(np.asarray(best, dtype=float))
        if len(snapped) >= 3 and math.hypot(*(snapped[0] - snapped[-1])) <= 1.5:
            snapped.pop()
        if len(snapped) >= 3:
            P = np.array(snapped)
            # snapping can fold a polygon across a wireframe node (self-
            # intersection) and balloon its area: fall back to the raw
            # simplified outline when the area moves more than 25 percent
            def _area(Q):
                return abs(float(np.sum(Q[:, 1] * np.roll(Q[:, 0], -1) - np.roll(Q[:, 1], -1) * Q[:, 0]))) / 2.0
            a_raw = _area(simp)
            if a_raw > 0 and not (0.75 <= _area(P) / a_raw <= 1.25):
                P = simp
            out[fid] = P
    return out
