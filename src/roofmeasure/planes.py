"""Plane-based roof facet segmentation.

A roof facet is a plane. Instead of growing regions on per-pixel slope and
aspect (which merges parallel-but-offset planes and swallows small facets),
every region is grown against its own fitted plane: a pixel joins only if
its elevation is within `tau` of the plane, and the plane is refitted as the
region grows. Seeds are taken in order of local planarity (lowest local
plane-fit residual first), so clean roof interiors seed before crest zones,
chimneys, or tree canopy.

Output: int32 label raster (0 = unassigned) and {label: (a, b, c)} with
z = a * X + b * Y + c, X/Y in metres (X = col * px_x, Y = row * px_y).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .pipeline_util import box_sum  # noqa: F401  (re-exported helper)

TAU_M = 0.07            # max |z - plane| to join a region (metres)
MIN_REGION_M2 = 1.6     # regions smaller than this are dissolved
MERGE_ANGLE_DEG = 4.0   # coplanar merge: normal angle tolerance
MERGE_OFFSET_M = 0.10   # coplanar merge: plane offset tolerance at the boundary
REFIT_EVERY = 3         # refit plane every N growth rings


def _dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _fit_plane(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[float, float, float]:
    n = len(Z)
    if n > 4000:
        idx = np.random.default_rng(0).choice(n, 4000, replace=False)
        X, Y, Z = X[idx], Y[idx], Z[idx]
    A = np.column_stack([X, Y, np.ones(len(Z))])
    coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def local_fit(dsm: np.ndarray, px_x: float, px_y: float, k: int = 3):
    """Per-pixel local plane (a, b) and RMS residual over a (2k+1)^2 window."""
    h, w = dsm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx * px_x
    Y = yy * px_y
    Z = dsm.astype(np.float64)
    n = float((2 * k + 1) ** 2)
    S = lambda a: box_sum(a, k)  # noqa: E731
    sx, sy, sz = S(X), S(Y), S(Z)
    sxx, syy, sxy = S(X * X), S(Y * Y), S(X * Y)
    sxz, syz, szz = S(X * Z), S(Y * Z), S(Z * Z)
    mx, my, mz = sx / n, sy / n, sz / n
    cxx, cyy, cxy = sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my
    cxz, cyz, czz = sxz / n - mx * mz, syz / n - my * mz, szz / n - mz * mz
    det = cxx * cyy - cxy * cxy
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (cxz * cyy - cyz * cxy) / det
        b = (cyz * cxx - cxz * cxy) / det
        var = czz - (a * cxz + b * cyz)
    a = np.where(np.isfinite(a), a, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    var = np.where(np.isfinite(var), np.maximum(var, 0.0), czz)
    return a, b, np.sqrt(var)


def segment_planes(dsm: np.ndarray, mask: np.ndarray, px_x: float, px_y: float,
                   tau: float = TAU_M) -> tuple[np.ndarray, dict]:
    h, w = dsm.shape
    mask = mask.astype(bool)
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx * px_x
    Y = yy * px_y
    Z = dsm.astype(np.float64)
    la, lb, lres = local_fit(dsm, px_x, px_y, k=3)

    labels = np.zeros((h, w), dtype=np.int32)
    planes: dict[int, tuple[float, float, float]] = {}
    tried = np.zeros((h, w), dtype=bool)   # seeds that produced tiny regions
    px_area = px_x * px_y
    min_px = int(MIN_REGION_M2 / px_area)

    # seed order: lowest local residual first, interior pixels only
    order_res = np.where(mask, lres, np.inf)
    flat_idx = np.argsort(order_res, axis=None)
    next_label = 1
    for fi in flat_idx:
        sy, sx = divmod(int(fi), w)
        if not mask[sy, sx] or labels[sy, sx] or tried[sy, sx]:
            continue
        if not np.isfinite(order_res[sy, sx]):
            break
        a, b = float(la[sy, sx]), float(lb[sy, sx])
        c = float(Z[sy, sx] - a * X[sy, sx] - b * Y[sy, sx])
        region = np.zeros((h, w), dtype=bool)
        region[sy, sx] = True
        free = mask & (labels == 0)
        ring = 0
        while True:
            pred = a * X + b * Y + c
            # steeper planes carry more interpolation error per pixel
            tau_eff = tau + 0.5 * max(px_x, px_y) * math.hypot(a, b)
            ok = free & (np.abs(Z - pred) <= tau_eff)
            grown = region.copy()
            # grow several rings against the current plane before refitting
            for _ in range(REFIT_EVERY):
                nxt = _dilate(grown) & ok
                if np.array_equal(nxt, grown):
                    break
                grown = nxt
            if np.array_equal(grown, region):
                break
            region = grown
            ring += 1
            if region.sum() >= 12:
                a, b, c = _fit_plane(X[region], Y[region], Z[region])
            if ring > 400:
                break
        n = int(region.sum())
        if n < max(min_px, 12):
            tried |= region
            continue
        labels[region] = next_label
        planes[next_label] = (a, b, c)
        next_label += 1

    labels = _assign_leftovers(labels, planes, mask, X, Y, Z)
    labels, planes = _merge_coplanar(labels, planes, X, Y, Z)
    labels, planes = _dissolve_crest_strips(labels, planes, mask, X, Y, Z, px_x)
    labels, planes = _merge_coplanar(labels, planes, X, Y, Z)
    labels, planes = _dissolve_small(labels, planes, mask, X, Y, Z, min_px)
    labels, planes = _merge_small_similar(labels, planes, X, Y, Z, px_area)
    labels = _reassign_edge_band(labels, planes, mask, X, Y, Z, px_x)
    labels, planes = _dissolve_small(labels, planes, mask, X, Y, Z, min_px)
    return labels, planes


def _region_shape(sel: np.ndarray, px: float):
    """(thickness_m, length_m) of a region from area / half-perimeter."""
    n = int(sel.sum())
    edge = sel & ~(np.roll(sel, 1, 0) & np.roll(sel, -1, 0) & np.roll(sel, 1, 1) & np.roll(sel, -1, 1))
    b = int(edge.sum())
    length = max(b / 2.0, 1.0) * px
    thick = n * px * px / length
    return thick, length



def _dissolve_crest_strips(labels, planes, mask, X, Y, Z, px, max_thick_m=2.6, min_elong=2.5):
    """Remove crest-blur 'facets'.

    The DSM rounds off every ridge/hip/valley, and a plane can be fitted to
    that rounded band, producing a thin strip region between the two real
    facets (two creases drawn where there is one). A strip is dissolved when
    it is thin and elongated and EITHER its pixels lie along the intersection
    line of its two dominant neighbors' planes OR its plane normal lies
    between those neighbors' normals (a shoulder between two real planes);
    pixels go to whichever neighbor plane predicts them best. A thin strip
    with one dominant neighbor (edge blur) merges into that neighbor.
    Real narrow facets (dormer cheeks, porch strips) have a normal outside
    the neighbors' cone and survive."""
    changed = True
    while changed:
        changed = False
        pairs = _adjacency(labels)
        for k in sorted(planes):
            sel = labels == k
            if not sel.any():
                continue
            thick, length = _region_shape(sel, px)
            if thick > max_thick_m or length / max(thick, 1e-6) < min_elong:
                continue
            nb = sorted([(cnt, (j if i == k else i)) for (i, j), cnt in pairs.items() if k in (i, j)],
                        reverse=True)
            if not nb:
                continue
            total = sum(c for c, _ in nb)
            if len(nb) == 1 or nb[0][0] > 0.7 * total:
                # edge-blur strip along one facet: merge only if its plane
                # is close to that facet's, it is very thin, or it sits
                # above that plane (a bump). A real narrow facet next to a
                # flat or differently sloped roof sits below and survives.
                tgt = nb[0][1]
                merge_ok = False
                if tgt in planes:
                    pt = planes[tgt]
                    offset = float(np.mean(Z[sel] - (pt[0] * X[sel] + pt[1] * Y[sel] + pt[2])))
                    area_m2 = float(sel.sum()) * px * px
                    merge_ok = (_normal_angle(planes[k], pt) <= 12.0 or thick <= 0.6
                                or (offset > 0.15 and area_m2 < 14.0))
                if merge_ok:
                    labels[sel] = tgt
                    planes.pop(k, None)
                    changed = True
                    break
                continue
            (ci, i), (cj, j) = nb[0], nb[1]
            if i not in planes or j not in planes or (ci + cj) < 0.6 * total:
                continue
            pi, pj, pk = planes[i], planes[j], planes[k]
            na = (pi[0] - pj[0], pi[1] - pj[1])
            nn = math.hypot(*na)
            on_line = False
            a_ij = _normal_angle(pi, pj)
            if nn >= 0.02:
                d = np.abs(na[0] * X[sel] + na[1] * Y[sel] + (pi[2] - pj[2])) / nn
                on_line = np.percentile(d, 80) <= 1.3
            else:
                # neighbors are parallel: the strip is the blurred step (wall)
                # between two offset copies of the same plane, or a seam in
                # one plane. Either way its pixels belong to the nearer one.
                on_line = thick <= 1.0 and length / max(thick, 1e-6) >= 5.0
            # shoulder test: k's normal between i's and j's normals
            a_ik = _normal_angle(pi, pk)
            a_jk = _normal_angle(pj, pk)
            shoulder = (a_ik + a_jk) <= a_ij + 12.0 and thick <= 1.4 and length / max(thick, 1e-6) >= 4.0
            if not (on_line or shoulder):
                continue
            zi = pi[0] * X[sel] + pi[1] * Y[sel] + pi[2]
            zj = pj[0] * X[sel] + pj[1] * Y[sel] + pj[2]
            to_i = np.abs(Z[sel] - zi) <= np.abs(Z[sel] - zj)
            ys, xs = np.where(sel)
            labels[ys[to_i], xs[to_i]] = i
            labels[ys[~to_i], xs[~to_i]] = j
            planes.pop(k, None)
            changed = True
            break
    return labels, planes



def _merge_small_similar(labels, planes, X, Y, Z, px_area, max_m2=8.0, ang=7.0, off=0.20):
    """Small regions (fragments at eaves, chimneys, canopy edges) merge into
    an adjacent facet with a similar plane under a looser tolerance."""
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = {int(i) for i, c in zip(ids, counts) if c * px_area < max_m2}
        if not small:
            break
        pairs = _adjacency(labels)
        merged = False
        for (i, j), cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if i not in planes or j not in planes:
                continue
            if i in small and j in small:
                continue
            if i not in small and j not in small:
                continue
            s_, big = (i, j) if i in small else (j, i)
            if _normal_angle(planes[s_], planes[big]) > ang:
                continue
            sel = labels == s_
            zb = planes[big][0] * X[sel] + planes[big][1] * Y[sel] + planes[big][2]
            if float(np.mean(np.abs(Z[sel] - zb))) > off:
                continue
            labels[sel] = big
            both = labels == big
            planes[big] = _fit_plane(X[both], Y[both], Z[both])
            del planes[s_]
            merged = True
            break
        if not merged:
            break
    return labels, planes



def _assign_leftovers(labels, planes, mask, X, Y, Z):
    """Pixels no plane accepted (crest blur, chimneys, canopy): give each the
    label of the touching region whose plane predicts it best, iteratively."""
    h, w = labels.shape
    for _ in range(200):
        todo = mask & (labels == 0)
        if not todo.any():
            break
        best_lab = np.zeros((h, w), dtype=np.int32)
        best_err = np.full((h, w), np.inf)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(labels, dy, 0), dx, 1)
            cand = todo & (nb > 0)
            if not cand.any():
                continue
            ys, xs = np.where(cand)
            for y, x in zip(ys, xs):
                lab = int(nb[y, x])
                a, b, c = planes[lab]
                err = abs(Z[y, x] - (a * X[y, x] + b * Y[y, x] + c))
                if err < best_err[y, x]:
                    best_err[y, x] = err
                    best_lab[y, x] = lab
        upd = best_lab > 0
        if not upd.any():
            break
        labels = np.where(upd, best_lab, labels)
    return labels



def _adjacency(labels):
    h, w = labels.shape
    pairs = defaultdict(int)
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: h - dy or None, : w - dx or None]
        b = labels[dy:, dx:]
        m = (a > 0) & (b > 0) & (a != b)
        for i, j in zip(a[m].tolist(), b[m].tolist()):
            pairs[(min(i, j), max(i, j))] += 1
    return pairs



def _normal_angle(p, q):
    n1 = np.array([-p[0], -p[1], 1.0]); n1 /= np.linalg.norm(n1)
    n2 = np.array([-q[0], -q[1], 1.0]); n2 /= np.linalg.norm(n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(n1 @ n2)))))



def _merge_coplanar(labels, planes, X, Y, Z):
    while True:
        pairs = _adjacency(labels)
        merged = False
        for (i, j), cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if i not in planes or j not in planes:
                continue
            if _normal_angle(planes[i], planes[j]) > MERGE_ANGLE_DEG:
                continue
            # offset between planes evaluated at the smaller region's centroid
            sel = labels == j
            cx, cy = float(X[sel].mean()), float(Y[sel].mean())
            zi = planes[i][0] * cx + planes[i][1] * cy + planes[i][2]
            zj = planes[j][0] * cx + planes[j][1] * cy + planes[j][2]
            if abs(zi - zj) > MERGE_OFFSET_M:
                continue
            labels[sel] = i
            both = labels == i
            planes[i] = _fit_plane(X[both], Y[both], Z[both])
            del planes[j]
            merged = True
            break
        if not merged:
            break
    return labels, planes



def _dissolve_small(labels, planes, mask, X, Y, Z, min_px):
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = [int(i) for i, c in zip(ids, counts) if c < min_px]
        if not small:
            break
        pairs = _adjacency(labels)
        changed = False
        for s in small:
            nb = [(cnt, (j if i == s else i)) for (i, j), cnt in pairs.items() if s in (i, j)]
            sel = labels == s
            if nb:
                # neighbor whose plane fits these pixels best
                best, best_err = None, np.inf
                for _, cand in nb:
                    if cand not in planes:
                        continue
                    a, b, c = planes[cand]
                    err = float(np.mean(np.abs(Z[sel] - (a * X[sel] + b * Y[sel] + c))))
                    if err < best_err:
                        best, best_err = cand, err
                if best is not None:
                    labels[sel] = best
                    planes.pop(s, None)
                    changed = True
                    continue
            labels[sel] = 0
            planes.pop(s, None)
            changed = True
        if not changed:
            break
    labels = _assign_leftovers(labels, planes, mask, X, Y, Z)
    return labels, planes


def _reassign_edge_band(labels, planes, mask, X, Y, Z, px, band_m=1.3):
    """The DSM blurs every roof edge into a band about a metre wide that reads
    as a gentle slope. A plane seeded on a genuinely flat area can grow all
    the way around the perimeter through that band, producing a RING facet
    (a region enclosing a hole much larger than itself). Such a facet is an
    artifact: its pixels inside the edge band are handed to the other planes
    (by best fit from the interior outward); only its non-band core (e.g. a
    real flat addition) keeps its label.
    """
    import contourpy

    def _area(l):
        x, y = l[:, 0], l[:, 1]
        return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2.0

    r = max(int(round(band_m / px)), 1)
    inner = mask.copy()
    for _ in range(r):
        e = inner.copy()
        e[1:, :] &= inner[:-1, :]; e[:-1, :] &= inner[1:, :]
        e[:, 1:] &= inner[:, :-1]; e[:, :-1] &= inner[:, 1:]
        inner = e
    band = mask & ~inner
    out = labels.copy()
    rings = []
    for k in list(planes):
        sel = out == k
        n = int(sel.sum())
        if n == 0:
            continue
        lines = contourpy.contour_generator(z=np.pad(sel.astype(np.float32), 1)).lines(0.5)
        if len(lines) < 2:
            continue
        areas = sorted((_area(l) for l in lines), reverse=True)
        hole = areas[1]
        if hole > 1.5 * n:           # encloses far more than it covers: a ring
            rings.append(k)
    if not rings:
        return out
    strip = np.zeros_like(out, dtype=bool)
    for k in rings:
        strip |= (out == k) & band
    out[strip] = 0
    # grow the other planes into the stripped band; the ring's own core is
    # hidden during this pass so it cannot reclaim the band
    core = np.zeros_like(out, dtype=bool)
    for k in rings:
        core |= out == k
    tmp = np.where(core, 0, out)
    others = {k: v for k, v in planes.items() if k not in rings}
    tmp = _assign_leftovers(tmp, others, mask & ~core, X, Y, Z)
    out = np.where(core, out, tmp)
    # whatever is still unassigned (band adjacent only to the ring's core)
    # returns to the ring facet
    out = _assign_leftovers(out, planes, mask, X, Y, Z)
    for k in list(planes):
        if not (out == k).any():
            planes.pop(k, None)
    return out
