"""Roof-geometry regularization: fit the measured planes to how US homes are
actually framed, then let the wireframe follow from the rules.

Conventions encoded (conventional stick or truss framing):
  * Every roof plane faces one of the building's four axis directions. The
    only exception is a bay or turret face at 45 degrees, which must be a
    real-sized facet. A "plane" facing 22 or 60 degrees off the house is
    not a roof plane; it is the rounded shoulder of a hip or valley in the
    elevation model and is dissolved into its neighbors.
  * Pitch is a whole number of inches of rise per 12 of run. All planes on
    a wing share the wing's pitch, so hips and valleys between equal-pitch
    planes run at exactly 45 degrees in plan and ridges sit centred.
  * Low-slope roofs (0/12 to 2/12) keep their measured direction; they are
    not part of the pitched system.
  * No gambrels. A residential roof plane never changes pitch part way down
    the slope (that is a barn or shed profile). Two adjacent facets facing
    the same way at different pitches, with no height step between them,
    are one plane that region growing split; they are merged at the wing's
    pitch. A porch or addition at a lower pitch meets the main roof at a
    WALL (height step), which keeps it separate.
  * Every ridge is an A-peak between exactly two slopes.

Each regularized plane keeps only its height offset from the data (least
squares over its pixels); its direction and pitch come from the rules.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

AXIS_TOL_DEG = 16.0       # snap to a building axis within this
BAY_TOL_DEG = 10.0        # snap to a 45-degree axis within this ...
BAY_MIN_M2 = 6.0          # ... only for a facet at least this big
PITCH_CONSENSUS = 1       # rises within +-1 of the wing's pitch become the wing's pitch
LOW_SLOPE_RISE = 2        # <= 2/12 is low slope: direction not regularized
CONNECTOR_MIN_OFFSET_M = 0.25  # parallel planes offset by at least this need a connector facet
PARALLEL_MERGE_MAX_GAP_M = 0.6  # parallel same-pitch neighbours offset less than this, with a
                                # continuous surface between them, are one plane
GAMBREL_STEP_M = 0.45          # same-aspect facets whose planes differ by less than this at
                               # the shared boundary are one plane (no gambrel breaks)


def _grad_dir(a: float, b: float) -> float:
    """Gradient direction in the X-Y plane (deg, 0..360)."""
    return math.degrees(math.atan2(b, a)) % 360.0


def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def dominant_axis(labels: np.ndarray, planes: dict) -> float:
    """Building axis (deg, 0..90) from area-weighted facet gradient directions."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    cnt = dict(zip(ids.tolist(), counts.tolist()))
    sx = sy = 0.0
    for fid, (a, b, _c) in planes.items():
        g = math.hypot(a, b)
        if g < math.tan(math.radians(14.0)):   # ~3/12: flat planes carry no axis
            continue
        th = math.atan2(b, a)
        w = cnt.get(fid, 0) * g
        sx += w * math.cos(4 * th)
        sy += w * math.sin(4 * th)
    if sx == 0 and sy == 0:
        return 0.0
    return math.degrees(math.atan2(sy, sx) / 4.0) % 90.0


def regularize(labels: np.ndarray, planes: dict, X: np.ndarray, Y: np.ndarray,
               Z: np.ndarray, px: float) -> tuple[np.ndarray, dict, dict]:
    """Return (labels, planes, info). Planes are rebuilt from snapped
    direction + pitch with a refitted offset; unsnappable facets are
    dissolved into neighbors by best plane fit. info holds the axis and
    the consensus pitch for the notes."""
    labels = labels.copy()
    planes = dict(planes)
    axis = dominant_axis(labels, planes)
    axes4 = [(axis + k * 90.0) % 360.0 for k in range(4)]
    axes45 = [(axis + 45.0 + k * 90.0) % 360.0 for k in range(4)]
    px_area = px * px

    # ---- per-facet measured direction and integer rise ----
    meas = {}
    for fid, (a, b, c) in planes.items():
        n = int((labels == fid).sum())
        slope = math.degrees(math.atan(math.hypot(a, b)))
        rise = int(round(math.tan(math.radians(slope)) * 12))
        meas[fid] = dict(n=n, dir=_grad_dir(a, b), rise=rise, area_m2=n * px_area)

    # ---- wing consensus pitch: area-weighted mode of pitched facets ----
    votes = defaultdict(float)
    for fid, m in meas.items():
        if m["rise"] > LOW_SLOPE_RISE:
            votes[m["rise"]] += m["area_m2"]
    consensus = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None

    # ---- decide each facet's snapped direction / rise ----
    decided = {}
    dissolve = []
    for fid, m in meas.items():
        if m["rise"] <= LOW_SLOPE_RISE:
            decided[fid] = (m["dir"], m["rise"])          # low slope: keep as measured
            continue
        d4 = min(axes4, key=lambda ax: _ang_dist(ax, m["dir"]))
        d45 = min(axes45, key=lambda ax: _ang_dist(ax, m["dir"]))
        if _ang_dist(d4, m["dir"]) <= AXIS_TOL_DEG:
            direction = d4
        elif _ang_dist(d45, m["dir"]) <= BAY_TOL_DEG and m["area_m2"] >= BAY_MIN_M2:
            direction = d45
        else:
            dissolve.append(fid)
            continue
        rise = m["rise"]
        if consensus is not None and abs(rise - consensus) <= PITCH_CONSENSUS:
            rise = consensus
        decided[fid] = (direction, max(rise, 1))

    # ---- off-axis facets: connector or shoulder? ----
    # Two parallel planes of the same pitch at different heights cannot meet
    # along a line on a continuous roof; framing puts a sideways-facing
    # connector facet between them (bounded by a hip and a valley). An
    # off-axis strip between two such planes is that connector: snap it to
    # the perpendicular axis. Between planes with no height offset it is a
    # blurred shoulder and dissolves.
    from .planes import _adjacency, _assign_leftovers

    pairs = _adjacency(labels)
    still_dissolve = []
    info_conn = []
    for fid in dissolve:
        nb = sorted([(cnt, (j if i == fid else i)) for (i, j), cnt in pairs.items() if fid in (i, j)],
                    reverse=True)
        nb = [(c, j) for c, j in nb if j in decided]
        connector = False
        if len(nb) >= 2:
            (ci, i), (cj, j) = nb[0], nb[1]
            (di, ri), (dj, rj) = decided[i], decided[j]
            if ri == rj and ri > LOW_SLOPE_RISE and _ang_dist(di, dj) < 1.0:
                # offset between the two parallel planes at the strip
                sel = labels == fid
                cx, cy = float(X[sel].mean()), float(Y[sel].mean())
                g = ri / 12.0
                ax_, by_ = g * math.cos(math.radians(di)), g * math.sin(math.radians(di))
                ni = labels == i
                nj = labels == j
                off_i = float(np.mean(Z[ni] - ax_ * X[ni] - by_ * Y[ni]))
                off_j = float(np.mean(Z[nj] - ax_ * X[nj] - by_ * Y[nj]))
                from .planes import _region_shape
                width_m, _len = _region_shape(sel, px)
                info_conn.append((fid, i, j, round(abs(off_i - off_j), 2), round(meas[fid]["area_m2"], 1),
                                  round(width_m, 2)))
                if abs(off_i - off_j) >= CONNECTOR_MIN_OFFSET_M:
                    perp = [(di + 90.0) % 360.0, (di - 90.0) % 360.0]
                    direction = min(perp, key=lambda ax: _ang_dist(ax, meas[fid]["dir"]))
                    decided[fid] = (direction, ri)
                    connector = True
        if not connector:
            still_dissolve.append(fid)
    dissolve = still_dissolve
    if dissolve:
        keep = {fid: planes[fid] for fid in decided}
        roof = labels > 0
        for fid in dissolve:
            labels[labels == fid] = 0
        labels = _assign_leftovers(labels, keep, roof, X, Y, Z)
        planes = keep

    # ---- rebuild planes: fixed gradient, refitted offset ----
    new_planes = {}
    for fid, (direction, rise) in decided.items():
        sel = labels == fid
        if not sel.any():
            continue
        g = rise / 12.0
        a = g * math.cos(math.radians(direction))
        b = g * math.sin(math.radians(direction))
        c = float(np.mean(Z[sel] - a * X[sel] - b * Y[sel]))
        new_planes[fid] = (a, b, c)

    # ---- merge facets that became the same plane ----
    labels, new_planes = _merge_identical(labels, new_planes, X, Y, Z)
    labels, new_planes, par = _merge_continuous_parallel(labels, new_planes, X, Y, Z, px)
    # ---- no gambrels: same aspect, different pitch, no wall between ----
    labels, new_planes, gambrels = _merge_gambrels(labels, new_planes, X, Y, Z, consensus)
    info = {"axis_deg": axis, "consensus_rise": consensus, "dissolved": len(dissolve),
            "connector_candidates": info_conn, "gambrel_merges": gambrels,
            "parallel_merges": par}
    return labels, new_planes, info


def _boundary_gap_m(labels, i, j, pi, pj, X, Y):
    """Median height difference between planes i and j along their shared
    boundary pixels (0 for a true crease, the wall height for a step)."""
    h, w = labels.shape
    gaps = []
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: h - dy or None, : w - dx or None]
        b = labels[dy:, dx:]
        m = ((a == i) & (b == j)) | ((a == j) & (b == i))
        ys, xs = np.where(m)
        if len(ys):
            x = X[ys, xs]; y = Y[ys, xs]
            zi = pi[0] * x + pi[1] * y + pi[2]
            zj = pj[0] * x + pj[1] * y + pj[2]
            gaps.append(np.abs(zi - zj))
    if not gaps:
        return float("inf")
    return float(np.median(np.concatenate(gaps)))


def _boundary_jump_m(labels, i, j, Z):
    """Median |dZ| between 4-adjacent pixel pairs across the i/j boundary."""
    h, w = labels.shape
    jumps = []
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: h - dy or None, : w - dx or None]
        b = labels[dy:, dx:]
        za = Z[: h - dy or None, : w - dx or None]
        zb = Z[dy:, dx:]
        m = ((a == i) & (b == j)) | ((a == j) & (b == i))
        if m.any():
            jumps.append(np.abs(za[m] - zb[m]))
    if not jumps:
        return float("inf")
    return float(np.median(np.concatenate(jumps)))


def _merge_continuous_parallel(labels, planes, X, Y, Z, px):
    """Two adjacent facets with the same snapped direction and pitch are the
    same roof surface unless a wall stands between them. The elevation model
    decides: a wall shows as a jump across the boundary (well above the
    slope's own rise per pixel); a continuous surface does not, and the
    plane offset between the two fits is a fitting artifact (a blurred
    valley shoulder, a rounded crest) rather than a step. Merge those."""
    from .planes import _adjacency

    merges = 0
    while True:
        pairs = _adjacency(labels)
        done = True
        for (i, j), _cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if i not in planes or j not in planes:
                continue
            ai, bi, ci = planes[i]
            aj, bj, cj = planes[j]
            g = math.hypot(ai, bi)
            if abs(ai - aj) > 1e-6 or abs(bi - bj) > 1e-6 or g * 12 <= LOW_SLOPE_RISE:
                continue
            if abs(ci - cj) > PARALLEL_MERGE_MAX_GAP_M:
                continue
            jump = _boundary_jump_m(labels, i, j, Z)
            if jump > max(1.5 * g * px, 0.06):
                continue                      # a real step: wall between the two
            labels[labels == j] = i
            sel = labels == i
            planes[i] = (ai, bi, float(np.median(Z[sel] - ai * X[sel] - bi * Y[sel])))
            del planes[j]
            merges += 1
            done = False
            break
        if done:
            break
    return labels, planes, merges


def _boundary_direction(labels, i, j, X, Y):
    """Unit direction (X, Y) of the shared boundary between facets i and j."""
    h, w = labels.shape
    pts = []
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: h - dy or None, : w - dx or None]
        b = labels[dy:, dx:]
        m = ((a == i) & (b == j)) | ((a == j) & (b == i))
        ys, xs = np.where(m)
        if len(ys):
            pts.append(np.column_stack([X[ys, xs], Y[ys, xs]]))
    if not pts:
        return None
    P = np.vstack(pts)
    if len(P) < 4:
        return None
    cc = P - P.mean(axis=0)
    ev, evec = np.linalg.eigh(cc.T @ cc)
    u = evec[:, int(np.argmax(ev))]
    return (float(u[0]), float(u[1]))


def _merge_gambrels(labels, planes, X, Y, Z, consensus, aspect_tol_deg: float = 1.0):
    """Merge adjacent pitched facets with the same snapped direction and a
    different rise when their planes meet without a height step. The merged
    plane keeps the direction, takes the wing's consensus rise when either
    side has it (else the larger facet's rise) and refits its offset."""
    from .planes import _adjacency

    merges = 0
    while True:
        pairs = _adjacency(labels)
        done = True
        for (i, j), _cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if i not in planes or j not in planes:
                continue
            ai, bi, ci = planes[i]
            aj, bj, cj = planes[j]
            gi, gj = math.hypot(ai, bi), math.hypot(aj, bj)
            ri, rj = int(round(gi * 12)), int(round(gj * 12))
            if ri <= LOW_SLOPE_RISE or rj <= LOW_SLOPE_RISE or ri == rj:
                continue
            if _ang_dist(_grad_dir(ai, bi), _grad_dir(aj, bj)) > aspect_tol_deg:
                continue
            if _boundary_gap_m(labels, i, j, planes[i], planes[j], X, Y) > GAMBREL_STEP_M:
                continue                      # a wall: porch or addition, not a gambrel
            # A gambrel break runs parallel to the ridge with the flatter
            # plane above. A pitch change whose boundary runs down the slope
            # is a side-by-side section (its sides are step-flashed walls),
            # and a steeper-above / flatter-below break is a porch or
            # overhang transition; both are real and stay separate.
            ux, uy = math.cos(math.radians(_grad_dir(ai, bi))), math.sin(math.radians(_grad_dir(ai, bi)))
            bdir = _boundary_direction(labels, i, j, X, Y)
            if bdir is None or abs(bdir[0] * ux + bdir[1] * uy) > math.sin(math.radians(30.0)):
                continue                      # boundary not parallel to the ridge
            si = labels == i; sj = labels == j
            up_i = float(X[si].mean() * ux + Y[si].mean() * uy)
            up_j = float(X[sj].mean() * ux + Y[sj].mean() * uy)
            upper, lower = (i, j) if up_i > up_j else (j, i)
            if (ri if upper == i else rj) > (rj if upper == i else ri):
                continue                      # steeper above, flatter below: not a gambrel
            ni, nj = int((labels == i).sum()), int((labels == j).sum())
            big, small = (i, j) if ni >= nj else (j, i)
            rise = consensus if consensus in (ri, rj) else (ri if big == i else rj)
            direction = _grad_dir(*planes[big][:2])
            labels[labels == small] = big
            sel = labels == big
            g = rise / 12.0
            a = g * math.cos(math.radians(direction))
            b = g * math.sin(math.radians(direction))
            planes[big] = (a, b, float(np.mean(Z[sel] - a * X[sel] - b * Y[sel])))
            del planes[small]
            merges += 1
            done = False
            break
        if done:
            break
    return labels, planes, merges


def _merge_identical(labels, planes, X, Y, Z, off_tol_m: float = 0.12):
    """Adjacent facets with identical snapped gradient and matching offset are
    one plane (region growing split it, or a dissolved shoulder sat between)."""
    from .planes import _adjacency

    while True:
        pairs = _adjacency(labels)
        merged = False
        for (i, j), _cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if i not in planes or j not in planes:
                continue
            ai, bi, ci = planes[i]
            aj, bj, cj = planes[j]
            if abs(ai - aj) > 1e-6 or abs(bi - bj) > 1e-6:
                continue
            if abs(ci - cj) > off_tol_m:
                continue
            labels[labels == j] = i
            sel = labels == i
            planes[i] = (ai, bi, float(np.mean(Z[sel] - ai * X[sel] - bi * Y[sel])))
            del planes[j]
            merged = True
            break
        if not merged:
            break
    return labels, planes
