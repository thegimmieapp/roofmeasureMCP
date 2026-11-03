"""Framing-model roof generator: the ridge / hip / valley graph is derived
from the eave outline the way a framer lays it out, and the elevation model
only supplies what geometry cannot (pitch, gable ends, flat or lower
sections, a fit check).

Rules encoded (production US residential framing, confirmed against 27
production-builder roofs in TX and AZ and the six EagleView reports):

  * Every pitched roof plane rises from an eave edge at the roof pitch. Two
    neighbouring planes meet where they reach the same height, so with one
    pitch every hip and valley runs at 45 degrees from its outline corner
    and every ridge sits centred over its wing. That surface is the
    straight skeleton of the eave polygon (grassfire), and its height at a
    point is pitch x distance to the nearest eave.
  * A hip starts at an outside corner of the eave line and ends at the point
    where a ridge begins. A valley starts at an inside corner and runs to the
    ridge system. A narrower wing's ridge is lower and ends where its two
    valleys meet, on the main slope, joined to the main ridge by a short
    hip; a wing as wide as the main run meets the main ridge at a junction.
  * A gable end is an eave-line edge from which no plane rises: the end
    planes extend to it (rakes) and the ridge runs to it.
  * A flat or low-slope section (porch, patio cover, carport) is not part of
    the pitched system. It is cut out, measured as its own flat facet, and
    the pitched roof keeps its eave along the junction.
  * No gambrels, no facet changes pitch part way down, no diamonds: a roof
    sheds water along straight planes to the eaves.

The result is a connected graph by construction: every line ends where it
meets another line.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

import numpy as np

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671

LOW_SLOPE_RISE = 2            # <= 2/12 is a flat section, outside the pitched system
FLAT_RISE_GAP = 3             # ... and at least this much flatter than the roof around it
FLAT_MIN_M2 = 6.0             # smaller flat regions are ignored (DSM blur)
GABLE_MIN_RISE_M = 0.8        # height variation along an edge that marks a gable end
EAVE_BAND_M = 0.6             # distance inside the outline where eave height is sampled
SPAN_SNAP_M = 0.7             # a gable span this close to an outline corner shares the corner


@dataclass
class GEdge:
    kind: str                 # ridge | hip | valley | eave | rake | flashing
    p0: tuple                 # (y, x) raster px
    p1: tuple
    length_ft: float
    plan_ft: float
    faces: tuple = ()

    @property
    def facets(self) -> tuple:
        return self.faces


@dataclass
class RoofGraph:
    edges: list = field(default_factory=list)
    faces: dict = field(default_factory=dict)      # fid -> Nx2 (y, x) px polygon
    face_plan_sqft: dict = field(default_factory=dict)
    face_area_sqft: dict = field(default_factory=dict)
    face_pitch: dict = field(default_factory=dict)
    face_slope_deg: dict = field(default_factory=dict)
    face_azimuth: dict = field(default_factory=dict)
    face_centroids: dict = field(default_factory=dict)
    face_downslope: dict = field(default_factory=dict)
    face_eave_px: dict = field(default_factory=dict)   # fid -> ((y,x),(y,x)) line the plane rises from
    totals: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    outline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    polys_ready: bool = True          # faces are polygons already (diagrams need no tracing)
    topology: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.edges) > 0


# ------------------------------------------------------------ geometry utils

def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _poly_area(P):
    return abs(_signed_area(P))


def _point_in_poly(pt, P):
    x, y = pt
    inside = False
    n = len(P)
    for i in range(n):
        x0, y0 = P[i]; x1, y1 = P[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xi > x:
                inside = not inside
    return inside


def _edt(mask: np.ndarray, px: float) -> np.ndarray:
    """Euclidean distance (m) from each inside pixel to the nearest outside pixel."""
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(mask) * px
    except Exception:  # pragma: no cover - scipy is a dependency, this is a guard
        h, w = mask.shape
        out = np.where(mask, np.inf, 0.0)
        for _ in range(2):
            for y in range(h):
                for x in range(w):
                    if mask[y, x]:
                        v = out[y, x]
                        if y > 0: v = min(v, out[y - 1, x] + 1)
                        if x > 0: v = min(v, out[y, x - 1] + 1)
                        out[y, x] = v
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if mask[y, x]:
                        v = out[y, x]
                        if y < h - 1: v = min(v, out[y + 1, x] + 1)
                        if x < w - 1: v = min(v, out[y, x + 1] + 1)
                        out[y, x] = v
        return out * px


# ------------------------------------------------------------ DSM queries

def eave_height(dsm: np.ndarray, mask: np.ndarray, px: float, known: np.ndarray | None = None) -> float:
    """Median elevation in the band just inside the outline (the eave line).
    `known` limits the samples to pixels whose elevation is roof (not canopy)."""
    d = _edt(mask, px)
    k = mask if known is None else (mask & known)
    band = k & (d > 0.15) & (d <= EAVE_BAND_M)
    if band.sum() < 20:
        band = k & (d <= EAVE_BAND_M * 2)
    if band.sum() < 20:
        band = mask & (d > 0.15) & (d <= EAVE_BAND_M)
    return float(np.median(dsm[band])) if band.any() else float(np.median(dsm[mask]))


def fit_pitch(dsm: np.ndarray, mask: np.ndarray, px: float, z_eave: float,
              known: np.ndarray | None = None) -> tuple[int, float, float]:
    """Fit rise/12 of a single-pitch hip roof: height above the eave is
    pitch x distance to the outline. Returns (rise, tan, rms_m) with the
    integer rise, the fitted tangent, and the fit residual."""
    d = _edt(mask, px)
    sel = mask & (d > 0.8)
    if known is not None and (sel & known).sum() >= 50:
        sel = sel & known
    if sel.sum() < 50:
        sel = mask & (d > 0.3)
    dd = d[sel]; zz = dsm[sel] - z_eave
    # robust slope: median of z/d over interior pixels, then refine by LSQ on inliers
    ratio = zz / np.maximum(dd, 1e-6)
    t0 = float(np.median(ratio))
    resid = zz - t0 * dd
    inl = np.abs(resid) < max(0.3, 2.0 * float(np.median(np.abs(resid))))
    if inl.sum() > 30:
        t = float(np.sum(dd[inl] * zz[inl]) / max(np.sum(dd[inl] * dd[inl]), 1e-9))
    else:
        t = t0
    t = max(t, 0.0)
    rise = int(round(t * 12))
    rms = float(np.sqrt(np.mean((zz - t * dd) ** 2)))
    return rise, t, rms


def flat_sections(labels: np.ndarray, planes: dict, px: float, roof_rise: int | None = None) -> np.ndarray:
    """Mask of low-slope facets large enough to be a real flat section. A
    section counts as flat only when it is clearly flatter than the roof
    around it (FLAT_RISE_GAP): on a 3/12 roof a facet the plane fit calls
    1/12 is ridge blur, not a porch cover."""
    out = np.zeros(labels.shape, dtype=bool)
    for fid, (a, b, _c) in planes.items():
        r = round(math.hypot(a, b) * 12)
        if r <= LOW_SLOPE_RISE and (roof_rise is None or roof_rise - r >= FLAT_RISE_GAP):
            sel = labels == fid
            if sel.sum() * px * px >= FLAT_MIN_M2:
                out |= sel
    return out


def gable_edges(outline: np.ndarray, dsm: np.ndarray, mask: np.ndarray, z_eave: float,
                px: float, tan: float) -> dict:
    """Gable ends along the outline: {edge index: (t0, t1, peak_m)}.

    Along an eave the roof just inside the edge sits at a constant height.
    Along a gable end it rises from both rakes to the ridge: a tent whose
    height 1 m inside the edge is pitch x distance from the nearer rake, so
    the tent's peak gives the gable's width (2 x peak / pitch) and its
    position gives the centre. A gable can therefore occupy part of a
    straight eave (a wing whose end wall is flush with the eave beside it);
    (t0, t1) is that span as fractions along the edge, snapped to the edge
    ends when within SPAN_SNAP_M of them."""
    n = len(outline)
    h, w = dsm.shape
    out: dict = {}
    for e in range(n):
        a = outline[e].astype(float); b = outline[(e + 1) % n].astype(float)
        v = b - a
        L = math.hypot(*v)
        Lm = L * px
        if Lm < GABLE_MIN_EDGE_M:
            continue
        u = v / L
        nrm = np.array([-u[1], u[0]])
        mid = (a + b) / 2
        inward = None
        for sgn in (1.0, -1.0):
            p = mid + nrm * sgn * (1.2 / px)
            y, x = int(round(p[0])), int(round(p[1]))
            if 0 <= y < h and 0 <= x < w and mask[y, x]:
                inward = sgn
                break
        if inward is None:
            continue
        depth = 1.0 / px
        ts = np.linspace(0.03, 0.97, max(int(Lm / 0.3), 12))
        zs = []
        for t in ts:
            p = a + v * t + nrm * inward * depth
            y, x = int(round(p[0])), int(round(p[1]))
            if 0 <= y < h and 0 <= x < w and mask[y, x]:
                zs.append(dsm[y, x] - z_eave)
            else:
                zs.append(np.nan)
        zs = np.array(zs, dtype=float)
        valid = ~np.isnan(zs)
        if valid.mean() < 0.6:
            continue
        zs = np.clip(zs, -1.0, tan * Lm / 2.0 + 0.8)
        sm = np.array([np.nanmedian(zs[max(i - 1, 0): i + 2]) for i in range(len(zs))])
        hip_lvl = tan * 1.0                      # height 1 m inside a plain eave
        kmax = int(np.nanargmax(sm))
        peak = float(sm[kmax])
        # the tent must rise clearly above what a hip/eave plane gives there
        if peak - hip_lvl < max(GABLE_MIN_RISE_M, 0.35 * tan * Lm / 2.0):
            continue
        # the span: the tent's flanks have the roof pitch, so each flank is a
        # line of slope +/-tan. Fitting the flank (points between 25% and 90%
        # of the peak) and extrapolating to eave level gives the rake position
        # without depending on the blurred top. One metre inside the edge the
        # wing plane and the eave plane meet exactly at the valley foot, so
        # for a flush gable that is where the rake starts; for a wing-wide
        # gable it lands at the outline corner.
        xm = ts * Lm
        lo, hi = 0.25 * peak, 0.9 * peak
        left = [(xm[i], sm[i]) for i in range(0, kmax) if lo <= sm[i] <= hi and not np.isnan(sm[i])]
        right = [(xm[i], sm[i]) for i in range(kmax + 1, len(sm)) if lo <= sm[i] <= hi and not np.isnan(sm[i])]
        if len(left) >= 2:
            xl = float(np.median([x - z / tan for x, z in left]))
        else:
            xl = xm[kmax] - peak / tan
        if len(right) >= 2:
            xr = float(np.median([x + z / tan for x, z in right]))
        else:
            xr = xm[kmax] + peak / tan
        t0 = xl / Lm; t1 = xr / Lm
        if t0 * Lm < SPAN_SNAP_M:
            t0 = 0.0
        if (1.0 - t1) * Lm < SPAN_SNAP_M:
            t1 = 1.0
        t0 = max(0.0, t0); t1 = min(1.0, t1)
        if (t1 - t0) * Lm < GABLE_MIN_EDGE_M:
            continue
        # the flank fit implies a ridge height; the measured top may be blurred
        # down but not by more than about 40%
        h_est = tan * (t1 - t0) * Lm / 2.0
        if peak < 0.6 * h_est:
            continue
        # the profile must really be a tent over the span: the ends of the
        # span sit near eave level, the quarter points about half the peak
        def z_at(t):
            k = int(np.clip(round((t - ts[0]) / (ts[1] - ts[0])), 0, len(sm) - 1))
            return float(sm[k])
        q1, q3 = z_at(t0 + 0.25 * (t1 - t0)), z_at(t0 + 0.75 * (t1 - t0))
        if not (q1 > 0.3 * peak and q3 > 0.3 * peak):
            continue
        if t0 > 0.0 and z_at(max(t0 - 0.6 / Lm, 0.0)) > 0.45 * peak:
            continue                              # does not come back down: not a gable end
        if t1 < 1.0 and z_at(min(t1 + 0.6 / Lm, 1.0)) > 0.45 * peak:
            continue
        if (t1 - t0) * Lm < GABLE_MIN_EDGE_M:
            continue
        out[e] = (round(t0, 3), round(t1, 3), round(peak, 2))
    return out


# ------------------------------------------------------------ skeleton

def _skeleton(poly_m: np.ndarray, tan: float):
    """Faces of the equal-pitch hip roof over poly_m (Nx2 metres, CCW).
    Returns (verts [(x, y, z)], faces [[idx...]]) with the outline vertices
    first, in order, and skeleton nodes appended."""
    from mathutils import Vector
    import bpypolyskel.bpypolyskel as bps

    verts = [Vector((float(x), float(y), 0.0)) for x, y in poly_m]
    faces = bps.polygonize(verts, 0, len(poly_m), None, 0.0, float(tan))
    V = [(v.x, v.y, v.z) for v in verts]
    return V, [list(f) for f in faces]


def _valid_ring(outline_px: np.ndarray) -> bool:
    from shapely.geometry import Polygon
    if len(outline_px) < 3:
        return False
    poly = Polygon(np.column_stack([outline_px[:, 1], -outline_px[:, 0]]))
    return bool(poly.is_valid and poly.area > 0)


def _clean_ring(outline_px: np.ndarray) -> np.ndarray:
    """Drop repeated vertices and exact back-tracks (A-B-A)."""
    P = [np.asarray(outline_px[0], float)]
    for q in outline_px[1:]:
        q = np.asarray(q, float)
        if np.hypot(*(q - P[-1])) < 0.5:
            continue
        if len(P) >= 2 and np.hypot(*(q - P[-2])) < 0.5:
            P.pop(); continue
        P.append(q)
    if len(P) > 1 and np.hypot(*(P[0] - P[-1])) < 0.5:
        P.pop()
    return np.array(P)


def simplify_outline(outline_px: np.ndarray, px: float, min_jog_m: float = 1.0) -> np.ndarray:
    """Jog removal that never returns a self-intersecting ring: the jog
    threshold is relaxed until the result is a valid polygon, falling back
    to the cleaned input."""
    base = _clean_ring(outline_px)
    for mj in (min_jog_m, min_jog_m * 0.6, min_jog_m * 0.3):
        cand = _clean_ring(_simplify_outline_raw(base, px, mj))
        if _valid_ring(cand):
            return cand
    return base if _valid_ring(base) else outline_px


def _simplify_outline_raw(outline_px: np.ndarray, px: float, min_jog_m: float = 1.0) -> np.ndarray:
    """Remove jogs shorter than min_jog_m from a rectilinear outline: a short
    edge between two parallel edges is deleted and the longer neighbour is
    extended to meet the other (the eave line a framer would snap)."""
    P = outline_px.astype(float).copy()
    changed = True
    guard = 0
    while changed and len(P) > 4 and guard < 200:
        changed = False; guard += 1
        n = len(P)
        lens = [math.hypot(*(P[(i + 1) % n] - P[i])) * px for i in range(n)]
        order = sorted(range(n), key=lambda i: lens[i])
        for i in order:
            if lens[i] >= min_jog_m:
                break
            a, b = P[i], P[(i + 1) % n]
            prev_v = P[i] - P[(i - 1) % n]
            next_v = P[(i + 2) % n] - P[(i + 1) % n]
            if math.hypot(*prev_v) < 1e-9 or math.hypot(*next_v) < 1e-9:
                continue
            # neighbours must be (anti)parallel to each other: a jog
            cosang = (prev_v @ next_v) / (math.hypot(*prev_v) * math.hypot(*next_v))
            if abs(cosang) < 0.9:
                continue
            # shift the short edge's endpoints onto the longer neighbour's line
            lp, ln = lens[(i - 1) % n], lens[(i + 1) % n]
            if lp >= ln:
                # extend the previous edge: move b (and the following edge start) onto prev line
                u = prev_v / math.hypot(*prev_v)
                nxt = P[(i + 2) % n]
                t = (nxt - a) @ u
                newpt = a + u * t
                P = np.delete(P, [(i + 1) % n], axis=0)
                j = i % len(P)
                P[(j + 1) % len(P)] = newpt
            else:
                u = next_v / math.hypot(*next_v)
                prv = P[(i - 1) % n]
                t = (prv - b) @ u
                newpt = b + u * t
                P = np.delete(P, [i], axis=0)
                P[(i - 1) % len(P)] = newpt
            changed = True
            break
    # drop collinear / duplicate vertices
    out = []
    n = len(P)
    for i in range(n):
        p0, p1, p2 = P[(i - 1) % n], P[i], P[(i + 1) % n]
        v1, v2 = p1 - p0, p2 - p1
        if math.hypot(*v2) < 1e-6:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if math.hypot(*v1) > 1e-9 and abs(cross) / (math.hypot(*v1) * math.hypot(*v2)) < 0.02:
            continue
        out.append(p1)
    return np.array(out) if len(out) >= 3 else P


def _norm_gables(gables, n: int) -> dict:
    """Accept a set of edge indices (whole-edge gables) or {e: (t0, t1[, peak])}."""
    out = {}
    if not gables:
        return out
    if isinstance(gables, dict):
        for e, v in gables.items():
            t0, t1 = float(v[0]), float(v[1])
            if t1 - t0 > 1e-6:
                out[int(e) % n] = (max(0.0, t0), min(1.0, t1))
    else:
        for e in gables:
            out[int(e) % n] = (0.0, 1.0)
    return out


def build_roof_graph(outline_px: np.ndarray, px_x: float, px_y: float, rise: int,
                     gables=None) -> RoofGraph:
    """Ridge / hip / valley / eave / rake graph of a single-pitch roof over
    the eave outline (Nx2 (y, x) px). `gables` is a set of outline edge
    indices, or {edge: (t0, t1)} with the gable's span along the edge as
    fractions (a wing end wall flush with the eave beside it).

    Gable ends are handled by extend-and-clip: the gable span is pushed
    outward far enough that the wing's hips fall outside the real outline,
    the hip skeleton is computed, and every face is clipped back to the real
    outline. The ridge then runs to the gable line and the side planes meet
    it along two rakes, which is how the wing is framed. For a partial
    gable the pushed span becomes a rectangular tab on the outline; the
    tab's side faces are the wing's own planes and their boundary with the
    eave plane beside them is the valley a framer would cut there."""
    from shapely.geometry import Polygon, LineString, Point

    g = RoofGraph()
    n = len(outline_px)
    if n < 3:
        return g
    gab = _norm_gables(gables, n)
    tan = rise / 12.0
    # metres, y up so that image-CW outlines become CCW consistently
    P = np.column_stack([outline_px[:, 1] * px_x, -outline_px[:, 0] * px_y])
    if _signed_area(P) < 0:
        P = P[::-1].copy()
        gab = {(n - 2 - e) % n: (1.0 - t1, 1.0 - t0) for e, (t0, t1) in gab.items()}
    real = Polygon(P)
    if not real.is_valid:
        fixed = real.buffer(0)
        if fixed.geom_type == "MultiPolygon":
            fixed = max(fixed.geoms, key=lambda q: q.area)
        if fixed.geom_type != "Polygon" or fixed.is_empty:
            return g
        # rebuild the vertex list from the repaired ring so indices stay consistent
        ring = np.array(fixed.exterior.coords[:-1])
        if _signed_area(ring) < 0:
            ring = ring[::-1].copy()
        P = ring; n = len(P); real = Polygon(P)
        gab = {}                                  # edge indices no longer correspond; gables re-detected next pass

    # ---- build the pushed polygon Q with a source tag per Q edge ----
    # tag: ("eave", e, t0, t1) a portion of real edge e; ("gable", e) the pushed
    # gable line; ("side", e) a tab side (the wing plane rises from it)
    Qv: list = []
    Qtag: list = []
    gable_lines = []

    def add(pt, tag):
        if Qv and np.allclose(Qv[-1], pt, atol=1e-6):
            return
        Qv.append(np.asarray(pt, float)); Qtag.append(tag)

    skip_start = False                           # next real edge starts at an already-added pushed corner
    # walk the outline starting at a plain eave so the wrap-around is never a gable
    non_gable = [e for e in range(n) if e not in gab]
    if not non_gable:
        gab = {}
        non_gable = [0]
    off = non_gable[0]
    for e in [(off + i) % n for i in range(n)]:
        a, b = P[e], P[(e + 1) % n]
        d = b - a; L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        if e not in gab:
            if not skip_start:
                add(a, ("eave", e, 0.0, 1.0))
            else:
                Qtag[-1] = ("eave", e, 0.0, 1.0); skip_start = False
            continue
        t0, t1 = gab[e]
        A = a + d * t0; B = a + d * t1
        outward = np.array([d[1], -d[0]]) / L      # right of a CCW edge is outside
        span = float(np.hypot(*(B - A)))
        ext = span / 2.0 + 1.0
        # the pushed tab may not overlap the footprint itself (an inside
        # corner's facing wall); anything outside the footprint, such as a
        # lower porch or garage roof cut out earlier, is fine
        tab = Polygon([tuple(A), tuple(B), tuple(B + outward * ext), tuple(A + outward * ext)])
        if (not tab.is_valid) or tab.intersection(real).area > 0.05 * tab.area:
            if not skip_start:
                add(a, ("eave", e, 0.0, 1.0))
            else:
                Qtag[-1] = ("eave", e, 0.0, 1.0); skip_start = False
            continue                              # cannot push into the footprint
        prev_dir = P[e] - P[(e - 1) % n]; prev_dir /= (np.hypot(*prev_dir) or 1.0)
        next_dir = P[(e + 2) % n] - P[(e + 1) % n]; next_dir /= (np.hypot(*next_dir) or 1.0)
        # start of the span
        if t0 > 1e-6:
            if not skip_start:
                add(a, ("eave", e, 0.0, t0))
            else:
                Qtag[-1] = ("eave", e, 0.0, t0); skip_start = False
            add(A, ("side", e))
        elif float(np.dot(prev_dir, outward)) > 0.95 and not skip_start:
            pass                                  # previous edge is the wing's side wall: it extends to the pushed corner
        else:
            if skip_start:
                skip_start = False
            add(A, ("side", e))
        add(A + outward * ext, ("gable", e))
        # end of the span
        if t1 < 1.0 - 1e-6:
            add(B + outward * ext, ("side", e))
            add(B, ("eave", e, t1, 1.0))
        elif float(np.dot(next_dir, -outward)) > 0.95:
            add(B + outward * ext, None)          # the next real edge (the wing's other wall) starts here
            skip_start = True
        else:
            add(B + outward * ext, ("side", e))
            add(B, None); skip_start = True       # the next real edge starts at B
        gable_lines.append((A.copy(), B.copy(), e, t0, t1))
    if skip_start and Qtag and Qtag[-1] is None:
        # wrapped around: the pending pushed corner is where edge `off` starts
        Qtag[-1] = ("eave", off, 0.0, 1.0)
        Qv.pop(0); Qtag.pop(0)
    for i in range(len(Qtag)):
        if Qtag[i] is None:
            Qtag[i] = ("eave", off, 0.0, 1.0)
    Q = np.array(Qv)
    m = len(Q)
    if m < 3 or not Polygon(Q).is_valid:
        Q = P.copy(); m = n; gable_lines = []
        Qtag = [("eave", e, 0.0, 1.0) for e in range(n)]
    qedge_tag = {i: Qtag[i] for i in range(m)}
    V, faces = _skeleton(Q, tan)
    if not faces:
        return g
    V = np.array(V, dtype=float)

    # ---- faces as shapely polygons clipped to the real outline, with their eave line ----
    face_polys = {}
    face_eave = {}
    face_tag = {}
    for fi, f in enumerate(faces):
        poly = Polygon(V[f, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(real)
        if clipped.is_empty or clipped.area < 0.05:
            continue
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda q: q.area)
        if clipped.geom_type != "Polygon":
            continue
        qe = f[0] if (f[0] < m and f[1] < m and (f[0] + 1) % m == f[1]) else None
        if qe is None:
            continue
        tag = qedge_tag[qe]
        if tag[0] == "gable":
            continue                              # the pushed gable face: nothing real
        face_polys[fi] = clipped
        face_eave[fi] = (Q[qe], Q[(qe + 1) % m])
        face_tag[fi] = tag

    def face_z(fi, pt):
        a, b = face_eave[fi]
        d = b - a; L = np.hypot(*d) or 1.0
        return tan * abs((pt[0] - a[0]) * d[1] - (pt[1] - a[1]) * d[0]) / L

    def to_px(pt):
        return (-pt[1] / px_y, pt[0] / px_x)

    # ---- interior edges: shared boundaries between faces ----
    fids = sorted(face_polys)
    for i, fa in enumerate(fids):
        for fb in fids[i + 1:]:
            inter = face_polys[fa].boundary.intersection(face_polys[fb].boundary)
            if inter.is_empty:
                continue
            lines = [inter] if inter.geom_type == "LineString" else [l for l in getattr(inter, "geoms", []) if l.geom_type == "LineString"]
            for ln in lines:
                if ln.length < 0.15:
                    continue
                pa, pb = np.array(ln.coords[0]), np.array(ln.coords[-1])
                za, zb = face_z(fa, pa), face_z(fa, pb)
                if abs(za - zb) < 0.02 * max(1.0, ln.length):
                    kind = "ridge"
                else:
                    # convex if, stepping into face A, plane B is above plane A
                    mid = (pa + pb) / 2
                    ca = np.array(face_polys[fa].representative_point().coords[0])
                    sv = ca - mid
                    sv = sv / (np.hypot(*sv) or 1.0) * 0.3
                    q = mid + sv
                    kind = "hip" if face_z(fb, q) > face_z(fa, q) else "valley"
                length = math.hypot(ln.length, zb - za)
                g.edges.append(GEdge(kind, to_px(pa), to_px(pb), round(length * M_TO_FT, 1),
                                     round(ln.length * M_TO_FT, 1), (fa, fb)))
    # ---- outline edges: eave portions, and two rakes per gable span split at the ridge foot ----
    owner = {}                                     # (e, t0, t1) -> face
    for fi in fids:
        tg = face_tag[fi]
        if tg[0] == "eave":
            owner[(tg[1], round(tg[2], 3), round(tg[3], 3))] = fi
    spans = {e: (t0, t1) for _A, _B, e, t0, t1 in gable_lines}
    for e in range(n):
        a, b = P[e], P[(e + 1) % n]
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        if e not in spans:
            g.edges.append(GEdge("eave", to_px(a), to_px(b), round(L * M_TO_FT, 1), round(L * M_TO_FT, 1),
                                 (owner.get((e, 0.0, 1.0)),)))
            continue
        t0, t1 = spans[e]
        A = a + d * t0; B = a + d * t1
        if t0 > 1e-6:
            g.edges.append(GEdge("eave", to_px(a), to_px(A), round(t0 * L * M_TO_FT, 1), round(t0 * L * M_TO_FT, 1),
                                 (owner.get((e, 0.0, round(t0, 3))),)))
        # the ridge foot: the ridge endpoint lying on the gable line
        seg = LineString([tuple(A), tuple(B)])
        foot = None
        for ed in g.edges:
            if ed.kind != "ridge":
                continue
            for pt in (ed.p0, ed.p1):
                pm = np.array([pt[1] * px_x, -pt[0] * px_y])
                if seg.distance(Point(pm)) < 0.08:
                    foot = pm
        if foot is None:
            foot = (A + B) / 2.0
        zf = 0.0
        for fi in fids:
            if face_polys[fi].distance(Point(foot)) < 0.1:
                zf = max(zf, face_z(fi, foot))
        for p0, p1 in ((A, foot), (foot, B)):
            plan = float(np.hypot(*(p1 - p0)))
            if plan < 0.05:
                continue
            own = None
            for fi in fids:
                if face_polys[fi].distance(LineString([tuple(p0), tuple(p1)]).centroid) < 0.1:
                    own = fi; break
            g.edges.append(GEdge("rake", to_px(p0), to_px(p1), round(math.hypot(plan, zf) * M_TO_FT, 1),
                                 round(plan * M_TO_FT, 1), (own,)))
        if t1 < 1.0 - 1e-6:
            g.edges.append(GEdge("eave", to_px(B), to_px(b), round((1 - t1) * L * M_TO_FT, 1),
                                 round((1 - t1) * L * M_TO_FT, 1), (owner.get((e, round(t1, 3), 1.0)),)))
    # ---- faces ----
    slope = math.degrees(math.atan(tan))
    for fi in fids:
        poly = face_polys[fi]
        coords = np.array(poly.exterior.coords[:-1])
        g.faces[fi] = np.array([to_px(c) for c in coords])
        plan_m2 = poly.area
        g.face_plan_sqft[fi] = plan_m2 * SQM_TO_SQFT
        g.face_area_sqft[fi] = plan_m2 * SQM_TO_SQFT / math.cos(math.radians(slope))
        g.face_pitch[fi] = f"{rise}/12"
        g.face_slope_deg[fi] = slope
        a, b = face_eave[fi]
        d = b - a; L = np.hypot(*d) or 1.0
        inward = np.array([-d[1], d[0]]) / L
        dsx, dsy = -inward[0], -inward[1]           # downslope (x east, y north)
        g.face_azimuth[fi] = (math.degrees(math.atan2(dsx, dsy)) + 360.0) % 360.0
        g.face_downslope[fi] = (-dsy, dsx)          # (drow, dcol)
        c = poly.centroid
        g.face_centroids[fi] = to_px((c.x, c.y))
        g.face_eave_px[fi] = (to_px(a), to_px(b))
    g.info = {"rise": rise, "gables": sorted(e for _a, _b, e, _t0, _t1 in gable_lines),
              "gable_spans": {int(e): (round(t0, 2), round(t1, 2)) for _a, _b, e, t0, t1 in gable_lines},
              "gables_requested": sorted(gab), "n_outline": n}
    _totals(g)
    return g


def add_flat_structure(g: RoofGraph, poly_px: np.ndarray, px_x: float, px_y: float,
                       fid: int, shared_with_main: np.ndarray | None = None) -> None:
    """Add a flat section as its own facet: its free edges are eaves, edges
    along the pitched roof's wall are flashing."""
    n = len(poly_px)
    plan_m2 = _poly_area(np.column_stack([poly_px[:, 1] * px_x, poly_px[:, 0] * px_y]))
    g.faces[fid] = poly_px.astype(float)
    g.face_plan_sqft[fid] = plan_m2 * SQM_TO_SQFT
    g.face_area_sqft[fid] = plan_m2 * SQM_TO_SQFT
    g.face_pitch[fid] = "1/12"
    g.face_slope_deg[fid] = math.degrees(math.atan(1 / 12))
    g.face_azimuth[fid] = 0.0
    g.face_downslope[fid] = (0.0, 0.0)
    g.face_centroids[fid] = (float(poly_px[:, 0].mean()), float(poly_px[:, 1].mean()))
    for e in range(n):
        a, b = poly_px[e].astype(float), poly_px[(e + 1) % n].astype(float)
        plan = math.hypot((b[0] - a[0]) * px_y, (b[1] - a[1]) * px_x)
        if plan < 0.2:
            continue
        kind = "eave"
        if shared_with_main is not None:
            mid = (a + b) / 2
            y, x = int(round(mid[0])), int(round(mid[1]))
            h, w = shared_with_main.shape
            # if the pitched roof is within 0.5 m on either side, this is the wall junction
            r = max(int(0.5 / px_x), 2)
            y0, y1 = max(y - r, 0), min(y + r + 1, h); x0, x1 = max(x - r, 0), min(x + r + 1, w)
            if shared_with_main[y0:y1, x0:x1].any():
                kind = "flashing"
        g.edges.append(GEdge(kind, tuple(a), tuple(b), round(plan * M_TO_FT, 1), round(plan * M_TO_FT, 1), (fid,)))
    _totals(g)


def _totals(g: RoofGraph) -> None:
    tot = defaultdict(float); cnt = defaultdict(int)
    for e in g.edges:
        tot[e.kind] += e.length_ft; cnt[e.kind] += 1
    g.totals = {
        "ridges_ft": round(tot["ridge"]), "hips_ft": round(tot["hip"]), "valleys_ft": round(tot["valley"]),
        "eaves_ft": round(tot["eave"]), "rakes_ft": round(tot["rake"]), "flashing_ft": round(tot["flashing"]),
        "step_flashing_ft": round(tot["step_flashing"]),
        "ridge_count": cnt["ridge"], "hip_count": cnt["hip"], "valley_count": cnt["valley"],
        "eave_count": cnt["eave"], "rake_count": cnt["rake"], "flashing_count": cnt["flashing"],
        "step_flashing_count": cnt["step_flashing"],
        "plan_area_sqft": round(sum(g.face_plan_sqft.values())),
        "surface_area_sqft": round(sum(g.face_area_sqft.values())),
    }


# ------------------------------------------------------------ fit check

def model_residual(dsm: np.ndarray, mask: np.ndarray, px: float, z_eave: float, tan: float,
                   gable_mask: np.ndarray | None = None) -> dict:
    """RMS and 90th-percentile residual between the DSM and the grassfire
    roof surface (pitch x distance to the outline) over the pitched footprint."""
    d = _edt(mask, px)
    sel = mask & (d > 0.5)
    if gable_mask is not None:
        sel &= ~gable_mask
    if sel.sum() < 30:
        return {"rms_m": float("nan"), "p90_m": float("nan"), "n": int(sel.sum())}
    r = dsm[sel] - (z_eave + tan * d[sel])
    return {"rms_m": float(np.sqrt(np.mean(r * r))), "p90_m": float(np.percentile(np.abs(r), 90)),
            "bias_m": float(np.mean(r)), "n": int(sel.sum())}


# ------------------------------------------------------------ orchestration

LOWER_SECTION_DROP_M = 0.6    # DSM this far below the roof model is another roof level
LOWER_SECTION_MIN_M2 = 6.0
MAX_LEVELS = 4
RAKE_SLOPE_DEG = 7.0          # along-edge slope of the facet plane above this = rake
MIN_ROOF_HEIGHT_M = 1.8       # footprint pixels lower than this above ground are not roof
MIN_JOG_M = 1.1               # eave-line jogs shorter than this are not framed; snap them out
GABLE_MIN_EDGE_M = 2.5        # a gable end is a wing's full width, never a short jog
PEEL_MIN_M2 = 12.0            # a lower roof level must be at least this big ...
PEEL_MIN_STEP_M = 0.4         # ... and sit behind a real step in the elevation model
MAX_PEEL_DEPTH = 2
MAX_GABLE_CANDIDATES = 4      # gable hypotheses tested in combination (2^4 plans)
PLANE_GABLE_CREDIT_M = 0.10   # fit-score credit for a gable the fitted planes also show


def _components(mask: np.ndarray):
    from .pipeline import _label_components
    r = _label_components(mask)
    lab = r[0] if isinstance(r, tuple) else r
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    return lab, list(zip(ids.tolist(), cnt.tolist()))


def _grad_mag(dsm: np.ndarray, px: float) -> np.ndarray:
    """|gradient| of a 3x3-mean-smoothed DSM (rise per metre)."""
    k = np.ones((3, 3)) / 9.0
    try:
        from scipy import ndimage
        sm = ndimage.convolve(dsm, k, mode="nearest")
    except Exception:  # pragma: no cover
        sm = dsm
    gy, gx = np.gradient(sm, px)
    return np.hypot(gx, gy)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, comps = _components(mask)
    if not comps:
        return mask
    cid = max(comps, key=lambda c: c[1])[0]
    return lab == cid


def _dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]; out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def consensus_rise(labels: np.ndarray, planes: dict, region: np.ndarray, px: float) -> int | None:
    """Area-weighted mode of the pitched facets' rise inside region."""
    votes = defaultdict(float)
    for fid, (a, b, _c) in planes.items():
        rise = int(round(math.hypot(a, b) * 12))
        if rise <= LOW_SLOPE_RISE:
            continue
        n = int(((labels == fid) & region).sum())
        if n:
            votes[rise] += n
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def rake_edges_from_planes(outline: np.ndarray, labels: np.ndarray, planes: dict,
                           px_x: float, px_y: float, spans: dict | None = None) -> set:
    """Outline edges that are gable ends: the facet plane just inside the
    edge slopes ALONG the edge (a rake) over most of its length. With
    `spans` ({e: (t0, t1, ...)}) only those edges are tested and only the
    span is sampled."""
    n = len(outline)
    h, w = labels.shape
    out = set()
    for e in (sorted(spans) if spans is not None else range(n)):
        a0 = outline[e].astype(float); b0 = outline[(e + 1) % n].astype(float)
        if spans is not None:
            t0, t1 = float(spans[e][0]), float(spans[e][1])
            a = a0 + (b0 - a0) * t0; b = a0 + (b0 - a0) * t1
        else:
            a, b = a0, b0
        v = b - a; L = math.hypot(*v)
        if L * (px_x + px_y) / 2 < 1.5:
            continue
        u = v / L
        cand = np.array([-u[1], u[0]])
        score = 0.0
        for t in (0.25, 0.5, 0.75):
            for depth in (3.0, 6.0, 10.0):
                for sgn in (1.0, -1.0):
                    p = a + v * t + cand * depth * sgn
                    y, x = int(round(p[0])), int(round(p[1]))
                    if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                        score += sgn
        nrm = cand * (1.0 if score >= 0 else -1.0)
        ux_m, uy_m = u[1] * px_x, u[0] * px_y
        um = math.hypot(ux_m, uy_m); ux_m, uy_m = ux_m / um, uy_m / um
        nsamp = max(int(L / 3), 4)
        rake = 0; tot = 0
        for k in range(nsamp):
            t = (k + 0.5) / nsamp
            lab = 0
            for depth in (3, 6, 10, 15):
                p = a + v * t + nrm * depth
                y, x = int(round(p[0])), int(round(p[1]))
                if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                    lab = int(labels[y, x]); break
            if lab in planes:
                pa = planes[lab]
                g = pa[0] * ux_m + pa[1] * uy_m
                tot += 1
                if math.degrees(math.atan(abs(g))) > RAKE_SLOPE_DEG:
                    rake += 1
        if tot and rake / tot > 0.5:
            out.add(e)
    return out


def rasterize_model(g: RoofGraph, mask: np.ndarray, px_x: float, px_y: float, z_eave: float,
                    tan: float) -> np.ndarray:
    """Model roof height over mask from the graph's faces: each face rises
    from its eave edge at the pitch. NaN where no face covers a pixel."""
    from matplotlib.path import Path

    h, w = mask.shape
    model = np.full((h, w), np.nan)
    eave_of_face = {fid: (np.asarray(a, float), np.asarray(b, float)) for fid, (a, b) in g.face_eave_px.items()}
    for e in g.edges:
        if e.kind in ("eave", "flashing") and e.faces and e.faces[0] is not None:
            eave_of_face.setdefault(e.faces[0], (np.asarray(e.p0, float), np.asarray(e.p1, float)))
    for fid, poly in g.faces.items():
        if len(poly) < 3:
            continue
        y0, y1 = int(max(poly[:, 0].min() - 1, 0)), int(min(poly[:, 0].max() + 2, h))
        x0, x1 = int(max(poly[:, 1].min() - 1, 0)), int(min(poly[:, 1].max() + 2, w))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        inside = Path(poly).contains_points(pts).reshape(yy.shape) & mask[y0:y1, x0:x1]
        if not inside.any():
            continue
        slope = g.face_slope_deg.get(fid, 0.0)
        if fid in eave_of_face and slope > 1.0:
            a, b = eave_of_face[fid]
            # perpendicular distance (m) to the eave line
            vx, vy = (b[1] - a[1]) * px_x, (b[0] - a[0]) * px_y
            L = math.hypot(vx, vy) or 1.0
            dx = (xx - a[1]) * px_x; dy = (yy - a[0]) * px_y
            dist = np.abs(dx * vy - dy * vx) / L
            z = z_eave + tan * dist
        else:
            z = np.full(yy.shape, z_eave)
        sub = model[y0:y1, x0:x1]
        sub[inside] = z[inside]
    return model


def generate_roof(dsm: np.ndarray, roof: np.ndarray, px_x: float, px_y: float,
                  axis_hint: float | None = None, depth: int = 0,
                  labels: np.ndarray | None = None, planes: dict | None = None,
                  force_flat: bool = False, ground: np.ndarray | None = None,
                  known: np.ndarray | None = None) -> RoofGraph:
    """Full framing-model roof for one footprint.

    1. Flat sections known from the fitted planes are cut out first (stage A).
    2. Square the eave outline, take the eave height, the pitch (consensus of
       the fitted planes, else a fit), the gable ends (planes sloping along
       the edge), and build the skeleton graph.
    3. Where the elevation model sits well below that roof (a lower wing, a
       porch roof), cut the region out as its own level and rebuild (stage B,
       up to twice). Lower levels get their own eave height and pitch;
       their edges along the upper roof's wall are flashing.
    """
    from .vectorize import trace_boundary
    from .wireframe import regularize_outline

    px = (px_x + px_y) / 2.0
    g = RoofGraph()
    main = roof.copy()
    # `known`: footprint pixels whose elevation is the roof itself. Under tree
    # canopy the footprint is completed from the house geometry and the DSM
    # there is tree top: those pixels shape the outline but never the eave
    # height, pitch, gable profiles, fit residual or lower-level search.
    known = np.ones_like(roof, dtype=bool) if known is None else (known.astype(bool))
    if ground is not None and depth == 0:
        # nothing lower than about 6 ft above ground is roof: drop the ground
        # strips, fences and shadows the footprint recovery can pick up
        main &= (dsm - ground) > MIN_ROOF_HEIGHT_M
        main = _largest_component(main) if main.any() else main
    lower_regions = []
    d_orig = _edt(roof, px)
    gm = _grad_mag(dsm, px)
    have_planes = labels is not None and planes is not None
    # ---- stage A: flat sections from the fitted planes ----
    if have_planes and not force_flat:
        r_roof = consensus_rise(labels, planes, main, px)
        flats = flat_sections(labels, planes, px, r_roof) & main
        lab, comps = _components(flats)
        if comps:
            ze0 = eave_height(dsm, main, px, known)
            r0 = r_roof
            tan0 = (r0 or 6) / 12.0
            grow_ok = main & ((gm < 0.5 * tan0) | (dsm <= ze0 + 0.25) | ((gm > 1.6 * tan0) & (d_orig <= 1.2)))
            for cid, cnt in comps:
                if cnt * px * px < LOWER_SECTION_MIN_M2:
                    continue
                reg = lab == cid
                for _ in range(int(2.0 / px)):
                    grown = _dilate(reg) & grow_ok
                    if not (grown & ~reg).any():
                        break
                    reg |= grown
                lower_regions.append((reg, True))
                main &= ~reg
            if lower_regions:
                main = _largest_component(main)
    if main.sum() * px * px < 4.0:
        return g

    # ---- stage A2: flat sections the plane fit missed. A pitched plane is
    # at least pitch x distance above the eave; a region sitting far below
    # that, more than 1 m in from the edge, is a flat cover (porch, patio,
    # carport) under or against the main roof ----
    r0 = consensus_rise(labels, planes, main, px) if have_planes else None
    if not force_flat and (r0 is None or r0 >= LOW_SLOPE_RISE + FLAT_RISE_GAP):
        ze0 = eave_height(dsm, main, px, known)
        if r0 is None:
            r0, _t, _r = fit_pitch(dsm, main, px, ze0, known)
        tan0 = max(r0, 3) / 12.0
        d0 = _edt(main, px)
        core = main & known & (d0 > 1.0) & ((dsm - ze0) < 0.5 * tan0 * d0 - 0.05) & (gm < 0.5 * tan0)
        lab, comps = _components(core)
        grow_ok = main & ((gm < 0.5 * tan0) | (dsm <= ze0 + 0.25) | ((gm > 1.6 * tan0) & (d_orig <= 1.2)))
        added = False
        for cid, cnt in comps:
            if cnt * px * px < LOWER_SECTION_MIN_M2:
                continue
            reg = lab == cid
            for _ in range(int(2.0 / px)):
                grown = _dilate(reg) & grow_ok
                if not (grown & ~reg).any():
                    break
                reg |= grown
            if reg.sum() * px * px < LOWER_SECTION_MIN_M2 * 1.5:
                continue
            lower_regions.append((reg, True))
            main &= ~reg
            added = True
        if added:
            main = _largest_component(main)
    if main.sum() * px * px < 4.0:
        return g

    # ---- build, check against the DSM, peel off lower levels, rebuild ----
    model = None; ze = 0.0; tan = 0.0; rise = 0; outline = None; gab = set()
    for _it in range(3):
        chain = trace_boundary(main)
        outline = regularize_outline(chain, px, axis_hint=axis_hint)
        if len(outline) < 3:
            return g
        outline = simplify_outline(outline, px, MIN_JOG_M)
        outline = rectilinearize(outline, axis_hint, px=px)
        if not _valid_ring(outline):
            from shapely.geometry import Polygon
            fixed = Polygon(np.column_stack([outline[:, 1], -outline[:, 0]])).buffer(0)
            if fixed.geom_type == "MultiPolygon":
                fixed = max(fixed.geoms, key=lambda q: q.area)
            if fixed.geom_type == "Polygon" and not fixed.is_empty:
                ring = np.array(fixed.exterior.coords[:-1])
                outline = np.column_stack([-ring[:, 1], ring[:, 0]])
        ze = eave_height(dsm, main, px, known)
        rise = consensus_rise(labels, planes, main, px) if have_planes else None
        fit_rise, fit_tan, fit_rms = fit_pitch(dsm, main, px, ze, known)
        if rise is None:
            rise = fit_rise
        if force_flat or rise <= LOW_SLOPE_RISE:
            g = RoofGraph()
            add_flat_structure(g, outline, px_x, px_y, 9000 + depth)
            g.info = {"rise": rise, "levels": 1, "flat_main": True, "depth": depth}
            return g
        tan = rise / 12.0
        nn = len(outline)
        elen = {e: math.hypot(*(outline[(e + 1) % nn] - outline[e])) * px for e in range(nn)}
        spans = gable_edges(outline, dsm, main & known, ze, px, tan)
        if have_planes:
            # the fitted planes are a second, independent gable detector: a
            # plane that slopes along an outline edge is a rake. Edges it
            # flags join the candidate set as whole-edge gables; the
            # elevation fit below decides which hypotheses survive.
            for e in rake_edges_from_planes(outline, labels, planes, px_x, px_y):
                if e not in spans and elen[e] >= GABLE_MIN_EDGE_M:
                    spans[e] = (0.0, 1.0, tan * elen[e] / 2.0)
            confirmed = rake_edges_from_planes(outline, labels, planes, px_x, px_y, spans)
            # rank: candidates both detectors agree on first, then by tent height
            spans = {e: (v[0], v[1], v[2] + (10.0 if e in confirmed else 0.0)) for e, v in spans.items()}
        # ---- gables are hypotheses: build every combination of the candidates
        # (strongest MAX_GABLE_CANDIDATES) and keep the plan the elevation
        # model agrees with best; the plain hip plan is always a candidate ----
        cands = sorted(spans, key=lambda e: -spans[e][2])[:MAX_GABLE_CANDIDATES]
        best = None
        for k in range(len(cands) + 1):
            for combo in combinations(cands, k):
                gab_try = {e: spans[e] for e in combo}
                g_try = build_roof_graph(outline, px_x, px_y, rise, gab_try)
                if not g_try.ok or not g_try.totals.get("eaves_ft"):
                    continue
                model_try = rasterize_model(g_try, main, px_x, px_y, ze, tan)
                sel = main & known & np.isfinite(model_try) & (d_orig > 0.4)
                if sel.sum() < 0.6 * (main & known).sum():
                    continue
                r = (dsm - model_try)[sel]
                p90 = float(np.percentile(np.abs(r), 90)); rms = float(np.sqrt(np.mean(r * r)))
                # prefer the simpler plan on a pure tie, but a gable the fitted
                # planes identify (a rake plane along the edge) is evidence the
                # height fit alone cannot see at low pitch: credit it
                n_conf = sum(1 for e in combo if spans[e][2] >= 10.0)
                score = p90 + 0.5 * rms + 0.02 * k - PLANE_GABLE_CREDIT_M * n_conf
                if best is None or score < best[0]:
                    best = (score, g_try, model_try, gab_try, p90, rms)
        if best is None:
            g = build_roof_graph(outline, px_x, px_y, rise, {})
            if not g.ok:
                return g
            model = rasterize_model(g, main, px_x, px_y, ze, tan)
            gab = {}
        else:
            _score, g, model, gab, _p90, _rms = best
        g.info["gable_candidates"] = {int(e): spans[e] for e in cands}
        g.info["gables_chosen"] = {int(e): spans[e] for e in gab}
        resid = dsm - model
        d = _edt(main, px)
        low = main & known & (d > 0.3) & np.isfinite(resid) & (resid < -LOWER_SECTION_DROP_M)
        lab, comps = _components(low)
        found = False
        grow_ok = main & known & np.isfinite(resid) & (resid < -0.3)
        for cid, cnt in comps:
            if cnt * px * px >= LOWER_SECTION_MIN_M2 and depth < MAX_PEEL_DEPTH:
                reg = lab == cid
                for _ in range(int(2.0 / px)):
                    grown = _dilate(reg) & grow_ok
                    if not (grown & ~reg).any():
                        break
                    reg |= grown
                if reg.sum() * px * px < PEEL_MIN_M2:
                    continue
                if _step_into(dsm, reg, main & ~reg, px) < PEEL_MIN_STEP_M:
                    continue                          # no wall: fit error, not another level
                lower_regions.append((reg, False))
                main &= ~reg
                found = True
        if not found or force_flat:
            break
        main = _largest_component(main)
    sel = main & known & np.isfinite(model) & (_edt(main, px) > 0.4)
    r = (dsm - model)[sel] if sel.any() else np.array([0.0])
    g.info.update({"eave_z": round(ze, 2), "fit_rms_m": round(float(np.sqrt(np.mean(r * r))), 2),
                   "p90_m": round(float(np.percentile(np.abs(r), 90)), 2),
                   "bias_m": round(float(np.mean(r)), 2), "lower_regions": len(lower_regions), "depth": depth})
    # ---- lower levels ----
    next_fid = 1000 * (depth + 1)
    for reg, is_flat in lower_regions:
        if depth + 1 >= MAX_LEVELS:
            break
        sub = generate_roof(dsm, reg, px_x, px_y, axis_hint=axis_hint, depth=depth + 1,
                            labels=labels, planes=planes, force_flat=is_flat, ground=ground, known=known)
        if not sub.ok:
            continue
        _merge_sub(g, sub, main, px, next_fid)
        next_fid += 100
    _totals(g)
    return g


def _step_into(dsm: np.ndarray, reg: np.ndarray, rest: np.ndarray, px: float = 0.1) -> float:
    """Height drop from `rest` to `reg` across their shared boundary: median
    DSM of rest pixels within 0.6 m of the boundary minus median DSM of reg
    pixels within 0.6 m of it (the wall between two roof levels)."""
    r = max(int(0.6 / px), 2)
    near_reg = reg.copy(); near_rest = rest.copy()
    grow_reg = reg.copy(); grow_rest = rest.copy()
    for _ in range(r):
        grow_reg = _dilate(grow_reg); grow_rest = _dilate(grow_rest)
    band_rest = rest & grow_reg          # rest pixels within r of the region
    band_reg = reg & grow_rest           # region pixels within r of the rest
    if band_rest.sum() < 10 or band_reg.sum() < 10:
        return 0.0
    return float(np.median(dsm[band_rest]) - np.median(dsm[band_reg]))


def _merge_sub(g: RoofGraph, sub: RoofGraph, main_mask: np.ndarray, px: float, base_fid: int) -> None:
    """Add a lower roof level: its outline edges that run along the upper
    roof's footprint become flashing (sloped against the wall: step
    flashing); everything else is kept as is."""
    h, w = main_mask.shape
    r = max(int(0.5 / px), 2)
    for e in sub.edges:
        kind = e.kind
        if kind in ("eave", "rake"):
            mid = ((e.p0[0] + e.p1[0]) / 2, (e.p0[1] + e.p1[1]) / 2)
            y, x = int(round(mid[0])), int(round(mid[1]))
            y0, y1 = max(y - r, 0), min(y + r + 1, h); x0, x1 = max(x - r, 0), min(x + r + 1, w)
            if main_mask[y0:y1, x0:x1].any():
                kind = "step_flashing" if e.kind == "rake" else "flashing"
        g.edges.append(GEdge(kind, e.p0, e.p1, e.length_ft, e.plan_ft, tuple(base_fid + (f or 0) for f in e.faces)))
    for fid in sub.faces:
        nf = base_fid + fid
        g.faces[nf] = sub.faces[fid]
        for attr in ("face_plan_sqft", "face_area_sqft", "face_pitch", "face_slope_deg", "face_azimuth",
                     "face_centroids", "face_downslope", "face_eave_px"):
            if fid in getattr(sub, attr):
                getattr(g, attr)[nf] = getattr(sub, attr)[fid]
    g.info.setdefault("levels", []).append(sub.info)


# ------------------------------------------------------------ plan support test

def plan_support(plan_edges, plane_edges, px: float, tol_m: float = 1.0,
                 known: np.ndarray | None = None) -> dict:
    """How well the measured plane intersections support a framed plan.

    Both inputs are edge lists with .kind, .p0, .p1 in raster px. Only
    ridges, hips and valleys count (the lines generated by the plan). For
    each framed crease, points every 0.5 m along it are tested for lying
    within tol_m of any measured crease; `support` is the supported fraction
    of framed crease length. `coverage` is the converse: the fraction of
    measured crease length that lies near a framed crease. A plan the
    planes agree with scores high on both; a plan that invents hips where
    the roof has gables scores low on support."""
    kinds = ("ridge", "hip", "valley")
    A = [(np.asarray(e.p0, float), np.asarray(e.p1, float)) for e in plan_edges if e.kind in kinds]
    B = [(np.asarray(e.p0, float), np.asarray(e.p1, float)) for e in plane_edges if e.kind in kinds]
    if not A or not B:
        return {"support": 0.0, "coverage": 0.0, "plan_len_m": 0.0, "plane_len_m": 0.0}

    def samples(segs):
        pts = []
        for a, b in segs:
            L = float(np.hypot(*(b - a))) * px
            k = max(int(L / 0.5), 1)
            for i in range(k + 1):
                pts.append(a + (b - a) * (i / k))
        return np.array(pts)

    def seg_dist(pts, segs):
        d = np.full(len(pts), np.inf)
        for a, b in segs:
            ab = b - a; L2 = float(ab @ ab) or 1e-9
            t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.minimum(d, np.hypot(*(pts - proj).T))
        return d * px

    pa = samples(A); pb = samples(B)
    if known is not None:
        # lines under tree canopy have no measured plane to support them:
        # they are neither for nor against the plan
        h, w = known.shape
        def keep(pts):
            yy = np.clip(np.round(pts[:, 0]).astype(int), 0, h - 1)
            xx = np.clip(np.round(pts[:, 1]).astype(int), 0, w - 1)
            return known[yy, xx]
        ka = keep(pa)
        if ka.sum() >= 10:
            pa = pa[ka]
        kb = keep(pb)
        if kb.sum() >= 10:
            pb = pb[kb]
    sup = float((seg_dist(pa, B) <= tol_m).mean())
    cov = float((seg_dist(pb, A) <= tol_m).mean())
    la = len(pa) * 0.5
    lb = len(pb) * 0.5
    return {"support": round(sup, 3), "coverage": round(cov, 3), "plan_len_m": round(la, 1), "plane_len_m": round(lb, 1)}


def rectilinearize(outline_px: np.ndarray, axis_deg: float | None, max_len_m: float = 6.0,
                   px: float = 0.1) -> np.ndarray:
    """A house outline has edges on two axes only. Any edge more than 12
    degrees off both axes (a tree line, a raster artefact) is replaced by
    the two axis-aligned edges through the corner that lies OUTSIDE the
    polygon, so the completion errs long, never short. Edges longer than
    max_len_m are left alone (a long diagonal is a real bay or a wrong
    axis, not an artefact)."""
    from shapely.geometry import Polygon, Point

    P = np.asarray(outline_px, float)
    n = len(P)
    if n < 4 or axis_deg is None:
        return P
    th = math.radians(axis_deg)
    ca, sa = math.cos(th), math.sin(th)
    ctr = P.mean(axis=0)

    def to_uv(p):
        y, x = p[0] - ctr[0], p[1] - ctr[1]
        return np.array([x * ca + y * sa, -x * sa + y * ca])

    def to_yx(q):
        u, v = q
        return np.array([ctr[0] + (u * sa + v * ca), ctr[1] + (u * ca - v * sa)])

    Q = [to_uv(p) for p in P]
    poly = Polygon([(q[0], q[1]) for q in Q])
    if not poly.is_valid:
        poly = poly.buffer(0)
    out = []
    changed = False
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        d = b - a; L = float(np.hypot(*d))
        out.append(a)
        if L < 1e-6:
            continue
        off = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
        off = min(off, 90.0 - off)
        if off <= 12.0 or L * px > max_len_m:
            continue
        c1 = np.array([b[0], a[1]]); c2 = np.array([a[0], b[1]])
        # the corner outside the polygon (or the one that adds area)
        c = c1 if not poly.contains(Point(c1[0], c1[1])) else c2
        if poly.contains(Point(c[0], c[1])):
            continue
        out.append(c)
        changed = True
    if not changed:
        return P
    R = np.array([to_yx(q) for q in out])
    # drop duplicates
    keep = [R[0]]
    for q in R[1:]:
        if np.hypot(*(q - keep[-1])) > 0.5:
            keep.append(q)
    if len(keep) > 1 and np.hypot(*(keep[0] - keep[-1])) <= 0.5:
        keep.pop()
    R = np.array(keep)
    return R if _valid_ring(R) else P
