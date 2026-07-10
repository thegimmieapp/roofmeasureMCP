"""Roof geometry engine.

Extracts roof facets from a DSM (digital surface model) raster and classifies
every edge as ridge / hip / valley / eave / rake, with slope-corrected lengths.

Pipeline:
  1. Smooth the masked DSM lightly.
  2. Compute per-pixel slope (deg) and aspect (downslope azimuth).
  3. Region-grow pixels into facets where slope and aspect are locally similar.
  4. Merge slivers into their largest neighbor.
  5. Walk facet boundaries:
       - shared boundary between two facets -> ridge / hip / valley
       - boundary against non-roof -> eave / rake (by edge direction vs aspect)
  6. Straight-segment lengths via PCA extent of boundary pixel clusters,
     slope-corrected using elevation change along the segment.

Everything is numpy-only (no scipy / rasterio / shapely).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .model import EdgeSummary, Facet, RoofMeasurements, slope_deg_to_pitch

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671

MIN_FACET_AREA_M2 = 2.0        # merge anything smaller
MIN_EDGE_LEN_FT = 3.0          # ignore shorter classified segments
ASPECT_TOL_DEG = 30.0          # region-growing aspect tolerance
SLOPE_TOL_DEG = 7.0            # region-growing slope tolerance
FLAT_SLOPE_DEG = 8.0           # below this a facet is treated as "flat/low slope"


# ---------------------------------------------------------------- utilities

def _smooth(z: np.ndarray, mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """Masked 3x3 box smoothing."""
    zs = z.copy()
    m = mask.astype(np.float32)
    for _ in range(iters):
        acc = np.zeros_like(zs)
        cnt = np.zeros_like(zs)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(zs * m, dy, 0), dx, 1)
                cnt += np.roll(np.roll(m, dy, 0), dx, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = acc / cnt
        zs = np.where(mask & (cnt > 0), out, zs)
    return zs


def _slope_aspect(z: np.ndarray, px_x: float, px_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel slope (deg) and downslope aspect (deg, 0=N, CW) from DSM."""
    gy, gx = np.gradient(z, px_y, px_x)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # aspect = compass direction water flows (downslope): gradient points uphill
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    # NOTE: raster row 0 is north; +y in array = south. gy is dz/dsouth.
    # downhill vector = -(gx, gy) in (east, south) coords ->
    # compass angle = atan2(east_component, north_component) = atan2(-gx, gy)
    return slope, aspect


def _ang_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d) if isinstance(d, np.ndarray) else (360.0 - d if d > 180.0 else d)


def _erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.copy()
    for _ in range(iters):
        shifted = m.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted &= np.roll(np.roll(m, dy, 0), dx, 1)
        m = shifted
    return m


def _grow_labels(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assign unlabeled masked pixels the label of their nearest labeled neighbor."""
    out = labels.copy()
    h, w = out.shape
    for _ in range(64):
        todo = mask & (out == 0)
        if not todo.any():
            break
        updates = []
        ys, xs = np.where(todo)
        for y, x in zip(ys, xs):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx]:
                    updates.append((y, x, out[ny, nx]))
                    break
        if not updates:
            break
        for y, x, lab in updates:
            out[y, x] = lab
    return out


def _segment_facets(slope: np.ndarray, aspect: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Region-grow connected facets. Returns int label array (0 = background)."""
    h, w = slope.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    flat = slope < FLAT_SLOPE_DEG
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            seed_flat = flat[sy, sx]
            q = deque([(sy, sx)])
            labels[sy, sx] = next_label
            # running means for robust growth
            n = 0
            mean_slope = 0.0
            sum_sin = 0.0
            sum_cos = 0.0
            while q:
                y, x = q.popleft()
                n += 1
                mean_slope += (slope[y, x] - mean_slope) / n
                a = math.radians(aspect[y, x])
                sum_sin += math.sin(a)
                sum_cos += math.cos(a)
                mean_aspect = math.degrees(math.atan2(sum_sin, sum_cos)) % 360.0
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if not mask[ny, nx] or labels[ny, nx]:
                        continue
                    if seed_flat:
                        ok = flat[ny, nx]
                    else:
                        ok = (
                            not flat[ny, nx]
                            and abs(slope[ny, nx] - mean_slope) <= SLOPE_TOL_DEG
                            and _ang_diff(aspect[ny, nx], mean_aspect) <= ASPECT_TOL_DEG
                        )
                    if ok:
                        labels[ny, nx] = next_label
                        q.append((ny, nx))
            next_label += 1
    return labels


def _merge_slivers(labels: np.ndarray, px_area_m2: float) -> np.ndarray:
    """Merge facets smaller than MIN_FACET_AREA_M2 into their biggest neighbor."""
    h, w = labels.shape
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        sizes = dict(zip(ids.tolist(), counts.tolist()))
        small = [i for i, c in sizes.items() if c * px_area_m2 < MIN_FACET_AREA_M2]
        if not small:
            break
        changed = False
        small_set = set(small)
        for sid in small:
            neigh = defaultdict(int)
            ys, xs = np.where(labels == sid)
            for y, x in zip(ys, xs):
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        l2 = labels[ny, nx]
                        if l2 and l2 != sid:
                            neigh[l2] += 1
            if neigh:
                # prefer merging into a non-small neighbor
                cands = sorted(neigh.items(), key=lambda kv: (kv[0] in small_set, -kv[1]))
                labels[labels == sid] = cands[0][0]
                changed = True
            else:
                labels[labels == sid] = 0
                changed = True
        if not changed:
            break
    return labels


def _dissolve_thin_facets(labels: np.ndarray, mask: np.ndarray, px: float,
                          max_thickness_m: float = 0.6, min_elongation: float = 4.0) -> np.ndarray:
    """Dissolve thin strip 'facets' (smoothing artifacts along crest lines)
    into their neighbors. A crest between two planes blurs into a narrow band
    of intermediate slope; left alone it double-counts every ridge/hip/valley.
    """
    h, w = labels.shape
    while True:
        dissolved = False
        for i in [int(v) for v in np.unique(labels) if v > 0]:
            sel = labels == i
            n = int(sel.sum())
            if n == 0:
                continue
            # boundary pixel count (4-neighborhood, against anything not-i)
            b = 0
            ys, xs = np.where(sel)
            for y, x in zip(ys, xs):
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w) or labels[ny, nx] != i:
                        b += 1
                        break
            interior = n - b
            thickness_m = (n / max(b / 2.0, 1.0)) * px
            length_m = (b / 2.0) * px
            elongated = length_m / max(thickness_m, 1e-6) >= min_elongation
            if (thickness_m <= max_thickness_m and elongated) or interior == 0:
                labels[sel] = 0
                dissolved = True
        if not dissolved:
            break
        labels = _grow_labels(labels, mask)
    return labels


def _dilate_bool(m: np.ndarray, iters: int) -> np.ndarray:
    out = m.copy()
    for _ in range(iters):
        grown = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown |= np.roll(np.roll(out, dy, 0), dx, 1)
        out = grown
    return out


def _refine_by_planes(labels: np.ndarray, z: np.ndarray, mask: np.ndarray,
                      px_x: float, px_y: float, iters: int = 3) -> np.ndarray:
    """Snap facet boundaries to true plane intersections.

    Fits a plane to each facet, then reassigns each pixel to whichever nearby
    facet's plane best predicts its elevation. Cleans up crest-blur zones and
    resolves same-plane fragments so they can merge.
    """
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx * px_x
    Y = yy * px_y
    for _ in range(iters):
        ids = [int(i) for i in np.unique(labels) if i > 0]
        if len(ids) < 2:
            break
        resid = np.full((len(ids), h, w), np.inf, dtype=np.float64)
        for k, i in enumerate(ids):
            sel = labels == i
            n = int(sel.sum())
            if n < 6:
                continue
            A = np.column_stack([X[sel], Y[sel], np.ones(n)])
            coef, *_ = np.linalg.lstsq(A, z[sel].astype(np.float64), rcond=None)
            pred = coef[0] * X + coef[1] * Y + coef[2]
            allowed = _dilate_bool(sel, 3)
            r = np.abs(z - pred)
            resid[k] = np.where(allowed, r, np.inf)
        best = np.argmin(resid, axis=0)
        best_val = np.min(resid, axis=0)
        new_labels = np.where(mask & np.isfinite(best_val),
                              np.array(ids, dtype=np.int32)[best], labels)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels.astype(np.int32)
    # reassignment can disconnect a label into pieces: relabel connected parts
    return _relabel_connected(labels, mask)


def _relabel_connected(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros_like(labels)
    next_label = 1
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or out[sy, sx] or not labels[sy, sx]:
                continue
            src = labels[sy, sx]
            q = deque([(sy, sx)])
            out[sy, sx] = next_label
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                            and labels[ny, nx] == src and not out[ny, nx]):
                        out[ny, nx] = next_label
                        q.append((ny, nx))
            next_label += 1
    return out


def _merge_similar_planes(labels: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                          z: np.ndarray) -> np.ndarray:
    """Merge adjacent facets that belong to the same plane.

    Region growing can fragment a single plane (running-mean drift, junction
    noise). Two adjacent facets merge when their circular-mean aspects and
    median slopes agree and elevation is continuous across their boundary.
    """
    h, w = labels.shape
    while True:
        ids = [int(i) for i in np.unique(labels) if i > 0]
        stats = {}
        for i in ids:
            sel = labels == i
            a = np.radians(aspect[sel])
            stats[i] = (
                float(np.median(slope[sel])),
                math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0,
            )
        # adjacency with boundary elevation continuity
        boundary_dz: dict[tuple[int, int], list[float]] = defaultdict(list)
        for y in range(h):
            for x in range(w):
                l1 = labels[y, x]
                if not l1:
                    continue
                for dy, dx in ((1, 0), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        l2 = labels[ny, nx]
                        if l2 and l2 != l1:
                            key = (min(l1, l2), max(l1, l2))
                            boundary_dz[key].append(abs(float(z[y, x]) - float(z[ny, nx])))
        merged = False
        for (i, j), dzs in boundary_dz.items():
            si, sj = stats.get(i), stats.get(j)
            if si is None or sj is None:
                continue
            slope_ok = abs(si[0] - sj[0]) <= 5.0
            aspect_ok = _ang_diff(si[1], sj[1]) <= 20.0
            cont_ok = float(np.median(dzs)) <= 0.25
            if slope_ok and aspect_ok and cont_ok:
                labels[labels == j] = i
                merged = True
                break  # recompute stats after each merge
        if not merged:
            break
    return labels


def _cluster_pixels(pixels: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Split boundary pixels into 8-connected clusters (edge segments)."""
    pset = set(pixels)
    seen: set[tuple[int, int]] = set()
    clusters = []
    for p in pixels:
        if p in seen:
            continue
        q = deque([p])
        seen.add(p)
        cl = []
        while q:
            y, x = q.popleft()
            cl.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    np_ = (y + dy, x + dx)
                    if np_ in pset and np_ not in seen:
                        seen.add(np_)
                        q.append(np_)
        clusters.append(cl)
    return clusters


def _pca_segment(
    cluster: list[tuple[int, int]], z: np.ndarray, px_x: float, px_y: float
) -> tuple[float, float, float]:
    """Length (plan, m), compass direction (deg), elevation change (m) of a cluster."""
    pts = np.array([(x * px_x, y * px_y) for y, x in cluster], dtype=np.float64)
    if len(pts) < 2:
        return max(px_x, px_y), 0.0, 0.0
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]  # (east, south) components
    proj = centered @ axis
    length = float(proj.max() - proj.min()) + float(max(px_x, px_y))  # + one pixel
    # compass direction of the axis
    east, south = axis
    direction = (math.degrees(math.atan2(east, -south)) + 360.0) % 360.0
    # elevation change along axis via least-squares fit (robust to jagged pixels)
    zvals = np.array([z[y, x] for y, x in cluster], dtype=np.float64)
    span = float(proj.max() - proj.min())
    if span > 1e-6:
        grad = float(np.polyfit(proj, zvals, 1)[0])
        dz = abs(grad) * length
    else:
        dz = 0.0
    return length, direction, dz


def _slope_corrected(plan_m: float, dz_m: float) -> float:
    return math.hypot(plan_m, dz_m)


# ---------------------------------------------------------------- main entry

def measure_from_dsm(
    dsm: np.ndarray,
    mask: np.ndarray,
    px_x: float,
    px_y: float,
    address: str,
    lat: float,
    lng: float,
) -> RoofMeasurements:
    """Full facet + edge extraction from a DSM and roof mask."""
    mask = mask.astype(bool)
    z = _smooth(dsm, mask)
    slope, aspect = _slope_aspect(z, px_x, px_y)
    # Gradients are unreliable where the stencil touches background: erode the
    # mask before segmenting, then grow labels back out to the full mask.
    valid = _erode(mask, 2)
    labels = _segment_facets(slope, aspect, valid)
    labels = _grow_labels(labels, mask)
    px_area_m2 = px_x * px_y
    labels = _merge_slivers(labels, px_area_m2)
    labels = _dissolve_thin_facets(labels, mask, min(px_x, px_y))
    labels = _refine_by_planes(labels, z, mask, px_x, px_y)
    labels = _merge_similar_planes(labels, slope, aspect, z)
    labels = _merge_slivers(labels, px_area_m2)

    ids = [int(i) for i in np.unique(labels) if i > 0]

    # ---- facet stats ----
    facet_stats: dict[int, dict] = {}
    for i in ids:
        sel = labels == i
        n = int(sel.sum())
        med_slope = float(np.median(slope[sel]))
        a = np.radians(aspect[sel])
        med_aspect = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)
        plan_m2 = n * px_area_m2
        surf_m2 = plan_m2 / max(math.cos(math.radians(med_slope)), 0.2)
        facet_stats[i] = {
            "n": n,
            "slope": med_slope,
            "aspect": med_aspect,
            "plan_m2": plan_m2,
            "surf_m2": surf_m2,
            "mean_z": float(z[sel].mean()),
            "max_z": float(z[sel].max()),
            "min_z": float(z[sel].min()),
        }

    # ---- boundary collection ----
    h, w = labels.shape
    internal: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    perimeter: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for y in range(h):
        for x in range(w):
            l1 = labels[y, x]
            if not l1:
                continue
            is_perimeter = False
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                l2 = labels[ny, nx] if (0 <= ny < h and 0 <= nx < w) else 0
                if l2 == l1:
                    continue
                if l2 == 0:
                    is_perimeter = True
                elif (dy, dx) in ((1, 0), (0, 1)):
                    # record internal boundaries once (down/right only)
                    key = (min(l1, l2), max(l1, l2))
                    internal[key].append((y, x))
            if is_perimeter:
                perimeter[l1].append((y, x))

    edges = EdgeSummary()
    facet_mean_z = {i: s["mean_z"] for i, s in facet_stats.items()}

    # ---- classify internal boundaries: ridge / hip / valley ----
    for (i, j), pix in internal.items():
        si, sj = facet_stats[i], facet_stats[j]
        for cluster in _cluster_pixels(pix):
            plan_m, direction, dz = _pca_segment(cluster, z, px_x, px_y)
            length_ft = _slope_corrected(plan_m, dz) * M_TO_FT
            if length_ft < MIN_EDGE_LEN_FT:
                continue
            zb = float(np.mean([z[y, x] for y, x in cluster]))
            higher = zb >= (facet_mean_z[i] + facet_mean_z[j]) / 2.0
            aspect_gap = _ang_diff(si["aspect"], sj["aspect"])
            if not higher:
                edges.valleys_ft += length_ft
                edges.valley_count += 1
            else:
                # ridge: near-level boundary between opposing slopes
                level = dz < max(0.6, 0.05 * plan_m)
                if level and aspect_gap > 120.0:
                    edges.ridges_ft += length_ft
                    edges.ridge_count += 1
                elif level and aspect_gap <= 120.0:
                    # level high boundary between similar-facing facets: treat as ridge
                    edges.ridges_ft += length_ft
                    edges.ridge_count += 1
                else:
                    edges.hips_ft += length_ft
                    edges.hip_count += 1

    # ---- classify perimeter boundaries: eave / rake ----
    # Perimeter pixel runs can turn corners (eave meeting rake), so classify
    # each pixel by its LOCAL tangent direction first, then cluster per class.
    for i, pix in perimeter.items():
        s = facet_stats[i]
        pset = set(pix)
        eave_pix: list[tuple[int, int]] = []
        rake_pix: list[tuple[int, int]] = []
        for (y, x) in pix:
            neigh = [
                (ny, nx)
                for ny in range(y - 2, y + 3)
                for nx in range(x - 2, x + 3)
                if (ny, nx) in pset
            ]
            if len(neigh) < 3:
                eave_pix.append((y, x))
                continue
            pts = np.array([(nx * px_x, ny * px_y) for ny, nx in neigh])
            c = pts - pts.mean(axis=0)
            cov = c.T @ c / len(pts)
            evals, evecs = np.linalg.eigh(cov)
            east, south = evecs[:, np.argmax(evals)]
            direction = (math.degrees(math.atan2(east, -south)) + 360.0) % 360.0
            rel = _ang_diff(direction, s["aspect"])
            rel = min(rel, 180.0 - rel)  # tangent is bidirectional
            (eave_pix if rel > 45.0 else rake_pix).append((y, x))
        for cluster in _cluster_pixels(eave_pix):
            plan_m, _, _ = _pca_segment(cluster, z, px_x, px_y)
            if plan_m * M_TO_FT < MIN_EDGE_LEN_FT:
                continue
            edges.eaves_ft += plan_m * M_TO_FT  # eaves are level: plan length
            edges.eave_count += 1
        for cluster in _cluster_pixels(rake_pix):
            plan_m, _, dz = _pca_segment(cluster, z, px_x, px_y)
            if plan_m * M_TO_FT < MIN_EDGE_LEN_FT:
                continue
            edges.rakes_ft += _slope_corrected(plan_m, dz) * M_TO_FT
            edges.rake_count += 1

    # ---- assemble measurements ----
    order = sorted(ids, key=lambda i: facet_stats[i]["surf_m2"])
    facets: list[Facet] = []
    areas_per_pitch: dict[str, float] = defaultdict(float)
    for rank, i in enumerate(order):
        s = facet_stats[i]
        label = _facet_label(rank)
        pitch = slope_deg_to_pitch(s["slope"])
        surf_sqft = s["surf_m2"] * SQM_TO_SQFT
        facets.append(
            Facet(
                label=label,
                plan_area_sqft=round(s["plan_m2"] * SQM_TO_SQFT, 1),
                surface_area_sqft=round(surf_sqft, 1),
                pitch=pitch,
                slope_deg=round(s["slope"], 1),
                azimuth_deg=round(s["aspect"], 1),
                mean_height_ft=round(s["mean_z"] * M_TO_FT, 1),
            )
        )
        areas_per_pitch[pitch] += surf_sqft

    total_sqft = sum(f.surface_area_sqft for f in facets)
    predominant = max(areas_per_pitch.items(), key=lambda kv: kv[1])[0] if areas_per_pitch else "0/12"

    for k in ("ridges_ft", "hips_ft", "valleys_ft", "rakes_ft", "eaves_ft"):
        setattr(edges, k, round(getattr(edges, k)))

    return RoofMeasurements(
        address=address,
        latitude=lat,
        longitude=lng,
        total_area_sqft=round(total_sqft),
        facets=facets,
        edges=edges,
        predominant_pitch=predominant,
        areas_per_pitch={k: round(v, 1) for k, v in sorted(areas_per_pitch.items(), key=lambda kv: int(kv[0].split("/")[0]))},
        method="dsm",
    )


def _facet_label(rank: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    rank += 1
    while rank > 0:
        rank, rem = divmod(rank - 1, 26)
        label = letters[rem] + label
    return label


# -------------------------------------------------- fallback (segments only)

def estimate_from_segments(insights: dict, address: str, lat: float, lng: float) -> RoofMeasurements:
    """Estimate measurements from buildingInsights roofSegmentStats only.

    Used when the DSM data layer is unavailable. Edge lengths are heuristic
    estimates (flagged in notes) derived from segment areas and pitches.
    """
    sp = insights.get("solarPotential", {})
    segs = sp.get("roofSegmentStats", [])
    facets: list[Facet] = []
    areas_per_pitch: dict[str, float] = defaultdict(float)
    total_plan_m2 = 0.0
    total_surf_sqft = 0.0
    for idx, s in enumerate(sorted(segs, key=lambda s: s.get("stats", {}).get("areaMeters2", 0))):
        pitch_deg = float(s.get("pitchDegrees", 0.0))
        az = float(s.get("azimuthDegrees", 0.0))
        area_m2 = float(s.get("stats", {}).get("areaMeters2", 0.0))
        plan_m2 = float(s.get("stats", {}).get("groundAreaMeters2", area_m2 * math.cos(math.radians(pitch_deg))))
        pitch = slope_deg_to_pitch(pitch_deg)
        surf_sqft = area_m2 * SQM_TO_SQFT
        total_plan_m2 += plan_m2
        total_surf_sqft += surf_sqft
        facets.append(
            Facet(
                label=_facet_label(idx),
                plan_area_sqft=round(plan_m2 * SQM_TO_SQFT, 1),
                surface_area_sqft=round(surf_sqft, 1),
                pitch=pitch,
                slope_deg=round(pitch_deg, 1),
                azimuth_deg=round(az, 1),
            )
        )
        areas_per_pitch[pitch] += surf_sqft

    n = max(len(facets), 1)
    # Heuristic edge model calibrated to typical hip/gable residential geometry.
    footprint_m2 = total_plan_m2
    side = math.sqrt(max(footprint_m2, 1.0))
    perimeter_m = 4.4 * side          # cut-up factor vs perfect square
    complexity = min(1.0 + 0.05 * max(n - 4, 0), 1.8)
    perimeter_m *= complexity
    eaves_m = perimeter_m * 0.75
    rakes_m = perimeter_m * 0.25
    ridge_hip_m = side * 1.5 * complexity
    valleys_m = side * 0.5 * max(n - 6, 0) / 6.0

    edges = EdgeSummary(
        ridges_ft=round(ridge_hip_m * 0.4 * M_TO_FT),
        hips_ft=round(ridge_hip_m * 0.6 * M_TO_FT),
        valleys_ft=round(valleys_m * M_TO_FT),
        rakes_ft=round(rakes_m * M_TO_FT),
        eaves_ft=round(eaves_m * M_TO_FT),
    )
    predominant = max(areas_per_pitch.items(), key=lambda kv: kv[1])[0] if areas_per_pitch else "0/12"
    m = RoofMeasurements(
        address=address,
        latitude=lat,
        longitude=lng,
        total_area_sqft=round(total_surf_sqft),
        facets=facets,
        edges=edges,
        predominant_pitch=predominant,
        areas_per_pitch={k: round(v, 1) for k, v in areas_per_pitch.items()},
        method="segments",
    )
    m.notes.append(
        "Edge lengths (ridges/hips/valleys/rakes/eaves) are HEURISTIC ESTIMATES from "
        "roof segment statistics; high-resolution DSM was unavailable for this address. "
        "Field verify before material order."
    )
    m.imagery_date = _fmt_date(insights.get("imageryDate"))
    m.imagery_quality = insights.get("imageryQuality", "")
    return m


def _fmt_date(d: dict | None) -> str:
    if not d:
        return ""
    return f"{d.get('month', 0):02d}/{d.get('day', 0):02d}/{d.get('year', 0)}"
