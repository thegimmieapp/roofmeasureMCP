"""Straight-skeleton roof wireframe.

Builds a geometrically consistent roof model from the regularized footprint:
each eave edge carries a plane rising inward at its measured pitch; the roof
surface is the lower envelope of those planes; the creases where planes meet
are the ridges, hips, and valleys. By construction every ridge is parallel to
its eaves, hips bisect outside corners, valleys sit at inside corners, and all
lines meet at shared nodes, exactly like an EagleView wireframe.

Rake (gable) edges carry no plane: the neighboring eave planes extend to the
rake, so the ridge runs out to the gable end automatically.

The DSM contributes only: the footprint, eave-vs-rake classification, and the
pitch of each eave edge. All wireframe geometry is analytic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

from .geometry import MeasureDetail

M_TO_FT = 3.280839895
NODE_SNAP_M = 1.2         # merge nodes within this distance
MIN_CREASE_FT = 2.0       # drop creases shorter than this
GRID_RES_M = 0.15         # skeleton raster resolution


@dataclass
class SkelEdge:
    kind: str                 # ridge | hip | valley | eave | rake
    p0: tuple                 # (y, x) raster px coords
    p1: tuple
    length_ft: float          # slope-corrected where applicable
    plan_ft: float = 0.0


@dataclass
class RoofSkeleton:
    outline: np.ndarray                     # Nx2 (y, x) px
    edges: list = field(default_factory=list)          # [SkelEdge] perimeter + creases
    face_regions: np.ndarray | None = None  # region raster (edge index per px, -1 outside)
    face_pitch: dict = field(default_factory=dict)     # edge idx -> pitch str
    face_area_sqft: dict = field(default_factory=dict)  # edge idx -> surface sqft
    face_centroids: dict = field(default_factory=dict)  # edge idx -> (y, x) px
    face_downslope: dict = field(default_factory=dict)   # edge idx -> (dy, dx) unit
    grid_origin: tuple = (0.0, 0.0)         # (y0, x0) px offset of region raster
    grid_step_px: float = 1.0
    totals: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.edges) > 0


def _polygon_orientation(poly: np.ndarray) -> float:
    """>0 if CCW in (x, y) with y flipped (raster). We just need consistency."""
    x = poly[:, 1]
    y = poly[:, 0]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def build_skeleton(detail: MeasureDetail, outline: np.ndarray,
                   edge_kinds: list[str], edge_pitches: list[float],
                   px_m: float) -> RoofSkeleton:
    """Compute the roof wireframe.

    outline: Nx2 (y, x) px, simple polygon (vertex i -> i+1 is edge i)
    edge_kinds: 'eave' | 'rake' per outline edge
    edge_pitches: slope in degrees per edge (used for eaves)
    px_m: meters per raster pixel
    """
    n = len(outline)
    if n < 3:
        return RoofSkeleton(outline=outline)

    # ---- per-edge geometry ----
    path = MplPath(outline[:, ::-1])  # (x, y)

    # determine the rotation sign that points normals into the polygon,
    # empirically (robust to vertex order convention)
    flip = 1.0
    for i in range(n):
        a, b = outline[i].astype(float), outline[(i + 1) % n].astype(float)
        v = b - a
        L = math.hypot(v[0], v[1])
        if L < 4:
            continue
        u = v / L
        cand = np.array([-u[1], u[0]])
        mid = (a + b) / 2.0
        p_in = mid + cand * 4.0
        p_out = mid - cand * 4.0
        in1 = path.contains_point((p_in[1], p_in[0]))
        in2 = path.contains_point((p_out[1], p_out[0]))
        if in1 != in2:
            flip = 1.0 if in1 else -1.0
            break

    a_pts, dirs, normals, tans = [], [], [], []
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        v = b - a
        L = math.hypot(v[0], v[1])
        if L < 1e-9:
            v = np.array([1.0, 0.0]); L = 1.0
        u = v / L
        nrm = np.array([-u[1], u[0]]) * flip
        a_pts.append(a.astype(float))
        dirs.append(u)
        normals.append(nrm)
        tans.append(math.tan(math.radians(max(edge_pitches[i], 5.0))))

    eave_idx = [i for i in range(n) if edge_kinds[i] == "eave"]
    if len(eave_idx) < 2:
        return RoofSkeleton(outline=outline)

    # ---- reflex/convex flags per vertex (vertex i joins edge i-1 and edge i) ----
    S = _polygon_orientation(outline)
    reflex = []
    for i in range(n):
        u_prev = dirs[(i - 1) % n]
        u_next = dirs[i]
        cz = u_prev[1] * u_next[0] - u_prev[0] * u_next[1]  # cross in (x, y)
        reflex.append(bool((cz > 0) != (S > 0)) and abs(cz) > 1e-6)

    # ---- raster partition: region = argmin over eave planes of d_i * tan_i ----
    step = max(GRID_RES_M / px_m, 0.5)  # grid step in px units
    y0, y1 = outline[:, 0].min(), outline[:, 0].max()
    x0, x1 = outline[:, 1].min(), outline[:, 1].max()
    gy = np.arange(y0, y1 + step, step)
    gx = np.arange(x0, x1 + step, step)
    YY, XX = np.meshgrid(gy, gx, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])  # (x, y) for Path
    inside = path.contains_points(pts).reshape(YY.shape)

    Z = np.full((len(eave_idx), *YY.shape), np.inf)
    for k, i in enumerate(eave_idx):
        d = (YY - a_pts[i][0]) * normals[i][0] + (XX - a_pts[i][1]) * normals[i][1]
        z = d * tans[i] * px_m  # meters of rise
        z[d < -1.0] = np.inf    # only the interior side of the supporting line
        # Face band: a face only claims territory within its segment's sweep.
        # At a convex corner the face shrinks under the hip; at a reflex corner
        # it expands past the valley; at a rake (gable) it stops at the edge.
        proj = (YY - a_pts[i][0]) * dirs[i][0] + (XX - a_pts[i][1]) * dirs[i][1]
        L = math.hypot(*(outline[(i + 1) % n] - outline[i]))

        def _end_slack(vertex: int, neighbor: int):
            if edge_kinds[neighbor] == "rake":
                return 0.0 * d
            rate = tans[i] / max(tans[neighbor], 0.05)
            return (-d * rate) if reflex[vertex] else (d * rate)

        lo = _end_slack(i, (i - 1) % n)
        hi = L - _end_slack((i + 1) % n, (i + 1) % n)
        # soft penalty outside the band: keeps the partition hole-free while
        # suppressing claims far beyond the face's real extent
        outside = np.maximum(np.maximum(lo - 2.0 - proj, proj - (hi + 2.0)), 0.0)
        Z[k] = z + outside * tans[i] * px_m * 2.0
    region_k = np.argmin(Z, axis=0)
    zmin = np.min(Z, axis=0)
    region = np.where(inside & np.isfinite(zmin), region_k, -1)

    # ---- collect crease boundaries between region pairs ----
    pair_pixels: dict[tuple[int, int], list] = defaultdict(list)
    h, w = region.shape
    for dy, dx in ((1, 0), (0, 1)):
        r1 = region[: h - dy or None, : w - dx or None]
        r2 = region[dy:, dx:]
        mism = (r1 >= 0) & (r2 >= 0) & (r1 != r2)
        ys, xs = np.where(mism)
        for y, x in zip(ys, xs):
            i, j = int(r1[y, x]), int(r2[y, x])
            key = (min(i, j), max(i, j))
            pair_pixels[key].append((gy[y], gx[x]))

    def z_at(p: np.ndarray) -> float:
        best = np.inf
        for k, i in enumerate(eave_idx):
            d = (p[0] - a_pts[i][0]) * normals[i][0] + (p[1] - a_pts[i][1]) * normals[i][1]
            if d >= -1.0:
                best = min(best, d * tans[i] * px_m)
        return 0.0 if not np.isfinite(best) else max(best, 0.0)

    # ---- analytic crease lines, clipped to their pixel evidence ----
    creases = []
    min_len_px = (MIN_CREASE_FT / M_TO_FT) / px_m
    for (ki, kj), pix in pair_pixels.items():
        if len(pix) < 3:
            continue
        i, j = eave_idx[ki], eave_idx[kj]
        # plane equality: t_i * (n_i . p - c_i) = t_j * (n_j . p - c_j)
        Nvec = tans[i] * normals[i] - tans[j] * normals[j]
        ci = normals[i] @ a_pts[i]
        cj = normals[j] @ a_pts[j]
        rhs = tans[i] * ci - tans[j] * cj
        pixarr = np.array(pix)
        nn = math.hypot(Nvec[0], Nvec[1])
        if nn < 1e-9:
            # parallel identical-weight planes: PCA fallback
            c = pixarr.mean(axis=0)
            cc = pixarr - c
            evals, evecs = np.linalg.eigh(cc.T @ cc / len(cc))
            u = evecs[:, int(np.argmax(evals))]
            p_on = c
        else:
            Nn = Nvec / nn
            u = np.array([-Nn[1], Nn[0]])
            c = pixarr.mean(axis=0)
            p_on = c - ((c @ Nvec - rhs) / nn) * Nn
        proj = (pixarr - p_on) @ u
        p0 = p_on + u * float(proj.min())
        p1 = p_on + u * float(proj.max())
        if math.hypot(*(p1 - p0)) < min_len_px:
            continue
        # classify: adjacent edges share a vertex -> hip (convex) / valley (reflex)
        # vertices of edge i: i and i+1; of edge j: j and j+1
        vi = {i, (i + 1) % n}
        vj = {j, (j + 1) % n}
        shared = vi & vj
        anti = normals[i] @ normals[j] < -0.7
        if shared:
            v = shared.pop()
            kind = "valley" if reflex[v] else "hip"
        elif anti:
            kind = "ridge"
        else:
            # non-adjacent, non-parallel: valley if the crease starts near a
            # reflex vertex, else hip
            kind = "hip"
            for v in range(n):
                if reflex[v]:
                    dv = min(math.hypot(*(p0 - outline[v])), math.hypot(*(p1 - outline[v])))
                    if dv < NODE_SNAP_M / px_m * 2.5:
                        kind = "valley"
                        break
        creases.append({"p0": p0, "p1": p1, "kind": kind, "pair": (i, j)})

    # ---- node snapping: creases to polygon vertices, rakes, and each other ----
    snap_px = NODE_SNAP_M / px_m

    def snap_point(p: np.ndarray) -> np.ndarray:
        # to polygon vertex
        dv = np.hypot(outline[:, 0] - p[0], outline[:, 1] - p[1])
        k = int(np.argmin(dv))
        if dv[k] < snap_px * 1.6:
            return outline[k].astype(float)
        # to rake edge line (gable ridge end)
        for i in range(n):
            if edge_kinds[i] != "rake":
                continue
            a, u = a_pts[i], dirs[i]
            t = (p - a) @ u
            L = math.hypot(*(outline[(i + 1) % n] - outline[i]))
            t = min(max(t, 0.0), L)
            q = a + u * t
            if math.hypot(*(p - q)) < snap_px * 1.6:
                return q
        return p

    endpoints = []
    for c in creases:
        c["p0"] = snap_point(np.asarray(c["p0"]))
        c["p1"] = snap_point(np.asarray(c["p1"]))
        endpoints.append(c["p0"])
        endpoints.append(c["p1"])
    # cluster remaining endpoints
    endpoints = [np.asarray(p) for p in endpoints]
    merged: list[np.ndarray] = []
    assign = {}
    for idx, p in enumerate(endpoints):
        placed = False
        for mi, mp in enumerate(merged):
            if math.hypot(*(p - mp)) < snap_px:
                merged[mi] = (mp * assign.get(mi, 1) + p) / (assign.get(mi, 1) + 1)
                assign[mi] = assign.get(mi, 1) + 1
                placed = True
                assign[("pt", idx)] = mi
                break
        if not placed:
            merged.append(p.copy())
            assign[("pt", idx)] = len(merged) - 1
    for cidx, c in enumerate(creases):
        c["p0"] = merged[assign[("pt", cidx * 2)]]
        c["p1"] = merged[assign[("pt", cidx * 2 + 1)]]

    # ---- build SkelEdge list ----
    ft = px_m * M_TO_FT
    edges: list[SkelEdge] = []
    for i in range(n):
        a, b = outline[i].astype(float), outline[(i + 1) % n].astype(float)
        plan = math.hypot(*(b - a)) * ft
        if edge_kinds[i] == "rake":
            dz = abs(z_at(b) - z_at(a)) * M_TO_FT
            length = math.hypot(plan, dz)
        else:
            length = plan
        edges.append(SkelEdge(kind=edge_kinds[i], p0=tuple(a), p1=tuple(b),
                              length_ft=round(length, 1), plan_ft=round(plan, 1)))
    for c in creases:
        p0, p1 = np.asarray(c["p0"]), np.asarray(c["p1"])
        plan = math.hypot(*(p1 - p0)) * ft
        if plan < MIN_CREASE_FT:
            continue
        dz = abs(z_at(p1) - z_at(p0)) * M_TO_FT
        length = math.hypot(plan, dz) if c["kind"] in ("hip", "valley") else plan
        edges.append(SkelEdge(kind=c["kind"], p0=tuple(p0), p1=tuple(p1),
                              length_ft=round(length, 1), plan_ft=round(plan, 1)))

    # ---- faces: area / pitch / centroid per eave region ----
    sk = RoofSkeleton(outline=outline, edges=edges, face_regions=region,
                      grid_origin=(float(gy[0]), float(gx[0])), grid_step_px=step)
    px_area_sqft = (step * px_m) ** 2 * (M_TO_FT ** 2)
    for k, i in enumerate(eave_idx):
        sel = region == k
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        slope = math.degrees(math.atan(tans[i]))
        plan_sqft = cnt * px_area_sqft
        sk.face_area_sqft[i] = plan_sqft / max(math.cos(math.radians(slope)), 0.2)
        rise = round(tans[i] * 12)
        sk.face_pitch[i] = f"{rise}/12"
        ys, xs = np.where(sel)
        sk.face_centroids[i] = (float(gy[ys].mean()), float(gx[xs].mean()))
        sk.face_downslope[i] = (float(-normals[i][0]), float(-normals[i][1]))

    # ---- totals ----
    tot = defaultdict(float)
    cnts = defaultdict(int)
    for e in edges:
        tot[e.kind] += e.length_ft
        cnts[e.kind] += 1
    sk.totals = {
        "ridges_ft": round(tot["ridge"]), "hips_ft": round(tot["hip"]),
        "valleys_ft": round(tot["valley"]), "eaves_ft": round(tot["eave"]),
        "rakes_ft": round(tot["rake"]),
        "ridge_count": cnts["ridge"], "hip_count": cnts["hip"],
        "valley_count": cnts["valley"], "eave_count": cnts["eave"],
        "rake_count": cnts["rake"],
        "plan_area_sqft": round(sum(a * math.cos(math.atan(tans[i]))
                                    for i, a in sk.face_area_sqft.items())),
        "surface_area_sqft": round(sum(sk.face_area_sqft.values())),
    }
    return sk


def edge_pitch_from_dsm(detail: MeasureDetail, outline: np.ndarray,
                        edge_kinds: list[str], facet_slopes: dict[str, float],
                        default_slope: float) -> list[float]:
    """Assign each outline edge the DSM-measured slope of the facet behind it."""
    n = len(outline)
    labels = detail.labels
    h, w = labels.shape
    id_to_letter = {v: k for k, v in detail.facet_ids.items()}
    orient = _polygon_orientation(outline)
    out = []
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        v = b - a
        L = math.hypot(v[0], v[1])
        if L < 1e-9:
            out.append(default_slope)
            continue
        u = v / L
        nrm = np.array([-u[1], u[0]]) if orient > 0 else np.array([u[1], -u[0]])
        slopes = []
        for t in np.linspace(0.15, 0.85, 7):
            for depth in (8, 14, 20):  # px inward
                p = a + v * t + nrm * depth
                y, x = int(round(p[0])), int(round(p[1]))
                if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
                    letter = id_to_letter.get(int(labels[y, x]))
                    if letter in facet_slopes:
                        slopes.append(facet_slopes[letter])
                    break
        out.append(float(np.median(slopes)) if slopes else default_slope)
    return out
