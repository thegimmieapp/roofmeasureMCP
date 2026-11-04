"""LotMeasure: the property lot for landscape design.

From a street address: lot boundary and area (parcels.py), the house
footprint RoofMeasure already measures, the yard split into front / back /
left / right zones with surface classes (house, detached structure, tree
canopy, hardscape, lawn and planting, water, bare soil, unknown), setbacks
from each house wall to the lot line it faces, the largest clear plantable
rectangle per zone, sun exposure from the Solar API hourly shade layers,
and trees.

Rules carried over from RoofMeasure (non-negotiable):
- never fabricate or pad a number: an area is a pixel count of what the
  data shows; when there is no parcel polygon no lot size is reported
  (lot_source "none"); unknown stays unknown.
- tree canopy is never a boundary: lot lines come from the county parcel
  polygon, the house from RoofMeasure's completed (rectilinear) outline.

Geometry frame: tile pixels (row, col) with integer pixel centres
(geo.TileGeo); metric work in X east = col * px_x, Y north = -row * px_y.
All areas are plan (horizontal) areas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .geo import TileGeo

SQM_TO_SQFT = 10.76391041671
M_TO_FT = 3.280839895
SQFT_PER_ACRE = 43560.0

# ---- surface classes
OUT, HOUSE, STRUCT, CANOPY, HARD, LAWN, WATER, BARE, UNKNOWN = range(9)
CLASS_NAMES = {HOUSE: "house", STRUCT: "detached_structure", CANOPY: "tree_canopy", HARD: "hardscape",
               LAWN: "lawn_planting", WATER: "pool_water", BARE: "bare_soil_or_mulch", UNKNOWN: "unknown"}

# ---- classifier thresholds (calibrated on the 8 Springfield / Republic tiles, tests/calibrate_lot.py)
CANOPY_RESID_M = 0.12      # same canopy test as pipeline._assess_confidence
CANOPY_MIN_H_M = 2.0
CANOPY_GREEN_H_M = 2.5     # smooth-crowned trees: green and this tall
LOW_H_M = 0.30             # hardscape / lawn: height above local ground below this
SHRUB_MAX_H_M = 2.5        # green and below this: planting (shrubs, beds)
STRUCT_MIN_H_M = 1.8
STRUCT_PLANAR_M = 0.06
STRUCT_MIN_M2 = 6.0
EXG_GREEN = 0.04           # chromatic excess green (2g - r - b), g = G / (R + G + B)
EXG_GREEN_SHADOW = 0.015   # the same test for dark (shadowed) pixels
SHADOW_BRIGHT = 48         # mean RGB below this and not canopy: shadow -> unknown
GRAY_SPREAD = 38           # max - min channel below this: concrete / asphalt (hardscape)
SPECK_M2 = 1.0
GROUND_OPEN_M = 6.0        # local ground = grey opening of the DSM (removes objects narrower than this)

FULL_SUN_H = 6.0
PART_SUN_H = 3.0


# ====================================================================== data

@dataclass
class LotTiles:
    """Rasters for one address on the DSM grid (geo), plus the shade grids."""
    dsm: np.ndarray
    mask: np.ndarray
    geo: TileGeo
    px_x: float
    px_y: float
    rgb: np.ndarray | None = None
    flux: np.ndarray | None = None
    flux_geo: TileGeo | None = None
    shade: dict = field(default_factory=dict)       # month (1-12) -> (H x W x 24 int32, TileGeo)
    imagery_date: str = ""
    imagery_quality: str = ""


@dataclass
class Side:
    """One straight lot line (collinear parcel vertices merged)."""
    a: np.ndarray      # (X, Y) metres
    b: np.ndarray
    idx: int = 0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def mid(self) -> np.ndarray:
        return (self.a + self.b) / 2.0

    @property
    def dir(self) -> np.ndarray:
        d = self.b - self.a
        return d / (np.linalg.norm(d) + 1e-12)


# ====================================================================== geometry helpers

def px_to_xy(rows, cols, px_x, px_y):
    return np.asarray(cols, float) * px_x, -np.asarray(rows, float) * px_y


def xy_to_px(X, Y, px_x, px_y):
    return -np.asarray(Y, float) / px_y, np.asarray(X, float) / px_x


def rasterize_polygon(poly_xy, shape, px_x, px_y, r0=0, c0=0) -> np.ndarray:
    """Boolean mask (window starting at r0, c0) of pixel centres inside a shapely
    polygon given in metres. Handles MultiPolygon and holes."""
    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    if poly_xy is None or poly_xy.is_empty:
        return out
    for g in polygons_of(poly_xy):
        if not isinstance(g, Polygon) or g.area <= 0:
            continue
        minx, miny, maxx, maxy = g.bounds
        rr0, cc0 = xy_to_px(minx, maxy, px_x, px_y)
        rr1, cc1 = xy_to_px(maxx, miny, px_x, px_y)
        ya, yb = max(int(math.floor(rr0)) - r0, 0), min(int(math.ceil(rr1)) - r0 + 1, h)
        xa, xb = max(int(math.floor(cc0)) - c0, 0), min(int(math.ceil(cc1)) - c0 + 1, w)
        if ya >= yb or xa >= xb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        X, Y = px_to_xy(yy + r0, xx + c0, px_x, px_y)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = MplPath(np.asarray(g.exterior.coords)).contains_points(pts)
        for hole in g.interiors:
            inside &= ~MplPath(np.asarray(hole.coords)).contains_points(pts)
        out[ya:yb, xa:xb] |= inside.reshape(yy.shape)
    return out


def polygons_of(geom) -> list:
    """Polygon parts of any shapely geometry (Polygon, MultiPolygon, or the
    GeometryCollection an intersection returns when it also yields slivers of
    line or point)."""
    from shapely.geometry import Polygon

    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    out = []
    for g in getattr(geom, "geoms", []):
        out.extend(polygons_of(g))
    return out


def as_area(geom):
    """Drop line / point slivers: a (Multi)Polygon or an empty Polygon."""
    from shapely.geometry import MultiPolygon, Polygon

    parts = [p for p in polygons_of(geom) if p.area > 1e-6]
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def merge_sides(poly, angle_tol_deg: float = 10.0, simplify_m: float = 0.3) -> list[Side]:
    """Parcel polygon -> straight lot lines (collinear runs merged; a curved
    cul-de-sac frontage of many short segments stays one run while it bends
    less than angle_tol between consecutive segments... accumulated bend is
    capped at 35 degrees)."""
    p = poly.simplify(simplify_m, preserve_topology=True)
    pts = np.asarray(p.exterior.coords)[:-1]
    if len(pts) < 3:
        return []
    # start at the sharpest corner so a run never wraps across the ring start
    n = len(pts)

    def ang(i):
        d0 = pts[i] - pts[i - 1]
        d1 = pts[(i + 1) % n] - pts[i]
        c = np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-12)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    start = int(np.argmax([ang(i) for i in range(n)]))
    pts = np.roll(pts, -start, axis=0)
    sides = []
    a = pts[0]
    run_dir0 = None
    prev = pts[0]
    for i in range(1, n + 1):
        q = pts[i % n]
        seg = q - prev
        if np.linalg.norm(seg) < 1e-6:
            continue
        d = seg / np.linalg.norm(seg)
        if run_dir0 is None:
            run_dir0, last_dir = d, d
        else:
            bend = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(last_dir, d))))))
            total = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(run_dir0, d))))))
            if bend > angle_tol_deg or total > 35.0:
                sides.append(Side(np.array(a), np.array(prev)))
                a = prev
                run_dir0 = d
            last_dir = d
        prev = q
    sides.append(Side(np.array(a), np.array(prev)))
    for k, s in enumerate(sides):
        s.idx = k
    return sides


def _angle_between(d0, d1) -> float:
    c = abs(float(np.dot(d0, d1))) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def compass(vec) -> str:
    """Compass name of an (X east, Y north) direction."""
    az = (math.degrees(math.atan2(vec[0], vec[1])) + 360.0) % 360.0
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return names[int((az + 22.5) // 45) % 8]


def house_axes(house_poly) -> tuple[np.ndarray, np.ndarray]:
    """Two unit axes of the (rectilinear) house outline: length-weighted
    dominant edge direction modulo 90 degrees."""
    pts = np.asarray(house_poly.exterior.coords)
    acc = np.zeros(2)
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        th = math.atan2(d[1], d[0]) * 4.0      # mod 90 degrees
        acc += L * np.array([math.cos(th), math.sin(th)])
    th = math.atan2(acc[1], acc[0]) / 4.0
    a1 = np.array([math.cos(th), math.sin(th)])
    return a1, np.array([-a1[1], a1[0]])


def snap_to_axis(vec, axes) -> np.ndarray:
    best = None
    for a in axes:
        for s in (1.0, -1.0):
            c = float(np.dot(vec, s * a))
            if best is None or c > best[0]:
                best = (c, s * a)
    return best[1]


# ====================================================================== zones and setbacks

def frontage_outward(side: Side, lot_poly) -> np.ndarray:
    """Unit normal of a lot line pointing out of the lot (towards the street)."""
    n = np.array([side.dir[1], -side.dir[0]])
    from shapely.geometry import Point

    probe = side.mid + 0.5 * n
    return n if not lot_poly.contains(Point(probe[0], probe[1])) else -n


def split_zones(lot_poly, house_poly, u: np.ndarray, region=None) -> dict:
    """Front / back / left / right yard polygons.

    u: unit vector from the house towards the street (front). Front yard =
    lot beyond the house's most street-ward point (full lot width); back =
    beyond its rearmost point; the strip between is split left / right of the
    house as seen from the street (viewer facing -u). region: optional clip
    (tile coverage)."""
    from shapely.geometry import Polygon

    yard = lot_poly.difference(house_poly)
    if region is not None:
        yard = yard.intersection(region)
    hp = np.asarray(house_poly.exterior.coords)
    proj = hp @ u
    hmax, hmin = float(proj.max()), float(proj.min())
    right = np.array([-u[1], u[0]])        # viewer faces d = -u; right of d = (d_y, -d_x) = (-u_y, u_x)
    c = np.asarray(house_poly.centroid.coords[0])
    vc = float(c @ right)
    big = 1e4

    def halfplane(n, t):
        # {p : p . n >= t}
        o = n * t
        tdir = np.array([-n[1], n[0]])
        return Polygon([o + tdir * big, o - tdir * big, o - tdir * big + n * big, o + tdir * big + n * big])

    front = yard.intersection(halfplane(u, hmax))
    back = yard.intersection(halfplane(-u, -hmin))
    mid = yard.difference(front).difference(back)
    rgt = mid.intersection(halfplane(right, vc))
    lft = mid.difference(rgt)
    return {"front": as_area(front), "back": as_area(back), "left": as_area(lft), "right": as_area(rgt),
            "_u": u, "_right": right}


def wall_setbacks(house_poly, lot_poly, min_wall_m: float = 1.5, parallel_tol_deg: float = 25.0) -> list[dict]:
    """Perpendicular distance from each house wall to the lot line it faces.

    Rays leave the wall along its outward normal every 0.5 m; the first lot-line
    crossing is taken. The setback is the smallest such distance over the wall
    (zoning convention: nearest point). parallel = the lot line hit is within
    25 degrees of the wall."""
    from shapely.geometry import LineString, Point

    hp = np.asarray(house_poly.exterior.coords)
    ccw = house_poly.exterior.is_ccw
    boundary = lot_poly.exterior
    lot_pts = np.asarray(boundary.coords)
    out = []
    for i in range(len(hp) - 1):
        a, b = hp[i], hp[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < min_wall_m:
            continue
        d = d / L
        n = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])   # outward
        best = None
        ts = np.arange(0.25, L - 0.24, 0.5) if L > 0.5 else [L / 2]
        for t in ts:
            p0 = a + d * t + n * 0.01
            if not lot_poly.contains(Point(p0[0], p0[1])):
                continue
            ray = LineString([p0, p0 + n * 500.0])
            hit = ray.intersection(boundary)
            if hit.is_empty:
                continue
            pts = [np.asarray(g.coords[0]) for g in getattr(hit, "geoms", [hit])]
            dists = [float(np.dot(q - p0, n)) + 0.01 for q in pts]   # from the wall, not the probe start
            k = int(np.argmin(dists))
            dist, q = dists[k], pts[k]
            # the lot line segment that was hit
            segd = None
            for j in range(len(lot_pts) - 1):
                s = LineString([lot_pts[j], lot_pts[j + 1]])
                if s.distance(Point(q[0], q[1])) < 1e-6:
                    segd = lot_pts[j + 1] - lot_pts[j]
                    break
            par = segd is not None and _angle_between(segd, d) <= parallel_tol_deg
            if best is None or dist < best[0]:
                best = (dist, par, q)
        if best is None:
            continue
        out.append({"wall_index": i, "wall_length_m": L, "normal": n, "mid": (a + b) / 2,
                    "setback_m": best[0], "parallel_lot_line": bool(best[1]), "hit": best[2]})
    return out


def zone_setbacks(walls: list[dict], u: np.ndarray, right: np.ndarray) -> dict:
    """Setback per zone: nearest wall facing that zone's direction (within 45 degrees)."""
    dirs = {"front": u, "back": -u, "right": right, "left": -right}
    out = {}
    for z, v in dirs.items():
        cands = [w for w in walls if float(np.dot(w["normal"], v)) >= math.cos(math.radians(45))]
        if not cands:
            out[z] = None
            continue
        w = min(cands, key=lambda w: w["setback_m"])
        out[z] = {"m": w["setback_m"], "ft": round(w["setback_m"] * M_TO_FT, 1),
                  "parallel_lot_line": w["parallel_lot_line"], "wall_index": w["wall_index"],
                  "from_xy": (w["hit"] - w["normal"] * w["setback_m"]).tolist(), "to_xy": w["hit"].tolist()}
    return out


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle in a boolean grid.
    Returns (area_cells, row0, col0, rows, cols). Histogram-stack method, O(H*W)."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best = (0, 0, 0, 0, 0)
    for r in range(h):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(w + 1):
            hc = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][1] >= hc:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best[0]:
                    best = (area, r - sh + 1, s, sh, c - s)
                start = s
            stack.append((start, hc))
    return best


def clear_rectangle(zone_mask_fn, zone_poly, axes, step_m: float = 0.25):
    """Largest rectangle of plantable ground inside a zone, aligned to the house
    axes. zone_mask_fn(X, Y) -> bool array (plantable & in zone)."""
    if zone_poly is None or zone_poly.is_empty or zone_poly.area < 1.0:
        return None
    a1, a2 = axes
    pts = []
    for g in polygons_of(zone_poly):
        pts.extend(np.asarray(g.exterior.coords))
    if not pts:
        return None
    P = np.asarray(pts)
    s, t = P @ a1, P @ a2
    s0, s1, t0, t1 = s.min(), s.max(), t.min(), t.max()
    ns, nt = int((s1 - s0) / step_m) + 1, int((t1 - t0) / step_m) + 1
    if ns * nt > 2_500_000:
        step_m = math.sqrt((s1 - s0) * (t1 - t0) / 2_500_000)
        ns, nt = int((s1 - s0) / step_m) + 1, int((t1 - t0) / step_m) + 1
    S, T = np.meshgrid(s0 + (np.arange(ns) + 0.5) * step_m, t0 + (np.arange(nt) + 0.5) * step_m)
    X = S * a1[0] + T * a2[0]
    Y = S * a1[1] + T * a2[1]
    grid = zone_mask_fn(X, Y)
    area, r0, c0, nr, nc = largest_rectangle(grid)
    if area == 0:
        return None
    w_m, d_m = nc * step_m, nr * step_m
    cs = s0 + (c0 + nc / 2.0) * step_m
    ct = t0 + (r0 + nr / 2.0) * step_m
    centre = cs * a1 + ct * a2
    corners = [(s0 + c0 * step_m) * a1 + (t0 + r0 * step_m) * a2,
               (s0 + (c0 + nc) * step_m) * a1 + (t0 + r0 * step_m) * a2,
               (s0 + (c0 + nc) * step_m) * a1 + (t0 + (r0 + nr) * step_m) * a2,
               (s0 + c0 * step_m) * a1 + (t0 + (r0 + nr) * step_m) * a2]
    return {"width_m": w_m, "depth_m": d_m, "area_m2": w_m * d_m, "centre_xy": centre, "corners_xy": corners}


# ====================================================================== raster helpers

def popcount32(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.int64) & 0xFFFFFFFF
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) & 0xFFFFFFFF) >> 24


def sun_hours(shade: np.ndarray, days: int) -> np.ndarray:
    """Mean hours of direct sun per day from an hourly-shade month (H x W x 24
    int32 bitmasks, one bit per day). Invalid (-9999) -> NaN."""
    invalid = np.any(shade == -9999, axis=-1)
    hrs = popcount32(shade).sum(axis=-1).astype(float) / float(days)
    hrs[invalid] = np.nan
    return hrs


def resample_to(src: np.ndarray, src_geo: TileGeo, dst_geo: TileGeo, rows, cols) -> np.ndarray:
    """Nearest-neighbour sample of src at dst pixel centres (rows, cols arrays)."""
    if src_geo.epsg == dst_geo.epsg:
        X = dst_geo.x0 + (np.asarray(cols, float) + 0.5) * dst_geo.sx
        Y = dst_geo.y0 - (np.asarray(rows, float) + 0.5) * dst_geo.sy
        c = np.floor((X - src_geo.x0) / src_geo.sx).astype(int)
        r = np.floor((src_geo.y0 - Y) / src_geo.sy).astype(int)
    else:
        lat, lng = dst_geo.px_to_latlng(rows, cols)
        rf, cf = src_geo.latlng_to_px(lat, lng)
        r, c = np.round(rf).astype(int), np.round(cf).astype(int)
    h, w = src.shape[:2]
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    out = np.full(np.shape(rows) + src.shape[2:], np.nan, dtype=float)
    out[ok] = src[r[ok], c[ok]]
    return out


def majority3(cls: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """3x3 majority filter on class labels; frozen pixels keep their class."""
    counts = np.stack([ndi.uniform_filter((cls == k).astype(np.float32), size=3) for k in range(9)])
    maj = counts.argmax(axis=0).astype(cls.dtype)
    return np.where(frozen, cls, maj)


def drop_specks(cls: np.ndarray, min_px: int, frozen: np.ndarray) -> np.ndarray:
    """Components smaller than min_px take the most common neighbouring class."""
    out = cls.copy()
    for k in (STRUCT, CANOPY, HARD, LAWN, WATER, BARE, UNKNOWN):
        lab, n = ndi.label(out == k)
        if n == 0:
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        small = np.where(sizes < min_px)[0] + 1
        if len(small) == 0:
            continue
        objs = ndi.find_objects(lab)
        for i in small:
            sl = objs[i - 1]
            ys = slice(max(sl[0].start - 1, 0), sl[0].stop + 1)
            xs = slice(max(sl[1].start - 1, 0), sl[1].stop + 1)
            comp = lab[ys, xs] == i
            ring = ndi.binary_dilation(comp) & ~comp
            vals = out[ys, xs][ring]
            vals = vals[vals != OUT]
            if vals.size == 0:
                continue
            sub = out[ys, xs]
            sub[comp & ~frozen[ys, xs]] = np.bincount(vals, minlength=9).argmax()
    return out


def classify_surfaces(dsm, rgb, house, ground_coarse, px, nodata=None):
    """Per-pixel surface class over a window (see CLASS_NAMES). Returns
    (classes, hag, resid) where hag is height above local ground."""
    from .pipeline import local_plane_residual

    resid = local_plane_residual(dsm, px, px, k=4)
    ndsm = dsm - ground_coarse
    canopy = (resid > CANOPY_RESID_M) & (ndsm > CANOPY_MIN_H_M)
    # local ground: grey opening of the DSM with tall things replaced by the
    # coarse ground model (opening keeps a sloped yard's slope exactly)
    tall = house | canopy | (ndsm > STRUCT_MIN_H_M)
    base = np.where(tall, np.minimum(dsm, ground_coarse), dsm)
    k = max(int(GROUND_OPEN_M / px) | 1, 3)
    ground_local = ndi.grey_opening(base, size=(k, k))
    ground_local = ndi.uniform_filter(ground_local, size=max(int(1.0 / px), 1))
    hag = dsm - np.minimum(ground_local, dsm.max())
    cls = np.full(dsm.shape, UNKNOWN, dtype=np.uint8)
    if rgb is not None:
        R, G, B = (rgb[..., i].astype(np.float32) for i in range(3))
        S = R + G + B + 1e-3
        exg = (2 * G - R - B) / S
        bright = S / 3.0
        spread = np.max(rgb, axis=-1).astype(np.float32) - np.min(rgb, axis=-1).astype(np.float32)
        # in shadow the colour flattens: a weaker green excess still means vegetation
        green = (exg > EXG_GREEN) | ((bright < SHADOW_BRIGHT * 1.6) & (exg > EXG_GREEN_SHADOW) & (G > R))
        blue = (B > R + 25) & (B > G + 5) & (bright > 60)
        shadow = bright < SHADOW_BRIGHT
    else:
        green = blue = shadow = np.zeros(dsm.shape, bool)
        spread = np.zeros(dsm.shape, np.float32)
    low = hag < LOW_H_M
    planar_tall = (ndsm > STRUCT_MIN_H_M) & (resid < STRUCT_PLANAR_M)
    struct = ndi.binary_opening(planar_tall & ~house, iterations=2)
    lab, n = ndi.label(struct)
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        # roofs are not green: a smooth, tall, mostly green blob is a tree crown
        gfrac = ndi.mean(green.astype(float), lab, index=np.arange(1, n + 1))
        ok = (sizes * px * px >= STRUCT_MIN_M2) & (np.asarray(gfrac) < 0.35)
        keep = np.isin(lab, np.where(ok)[0] + 1)
        struct = ndi.binary_fill_holes(keep)
    canopy = canopy | (green & (hag > CANOPY_GREEN_H_M))
    if rgb is not None:
        cls[low & green] = LAWN
        cls[~low & green & (hag <= SHRUB_MAX_H_M)] = LAWN          # shrubs and beds
        cls[low & ~green & (spread < GRAY_SPREAD)] = HARD
        cls[low & ~green & (spread >= GRAY_SPREAD)] = BARE
        cls[low & blue & (resid < 0.05)] = WATER
        cls[shadow & ~green & ~canopy] = UNKNOWN
    cls[canopy & ~struct] = CANOPY
    cls[struct] = STRUCT
    cls[house] = HOUSE
    if nodata is not None:
        cls[nodata & ~house] = UNKNOWN
    return cls, hag, resid


def find_trees(canopy: np.ndarray, hag: np.ndarray, px: float, min_area_m2: float = 4.0):
    """Split canopy into trees: one marker per crown top (local maximum of the
    smoothed height, 3 m apart, taller than 3 m), canopy pixels go to the
    nearest marker inside the same canopy blob."""
    if not canopy.any():
        return []
    sm = ndi.uniform_filter(np.where(canopy, hag, 0.0), size=max(int(1.0 / px), 1))
    win = max(int(3.0 / px) | 1, 3)
    peaks = (sm == ndi.maximum_filter(sm, size=win)) & canopy & (sm > 3.0)
    mlab, nm = ndi.label(peaks)
    if nm == 0:
        return []
    _, (iy, ix) = ndi.distance_transform_edt(mlab == 0, return_indices=True)
    owner = mlab[iy, ix]
    blob, nb = ndi.label(canopy)
    # a marker only owns pixels in its own blob
    mblob = np.zeros(nm + 1, dtype=int)
    for i, (y, x) in enumerate(ndi.center_of_mass(peaks, mlab, range(1, nm + 1)), start=1):
        yy, xx = int(round(y)), int(round(x))
        mblob[i] = blob[yy, xx] if canopy[yy, xx] else 0
    owner = np.where(canopy & (mblob[owner] == blob), owner, 0)
    trees = []
    for i in range(1, nm + 1):
        m = owner == i
        n = int(m.sum())
        if n * px * px < min_area_m2:
            continue
        ys, xs = np.where(m)
        trees.append({"mask_idx": i, "area_m2": n * px * px, "row": float(ys.mean()), "col": float(xs.mean()),
                      "height_m": float(hag[m].max()), "pixels": (ys, xs)})
    return trees


# ====================================================================== orchestration

ROOF_WINDOW_M = 60.0      # RoofMeasure's own tile radius: the house is measured on the same window


def _ring_xy(ring_latlng, geo: TileGeo, px_x, px_y):
    r = np.asarray(ring_latlng, float)
    rows, cols = geo.latlng_to_px(r[:, 0], r[:, 1])
    return np.column_stack(px_to_xy(rows, cols, px_x, px_y))


def _xy_latlng(xy, geo: TileGeo, px_x, px_y):
    xy = np.atleast_2d(np.asarray(xy, float))
    rows, cols = xy_to_px(xy[:, 0], xy[:, 1], px_x, px_y)
    lat, lng = geo.px_to_latlng(rows, cols)
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lng))]


def _sqft(n_px, px_x, px_y) -> int:
    return int(round(float(n_px) * px_x * px_y * SQM_TO_SQFT))


def run_roof(tiles: LotTiles, gc: dict, insights: dict):
    """RoofMeasure on the central 120 m window of the lot tile, so the house
    footprint is the one the roof report would draw. Returns (m, detail,
    house_mask_full, outline_rc_full, (r0, c0))."""
    from . import pipeline

    rc, cc = tiles.geo.latlng_to_px(gc["lat"], gc["lng"])
    half = int(round(ROOF_WINDOW_M / tiles.px_x))
    h, w = tiles.dsm.shape
    r0, c0 = max(int(round(float(rc))) - half, 0), max(int(round(float(cc))) - half, 0)
    r1, c1 = min(r0 + 2 * half, h), min(c0 + 2 * half, w)
    dsm = tiles.dsm[r0:r1, c0:c1]
    mask = tiles.mask[r0:r1, c0:c1] > 0
    m, detail = pipeline.measure_from_rasters(dsm, mask, tiles.px_x, tiles.px_y, gc["formatted"],
                                              gc["lat"], gc["lng"], insights=insights)
    y0, y1, x0, x1 = detail.crop
    foot = detail.labels > 0
    if detail.roof_mask is not None:
        foot = foot | detail.roof_mask
    house = np.zeros((h, w), dtype=bool)
    house[r0 + y0:r0 + y1, c0 + x0:c0 + x1] = foot
    outline = None
    sk = getattr(detail, "skeleton", None)
    if sk is not None and getattr(sk, "outline", None) is not None and len(sk.outline) >= 3:
        outline = np.asarray(sk.outline, float) + np.array([r0 + y0, c0 + x0])
    return m, detail, house, outline, (r0, c0)


def _house_polygon(outline_rc, house_mask, px_x, px_y):
    """House polygon in metres: the completed rectilinear outline when the
    roof pipeline produced one, else the footprint's rotated bounding box
    trimmed to the mask (never the raw raster edge, which follows trees)."""
    from shapely.geometry import MultiPoint, Polygon

    if outline_rc is not None:
        X, Y = px_to_xy(outline_rc[:, 0], outline_rc[:, 1], px_x, px_y)
        p = Polygon(np.column_stack([X, Y])).buffer(0)
        if p.area > 5:
            if p.geom_type == "MultiPolygon":
                p = max(p.geoms, key=lambda g: g.area)
            return p, "completed rectilinear outline from RoofMeasure"
    ys, xs = np.where(house_mask)
    X, Y = px_to_xy(ys, xs, px_x, px_y)
    return MultiPoint(np.column_stack([X, Y])[::7]).minimum_rotated_rectangle, "rotated bounding box of the roof footprint"


def _frontage(sides, lot_poly, house_poly, cls_fn, neighbours_xy, roads_xy, route: str):
    """Pick the street-side lot line. Returns (side, rule, detail)."""
    from shapely.geometry import LineString, Point

    long_sides = [s for s in sides if s.length >= 3.0]
    if not long_sides:
        return None, "none", "no lot line longer than 3 m"
    # 1. OSM road centreline (named route preferred)
    if roads_xy:
        best = None
        for s in long_sides:
            for rd in roads_xy:
                line = LineString(rd["xy"])
                dist = line.distance(Point(*s.mid))
                if dist > 30.0:
                    continue
                # direction of the road near the side
                t = line.project(Point(*s.mid))
                p0 = np.asarray(line.interpolate(max(t - 3, 0)).coords[0])
                p1 = np.asarray(line.interpolate(t + 3).coords[0])
                if _angle_between(p1 - p0, s.dir) > 30:
                    continue
                named = bool(route) and rd.get("name", "").lower() == route.lower()
                key = (0 if named else 1, dist)
                if best is None or key < best[0]:
                    best = (key, s, rd.get("name", ""))
        if best is not None:
            return best[1], "osm_road", f"lot line nearest the {best[2] or 'road'} centreline ({best[0][1]:.0f} m)"
    # 2. road gap: no neighbouring parcel across this line
    gap = []
    if neighbours_xy:
        from shapely.ops import unary_union
        nb = unary_union([p for p in neighbours_xy if p.is_valid and p.area > 1])
        for s in long_sides:
            out_n = frontage_outward(s, lot_poly)
            probes = [s.a + (s.b - s.a) * f + out_n * 4.0 for f in (0.25, 0.5, 0.75)]
            free = sum(1 for p in probes if not nb.contains(Point(*p)))
            if free >= 2:
                gap.append(s)
    # driveway: hardscape inside the lot touching the line
    def drive_px(s):
        out_n = frontage_outward(s, lot_poly)
        cnt = 0
        for f in np.linspace(0.05, 0.95, 19):
            for dd in (0.5, 1.0, 1.5):
                p = s.a + (s.b - s.a) * f - out_n * dd
                cnt += int(cls_fn(p) == HARD)
        return cnt
    if gap:
        if len(gap) == 1:
            return gap[0], "neighbor_gap", "only lot line with no neighbouring parcel across it"
        scored = sorted(gap, key=lambda s: (-drive_px(s), house_poly.distance(LineString([s.a, s.b]))))
        why = "driveway touches it" if drive_px(scored[0]) >= 3 else "nearest the house"
        return scored[0], "neighbor_gap+driveway", f"{len(gap)} lot lines face open ground; chose the one the {why}"
    # 3. driveway alone
    scored = sorted(long_sides, key=lambda s: -drive_px(s))
    if drive_px(scored[0]) >= 3:
        return scored[0], "driveway", "lot line the driveway (hardscape) runs to"
    return None, "none", "street side could not be determined"


def measure_lot_from_tiles(tiles: LotTiles, gc: dict, insights: dict, parcel=None,
                           neighbours=None, roads=None, include_sun: bool = True,
                           provider_log=None, keep_rasters: bool = False) -> dict:
    """Everything after the downloads (offline calibration enters here)."""
    from shapely.geometry import Point, Polygon, box

    px_x, px_y = tiles.px_x, tiles.px_y
    px = px_x
    notes = []
    m, detail, house_full, outline_rc, _ = run_roof(tiles, gc, insights)
    house_poly, house_src = _house_polygon(outline_rc, house_full, px_x, px_y)
    h, w = tiles.dsm.shape
    tile_poly = box(0, -(h - 1) * px_y, (w - 1) * px_x, 0)

    lot_poly = None
    if parcel is not None:
        lot_poly = Polygon(_ring_xy(parcel.ring, tiles.geo, px_x, px_y)).buffer(0)
        if lot_poly.geom_type == "MultiPolygon":
            lot_poly = max(lot_poly.geoms, key=lambda g: g.area)
    # ---- analysis region
    if lot_poly is not None:
        region = lot_poly
        coverage = float(lot_poly.intersection(tile_poly).area / max(lot_poly.area, 1e-6))
    else:
        region = house_poly.buffer(15.0, join_style=2)
        coverage = 1.0
    work = as_area(region.intersection(tile_poly))
    minx, miny, maxx, maxy = work.buffer(6.0).intersection(tile_poly).bounds
    r0, c0 = xy_to_px(minx, maxy, px_x, px_y)
    r1, c1 = xy_to_px(maxx, miny, px_x, px_y)
    r0, c0 = max(int(math.floor(r0)), 0), max(int(math.floor(c0)), 0)
    r1, c1 = min(int(math.ceil(r1)) + 1, h), min(int(math.ceil(c1)) + 1, w)
    win = (slice(r0, r1), slice(c0, c1))
    shape = (r1 - r0, c1 - c0)
    dsm = tiles.dsm[win].astype(np.float64)
    nodata = dsm < -1000
    if nodata.any():
        dsm = np.where(nodata, np.nanmedian(np.where(nodata, np.nan, dsm)), dsm)
    from .pipeline import ground_model
    # coarse ground on a 20 m wider window so block edges do not cut the lot
    pr0, pc0 = max(r0 - 200, 0), max(c0 - 200, 0)
    pr1, pc1 = min(r1 + 200, h), min(c1 + 200, w)
    big = tiles.dsm[pr0:pr1, pc0:pc1].astype(np.float64)
    big = np.where(big < -1000, np.nanmedian(np.where(big < -1000, np.nan, big)), big)
    ground = ground_model(big, px_x, px_y)[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0]
    rgb = tiles.rgb[win] if tiles.rgb is not None else None
    house_w = house_full[win] | rasterize_polygon(house_poly, shape, px_x, px_y, r0, c0)
    cls, hag, _resid = classify_surfaces(dsm, rgb, house_w, ground, px, nodata=nodata)
    cls = majority3(cls, house_w)
    cls = drop_specks(cls, int(round(SPECK_M2 / (px_x * px_y))), house_w)
    inreg = rasterize_polygon(work, shape, px_x, px_y, r0, c0)
    cls_lot = np.where(inreg, cls, OUT).astype(np.uint8)

    def cls_at_xy(p):
        rr, cc = xy_to_px(p[0], p[1], px_x, px_y)
        rr, cc = int(round(float(rr))) - r0, int(round(float(cc))) - c0
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            return int(cls_lot[rr, cc])
        return OUT

    # ---- sun
    sun = {}
    rows, cols = np.mgrid[r0:r1, c0:c1]
    if include_sun:
        for mon, days in ((6, 30), (12, 31)):
            if mon in tiles.shade:
                arr, sg = tiles.shade[mon]
                sun[mon] = resample_to(sun_hours(arr, days), sg, tiles.geo, rows, cols)
        if tiles.flux is not None:
            fg = tiles.flux_geo or tiles.geo
            fl = resample_to(np.where(tiles.flux < -1000, np.nan, tiles.flux), fg, tiles.geo, rows, cols)
            sun["flux"] = fl
        if 6 not in sun:
            notes.append("Hourly shade layer unavailable for this location; sun classes not reported.")

    # ---- frontage, zones, setbacks
    axes = house_axes(house_poly)
    frontage = {"rule": "none", "detail": "no parcel boundary"}
    u = None
    if lot_poly is not None:
        sides = merge_sides(lot_poly)
        nb_xy = []
        for ring in (neighbours or []):
            try:
                nb_xy.append(Polygon(_ring_xy(ring, tiles.geo, px_x, px_y)).buffer(0))
            except Exception:
                pass
        roads_xy = []
        for rd in (roads or []):
            c = np.asarray(rd["coords"], float)
            rr, cc = tiles.geo.latlng_to_px(c[:, 0], c[:, 1])
            roads_xy.append({"name": rd.get("name", ""), "xy": np.column_stack(px_to_xy(rr, cc, px_x, px_y))})
        side, rule, why = _frontage(sides, lot_poly, house_poly, cls_at_xy, nb_xy, roads_xy, gc.get("route", ""))
        frontage = {"rule": rule, "detail": why}
        if side is not None:
            u = snap_to_axis(frontage_outward(side, lot_poly), axes)
            frontage["lot_line_latlng"] = _xy_latlng([side.a, side.b], tiles.geo, px_x, px_y)
            frontage["length_ft"] = round(side.length * M_TO_FT, 1)
            frontage["street_side_faces"] = compass(u)
    house_in_lot = 1.0
    if lot_poly is not None:
        house_in_lot = float(house_poly.intersection(lot_poly).area / max(house_poly.area, 1e-6))
    if house_in_lot < 0.5:
        # the building RoofMeasure found is not on this lot (typically the house
        # here is under tree cover and the neighbour's roof was picked up)
        notes.append(f"The building the satellite roof model found lies {100 - house_in_lot * 100:.0f}% outside "
                     "this lot, so it is most likely a neighbouring house (the house on this lot is probably "
                     "hidden by tree cover). House footprint, setbacks and front/back/side zones are not "
                     "reported; the lot is measured as one zone.")
        u = None
    named = u is not None and house_in_lot >= 0.5
    if u is None:
        # no street side: split by compass using the house axis nearest south
        u = snap_to_axis(np.array([0.0, -1.0]), axes)
    if house_in_lot < 0.5:
        from shapely.geometry import Polygon as _P
        zones_poly = {"whole_lot": as_area(lot_poly.intersection(tile_poly))}
        right = np.array([-u[1], u[0]])
        walls = []
        house_poly_eff = _P()
    else:
        zones_poly = split_zones(lot_poly if lot_poly is not None else region, house_poly, u, region=tile_poly)
        right = zones_poly.pop("_right")
        zones_poly.pop("_u")
        walls = wall_setbacks(house_poly, lot_poly) if lot_poly is not None else []
        house_poly_eff = house_poly
    zsb = zone_setbacks(walls, u, right) if walls else {}
    zdir = {"front": u, "back": -u, "right": right, "left": -right, "whole_lot": u}

    plantable = cls_lot == LAWN
    area_px = px_x * px_y

    def zone_mask(poly):
        return rasterize_polygon(poly, shape, px_x, px_y, r0, c0) & inreg

    def plant_fn_for(zm):
        def f(X, Y):
            rr, cc = xy_to_px(X, Y, px_x, px_y)
            rr = np.round(rr).astype(int) - r0
            cc = np.round(cc).astype(int) - c0
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            out = np.zeros(X.shape, bool)
            out[ok] = (plantable & zm)[rr[ok], cc[ok]]
            return out
        return f

    zones = {}
    plan_zones = {}
    for name, poly in zones_poly.items():
        key = name if (named or name == "whole_lot") else {
            "front": compass(zdir["front"]), "back": compass(zdir["back"]),
            "right": compass(zdir["right"]), "left": compass(zdir["left"])}[name]
        zm = zone_mask(poly)
        n = int(zm.sum())
        surf = {}
        for k, nm in CLASS_NAMES.items():
            if k == HOUSE:
                continue
            v = _sqft((cls_lot[zm] == k).sum(), px_x, px_y)
            if v:
                surf[nm + "_sqft"] = v
        z = {"area_sqft": _sqft(n, px_x, px_y), "faces": compass(zdir[name]) if name != "whole_lot" else "",
             "surfaces": surf}
        if n:
            z["lawn_planting_pct"] = round(100.0 * float((cls_lot[zm] == LAWN).sum()) / n)
        sb = zsb.get(name)
        z["setback_ft"] = sb["ft"] if sb else None
        if sb and not sb["parallel_lot_line"]:
            z["setback_note"] = "nearest lot line is not parallel to the wall (angled lot)"
        rect = clear_rectangle(plant_fn_for(zm), poly, axes)
        if rect and rect["area_m2"] >= 1.0:
            z["largest_clear_rectangle"] = {
                "width_ft": round(rect["width_m"] * M_TO_FT, 1), "depth_ft": round(rect["depth_m"] * M_TO_FT, 1),
                "area_sqft": round(rect["area_m2"] * SQM_TO_SQFT),
                "center_latlng": _xy_latlng(rect["centre_xy"], tiles.geo, px_x, px_y)[0]}
        else:
            z["largest_clear_rectangle"] = None
        if 6 in sun:
            pm = zm & plantable
            jun = sun[6][pm]
            full = int(np.sum(jun >= FULL_SUN_H)); part = int(np.sum((jun >= PART_SUN_H) & (jun < FULL_SUN_H)))
            shd = int(np.sum(jun < PART_SUN_H)); unk = int(np.sum(np.isnan(jun)))
            zsun = {"full_sun_sqft": _sqft(full, px_x, px_y), "part_sun_sqft": _sqft(part, px_x, px_y),
                    "shade_sqft": _sqft(shd, px_x, px_y)}
            if unk:
                zsun["no_sun_data_sqft"] = _sqft(unk, px_x, px_y)
            if pm.any():
                zsun["june_mean_sun_hours"] = round(float(np.nanmean(jun)), 1) if np.isfinite(jun).any() else None
                if 12 in sun:
                    dec = sun[12][pm]
                    zsun["december_mean_sun_hours"] = round(float(np.nanmean(dec)), 1) if np.isfinite(dec).any() else None
                if "flux" in sun:
                    fl = sun["flux"][pm]
                    zsun["annual_flux_kwh_per_kw"] = round(float(np.nanmean(fl))) if np.isfinite(fl).any() else None
            z["sun"] = zsun
        zones[key] = z
        plan_zones[key] = {"poly": poly, "rect": rect, "setback": sb}

    # ---- trees (crowns whose centre is inside the lot)
    canopy_w = cls == CANOPY
    trees_raw = find_trees(canopy_w, hag, px)
    dist_house = ndi.distance_transform_edt(~house_w) * px
    trees = []
    lot_or_region = lot_poly if lot_poly is not None else region
    for t in trees_raw:
        X, Y = px_to_xy(t["row"] + r0, t["col"] + c0, px_x, px_y)
        if not lot_or_region.contains(Point(float(X), float(Y))):
            continue
        ys, xs = t["pixels"]
        near = float(dist_house[ys, xs].min())
        lat, lng = tiles.geo.px_to_latlng(t["row"] + r0, t["col"] + c0)
        trees.append({"lat": round(float(lat), 7), "lng": round(float(lng), 7),
                      "canopy_diameter_ft": round(2 * math.sqrt(t["area_m2"] / math.pi) * M_TO_FT, 1),
                      "canopy_area_sqft": round(t["area_m2"] * SQM_TO_SQFT),
                      "height_ft": round(t["height_m"] * M_TO_FT),
                      "within_3m_of_house": near <= 3.0,
                      "distance_to_house_ft": round(near * M_TO_FT, 1),
                      "_xy": (float(X), float(Y)), "_r": math.sqrt(t["area_m2"] / math.pi)})
    trees.sort(key=lambda t: -t["canopy_area_sqft"])

    # ---- totals
    totals = {CLASS_NAMES[k] + "_sqft": _sqft((cls_lot == k).sum(), px_x, px_y) for k in CLASS_NAMES}
    yard_px = int(np.isin(cls_lot, [STRUCT, CANOPY, HARD, LAWN, WATER, BARE, UNKNOWN]).sum())
    canopy_frac = float((cls_lot == CANOPY).sum()) / max(int(inreg.sum()), 1)
    unknown_frac = float((cls_lot == UNKNOWN).sum()) / max(int(inreg.sum()), 1)

    # ---- confidence notes
    src_txt = (f"Lot line from {parcel.provider}" + ("" if parcel.verified_layer else " (unverified layer)")
               if parcel is not None else "No parcel boundary available")
    conf = [src_txt + "."]
    if parcel is not None and parcel.county_recorded:
        rec = parcel.county_recorded
        if "acres" in rec:
            conf.append(f"County recorded lot size {rec['acres']:.3f} acres ({rec['sqft']:,} sq ft, field {rec['field']}); "
                        f"the mapped polygon measures {parcel.area_acres:.3f} acres ({rec['difference_pct']:+.1f}%).")
        if "county_gis_sqft" in rec:
            conf.append(f"The county's own GIS area of the same polygon is {rec['county_gis_sqft']:,} sq ft "
                        f"({rec['county_gis_difference_pct']:+.2f}% from ours).")
    if parcel is None:
        conf.append("No lot size is reported. Yard figures cover a 15 m band around the house, not the lot.")
    if coverage < 0.999:
        conf.append(f"Only {coverage:.0%} of the lot lies inside the 100 m satellite tile; zone areas cover that part only.")
    conf.append(f"Tree canopy covers {canopy_frac:.0%} of the measured area; ground under canopy is reported as canopy, "
                "not lawn or hardscape.")
    if unknown_frac > 0.05:
        conf.append(f"{unknown_frac:.0%} of the area is unknown (deep shadow, vehicles, low objects or no data).")
    if lot_poly is not None:
        out_frac = float(house_poly.difference(lot_poly).area / max(house_poly.area, 1e-6))
        if out_frac > 0.03:
            conf.append(f"{out_frac:.0%} of the house outline falls outside the mapped lot line: county GIS lines and "
                        "imagery can be offset by a metre or two; setbacks near that side are approximate.")
    if frontage["rule"] == "none" and parcel is not None:
        conf.append("The street side could not be determined; zones are named by compass direction.")
    conf.append(f"Roof footprint confidence {getattr(m, 'confidence', '')} (canopy at the roof edge "
                f"{getattr(m, 'canopy_fraction', 0):.0%}); house outline: {house_src}.")
    conf.append("Surface classes come from elevation (DSM) plus the aerial colour image: shrubs and beds count as "
                "lawn/planting; bare soil or mulch is reported separately and NOT counted as plantable; "
                "pavers and brick can read as bare soil.")
    conf.extend(notes)
    if parcel is not None:
        conf.extend(parcel.notes)

    house_ring = np.asarray(house_poly.exterior.coords)
    result = {
        "address": gc["formatted"],
        "latitude": gc["lat"], "longitude": gc["lng"],
        "lot_source": parcel.source if parcel is not None else "none",
        "parcel": parcel.to_dict() if parcel is not None else None,
        "house": None if house_in_lot < 0.5 else {
            "plan_area_sqft": round(house_poly.area * SQM_TO_SQFT),
            "outline_latlng": _xy_latlng(house_ring[:-1], tiles.geo, px_x, px_y),
            "outline_source": house_src,
            "roof_area_sqft": m.total_area_sqft, "roof_squares": m.squares,
            "predominant_pitch": m.predominant_pitch, "roof_confidence": getattr(m, "confidence", ""),
            "nearby_detached_roofs": m.nearby_structures,
        },
        "frontage": frontage,
        "zones": zones,
        "zone_naming": ("front/back/left/right; left and right as seen from the street" if named
                        else "compass sides of the house (street side unknown)"),
        "setbacks": {z: zsb.get(z) and {"ft": zsb[z]["ft"], "parallel_lot_line": zsb[z]["parallel_lot_line"]}
                     for z in ("front", "back", "left", "right")} if walls else None,
        "walls": [{"faces": compass(wl["normal"]), "length_ft": round(wl["wall_length_m"] * M_TO_FT, 1),
                   "setback_ft": round(wl["setback_m"] * M_TO_FT, 1), "parallel_lot_line": wl["parallel_lot_line"]}
                  for wl in walls],
        "trees": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trees],
        "tree_count": len(trees),
        "trees_within_3m_of_house": sum(1 for t in trees if t["within_3m_of_house"]),
        "totals": {**totals, "yard_sqft": _sqft(yard_px, px_x, px_y),
                   "plantable_sqft": totals["lawn_planting_sqft"],
                   "hardscape_sqft": totals["hardscape_sqft"]},
        "sun_method": ("Solar API hourly shade, June and December; plantable area classed by June hours of direct "
                       "sun: full sun 6 h or more, part sun 3 to 6 h, shade under 3 h" if 6 in sun else "not available"),
        "confidence": {"notes": conf, "canopy_fraction": round(canopy_frac, 2), "tile_coverage": round(coverage, 3),
                       "parcel_source": parcel.source if parcel is not None else "none",
                       "provider_log": provider_log or []},
        "imagery_date": tiles.imagery_date,
        "imagery_quality": tiles.imagery_quality,
        "units": "plan (horizontal) square feet; lengths in feet",
        "verify": "Measured from satellite elevation data and aerial imagery; verify on site. Not a survey: do not use "
                  "for fence placement or property line questions.",
        "measured_with": "LotMeasure (roofmeasure 0.6.0)",
    }
    if 6 in sun:
        pm = plantable
        jun = sun[6][pm]
        result["sun_totals"] = {"full_sun_sqft": _sqft(np.sum(jun >= FULL_SUN_H), px_x, px_y),
                                "part_sun_sqft": _sqft(np.sum((jun >= PART_SUN_H) & (jun < FULL_SUN_H)), px_x, px_y),
                                "shade_sqft": _sqft(np.sum(jun < PART_SUN_H), px_x, px_y)}
    if keep_rasters:
        result["_plan"] = {"cls": cls_lot, "win": (r0, c0), "shape": shape, "geo": tiles.geo, "px": (px_x, px_y),
                           "lot_poly": lot_poly, "region": work, "house_poly": house_poly_eff, "zones": plan_zones,
                           "trees": trees, "walls": walls, "sun_jun": sun.get(6), "u": u, "named": named,
                           "rgb": rgb, "frontage": frontage}
    return result


# ====================================================================== live entry point

def lot_radius_m(lat, lng, ring) -> float:
    """dataLayers radius that covers the parcel around the geocoded point (60 to 100 m)."""
    from .geo import local_metres

    if not ring:
        return 60.0
    r = np.asarray(ring, float)
    x, y = local_metres(r[:, 0], r[:, 1], lat, lng)
    need = float(np.max(np.maximum(np.abs(x), np.abs(y)))) + 6.0
    return float(min(max(need, ROOF_WINDOW_M), 100.0))


def fetch_tiles(lat, lng, radius_m: float, include_sun: bool = True, quality: str = "LOW") -> LotTiles:
    from . import geometry, solar

    layers = solar.data_layers(lat, lng, radius_m=radius_m, quality=quality)
    if not layers.get("dsmUrl") or not layers.get("maskUrl"):
        raise solar.SolarAPIError("dataLayers response missing dsmUrl/maskUrl")
    dsm, px_x, px_y, geo = solar.fetch_geotiff_geo(layers["dsmUrl"])
    mask, _, _, _ = solar.fetch_geotiff_geo(layers["maskUrl"])
    t = LotTiles(dsm=dsm, mask=mask, geo=geo, px_x=px_x, px_y=px_y,
                 imagery_date=geometry._fmt_date(layers.get("imageryDate")),
                 imagery_quality=layers.get("imageryQuality", ""))
    if layers.get("rgbUrl"):
        rgb, _g = solar.read_geotiff_rgb_geo(solar._download(layers["rgbUrl"]))
        t.rgb = rgb
    if include_sun:
        if layers.get("annualFluxUrl"):
            t.flux, _, _, t.flux_geo = solar.fetch_geotiff_geo(layers["annualFluxUrl"])
        sh = layers.get("hourlyShadeUrls") or []
        for mon in (6, 12):
            if len(sh) >= mon:
                arr, _, _, g = solar.fetch_geotiff_geo(sh[mon - 1], bands=True)
                t.shade[mon] = (arr, g)
    return t


def measure_lot(address: str, include_sun: bool = True, keep_rasters: bool = False) -> dict:
    """Address -> lot measurements (see measure_lot_from_tiles)."""
    from . import parcels, solar

    gc = solar.geocode_full(address)
    parcel, log = parcels.find_parcel(gc["lat"], gc["lng"], gc.get("county", ""), gc.get("state", ""))
    neighbours = parcels.neighbour_rings(parcel) if parcel is not None else []
    roads = parcels.osm_roads(gc["lat"], gc["lng"]) if parcel is not None else []
    if parcel is not None and not roads:
        log.append("OpenStreetMap roads unavailable (Overpass unreachable or no named road nearby)")
    insights = solar.building_insights(gc["lat"], gc["lng"])
    radius = lot_radius_m(gc["lat"], gc["lng"], parcel.ring if parcel else None)
    tiles = fetch_tiles(gc["lat"], gc["lng"], radius, include_sun=include_sun)
    return measure_lot_from_tiles(tiles, gc, insights, parcel=parcel, neighbours=neighbours, roads=roads,
                                  include_sun=include_sun, provider_log=log, keep_rasters=keep_rasters)
