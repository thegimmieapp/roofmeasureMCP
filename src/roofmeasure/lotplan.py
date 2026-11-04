"""Lot plan drawing (PNG + PDF) for LotMeasure.

Parcel outline, rectilinear house outline, yard surfaces shaded by class,
sun hatching on plantable ground (part sun //, shade xx), trees as circles,
setback dimensions, the largest clear rectangle per zone, north arrow and
scale bar. Branding: Stronghouse logo for roofing; Dream Landscaper (or any
other company) gets a neutral header with its name.
"""

from __future__ import annotations

import math
import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Polygon as MplPolygon, Rectangle  # noqa: E402

from . import lot as L  # noqa: E402

M_TO_FT = 3.280839895
_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

CLASS_COLORS = {
    L.OUT: (1, 1, 1, 0),
    L.HOUSE: "#ffffff",
    L.STRUCT: "#b08968",
    L.CANOPY: "#2f5d34",
    L.HARD: "#b9b9b9",
    L.LAWN: "#9cc97a",
    L.WATER: "#5aa9e6",
    L.BARE: "#d9c39a",
    L.UNKNOWN: "#ece8df",
}
BRANDS = {
    "stronghouse": {"name": "Stronghouse Solutions", "primary": "#1a3b6e", "band": "#e8edf5", "logo": "stronghouse_logo_blue.png"},
    "dreamlandscaper": {"name": "Dream Landscaper", "primary": "#3B4A23", "band": "#F4F0DC", "logo": None,
                        "accent": "#8A9B6E"},
}


def _brand(brand: str) -> dict:
    b = (brand or "").strip()
    key = b.lower().replace(" ", "").replace("_", "")
    if key.startswith("stronghouse"):
        return BRANDS["stronghouse"]
    if key in ("dreamlandscaper", "dl", "dreamyard"):
        return BRANDS["dreamlandscaper"]
    return {"name": b or "Lot Plan", "primary": "#333333", "band": "#f0f0f0", "logo": None}


def _draw_plan(ax, res, show_sun=True, aerial=False):
    P = res["_plan"]
    r0, c0 = P["win"]
    h, w = P["shape"]
    px_x, px_y = P["px"]
    ext = [c0 * px_x - px_x / 2, (c0 + w) * px_x - px_x / 2, -(r0 + h) * px_y + px_y / 2, -r0 * px_y + px_y / 2]
    if aerial and P.get("rgb") is not None:
        ax.imshow(P["rgb"], extent=ext, origin="upper", interpolation="bilinear")
    else:
        cmap = ListedColormap([CLASS_COLORS[k] for k in range(9)])
        ax.imshow(P["cls"], cmap=cmap, vmin=-0.5, vmax=8.5, extent=ext, origin="upper", interpolation="nearest")
        sun = P.get("sun_jun")
        if show_sun and sun is not None:
            plant = (P["cls"] == L.LAWN).astype(float)
            part = np.where(plant > 0, ((sun >= L.PART_SUN_H) & (sun < L.FULL_SUN_H)).astype(float), 0.0)
            shade = np.where(plant > 0, (sun < L.PART_SUN_H).astype(float), 0.0)
            yy = np.linspace(ext[3], ext[2], h)
            xx = np.linspace(ext[0], ext[1], w)
            step = max(int(0.5 / px_x), 1)
            for arr, hatch in ((part, "///"), (shade, "xxx")):
                if arr.any():
                    ax.contourf(xx[::step], yy[::step], arr[::step, ::step], levels=[0.5, 1.5],
                                colors="none", hatches=[hatch])
    lp = P["lot_poly"]
    if lp is not None:
        ax.add_patch(MplPolygon(np.asarray(lp.exterior.coords), closed=True, fill=False, ec="#000000", lw=2.2, zorder=5))
    else:
        reg = P["region"]
        for g in L.polygons_of(reg):
            ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, fill=False, ec="#777777",
                                    lw=1.2, ls=":", zorder=5))
    hp = P["house_poly"]
    if hp is not None and not hp.is_empty:
        ax.add_patch(MplPolygon(np.asarray(hp.exterior.coords), closed=True, fc="#ffffff" if not aerial else "none",
                                ec="#c62828", lw=1.8, zorder=6))
    if aerial:
        return ext
    # zone boundaries and labels
    for name, z in P["zones"].items():
        poly = z["poly"]
        if poly is None or poly.is_empty:
            continue
        for g in L.polygons_of(poly):
            if g.area < 2:
                continue
            ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, fill=False, ec="#444444",
                                    lw=0.8, ls="--", zorder=4))
        big = max(L.polygons_of(poly), key=lambda g: g.area)
        c = big.representative_point()
        zr = res["zones"].get(name, {})
        ax.text(c.x, c.y, f"{name.upper()}\n{zr.get('area_sqft', 0):,} sf", ha="center", va="center", fontsize=7,
                fontweight="bold", color="#1b1b1b", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        rect = z.get("rect")
        if rect and rect["area_m2"] >= 4:
            ax.add_patch(MplPolygon(np.asarray(rect["corners_xy"]), closed=True, fill=False, ec="#1565c0",
                                    lw=1.0, ls=(0, (4, 2)), zorder=7))
        sb = z.get("setback")
        if sb:
            a, b = np.asarray(sb["from_xy"]), np.asarray(sb["to_xy"])
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color="#c62828", lw=1.1), zorder=8)
            m = a + (b - a) * 0.3
            ax.text(m[0], m[1], f"{sb['ft']:.0f}'", fontsize=7, color="#c62828", fontweight="bold", zorder=9,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    for t in P["trees"]:
        x, y = t["_xy"]
        ax.add_patch(Circle((x, y), t["_r"], fill=False, ec="#0f3d17", lw=1.0, zorder=7))
        ax.plot([x], [y], marker="+", color="#0f3d17", ms=4, zorder=7)
    return ext


def _furniture(ax, bounds):
    x0, x1, y0, y1 = bounds
    span = x1 - x0
    # scale bar: round feet
    target_ft = span * M_TO_FT / 5
    nice = min([10, 20, 25, 50, 100, 200], key=lambda v: abs(v - target_ft))
    L_m = nice / M_TO_FT
    bx, by = x0 + 0.04 * span, y0 + 0.04 * (y1 - y0)
    ax.add_patch(Rectangle((bx, by), L_m, 0.012 * (y1 - y0), fc="black", ec="black", zorder=10))
    ax.text(bx + L_m / 2, by + 0.025 * (y1 - y0), f"{nice} ft", ha="center", fontsize=7, zorder=10)
    # north arrow (UTM grid north, within about 2 degrees of true north)
    nx, ny = x1 - 0.06 * span, y1 - 0.14 * (y1 - y0)
    ax.add_patch(FancyArrow(nx, ny, 0, 0.08 * (y1 - y0), width=0.012 * span, head_width=0.035 * span,
                            head_length=0.03 * (y1 - y0), fc="black", ec="black", zorder=10))
    ax.text(nx, ny - 0.03 * (y1 - y0), "N", ha="center", fontsize=9, fontweight="bold", zorder=10)


def render_lot_plan(res: dict, png_path: str, pdf_path: str | None = None, brand: str = "Stronghouse Solutions",
                    prepared_for: str = "") -> dict:
    """Draw the plan. res must come from measure_lot(..., keep_rasters=True)."""
    P = res["_plan"]
    B = _brand(brand)
    reg = P["lot_poly"] if P["lot_poly"] is not None else P["region"]
    minx, miny, maxx, maxy = reg.bounds
    pad = 0.08 * max(maxx - minx, maxy - miny) + 2
    bounds = [minx - pad, maxx + pad, miny - pad, maxy + pad]

    fig = plt.figure(figsize=(8.5, 11))
    # header
    logo = os.path.join(_ASSETS, B["logo"]) if B.get("logo") else None
    if logo and os.path.exists(logo):
        axl = fig.add_axes([0.05, 0.925, 0.26, 0.055]); axl.imshow(plt.imread(logo)); axl.axis("off")
    else:
        fig.text(0.05, 0.948, B["name"], fontsize=19, fontweight="bold", color=B["primary"])
    fig.text(0.95, 0.962, "Lot Plan", fontsize=14, color=B["primary"], ha="right")
    fig.text(0.95, 0.944, date.today().strftime("%B %-d, %Y") if os.name != "nt" else date.today().isoformat(),
             fontsize=8.5, color="#444", ha="right")
    fig.patches.append(Rectangle((0.05, 0.893), 0.90, 0.03, transform=fig.transFigure, fc=B["primary"], ec="none"))
    fig.text(0.06, 0.908, res["address"], fontsize=10.5, color="white", va="center")
    if prepared_for:
        fig.text(0.94, 0.908, f"Prepared for {prepared_for}", fontsize=9, color="white", va="center", ha="right")

    ax = fig.add_axes([0.05, 0.36, 0.62, 0.52])
    _draw_plan(ax, res)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_aspect("equal"); ax.axis("off")
    _furniture(ax, bounds)

    axa = fig.add_axes([0.69, 0.63, 0.26, 0.235])
    _draw_plan(axa, res, aerial=True)
    axa.set_xlim(bounds[0], bounds[1]); axa.set_ylim(bounds[2], bounds[3]); axa.set_aspect("equal")
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_title(f"Aerial ({res.get('imagery_date') or 'date n/a'})", fontsize=7.5)

    # legend
    y = 0.615
    fig.text(0.69, y, "Legend", fontsize=8.5, fontweight="bold", color=B["primary"])
    items = [(L.LAWN, "Lawn / planting"), (L.HARD, "Hardscape"), (L.CANOPY, "Tree canopy"), (L.WATER, "Pool / water"),
             (L.STRUCT, "Detached structure"), (L.BARE, "Bare soil / mulch"), (L.UNKNOWN, "Unknown / shadow")]
    for k, label in items:
        y -= 0.018
        fig.patches.append(Rectangle((0.69, y - 0.003), 0.018, 0.012, transform=fig.transFigure,
                                     fc=CLASS_COLORS[k], ec="#555", lw=0.4))
        fig.text(0.715, y, label, fontsize=7.2, va="bottom")
    for txt in ("/// part sun (3 to 6 h, June)", "xxx shade (under 3 h, June)", "red: house, setbacks",
                "blue dashed: largest clear rectangle", "circles: tree crowns"):
        y -= 0.017
        fig.text(0.69, y, txt, fontsize=6.8, va="bottom", color="#333")

    # summary table
    p = res.get("parcel")
    tot = res["totals"]
    lines = []
    if p:
        lines.append(("Lot area", f"{p['area_sqft']:,} sq ft ({p['area_acres']:.3f} acres)"))
        rec = p.get("county_recorded")
        if rec and "acres" in rec:
            lines.append(("County record", f"{rec['acres']:.3f} acres ({rec['difference_pct']:+.1f}%)"))
    else:
        lines.append(("Lot area", "no parcel boundary available"))
    lines += [("House footprint", f"{res['house']['plan_area_sqft']:,} sq ft" if res.get("house")
               else "not reported (see notes)"),
              ("Plantable (lawn/planting)", f"{tot['plantable_sqft']:,} sq ft"),
              ("Hardscape", f"{tot['hardscape_sqft']:,} sq ft"),
              ("Tree canopy", f"{tot['tree_canopy_sqft']:,} sq ft, {res['tree_count']} trees"),
              ("Street side", f"{res['frontage'].get('street_side_faces', 'unknown')} ({res['frontage']['rule']})")]
    if res.get("sun_totals"):
        s = res["sun_totals"]
        lines.append(("Sun on plantable", f"full {s['full_sun_sqft']:,} / part {s['part_sun_sqft']:,} / shade {s['shade_sqft']:,} sq ft"))
    yy = 0.335
    for k, v in lines:
        fig.text(0.05, yy, k, fontsize=8, fontweight="bold", color=B["primary"])
        fig.text(0.25, yy, v, fontsize=8)
        yy -= 0.0185
    # zone table
    yy -= 0.006
    hdr = ["Zone", "Faces", "Area sf", "Lawn %", "Setback", "Clear rect", "Full / part / shade sf"]
    xs = [0.05, 0.13, 0.23, 0.32, 0.40, 0.49, 0.66]
    for x, t in zip(xs, hdr):
        fig.text(x, yy, t, fontsize=7.5, fontweight="bold", color=B["primary"])
    for name, z in res["zones"].items():
        yy -= 0.017
        r = z.get("largest_clear_rectangle")
        s = z.get("sun") or {}
        vals = [name, z["faces"], f"{z['area_sqft']:,}", f"{z.get('lawn_planting_pct', 0)}",
                f"{z['setback_ft']:.0f} ft" if z.get("setback_ft") is not None else "n/a",
                f"{r['width_ft']:.0f} x {r['depth_ft']:.0f} ft" if r else "none",
                f"{s.get('full_sun_sqft', 0):,} / {s.get('part_sun_sqft', 0):,} / {s.get('shade_sqft', 0):,}" if s else "n/a"]
        for x, t in zip(xs, vals):
            fig.text(x, yy, t, fontsize=7.5)
    yy -= 0.024
    notes = res["confidence"]["notes"][:7]
    import textwrap
    for n in notes:
        for ln in textwrap.wrap(n, 150):
            fig.text(0.05, yy, ln, fontsize=5.9, color="#333")
            yy -= 0.0115
        if yy < 0.045:
            break
    fig.text(0.5, 0.022, res["verify"], fontsize=6.2, ha="center", color="#555")
    fig.savefig(png_path, dpi=160)
    if pdf_path:
        fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path or ""}
