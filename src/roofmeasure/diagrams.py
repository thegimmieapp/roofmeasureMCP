"""EagleView-layout PDF measurement report (vector diagrams, 1:1 page set).

Pages (matching an EagleView Premium Report):
  Cover         logo, aerial, address, prepared-for block
  Summary       outline thumbnail, table of contents, measurements
  Images        top view aerial
  Length        color-coded lines with rotated length labels, totals header
  Pitch         shaded facets, pitch value + downslope arrow per facet
  Area          facet areas (rounded) per facet
  Notes         facet letters A-Z (smallest to largest)
  Report Summary per structure + All Structures:
                areas per pitch, structure complexity, waste calculation,
                lengths/areas/pitches, property location
  Legal notice
"""

from __future__ import annotations

import math
import os
import textwrap
from datetime import date

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, Rectangle  # noqa: E402

from .geometry import MeasureDetail  # noqa: E402
from .model import RoofMeasurements  # noqa: E402

# EagleView line colors
EDGE_COLORS = {
    "ridge": "#e8141c",      # red (ridges and hips)
    "hip": "#e8141c",
    "valley": "#1f3fd8",     # blue, dashed
    "eave": "#000000",       # black
    "rake": "#1ea83c",       # green
    "flashing": "#c87a2a",   # orange-brown, dashed
    "step_flashing": "#c87a2a",
}
NAVY = "#1a3b6e"
HEADER_BAND = "#dde7f1"
TABLE_HEAD = "#7a7a7a"
LIGHT_BLUE = "#d9ecf6"
PITCH_BLUE = "#dcecf5"
PITCH_GRAY = "#d6d6d6"
MIN_LABEL_FT = 5.0
PAGE_W, PAGE_H = 8.5, 11.0
_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

FOOT = ("Measurements derived from satellite elevation and imagery data (Google Solar API). "
        "Field verify before material order.")


# ------------------------------------------------------------ geometry bundle

class _Vec:
    """Vector geometry for one structure."""

    def __init__(self, m: RoofMeasurements, detail: MeasureDetail):
        from .wireframe import facet_polygons

        wf = getattr(detail, "skeleton", None)
        px = (detail.px_x + detail.px_y) / 2.0
        self.px = px
        self.ok = wf is not None and getattr(wf, "ok", False)
        if self.ok:
            self.outline = wf.outline
            self.edges = [{"p0": e.p0, "p1": e.p1, "kind": e.kind, "length_ft": e.length_ft}
                          for e in wf.edges]
            polys = wf.faces if getattr(wf, "polys_ready", False) else facet_polygons(detail.labels, wf, px)
            order = sorted(wf.face_area_sqft, key=lambda i: wf.face_area_sqft[i])
            self.faces = []
            for rank, fid in enumerate(order):
                self.faces.append({
                    "letter": _letter(rank), "fid": fid,
                    "centroid": wf.face_centroids.get(fid, (0, 0)),
                    "pitch": wf.face_pitch.get(fid, m.predominant_pitch),
                    "area_sqft": wf.face_area_sqft[fid],
                    "downslope": wf.face_downslope.get(fid, (0.0, 0.0)),
                    "poly": polys.get(fid),
                })
        else:
            from .vectorize import classify_outline_edges, internal_lines, outline_polygon
            self.outline = outline_polygon(detail)
            self.edges = classify_outline_edges(self.outline, detail) + internal_lines(detail)
            facet_by_letter = {f.label: f for f in m.facets}
            self.faces = []
            for letter, (cy, cx) in detail.facet_centroids.items():
                f = facet_by_letter.get(letter)
                if f is None:
                    continue
                az = math.radians(detail.facet_azimuths.get(letter, 0.0))
                self.faces.append({"letter": letter, "fid": None, "centroid": (cy, cx), "pitch": f.pitch,
                                   "area_sqft": f.surface_area_sqft,
                                   "downslope": (-math.cos(az), math.sin(az)), "poly": None})
        self.internal = [e for e in self.edges if e["kind"] in ("ridge", "hip", "valley", "flashing", "step_flashing")]
        pts = self.outline if len(self.outline) else np.zeros((1, 2))
        self.bbox = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())


def _letter(rank: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    rank += 1
    while rank > 0:
        rank, rem = divmod(rank - 1, 26)
        label = letters[rem] + label
    return label


def _union_bbox(vecs):
    y0 = min(v.bbox[0] for v in vecs); y1 = max(v.bbox[1] for v in vecs)
    x0 = min(v.bbox[2] for v in vecs); x1 = max(v.bbox[3] for v in vecs)
    pad = max((y1 - y0), (x1 - x0)) * 0.06 + 8
    return (y0 - pad, y1 + pad, x0 - pad, x1 + pad)


def _setup_ax(ax, bbox):
    y0, y1, x0, x1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------ page furniture

def _page(pdf_title: str, address: str, report_id: str, date_str: str, claim: str = "",
          company: str = "", section: str | None = None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    logo = os.path.join(_ASSETS, "stronghouse_logo_blue.png")
    if company.lower().startswith("stronghouse") and os.path.exists(logo):
        axl = fig.add_axes([0.06, 0.925, 0.26, 0.055])
        axl.imshow(plt.imread(logo))
        axl.axis("off")
    else:
        fig.text(0.06, 0.945, company or "Roof Report", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.94, 0.962, pdf_title, fontsize=13, color=NAVY, ha="right")
    fig.text(0.94, 0.945, date_str, fontsize=9, color="#444444", ha="right")
    if claim:
        fig.text(0.94, 0.930, f"Claim: {claim}", fontsize=9, color="#444444", ha="right")
    fig.patches.append(Rectangle((0.06, 0.895), 0.88, 0.028, transform=fig.transFigure,
                                 facecolor=NAVY, edgecolor="none"))
    fig.text(0.07, 0.909, address, fontsize=11, color="white", va="center")
    fig.text(0.93, 0.909, f"Report: {report_id}", fontsize=11, color="white", ha="right", va="center")
    if section:
        fig.patches.append(Rectangle((0.06, 0.855), 0.88, 0.035, transform=fig.transFigure,
                                     facecolor=HEADER_BAND, edgecolor="none"))
        fig.text(0.085, 0.8725, section, fontsize=17, color=NAVY, va="center")
    fig.lines.append(plt.Line2D([0.06, 0.94], [0.085, 0.085], transform=fig.transFigure, color=NAVY, lw=1.5))
    fig.text(0.5, 0.072, FOOT, fontsize=6, color="#333333", ha="center")
    return fig


def _page_num(fig, n: int):
    fig.text(0.90, 0.035, f"PAGE {n}", fontsize=11, color=NAVY, ha="right")


def _compass(fig, x=0.80, y=0.24, s=0.035):
    ax = fig.add_axes([x, y, s * 1.6, s * 1.6 * (PAGE_W / PAGE_H)])
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis("off")
    ax.annotate("", xy=(0, 1), xytext=(0, -1), arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.annotate("", xy=(1, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(0, 1.1, "N", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(0, -1.1, "S", ha="center", va="top", fontsize=6)
    ax.text(-1.1, 0, "W", ha="right", va="center", fontsize=6)
    ax.text(1.1, 0, "E", ha="left", va="center", fontsize=6)


# ------------------------------------------------------------ drawing

def _draw_outline(ax, vecs, fill="none", lw=1.0, color="#555555"):
    for v in vecs:
        if len(v.outline):
            ax.add_patch(MplPolygon(v.outline[:, ::-1], closed=True, facecolor=fill,
                                    edgecolor=color, linewidth=lw, zorder=1))


def _draw_faces(ax, vecs, facecolor_fn, edgecolor="#8a8a8a", lw=0.9):
    """Facet polygons (preferred) or outline + crease lines as fallback."""
    for v in vecs:
        drew = False
        for f in sorted(v.faces, key=lambda f: -f["area_sqft"]):
            if f["poly"] is not None and len(f["poly"]) >= 3:
                ax.add_patch(MplPolygon(f["poly"][:, ::-1], closed=True, facecolor=facecolor_fn(f),
                                        edgecolor=edgecolor, linewidth=lw, zorder=2))
                drew = True
        if not drew:
            _draw_outline(ax, [v], fill=facecolor_fn({"pitch": "8/12"}), color=edgecolor, lw=lw)
            for e in v.internal:
                (y0, x0), (y1, x1) = e["p0"], e["p1"]
                ax.plot([x0, x1], [y0, y1], color=edgecolor, lw=lw, zorder=3)
        else:
            _draw_outline(ax, [v], fill="none", color=edgecolor, lw=lw)


def _draw_length_lines(ax, vecs, fig_scale: float):
    placed: list[tuple[float, float]] = []
    for v in vecs:
        for e in v.edges:
            kind = e["kind"]
            color = EDGE_COLORS.get(kind, "#777777")
            (y0, x0), (y1, x1) = e["p0"], e["p1"]
            ls = (0, (4, 3)) if kind in ("valley", "flashing", "step_flashing") else "-"
            lw = 1.8 if kind in ("ridge", "hip") else 1.2
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=3)
            if e["length_ft"] < MIN_LABEL_FT:
                continue
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            off = 7.0 * fig_scale
            lx, ly = mx + nx * off, my + ny * off
            if any(math.hypot(lx - a, ly - b) < 9 * fig_scale for a, b in placed):
                continue
            placed.append((lx, ly))
            ang = math.degrees(math.atan2(-dy, dx))
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            txt = f"{e['length_ft']:.0f}"
            if txt in ("6", "9", "66", "99", "69", "96") and abs(ang) > 45:
                txt = "+" + txt
            ax.text(lx, ly, txt, fontsize=6.5, color=color if kind != "eave" else "black",
                    ha="center", va="center", rotation=ang, rotation_mode="anchor", zorder=5)


def _length_header(fig, m: RoofMeasurements):
    e = m.edges
    fig.text(0.07, 0.820, "Total Line Lengths:", fontsize=8.5)
    fig.text(0.125, 0.806, f"Ridges = {e.ridges_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["ridge"], fontweight="bold")
    fig.text(0.125, 0.792, f"Hips = {e.hips_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["hip"])
    fig.text(0.41, 0.820, f"Valleys = {e.valleys_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["valley"])
    fig.text(0.41, 0.806, f"Rakes = {e.rakes_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["rake"])
    fig.text(0.41, 0.792, f"Eaves = {e.eaves_ft:,.0f} ft", fontsize=8.5, color="black")
    fig.text(0.70, 0.820, f"Flashing = {e.flashing_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["flashing"])
    fig.text(0.70, 0.806, f"Step flashing = {e.step_flashing_ft:,.0f} ft", fontsize=8.5, color=EDGE_COLORS["flashing"])
    fig.text(0.70, 0.792, "Parapets = 0 ft", fontsize=8.5, color="#777777", fontweight="bold")
    if e.safety_factor != 1.0:
        pct = round((e.safety_factor - 1.0) * 100)
        fig.text(0.07, 0.776, f"Eave and rake totals include a +{pct}% ordering margin "
                 f"(measured: eaves {e.eaves_measured_ft:,.0f} ft, rakes {e.rakes_measured_ft:,.0f} ft; "
                 "segment labels are measured lengths).", fontsize=7, color="#555555")


# ------------------------------------------------------------ summary tables

def _gray_table(fig, y_top: float, title: str, rows: list[list[str]], col_w: list[float],
                row_h: float = 0.016, highlight_cols: set | None = None,
                x0: float = 0.065, width: float = 0.87):
    fig.patches.append(Rectangle((x0, y_top - 0.018), width, 0.018, transform=fig.transFigure,
                                 facecolor=TABLE_HEAD, edgecolor="none"))
    fig.text(x0 + 0.01, y_top - 0.009, title, fontsize=8.5, color="white", fontweight="bold", va="center")
    y = y_top - 0.018
    total_w = sum(col_w)
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            w = width * col_w[c] / total_w
            face = "white"
            if c > 0:
                face = LIGHT_BLUE if (highlight_cols and c in highlight_cols) else ("#f4f4f4" if r % 2 == 0 else "white")
            fig.patches.append(Rectangle((x, y - row_h), w, row_h, transform=fig.transFigure,
                                         facecolor=face, edgecolor="#bbbbbb", lw=0.4))
            fig.text(x + (0.006 if c == 0 else w / 2), y - row_h / 2, cell, fontsize=7.5,
                     ha="left" if c == 0 else "center", va="center",
                     fontweight="bold" if c == 0 else "normal")
            x += w
        y -= row_h
    return y


def _complexity_bar(fig, y_top: float, level: str):
    fig.patches.append(Rectangle((0.065, y_top - 0.018), 0.87, 0.018, transform=fig.transFigure,
                                 facecolor=TABLE_HEAD, edgecolor="none"))
    fig.text(0.075, y_top - 0.009, "Structure Complexity", fontsize=8.5, color="white", fontweight="bold", va="center")
    y = y_top - 0.018
    for i, name in enumerate(("Simple", "Normal", "Complex")):
        x = 0.065 + 0.87 * i / 3
        face = LIGHT_BLUE if name == level else "white"
        fig.patches.append(Rectangle((x, y - 0.018), 0.29, 0.018, transform=fig.transFigure,
                                     facecolor=face, edgecolor="#bbbbbb", lw=0.4))
        fig.text(x + 0.145, y - 0.009, name, fontsize=7.5, ha="center", va="center")
    return y - 0.018


def _waste_table(fig, y_top: float, m: RoofMeasurements):
    fig.patches.append(Rectangle((0.065, y_top - 0.040), 0.87, 0.040, transform=fig.transFigure,
                                 facecolor=TABLE_HEAD, edgecolor="none"))
    fig.text(0.075, y_top - 0.010, "Waste Calculation", fontsize=8.5, color="white", fontweight="bold", va="center")
    fig.text(0.075, y_top - 0.023, "NOTE: This waste calculation table is for asphalt shingle roofing applications. "
             "All values in table below only include roof areas of 3/12 pitch or greater.", fontsize=5.3, color="white", va="center")
    fig.text(0.075, y_top - 0.033, "For total measurements of all pitches, please refer to the Lengths, Areas, and Pitches section below.",
             fontsize=5.3, color="white", va="center")
    y = y_top - 0.040
    wt = m.waste_table()
    labels = ["Waste %", "Area (Sq ft)", "Squares *"]
    ncol = len(wt) + 1
    cw = 0.87 / ncol
    rh = 0.016
    sug = m.suggested_waste_pct()
    for r, lab in enumerate(labels):
        fig.text(0.065 + 0.004, y - rh / 2, lab, fontsize=7.5, va="center")
        for c, row in enumerate(wt, start=1):
            x = 0.065 + cw * c
            near = abs(row["waste_pct"] - sug) <= 2 and row["waste_pct"] != 0
            if near:
                fig.patches.append(Rectangle((x, y - rh), cw, rh, transform=fig.transFigure,
                                             facecolor=LIGHT_BLUE, edgecolor="none"))
            val = (f"{row['waste_pct']}%" if r == 0 else f"{row['area_sqft']:,}" if r == 1 else f"{row['squares']:.2f}")
            bold = row["waste_pct"] == 0 or row.get("suggested")
            fig.text(x + cw / 2, y - rh / 2, val, fontsize=7.5, ha="center", va="center",
                     fontweight="bold" if bold else "normal")
        y -= rh
    fig.text(0.065 + cw * 1.5, y - 0.006, "Measured", fontsize=7, ha="center", va="center", fontweight="bold")
    for c, row in enumerate(wt, start=1):
        if row.get("suggested"):
            x = 0.065 + cw * c
            fig.patches.append(Rectangle((x, y - 0.013), cw, 0.013, transform=fig.transFigure,
                                         facecolor=LIGHT_BLUE, edgecolor="none"))
            fig.text(x + cw / 2, y - 0.006, "Suggested", fontsize=7, ha="center", va="center", fontweight="bold")
    y -= 0.022
    fig.text(0.065, y, "* Squares are rounded up to the 1/3 of a square", fontsize=7.5, va="top")
    y -= 0.02
    fig.text(0.065, y,
             "Additional materials needed for ridge, hip, and starter lengths are not included in the above table. "
             "The provided suggested waste\nfactor is intended to serve as a guide; actual waste percentages may differ "
             "based upon several variables (installation techniques, crew\nexperience, shingle material subtleties, and "
             "potential salvage from the site). The suggested waste is not to replace or substitute for\nexperience or "
             "judgment as to any given replacement or repair work.", fontsize=7, va="top", linespacing=1.4)
    return y - 0.075


def _lengths_block(fig, y_top: float, m: RoofMeasurements, vecs, label: str):
    e = m.edges
    if label:
        fig.text(0.065, y_top, label, fontsize=9, fontweight="bold", va="top")
    axt = fig.add_axes([0.075, y_top - 0.19, 0.20, 0.15])
    _setup_ax(axt, _union_bbox(vecs))
    _draw_faces(axt, vecs, lambda f: PITCH_BLUE if int(f["pitch"].split("/")[0]) >= 3 else PITCH_GRAY,
                edgecolor="#9a9a9a", lw=0.5)
    fig.text(0.075, y_top - 0.205, f"Total Roof Facets = {m.facet_count}", fontsize=8, va="top")
    x = 0.33
    fig.text(x, y_top - 0.02, "Lengths, Areas and Pitches", fontsize=9, fontweight="bold", color=NAVY, va="top")
    lines = [
        f"Ridges = {e.ridges_ft:,.0f} ft ({e.ridge_count} Ridges)",
        f"Hips = {e.hips_ft:,.0f} ft ({e.hip_count} Hips).",
        f"Valleys = {e.valleys_ft:,.0f} ft ({e.valley_count} Valleys)",
        f"Rakes† = {e.rakes_ft:,.0f} ft ({e.rake_count} Rakes)" + (f", measured {e.rakes_measured_ft:,.0f} ft" if e.safety_factor != 1.0 else ""),
        f"Eaves/Starter‡ = {e.eaves_ft:,.0f} ft ({e.eave_count} Eaves)" + (f", measured {e.eaves_measured_ft:,.0f} ft" if e.safety_factor != 1.0 else ""),
        f"Drip Edge (Eaves + Rakes) = {e.drip_edge_ft:,.0f} ft ({e.eave_count + e.rake_count} Lengths)",
        "Parapet Walls = 0 (0 Lengths).",
        f"Flashing = {e.flashing_ft:,.0f} ft",
        f"Step flashing = {e.step_flashing_ft:,.0f} ft",
        f"Predominant Pitch = {m.predominant_pitch}",
    ]
    yy = y_top - 0.036
    for ln in lines:
        fig.text(x, yy, ln, fontsize=8, va="top"); yy -= 0.0135
    fig.text(x, yy, f"Total Area (All Pitches) = {m.total_area_sqft:,.0f} sq ft", fontsize=8, va="top", fontweight="bold")
    x = 0.70
    fig.text(x, y_top - 0.02, "Property Location", fontsize=9, fontweight="bold", color=NAVY, va="top")
    fig.text(x, y_top - 0.036, f"Longitude = {m.longitude:.7f}", fontsize=8, va="top")
    fig.text(x, y_top - 0.0495, f"Latitude = {m.latitude:.7f}", fontsize=8, va="top")
    fig.text(x, y_top - 0.066, "Notes", fontsize=9, fontweight="bold", color=NAVY, va="top")
    note = "Measured from satellite\nelevation data"
    if m.imagery_date:
        note += f"\n(imagery {m.imagery_date})."
    fig.text(x, y_top - 0.082, note, fontsize=8, va="top", linespacing=1.3)
    return y_top - 0.22


def _summary_page(pdf, m: RoofMeasurements, vecs, heading: str, hdr: dict, page_no: int):
    fig = _page("Premium Report", **hdr, section="REPORT SUMMARY")
    fig.text(0.065, 0.835, heading, fontsize=10, fontweight="bold", va="top")
    y = 0.818
    pitches = list(m.areas_per_pitch.items())
    total = m.total_area_sqft or 1.0
    rows = [["Roof Pitches"] + [p for p, _ in pitches],
            ["Area (sq ft)"] + [f"{a:,.1f}" for _, a in pitches],
            ["% of Roof"] + [f"{100 * a / total:.1f}%" for _, a in pitches]]
    y = _gray_table(fig, y, "Areas per Pitch", rows, [1.6] + [1.0] * max(len(pitches), 1),
                    highlight_cols=set(range(1, len(pitches) + 1)))
    fig.text(0.075, y - 0.004, "The table above lists each pitch on this roof and the total area and percent (both rounded) of the roof with that pitch.",
             fontsize=7, va="top")
    y -= 0.03
    y = _complexity_bar(fig, y, m.complexity())
    y -= 0.016
    y = _waste_table(fig, y, m)
    y = _lengths_block(fig, y, m, vecs, "All Structures Totals" if heading == "All Structures" else "")
    fig.text(0.065, 0.118, "†   Rakes are defined as roof edges that are sloped (not level).", fontsize=7)
    fig.text(0.065, 0.106, "‡   Eaves are defined as roof edges that are not sloped and level.", fontsize=7)
    _page_num(fig, page_no)
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------ main entry

def _vecs_for(m: RoofMeasurements, detail: MeasureDetail):
    subs = getattr(detail, "structures", None)
    if subs and m.structures:
        return [_Vec(sm, sd) for sm, sd in zip(m.structures, subs)]
    return [_Vec(m, detail)]


def render_diagram_png(m: RoofMeasurements, detail: MeasureDetail, out_path: str,
                       rgb: np.ndarray | None = None) -> str:
    """Composite PNG (aerial + line diagram) for the Xactimate docx."""
    vecs = _vecs_for(m, detail)
    bbox = _union_bbox(vecs)
    fig = plt.figure(figsize=(8.5, 5.2), dpi=200)
    if rgb is not None:
        y0, y1, x0, x1 = [int(v) for v in bbox]
        ax0 = fig.add_axes([0.03, 0.08, 0.42, 0.84])
        ax0.imshow(rgb[max(y0, 0):y1, max(x0, 0):x1]); ax0.axis("off")
        ax0.set_title("Aerial", fontsize=10, fontweight="bold", loc="left")
        rect = [0.49, 0.08, 0.48, 0.84]
    else:
        rect = [0.05, 0.08, 0.90, 0.84]
    ax = fig.add_axes(rect)
    _setup_ax(ax, bbox)
    _draw_faces(ax, vecs, lambda f: "#f4f6f9")
    _draw_length_lines(ax, vecs, fig_scale=(bbox[1] - bbox[0]) / 400.0)
    ax.set_title("Roof Diagram", fontsize=10, fontweight="bold", loc="left")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def render_pdf_report(m: RoofMeasurements, detail: MeasureDetail, out_path: str,
                      rgb: np.ndarray | None = None, company: str = "Stronghouse Solutions",
                      contact: str = "", phone: str = "", email: str = "",
                      company_address: str = "", claim_number: str = "",
                      report_id: str = "") -> str:
    vecs = _vecs_for(m, detail)
    per_structure = list(zip(m.structures, vecs)) if m.structures else [(m, vecs[0])]
    bbox = _union_bbox(vecs)
    today = date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"
    report_id = report_id or today.strftime("%y%m%d%H%M")
    short_addr = m.address.replace(", USA", "")
    hdr = dict(address=short_addr, report_id=report_id, date_str=date_str, claim=claim_number, company=company)
    scale = (bbox[1] - bbox[0]) / 400.0
    e = m.edges

    with PdfPages(out_path) as pdf:
        # ---------------- Cover ----------------
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.patches.append(Rectangle((0.82, 0.0), 0.18, 1.0, transform=fig.transFigure,
                                     facecolor="#7ac143", edgecolor="none"))
        fig.text(0.93, 0.5, "Precise Aerial Measurement Report", fontsize=17, color="white",
                 rotation=-90, ha="center", va="center")
        logo = os.path.join(_ASSETS, "stronghouse_logo_blue.png")
        if company.lower().startswith("stronghouse") and os.path.exists(logo):
            axl = fig.add_axes([0.22, 0.80, 0.38, 0.12]); axl.imshow(plt.imread(logo)); axl.axis("off")
        else:
            fig.text(0.41, 0.86, company, fontsize=24, fontweight="bold", color=NAVY, ha="center")
        if rgb is not None:
            y0, y1, x0, x1 = [int(v) for v in bbox]
            pad = int(max(y1 - y0, x1 - x0) * 0.25)
            ax = fig.add_axes([0.16, 0.42, 0.50, 0.32])
            ax.imshow(rgb[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]); ax.axis("off")
            if m.imagery_date:
                ax.text(0.98, 0.02, m.imagery_date, transform=ax.transAxes, fontsize=7, color="white",
                        ha="right", va="bottom", bbox=dict(fc="black", alpha=0.5, ec="none", pad=1.5))
        fig.text(0.41, 0.385, short_addr, fontsize=11, ha="center")
        fig.text(0.41, 0.33, f"Prepared for you by {company}", fontsize=13, color="#4a9a2a", ha="center")
        left = company + ("\n" + company_address.replace(", ", "\n") if company_address else "")
        fig.text(0.21, 0.20, left, fontsize=12, fontweight="bold", ha="center", va="center", linespacing=1.4)
        right = "\n".join(s for s in (contact, f"tel. {phone}" if phone else "", f"email: {email}" if email else "") if s)
        fig.text(0.62, 0.20, right, fontsize=10, color="#444444", ha="center", va="center", linespacing=1.5)
        fig.text(0.41, 0.03, "Measurements derived from Google Solar API elevation data; "
                 "prepared with RoofMeasure.", fontsize=6.5, color="#555555", ha="center")
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Summary / TOC page ----------------
        fig = _page("Premium Report", **hdr)
        fig.patches.append(Rectangle((0.07, 0.12), 0.48, 0.70, transform=fig.transFigure,
                                     facecolor="none", edgecolor="#777777", lw=0.8))
        ax = fig.add_axes([0.10, 0.50, 0.42, 0.30])
        ax.set_zorder(5)
        _setup_ax(ax, bbox)
        _draw_faces(ax, vecs, lambda f: "#e6e6e6", edgecolor="#555555", lw=0.8)
        fig.text(0.31, 0.47, "Roof model: facets drawn from measured roof planes.", fontsize=7.5, ha="center")
        if claim_number:
            fig.text(0.085, 0.43, "Claim Information", fontsize=9, fontweight="bold")
            fig.text(0.085, 0.415, f"Claim Number: {claim_number}", fontsize=8)
        fig.patches.append(Rectangle((0.085, 0.27), 0.45, 0.12, transform=fig.transFigure,
                                     facecolor=HEADER_BAND, edgecolor="none"))
        fig.text(0.10, 0.365, "PREPARED FOR", fontsize=14, fontweight="bold", color=NAVY)
        for i, (k, v) in enumerate((("Contact:", contact or company), ("Company:", company),
                                     ("Address:", company_address), ("Phone:", phone))):
            fig.text(0.11, 0.335 - i * 0.017, k, fontsize=9)
            fig.text(0.33, 0.335 - i * 0.017, v, fontsize=8.5)
        fig.patches.append(Rectangle((0.58, 0.79), 0.36, 0.025, transform=fig.transFigure,
                                     facecolor=HEADER_BAND, edgecolor="none"))
        fig.text(0.59, 0.802, "TABLE OF CONTENTS", fontsize=11, fontweight="bold", color=NAVY, va="center")
        toc = [("Images", 1), ("Length Diagram", 2), ("Pitch Diagram", 3), ("Area Diagram", 4),
               ("Notes Diagram", 5), ("Report Summary", 6)]
        for i, (name, pg) in enumerate(toc):
            fig.text(0.60, 0.765 - i * 0.016, f"{name} " + "." * 60 + f" {pg}", fontsize=8, color=NAVY)
        fig.patches.append(Rectangle((0.58, 0.63), 0.36, 0.025, transform=fig.transFigure,
                                     facecolor=HEADER_BAND, edgecolor="none"))
        fig.text(0.59, 0.642, "MEASUREMENTS", fontsize=11, fontweight="bold", color=NAVY, va="center")
        meas = [f"Total Roof Area ={m.total_area_sqft:,.0f} sq ft", f"Total Roof Facets ={m.facet_count}",
                f"Predominant Pitch ={m.predominant_pitch}", "Number of Stories <=1",
                f"Total Ridges/Hips ={e.ridges_hips_ft:,.0f} ft", f"Total Valleys ={e.valleys_ft:,.0f} ft",
                f"Total Rakes ={e.rakes_ft:,.0f} ft", f"Total Eaves ={e.eaves_ft:,.0f} ft"]
        for i, t in enumerate(meas):
            fig.text(0.585, 0.608 - i * 0.015, t, fontsize=8.5, color=NAVY)
        conf = getattr(m, "confidence", "")
        if conf:
            col = {"HIGH": "#1e7d32", "MEDIUM": "#b26a00", "LOW": "#b00020"}.get(conf, NAVY)
            fig.text(0.585, 0.482, f"Measurement confidence: {conf}", fontsize=8.5, color=col, fontweight="bold")
            if conf == "LOW":
                fig.text(0.585, 0.468, "Tree canopy hides part of this roof; values are a lower bound.", fontsize=7, color=col)
        if getattr(m, "nearby_structures", None):
            fig.text(0.585, 0.45, "Detached roofs nearby (not included):", fontsize=8, color="#8a1c1c")
            for i, s in enumerate(m.nearby_structures[:4]):
                fig.text(0.585, 0.436 - i * 0.013, "- " + s, fontsize=7.5, color="#8a1c1c")
        pdf.savefig(fig); plt.close(fig)

        page = 1
        # ---------------- Images ----------------
        fig = _page("Premium Report", **hdr, section="IMAGES")
        fig.text(0.065, 0.835, "The following aerial image shows the structure from directly above for your reference.", fontsize=8.5)
        fig.text(0.5, 0.808, "Top View", fontsize=9, ha="center")
        if rgb is not None:
            y0, y1, x0, x1 = [int(v) for v in bbox]
            pad = int(max(y1 - y0, x1 - x0) * 0.35)
            ax = fig.add_axes([0.10, 0.30, 0.80, 0.50])
            ax.imshow(rgb[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]); ax.axis("off")
            if m.imagery_date:
                ax.text(0.98, 0.02, f"{m.imagery_date}  {m.imagery_quality}", transform=ax.transAxes, fontsize=7,
                        color="white", ha="right", va="bottom", bbox=dict(fc="black", alpha=0.5, ec="none", pad=1.5))
        else:
            fig.text(0.5, 0.55, "Aerial imagery unavailable", fontsize=10, ha="center", color="#777777")
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # ---------------- Length Diagram ----------------
        fig = _page("Premium Report", **hdr, section="LENGTH DIAGRAM")
        _length_header(fig, m)
        ax = fig.add_axes([0.08, 0.17, 0.76, 0.60])
        _setup_ax(ax, bbox)
        _draw_length_lines(ax, vecs, scale)
        _compass(fig)
        fig.text(0.065, 0.118, "Note: This diagram contains segment lengths (rounded to the nearest whole number) over 5.0 Feet. "
                 "In some cases, segment labels\nhave been removed for readability. Plus signs preface some numbers to avoid "
                 "confusion when rotated (e.g. +6 and +9).", fontsize=8, va="top", linespacing=1.3)
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # ---------------- Pitch Diagram ----------------
        fig = _page("Premium Report", **hdr, section="PITCH DIAGRAM")
        fig.text(0.065, 0.835, "Pitch values are shown in inches per foot, and arrows indicate slope direction. "
                 f"The predominant pitch on this roof is {m.predominant_pitch}", fontsize=8.5)
        ax = fig.add_axes([0.08, 0.17, 0.76, 0.64])
        _setup_ax(ax, bbox)
        _draw_faces(ax, vecs, lambda f: PITCH_BLUE if int(f["pitch"].split("/")[0]) >= 3 else PITCH_GRAY)
        for v in vecs:
            for f in v.faces:
                if f["area_sqft"] < 20:
                    continue
                cy, cx = f["centroid"]
                dy, dx = f["downslope"]
                L = 10 * scale
                rise = f["pitch"].split("/")[0]
                ang = math.degrees(math.atan2(-dy, dx))
                ax.annotate("", xy=(cx + dx * L, cy + dy * L), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="black", lw=0.8), zorder=4)
                ax.text(cx - dx * L * 0.9, cy - dy * L * 0.9, rise, fontsize=7.5, ha="center", va="center",
                        rotation=(ang - 90) if abs(dy) > abs(dx) else 0, zorder=5)
        _compass(fig)
        fig.text(0.065, 0.118, "Note:  This diagram contains labeled pitches for facet areas larger than 20.0 square feet. "
                 "In some cases, pitch labels have been\nremoved for readability.  Blue shading indicates a pitch of 3/12 "
                 "and greater. Gray shading indicates flat, 1/12 or 2/12 pitches.", fontsize=8, va="top", linespacing=1.3)
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # ---------------- Area Diagram ----------------
        fig = _page("Premium Report", **hdr, section="AREA DIAGRAM")
        fig.text(0.065, 0.835, f"Total Area = {m.total_area_sqft:,.0f} sq ft, with {m.facet_count} facets.", fontsize=8.5)
        ax = fig.add_axes([0.08, 0.17, 0.76, 0.64])
        _setup_ax(ax, bbox)
        _draw_faces(ax, vecs, lambda f: "white")
        for v in vecs:
            for f in v.faces:
                if f["area_sqft"] < 10:
                    continue
                cy, cx = f["centroid"]
                a = f"{f['area_sqft']:.0f}"
                if a in ("6", "9", "66", "99", "86", "68", "89", "98"):
                    a = "+" + a
                ax.text(cx, cy, a, fontsize=7.5, ha="center", va="center", zorder=5)
        _compass(fig)
        fig.text(0.065, 0.118, "Note: This diagram shows the square feet of each roof facet (rounded to the nearest Foot). "
                 "The total area in square feet, at the top\nof this page, is based on the non-rounded values of each roof "
                 "facet (rounded to the nearest square foot after being totaled).", fontsize=8, va="top", linespacing=1.3)
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # ---------------- Notes Diagram ----------------
        fig = _page("Premium Report", **hdr, section="NOTES DIAGRAM")
        fig.text(0.065, 0.835, "Roof facets are labeled from smallest to largest (A to Z) for easy reference.", fontsize=8.5)
        ax = fig.add_axes([0.08, 0.17, 0.76, 0.64])
        _setup_ax(ax, bbox)
        _draw_faces(ax, vecs, lambda f: "white")
        all_faces = [f for v in vecs for f in v.faces]
        for rank, f in enumerate(sorted(all_faces, key=lambda f: f["area_sqft"])):
            f["gletter"] = _letter(rank)
        for f in all_faces:
            cy, cx = f["centroid"]
            ax.text(cx, cy, f["gletter"], fontsize=8, ha="center", va="center", zorder=5)
        _compass(fig)
        fig.text(0.065, 0.118, "Note: This diagram also appears in the Property Owner Report.", fontsize=8, va="top")
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # ---------------- Report Summary ----------------
        if len(per_structure) > 1:
            for sm, sv in per_structure:
                _summary_page(pdf, sm, [sv], f"Structure #{sm.structure_index}", hdr, page); page += 1
        _summary_page(pdf, m, vecs, "All Structures", hdr, page); page += 1

        # ---------------- Notes & legal ----------------
        fig = _page("Legal Notice and Disclaimer", **hdr, section="IMPORTANT NOTICE AND DISCLAIMER")
        y = 0.83
        fig.text(0.065, y, "Measurement Notes", fontsize=9, fontweight="bold", va="top"); y -= 0.02
        for n in m.notes:
            for ln in textwrap.wrap("- " + n, 120):
                fig.text(0.065, y, ln, fontsize=7.5, va="top"); y -= 0.013
            y -= 0.004
        y -= 0.02
        fig.text(0.065, y, "Notice and Disclaimer", fontsize=9, fontweight="bold", va="top"); y -= 0.02
        disc = ("No Warranty: This report is provided \"as is\" and you agree to use it at your own risk. "
                f"{company} makes no guarantees, representations or warranties of any kind, express or implied, "
                "as to the content, quality, accuracy, completeness, effectiveness, reliability or fitness for a "
                "particular purpose of these measurements. Measurements are derived from satellite elevation and "
                "imagery data and can be affected by tree coverage, image resolution, recent construction and other "
                "factors. Contractors agree to always conduct a preliminary site survey to verify the measurements "
                "before ordering material or submitting a claim.")
        for ln in textwrap.wrap(disc, 120):
            fig.text(0.065, y, ln, fontsize=7.5, va="top"); y -= 0.013
        _page_num(fig, page); pdf.savefig(fig); plt.close(fig)

    return out_path
