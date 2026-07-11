"""EagleView-style PDF measurement report with vector (straight-line) diagrams.

Roofs are combinations of straight-edged shapes; all diagrams here are drawn
from vectorized facet polygons (see vectorize.py), never raster pixels.

Pages:
  1. Cover: summary table + aerial image
  2. Length Diagram: color-coded ridges/hips/valleys/rakes/eaves, labeled
  3. Pitch Diagram: facets shaded, pitch labels + slope-direction arrows
  4. Area Diagram: facet letters + areas
  5. Tables: areas per pitch, waste calculation, facet detail
  6. Notes & disclaimer
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from .geometry import MeasureDetail  # noqa: E402
from .model import RoofMeasurements  # noqa: E402
from .vectorize import (classify_outline_edges, classify_polygon_edges,  # noqa: E402
                        facet_polygons, internal_lines, outline_polygon)

EDGE_COLORS = {
    "ridge": "#d62728",   # red
    "hip": "#ff7f0e",     # orange
    "valley": "#1f77b4",  # blue
    "eave": "#2ca02c",    # green
    "rake": "#9467bd",    # purple
}
UNMATCHED_COLOR = "#777777"
MIN_LABEL_FT = 5.0


class _Vec:
    """Vector geometry bundle computed once per report."""

    def __init__(self, m: RoofMeasurements, detail: MeasureDetail):
        self.outline = outline_polygon(detail)          # Nx2 (y, x), closed
        self.outline_edges = classify_outline_edges(self.outline, detail)
        self.internal = internal_lines(detail)          # straight ridge/hip/valley lines
        self.edges = self.outline_edges + self.internal
        pts = self.outline if len(self.outline) else np.zeros((1, 2))
        pad = 14
        self.bbox = (pts[:, 0].min() - pad, pts[:, 0].max() + pad,
                     pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        self.centroids = dict(detail.facet_centroids)


def _setup_ax(ax, bbox):
    y0, y1, x0, x1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # raster orientation, north up
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_roof(ax, vec: _Vec, fill: str = "#f2f4f7",
               outline: str = "#444444", lw: float = 1.2):
    """Roof outline (filled) + straight internal lines: the base drawing."""
    if len(vec.outline):
        ax.add_patch(MplPolygon(vec.outline[:, ::-1], closed=True, facecolor=fill,
                                edgecolor=outline, linewidth=lw, zorder=1))
    for e in vec.internal:
        (y0, x0), (y1, x1) = e["p0"], e["p1"]
        ax.plot([x0, x1], [y0, y1], color="#666666", lw=0.9, zorder=2)


def _draw_classified_edges(ax, vec: _Vec, label_lengths: bool = True, lw: float = 2.4):
    drawn_labels: list[tuple[float, float]] = []
    for e in vec.edges:
        color = EDGE_COLORS.get(e["kind"], UNMATCHED_COLOR)
        (y0, x0), (y1, x1) = e["p0"], e["p1"]
        ax.plot([x0, x1], [y0, y1], color=color,
                lw=lw if e["kind"] else 1.0, solid_capstyle="round", zorder=3)
        if not label_lengths or not e["kind"] or e["length_ft"] < MIN_LABEL_FT:
            continue
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # avoid stacking labels on top of each other
        if any(math.hypot(mx - ax_, my - ay_) < 12 for ax_, ay_ in drawn_labels):
            continue
        drawn_labels.append((mx, my))
        ang = math.degrees(math.atan2(-(y1 - y0), x1 - x0))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        ax.annotate(f"{e['length_ft']:.0f}'", xy=(mx, my), fontsize=6.5,
                    fontweight="bold", color="black", ha="center", va="center",
                    rotation=ang, rotation_mode="anchor", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.13", fc="white",
                              ec=EDGE_COLORS[e["kind"]], lw=0.7, alpha=0.92))


def _legend(ax, m: RoofMeasurements, totals: bool = True, ncol: int = 3):
    e = m.edges
    if totals:
        labels = {
            "ridge": f"Ridges = {e.ridges_ft:,.0f} ft",
            "hip": f"Hips = {e.hips_ft:,.0f} ft",
            "valley": f"Valleys = {e.valleys_ft:,.0f} ft",
            "eave": f"Eaves = {e.eaves_ft:,.0f} ft",
            "rake": f"Rakes = {e.rakes_ft:,.0f} ft",
        }
    else:
        labels = {k: k.capitalize() for k in EDGE_COLORS}
    handles = [Line2D([0], [0], color=c, lw=3, label=labels[k])
               for k, c in EDGE_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=ncol, fontsize=9 if totals else 7, frameon=False)


def _table(ax, col_labels, rows, title=None, fontsize=8, col_widths=None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8)
    t = ax.table(cellText=rows, colLabels=col_labels, loc="upper left",
                 cellLoc="center", colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.3)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#999999")
        if r == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(fontweight="bold")


def _pitch_colors(m: RoofMeasurements, vec: _Vec) -> tuple[dict, dict]:
    facet_by_letter = {f.label: f for f in m.facets}
    return facet_by_letter, {}


def render_diagram_png(
    m: RoofMeasurements,
    detail: MeasureDetail,
    out_path: str,
    rgb: np.ndarray | None = None,
) -> str:
    """Composite PNG (aerial + straight-line roof diagram) for the Xactimate docx."""
    vec = _Vec(m, detail)
    fig = plt.figure(figsize=(8.5, 5.2), dpi=200)

    if rgb is not None:
        y0, y1, x0, x1 = [int(v) for v in vec.bbox]
        ax0 = fig.add_axes([0.03, 0.08, 0.42, 0.84])
        ax0.imshow(rgb[max(y0, 0):y1, max(x0, 0):x1])
        ax0.axis("off")
        ax0.set_title("Aerial", fontsize=10, fontweight="bold", loc="left")
        diag_rect = [0.49, 0.08, 0.48, 0.84]
    else:
        diag_rect = [0.05, 0.08, 0.90, 0.84]

    ax = fig.add_axes(diag_rect)
    _setup_ax(ax, vec.bbox)
    facet_by_letter, _ = _pitch_colors(m, vec)
    _draw_roof(ax, vec)
    _draw_classified_edges(ax, vec, label_lengths=False)
    for letter, (cy, cx) in vec.centroids.items():
        f = facet_by_letter.get(letter)
        if f is None or f.surface_area_sqft < 20:
            continue
        ax.text(cx, cy, letter, fontsize=8, fontweight="bold", ha="center", va="center",
                zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black",
                                    lw=0.6, alpha=0.9))
    ax.set_title("Roof Diagram", fontsize=10, fontweight="bold", loc="left")
    _legend(ax, m, totals=False, ncol=5)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def render_pdf_report(
    m: RoofMeasurements,
    detail: MeasureDetail,
    out_path: str,
    rgb: np.ndarray | None = None,
    company: str = "Stronghouse Solutions",
    contact: str = "",
    phone: str = "",
) -> str:
    vec = _Vec(m, detail)
    e = m.edges
    today = date.today().strftime("%B %d, %Y")

    with PdfPages(out_path) as pdf:
        # ---------------- Page 1: cover ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Roof Measurement Report", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.915, m.address, fontsize=12)
        sub = today + (f"  |  Imagery: {m.imagery_date} ({m.imagery_quality})" if m.imagery_date else "")
        fig.text(0.08, 0.89, sub, fontsize=9, color="#444444")
        who = company + (f"  |  {contact}" if contact else "") + (f"  |  {phone}" if phone else "")
        fig.text(0.08, 0.868, f"Prepared by: {who}", fontsize=9, color="#444444")

        ax = fig.add_axes([0.08, 0.44, 0.84, 0.40])
        if rgb is not None:
            y0, y1, x0, x1 = [int(v) for v in vec.bbox]
            ax.imshow(rgb[max(y0, 0):y1, max(x0, 0):x1])
            ax.set_title("Aerial Image", fontsize=10, fontweight="bold", loc="left")
            ax.axis("off")
        else:
            _setup_ax(ax, vec.bbox)
            _draw_roof(ax, vec)
            ax.set_title("Roof Outline", fontsize=10, fontweight="bold", loc="left")

        axt = fig.add_axes([0.08, 0.06, 0.84, 0.33])
        rows = [
            ["Total Roof Area", f"{m.total_area_sqft:,.0f} sq ft"],
            ["Total Squares (measured)", f"{m.squares:.2f} SQ"],
            [f"Squares with suggested {m.suggested_waste_pct()}% waste",
             f"{m.squares_with_waste(m.suggested_waste_pct()):.2f} SQ"],
            ["Total Roof Facets", str(m.facet_count)],
            ["Predominant Pitch", m.predominant_pitch],
            ["Ridges", f"{e.ridges_ft:,.0f} ft  ({e.ridge_count})"],
            ["Hips", f"{e.hips_ft:,.0f} ft  ({e.hip_count})"],
            ["Ridges + Hips", f"{e.ridges_hips_ft:,.0f} ft"],
            ["Valleys", f"{e.valleys_ft:,.0f} ft  ({e.valley_count})"],
            ["Rakes", f"{e.rakes_ft:,.0f} ft  ({e.rake_count})"],
            ["Eaves / Starter", f"{e.eaves_ft:,.0f} ft  ({e.eave_count})"],
            ["Drip Edge (Eaves + Rakes)", f"{e.drip_edge_ft:,.0f} ft"],
            ["Longitude / Latitude", f"{m.longitude:.7f} / {m.latitude:.7f}"],
        ]
        _table(axt, ["Measurement", "Value"], rows, title="Report Summary",
               col_widths=[0.55, 0.45])
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 2: length diagram ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Length Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925, m.address, fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.18, 0.90, 0.70])
        _setup_ax(ax, vec.bbox)
        _draw_roof(ax, vec, fill="none")
        _draw_classified_edges(ax, vec, label_lengths=True)
        _legend(ax, m, totals=True)
        fig.text(0.08, 0.10,
                 "Segment labels shown for edges 5 ft and longer. Lengths are slope-corrected "
                 "where applicable (hips, valleys, rakes).",
                 fontsize=7.5, color="#555555")
        fig.text(0.08, 0.075, "Field verify measurements before material order.",
                 fontsize=7.5, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 3: pitch diagram ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Pitch Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925, f"{m.address}  |  Predominant pitch: {m.predominant_pitch}",
                 fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.15, 0.90, 0.75])
        _setup_ax(ax, vec.bbox)
        facet_by_letter, _ = _pitch_colors(m, vec)
        _draw_roof(ax, vec, fill="#dce8f5")
        for letter, (cy, cx) in vec.centroids.items():
            f = facet_by_letter.get(letter)
            if f is None or f.surface_area_sqft < 20:
                continue
            az = math.radians(detail.facet_azimuths.get(letter, 0.0))
            dx, dy = math.sin(az), math.cos(az)  # compass -> raster: north = -y
            dy = -dy
            L = 14
            ax.annotate("", xy=(cx + dx * L, cy + dy * L), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.0), zorder=4)
            ax.text(cx, cy - 6, f.pitch, fontsize=8, fontweight="bold", ha="center",
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.9))
        fig.text(0.08, 0.09,
                 "Pitch shown in inches of rise per 12 inches of run; arrows point downslope. "
                 "Labels shown for facets 20 sq ft and larger.",
                 fontsize=7.5, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 4: area diagram ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Area Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925,
                 f"{m.address}  |  Total = {m.total_area_sqft:,.0f} sq ft, {m.facet_count} facets",
                 fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.15, 0.90, 0.75])
        _setup_ax(ax, vec.bbox)
        _draw_roof(ax, vec)
        for letter, (cy, cx) in vec.centroids.items():
            f = facet_by_letter.get(letter)
            if f is None or f.surface_area_sqft < 10:
                continue
            ax.text(cx, cy, f"{letter}\n{f.surface_area_sqft:,.0f}", fontsize=7.5,
                    fontweight="bold", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.85))
        fig.text(0.08, 0.09,
                 "Facet surface areas in square feet (slope-corrected), labeled smallest to largest (A-Z).",
                 fontsize=7.5, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 5: tables ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Report Summary Tables", fontsize=16, fontweight="bold")

        ax1 = fig.add_axes([0.08, 0.72, 0.84, 0.18])
        rows = [[p, f"{a:,.1f}", f"{100*a/m.total_area_sqft:.1f}%"]
                for p, a in m.areas_per_pitch.items()]
        _table(ax1, ["Roof Pitch", "Area (sq ft)", "% of Roof"], rows, title="Areas per Pitch")

        ax2 = fig.add_axes([0.08, 0.46, 0.84, 0.20])
        wt = m.waste_table()
        _table(ax2,
               ["Waste %"] + [f"{r['waste_pct']}%" for r in wt],
               [["Area (sq ft)"] + [f"{r['area_sqft']:,}" for r in wt],
                ["Squares*"] + [f"{r['squares']:.2f}" for r in wt]],
               title="Waste Calculation (asphalt shingle)", fontsize=7)
        fig.text(0.08, 0.435,
                 f"*Squares rounded up to the nearest 1/3. Suggested waste factor: "
                 f"{m.suggested_waste_pct()}% = {m.squares_with_waste(m.suggested_waste_pct()):.2f} SQ. "
                 "Ridge/hip cap and starter material not included in waste table.",
                 fontsize=7.5, color="#555555")

        n_show = min(len(m.facets), 26)
        ax3 = fig.add_axes([0.08, 0.06, 0.84, 0.33])
        rows = [[f.label, f"{f.surface_area_sqft:,.1f}", f"{f.plan_area_sqft:,.1f}",
                 f.pitch, f"{f.azimuth_deg:.0f}"]
                for f in m.facets[-n_show:]][::-1]
        _table(ax3, ["Facet", "Surface (sq ft)", "Plan (sq ft)", "Pitch", "Slope Dir (deg)"],
               rows, title="Facet Detail (largest first)", fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- Page 6: notes / disclaimer ----------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Notes & Disclaimer", fontsize=16, fontweight="bold")
        y = 0.90
        for n in m.notes:
            fig.text(0.08, y, "- " + n, fontsize=9, wrap=True, verticalalignment="top")
            y -= 0.05
        fig.text(0.08, y - 0.02,
                 "This report was produced from satellite-derived elevation and imagery data.\n"
                 "Accuracy can be affected by tree coverage, image resolution, recent\n"
                 "construction, and other factors. This report does not carry an accuracy\n"
                 "guarantee; field verification of measurements is recommended before\n"
                 "ordering material.",
                 fontsize=9, verticalalignment="top")
        pdf.savefig(fig)
        plt.close(fig)

    return out_path
