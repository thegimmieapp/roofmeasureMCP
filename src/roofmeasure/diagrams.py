"""EagleView-style PDF measurement report with labeled diagrams.

Pages:
  1. Cover: summary table + aerial image
  2. Length Diagram: color-coded ridges/hips/valleys/rakes/eaves with labels
  3. Pitch Diagram: facets shaded, pitch labels + slope-direction arrows
  4. Area Diagram: facet letters + areas
  5. Tables: areas per pitch, waste calculation, facet detail
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

from .geometry import MeasureDetail  # noqa: E402
from .model import RoofMeasurements  # noqa: E402

EDGE_COLORS = {
    "ridge": "#d62728",   # red
    "hip": "#ff7f0e",     # orange
    "valley": "#1f77b4",  # blue
    "eave": "#2ca02c",    # green
    "rake": "#9467bd",    # purple
}
MIN_LABEL_FT = 5.0


def _roof_bbox(labels: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    ys, xs = np.where(labels > 0)
    h, w = labels.shape
    return (max(ys.min() - pad, 0), min(ys.max() + pad, h - 1),
            max(xs.min() - pad, 0), min(xs.max() + pad, w - 1))


def _setup_ax(ax, bbox):
    y0, y1, x0, x1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # invert y for raster orientation (north up)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_facet_outline(ax, labels, bbox, lw: float = 0.6):
    """Light outline of all facet boundaries."""
    b = np.zeros(labels.shape, dtype=bool)
    lab = labels
    b[:-1, :] |= (lab[:-1, :] != lab[1:, :])
    b[:, :-1] |= (lab[:, :-1] != lab[:, 1:])
    b &= labels > 0
    ys, xs = np.where(b)
    ax.scatter(xs, ys, s=lw, c="#555555", marker="s", linewidths=0)


def _draw_edges(ax, detail: MeasureDetail, label_lengths: bool = True):
    for seg in detail.edges:
        pix = np.asarray(seg.pixels)
        ax.scatter(pix[:, 1], pix[:, 0], s=1.6, c=EDGE_COLORS[seg.kind],
                   marker="s", linewidths=0, zorder=3)
        if label_lengths and seg.length_ft >= MIN_LABEL_FT:
            cy, cx = seg.mid_yx
            ax.annotate(f"{seg.length_ft:.0f}'", xy=(cx, cy), fontsize=6.5,
                        fontweight="bold", color="black", ha="center", va="center",
                        zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=EDGE_COLORS[seg.kind], lw=0.7, alpha=0.9))


def _facet_fill(ax, labels, detail, colors_by_letter: dict):
    img = np.ones((*labels.shape, 4))
    img[..., 3] = 0.0
    for letter, fid in detail.facet_ids.items():
        c = colors_by_letter.get(letter, (0.8, 0.8, 0.8, 1.0))
        img[labels == fid] = c
    ax.imshow(img, interpolation="nearest", zorder=1)


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


def render_diagram_png(
    m: RoofMeasurements,
    detail: MeasureDetail,
    out_path: str,
    rgb: np.ndarray | None = None,
) -> str:
    """Single composite PNG (aerial + pitch-shaded facet diagram) for
    embedding directly into the Xactimate .docx as a 'Roof Diagram' page."""
    labels = detail.labels
    bbox = _roof_bbox(labels)
    fig = plt.figure(figsize=(8.5, 5.2), dpi=200)

    if rgb is not None:
        y0, y1, x0, x1 = bbox
        ax0 = fig.add_axes([0.03, 0.08, 0.42, 0.84])
        ax0.imshow(rgb[y0:y1, x0:x1])
        ax0.axis("off")
        ax0.set_title("Aerial", fontsize=10, fontweight="bold", loc="left")
        diag_rect = [0.49, 0.08, 0.48, 0.84]
    else:
        diag_rect = [0.05, 0.08, 0.90, 0.84]

    ax = fig.add_axes(diag_rect)
    _setup_ax(ax, bbox)
    facet_by_letter = {f.label: f for f in m.facets}
    pitches = sorted({f.pitch for f in m.facets}, key=lambda p: int(p.split("/")[0]))
    cmap = plt.cm.Blues(np.linspace(0.25, 0.85, max(len(pitches), 1)))
    pitch_color = {p: cmap[i] for i, p in enumerate(pitches)}
    colors_by_letter = {l: pitch_color[facet_by_letter[l].pitch]
                        for l in detail.facet_ids if l in facet_by_letter}
    _facet_fill(ax, labels, detail, colors_by_letter)
    _draw_facet_outline(ax, labels, bbox)
    _draw_edges(ax, detail, label_lengths=False)
    for letter, (cy, cx) in detail.facet_centroids.items():
        f = facet_by_letter.get(letter)
        if f is None or f.surface_area_sqft < 20:
            continue
        ax.text(cx, cy, letter, fontsize=8, fontweight="bold", ha="center", va="center",
                zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.6, alpha=0.9))
    ax.set_title("Roof Diagram", fontsize=10, fontweight="bold", loc="left")
    handles = [Line2D([0], [0], color=c, lw=2.5, label=k.capitalize())
              for k, c in EDGE_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
             ncol=5, fontsize=7, frameon=False)

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
    labels = detail.labels
    bbox = _roof_bbox(labels)
    e = m.edges
    today = date.today().strftime("%B %d, %Y")

    with PdfPages(out_path) as pdf:
        # ---- Page 1: cover ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Roof Measurement Report", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.915, m.address, fontsize=12)
        sub = today + (f"  |  Imagery: {m.imagery_date} ({m.imagery_quality})" if m.imagery_date else "")
        fig.text(0.08, 0.89, sub, fontsize=9, color="#444444")
        who = company + (f"  |  {contact}" if contact else "") + (f"  |  {phone}" if phone else "")
        fig.text(0.08, 0.868, f"Prepared by: {who}", fontsize=9, color="#444444")

        ax = fig.add_axes([0.08, 0.44, 0.84, 0.40])
        if rgb is not None:
            y0, y1, x0, x1 = bbox
            ax.imshow(rgb[y0:y1, x0:x1])
            _setup = ax.set_title("Aerial Image", fontsize=10, fontweight="bold", loc="left")
            ax.axis("off")
        else:
            _setup_ax(ax, bbox)
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(detail.facet_ids), 1)))
            _facet_fill(ax, labels, detail, {l: colors[i] for i, l in enumerate(detail.facet_ids)})
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

        # ---- Page 2: length diagram ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Length Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925, m.address, fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.18, 0.90, 0.70])
        _setup_ax(ax, bbox)
        _draw_facet_outline(ax, labels, bbox)
        _draw_edges(ax, detail, label_lengths=True)
        handles = [Line2D([0], [0], color=c, lw=3,
                          label=f"{k.capitalize()}s = {getattr(e, k + 's_ft'):,.0f} ft")
                   for k, c in EDGE_COLORS.items()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=9, frameon=False)
        fig.text(0.08, 0.10,
                 "Segment labels shown for edges 5 ft and longer. Lengths are slope-corrected "
                 "where applicable (hips, valleys, rakes).",
                 fontsize=7.5, color="#555555")
        fig.text(0.08, 0.075,
                 "Field verify measurements before material order.",
                 fontsize=7.5, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: pitch diagram ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Pitch Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925, f"{m.address}  |  Predominant pitch: {m.predominant_pitch}",
                 fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.15, 0.90, 0.75])
        _setup_ax(ax, bbox)
        facet_by_letter = {f.label: f for f in m.facets}
        pitches = sorted({f.pitch for f in m.facets}, key=lambda p: int(p.split("/")[0]))
        cmap = plt.cm.Blues(np.linspace(0.25, 0.85, max(len(pitches), 1)))
        pitch_color = {p: cmap[i] for i, p in enumerate(pitches)}
        colors_by_letter = {l: pitch_color[facet_by_letter[l].pitch]
                            for l in detail.facet_ids if l in facet_by_letter}
        _facet_fill(ax, labels, detail, colors_by_letter)
        _draw_facet_outline(ax, labels, bbox)
        for letter, (cy, cx) in detail.facet_centroids.items():
            f = facet_by_letter.get(letter)
            if f is None or f.surface_area_sqft < 20:
                continue
            az = math.radians(detail.facet_azimuths.get(letter, 0.0))
            dy, dx = -math.cos(az), math.sin(az)  # compass -> raster (row 0 = north)
            dy = -dy  # raster rows grow southward
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

        # ---- Page 4: area diagram ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Area Diagram", fontsize=16, fontweight="bold")
        fig.text(0.08, 0.925,
                 f"{m.address}  |  Total = {m.total_area_sqft:,.0f} sq ft, {m.facet_count} facets",
                 fontsize=9, color="#444444")
        ax = fig.add_axes([0.05, 0.15, 0.90, 0.75])
        _setup_ax(ax, bbox)
        colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(detail.facet_ids), 1)))
        _facet_fill(ax, labels, detail, {l: colors[i] for i, l in enumerate(detail.facet_ids)})
        _draw_facet_outline(ax, labels, bbox)
        for letter, (cy, cx) in detail.facet_centroids.items():
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

        # ---- Page 5: tables ----
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

        # ---- Page 6: notes / disclaimer ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Notes & Disclaimer", fontsize=16, fontweight="bold")
        y = 0.90
        for n in m.notes:
            fig.text(0.08, y, "- " + n, fontsize=9, wrap=True,
                     verticalalignment="top")
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