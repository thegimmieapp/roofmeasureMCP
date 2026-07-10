"""EagleView-style roof measurement report (Markdown + JSON)."""

from __future__ import annotations

from datetime import date

from .model import RoofMeasurements


def render_markdown(m: RoofMeasurements, company: str = "Stronghouse Solutions",
                    contact: str = "", phone: str = "") -> str:
    e = m.edges
    today = date.today().strftime("%B %d, %Y")
    est_flag = " (ESTIMATED)" if m.method == "segments" else ""
    lines: list[str] = []
    a = lines.append

    a(f"# Roof Measurement Report")
    a("")
    a(f"**{m.address}**  ")
    a(f"{today}")
    a("")
    a(f"Prepared by: **{company}**" + (f" | {contact}" if contact else "") + (f" | {phone}" if phone else ""))
    a("")
    a("---")
    a("")
    a("## Report Summary")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Total Roof Area | **{m.total_area_sqft:,.0f} sq ft** ({m.squares:.2f} squares) |")
    a(f"| Total Roof Facets | {m.facet_count} |")
    a(f"| Predominant Pitch | {m.predominant_pitch} |")
    a(f"| Total Ridges{est_flag} | {e.ridges_ft:,.0f} ft ({e.ridge_count} ridges) |")
    a(f"| Total Hips{est_flag} | {e.hips_ft:,.0f} ft ({e.hip_count} hips) |")
    a(f"| Total Ridges + Hips | {e.ridges_hips_ft:,.0f} ft |")
    a(f"| Total Valleys{est_flag} | {e.valleys_ft:,.0f} ft ({e.valley_count} valleys) |")
    a(f"| Total Rakes{est_flag} | {e.rakes_ft:,.0f} ft ({e.rake_count} rakes) |")
    a(f"| Total Eaves/Starter{est_flag} | {e.eaves_ft:,.0f} ft ({e.eave_count} eaves) |")
    a(f"| Drip Edge (Eaves + Rakes) | {e.drip_edge_ft:,.0f} ft |")
    a(f"| Longitude / Latitude | {m.longitude:.7f} / {m.latitude:.7f} |")
    if m.imagery_date:
        a(f"| Imagery Date / Quality | {m.imagery_date} / {m.imagery_quality} |")
    a(f"| Measurement Method | {'High-resolution DSM (measured)' if m.method == 'dsm' else 'Segment statistics (estimated)'} |")
    a("")
    a("## Areas per Pitch")
    a("")
    a("| Roof Pitch | Area (sq ft) | % of Roof |")
    a("|---|---|---|")
    for pitch, area in m.areas_per_pitch.items():
        pct = 100.0 * area / m.total_area_sqft if m.total_area_sqft else 0
        a(f"| {pitch} | {area:,.1f} | {pct:.1f}% |")
    a("")
    a("## Waste Calculation (asphalt shingle)")
    a("")
    header = "| Waste % |" + "".join(f" {row['waste_pct']}% |" for row in m.waste_table())
    a(header)
    a("|---|" + "---|" * len(m.waste_table()))
    a("| Area (sq ft) |" + "".join(f" {row['area_sqft']:,} |" for row in m.waste_table()))
    a("| Squares* |" + "".join(f" {row['squares']:.2f} |" for row in m.waste_table()))
    a("")
    a(f"Suggested waste factor: **{m.suggested_waste_pct()}%** "
      f"({m.area_with_waste(m.suggested_waste_pct()):,.0f} sq ft / "
      f"{m.squares_with_waste(m.suggested_waste_pct()):.2f} squares)")
    a("")
    a("*Squares are rounded up to the nearest 1/3 square. Additional material for "
      "ridge, hip, and starter lengths is not included in the waste table.")
    a("")
    a("## Facet Detail")
    a("")
    a("| Facet | Surface Area (sq ft) | Plan Area (sq ft) | Pitch | Slope (deg) | Direction (deg) |")
    a("|---|---|---|---|---|---|")
    for f in m.facets:
        a(f"| {f.label} | {f.surface_area_sqft:,.1f} | {f.plan_area_sqft:,.1f} | {f.pitch} | {f.slope_deg} | {f.azimuth_deg} |")
    a("")
    steep = m.steep_area_split()
    a("## Steep-Slope Breakdown (for labor surcharges)")
    a("")
    a("| Band | Area (sq ft) | Squares |")
    a("|---|---|---|")
    a(f"| Below 8/12 | {steep['normal']:,.0f} | {steep['normal']/100:.2f} |")
    a(f"| 8/12 - 9/12 | {steep['steep_8_9']:,.0f} | {steep['steep_8_9']/100:.2f} |")
    a(f"| 10/12 and steeper | {steep['steep_gt9']:,.0f} | {steep['steep_gt9']/100:.2f} |")
    a("")
    if m.notes:
        a("## Notes")
        a("")
        for n in m.notes:
            a(f"- {n}")
        a("")
    a("---")
    a("")
    a("Disclaimer: This report was produced from satellite-derived elevation and "
      "imagery data. Accuracy can be affected by tree coverage, image resolution, "
      "recent construction, and other factors. This report does not carry an "
      "accuracy guarantee; field verification of measurements is recommended "
      "before ordering material.")
    return "\n".join(lines)
