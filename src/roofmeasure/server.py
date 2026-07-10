"""RoofMeasure MCP server.

Tools:
  measure_roof              - satellite roof measurements for an address
  generate_roof_report      - EagleView-style Markdown report file
  generate_xactimate_estimate - Xactimate-style .docx insurance estimate

Run:  roofmeasure-mcp          (stdio transport)
"""

from __future__ import annotations

import json
import os
import re
from datetime import date

from mcp.server.fastmcp import FastMCP

from . import pipeline, pricing, report, xactimate

mcp = FastMCP(
    "roofmeasure",
    instructions=(
        "Free satellite roof measurement + Xactimate-style estimating for roofing "
        "contractors. IMPORTANT INTAKE RULE: before calling "
        "generate_xactimate_estimate, make sure you have the property address, "
        "the homeowner's name, and the date of loss. If the user has not "
        "provided them, ASK THE USER for: (1) property address, (2) homeowner "
        "name and phone, (3) date of loss, and optionally claim number, policy "
        "number, and component counts (pipe jacks, vents, skylights, chimneys). "
        "Only the address is required for measure_roof and generate_roof_report."
    ),
)


def _out_dir() -> str:
    d = os.environ.get("ROOFMEASURE_OUT_DIR", os.path.join(os.path.expanduser("~"), "RoofMeasure Reports"))
    os.makedirs(d, exist_ok=True)
    return d


def _slug(address: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", address).strip("_")[:60]


@mcp.tool()
def measure_roof(address: str) -> str:
    """Measure a roof from satellite data. Returns full JSON measurements:
    total area (sq ft + squares), facet count and detail, predominant pitch,
    areas per pitch, ridges/hips/valleys/rakes/eaves lengths (ft), waste table
    (squares with 0-32% waste), suggested waste factor, and steep-slope split.

    Args:
        address: Full property address (street, city, state, zip).
    """
    m = pipeline.measure_address(address)
    return m.to_json()


@mcp.tool()
def generate_roof_report(address: str, company: str = "Stronghouse Solutions",
                         contact: str = "", phone: str = "") -> str:
    """Generate an EagleView-style roof measurement report (Markdown file) for
    an address. Returns JSON with the saved file path and a summary.

    Args:
        address: Full property address.
        company: Company name stamped on the report.
        contact: Contact person shown on the report (optional).
        phone: Contact phone shown on the report (optional).
    """
    m = pipeline.measure_address(address)
    md = report.render_markdown(m, company=company, contact=contact, phone=phone)
    out = os.path.join(_out_dir(), f"Roof_Report_{_slug(address)}.md")
    with open(out, "w") as f:
        f.write(md)
    return json.dumps({
        "report_file": out,
        "summary": {
            "total_area_sqft": m.total_area_sqft,
            "squares": m.squares,
            "facets": m.facet_count,
            "predominant_pitch": m.predominant_pitch,
            "ridges_ft": m.edges.ridges_ft,
            "hips_ft": m.edges.hips_ft,
            "valleys_ft": m.edges.valleys_ft,
            "rakes_ft": m.edges.rakes_ft,
            "eaves_ft": m.edges.eaves_ft,
            "suggested_waste_pct": m.suggested_waste_pct(),
            "method": m.method,
        },
    }, indent=2)


@mcp.tool()
def generate_xactimate_estimate(
    address: str,
    homeowner: str = "",
    homeowner_phone: str = "",
    date_of_loss: str = "",
    claim_number: str = "TBD",
    policy_number: str = "TBD",
    type_of_loss: str = "Wind/Hail",
    tax_rate: float = 0.0,
    price_list: str = "",
    waste_pct: float = 0.0,
    components_json: str = "{}",
    company_name: str = "Stronghouse Solutions",
    company_tagline: str = "Roofing & Exteriors",
    company_city: str = "",
    estimator_name: str = "Stronghouse Solutions",
    include_op: bool = False,
    measurements_json: str = "",
) -> str:
    """Generate an Xactimate-style insurance roofing estimate (.docx).

    INTAKE: homeowner name and date_of_loss are REQUIRED for a complete
    estimate. If missing, this tool returns needs_info listing what to ask the
    user for; collect it and call again.

    Args:
        address: Full property address (required).
        homeowner: Homeowner's full name (required; ask the user if unknown).
        homeowner_phone: Homeowner's phone number.
        date_of_loss: Date of loss, e.g. 5/14/2026 (required; ask the user).
        claim_number: Insurance claim number if available.
        policy_number: Insurance policy number if available.
        type_of_loss: e.g. Wind/Hail (default), Wind, Hail, Tree/Impact.
        tax_rate: Combined LOCAL sales tax rate as a decimal (e.g. 0.0825 for
            Austin TX). If 0, the agent should research the property's combined
            city/county/state rate and pass it in.
        price_list: Xactimate price list code, e.g. TXAU8X_JUL26. If empty, one
            is generated from the state and current month.
        waste_pct: Shingle waste percent. If 0, the suggested waste factor from
            the measurement's complexity analysis is used.
        components_json: JSON object of component counts, e.g.
            {"pipe_jacks": 3, "turtle_vents": 4, "ridge_vent_lf": 40,
             "exhaust_caps": 1, "skylights_flash": 1, "chimneys": 1}.
        company_name/company_tagline/company_city/estimator_name: Branding
            overrides so any contractor can use this tool.
        include_op: Include 10/10 Overhead & Profit (OFF by default).
        measurements_json: Optional pre-computed output of measure_roof to
            avoid re-measuring.
    """
    missing = []
    if not homeowner:
        missing.append("homeowner name")
    if not date_of_loss:
        missing.append("date of loss")
    if missing:
        return json.dumps({
            "needs_info": missing,
            "message": ("Ask the user for: " + ", ".join(missing) +
                        ". Also offer to collect phone, claim number, policy number, "
                        "and roof component counts (pipe jacks, vents, skylights, chimneys)."),
        })

    if measurements_json:
        m = _measurements_from_json(measurements_json, address)
    else:
        m = pipeline.measure_address(address)

    components = json.loads(components_json or "{}")
    rules = pricing.PricingRules()
    waste = waste_pct if waste_pct > 0 else None
    items = pricing.build_roof_line_items(m, waste_pct=waste, rules=rules, components=components)

    state = _state_from_address(m.address)
    if not price_list:
        price_list = f"{state or 'US'}{date.today().strftime('%b').upper()}{date.today().strftime('%y')}"
    if tax_rate <= 0:
        tax_rate = 0.0825  # generic default; agent should research and override

    today = date.today().strftime("%-m/%-d/%Y") if os.name != "nt" else date.today().strftime("%m/%d/%Y")
    prop_lines = [p.strip() for p in m.address.split(",")][:3]
    est_name = _slug(prop_lines[0] if prop_lines else address).upper()[:24]

    cfg = xactimate.EstimateConfig(
        out_file=os.path.join(_out_dir(), f"{company_name.replace(' ', '_')}_Estimate_{_slug(address)}.docx"),
        company_name=company_name,
        company_tagline=company_tagline,
        company_city=company_city,
        logo_path=os.environ.get("ROOFMEASURE_LOGO", ""),
        owner=homeowner,
        phone=homeowner_phone or "TBD",
        property_lines=prop_lines,
        estimator_name=estimator_name,
        contractor_company=company_name,
        contractor_city=company_city,
        claim_number=claim_number,
        policy_number=policy_number,
        type_of_loss=type_of_loss,
        date_of_loss=date_of_loss,
        date_inspected=today,
        date_entered=today,
        price_list=price_list,
        estimate_name=est_name,
        footer_date=today,
        tax_rate=tax_rate,
        include_op=include_op,
        structures=[{
            "heading": "Dwelling Roof",
            "summary_heading": "Summary for Dwelling",
            "totals_label": "Totals: Dwelling Roof",
            "recap_label": "Dwelling Roof",
            "measurements": pricing.build_measurement_block(m),
            "items": items,
        }],
    )
    totals = xactimate.build_estimate(cfg)
    totals["measurement_method"] = m.method
    totals["squares"] = m.squares
    totals["suggested_waste_pct"] = m.suggested_waste_pct()
    totals["notes"] = m.notes
    if tax_rate == 0.0825:
        totals["tax_note"] = ("Default 8.25% tax rate used. Verify the property's combined "
                              "local sales tax rate and re-run with tax_rate if different.")
    return json.dumps(totals, indent=2)


def _measurements_from_json(s: str, address: str):
    from .model import EdgeSummary, Facet, RoofMeasurements

    d = json.loads(s)
    ed = d.get("edges", {})
    edges = EdgeSummary(**{k: v for k, v in ed.items() if k in EdgeSummary.__dataclass_fields__})
    facets = [Facet(**{k: v for k, v in f.items() if k in Facet.__dataclass_fields__})
              for f in d.get("facets", [])]
    return RoofMeasurements(
        address=d.get("address", address),
        latitude=d.get("latitude", 0.0),
        longitude=d.get("longitude", 0.0),
        total_area_sqft=d.get("total_area_sqft", 0.0),
        facets=facets,
        edges=edges,
        predominant_pitch=d.get("predominant_pitch", "0/12"),
        areas_per_pitch=d.get("areas_per_pitch", {}),
        method=d.get("method", "dsm"),
        notes=d.get("notes", []),
    )


def _state_from_address(address: str) -> str:
    mm = re.search(r",\s*([A-Z]{2})\s+\d{5}", address)
    return mm.group(1) if mm else ""


def main():
    mcp.run()


if __name__ == "__main__":
    main()
