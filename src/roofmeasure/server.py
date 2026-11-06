"""RoofMeasure MCP server.

Tools:
  measure_roof              - satellite roof measurements for an address
  generate_roof_report      - EagleView-style Markdown report file
  generate_xactimate_estimate - Xactimate-style .docx insurance estimate
  measure_lot               - LotMeasure: parcel, yard zones, surfaces, setbacks, sun, trees
  generate_lot_plan         - LotMeasure plan drawing (PNG + PDF)

HTTP mode also serves a JSON API for websites (X-API-Key header):
  GET /<secret>/api/lot?address=...   GET /<secret>/api/roof?address=...   GET /<secret>/api/health

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
        "Only the address is required for measure_roof and generate_roof_report. "
        "LOT TOOLS: measure_lot and generate_lot_plan need only the address. Quote "
        "their confidence notes; never state a lot size when lot_source is 'none'; "
        "lot lines come from county GIS and are not a survey."
    ),
)


def _out_dir() -> str:
    d = os.environ.get("ROOFMEASURE_OUT_DIR", os.path.join(os.path.expanduser("~"), "RoofMeasure Reports"))
    os.makedirs(d, exist_ok=True)
    return d


def _slug(address: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", address).strip("_")[:60]


def _public_url(path: str) -> str:
    """Download link for a generated file when the server runs remotely
    (ROOFMEASURE_PUBLIC_URL set); empty when it runs on the user's machine."""
    base = os.environ.get("ROOFMEASURE_PUBLIC_URL", "").rstrip("/")
    if not base or not path:
        return ""
    secret = os.environ.get("ROOFMEASURE_PATH_SECRET", "").strip("/")
    prefix = f"{base}/{secret}" if secret else base
    return f"{prefix}/files/{os.path.basename(path)}"


@mcp.tool()
def measure_roof(address: str, include_outbuildings: bool = False) -> str:
    """Measure a roof from satellite data. Returns full JSON measurements:
    total area (sq ft + squares), facet count and detail, predominant pitch,
    areas per pitch, ridges/hips/valleys/rakes/eaves lengths (ft) from a
    plane-intersection wireframe, EagleView-style waste table (3/12+ area,
    suggested factor), structure complexity, and steep-slope split.

    Detached roofs near the house (garage, shed) are reported in
    nearby_structures but NOT added to totals unless include_outbuildings is
    true, because parcel lines are unknown and they may be a neighbor's.

    Args:
        address: Full property address (street, city, state, zip).
        include_outbuildings: Add detached roofs as Structure #2.. and report
            All-Structures totals (set true when the user confirms they are
            on the same property).
    """
    m, _d, _r = pipeline.measure_address_full(address, want_rgb=False,
                                              include_outbuildings=include_outbuildings)
    return m.to_json()


@mcp.tool()
def generate_roof_report(address: str, company: str = "Stronghouse Solutions",
                         contact: str = "", phone: str = "", email: str = "",
                         company_address: str = "", claim_number: str = "",
                         include_outbuildings: bool = False) -> str:
    """Generate an EagleView-layout roof measurement report PDF for an address
    (cover, images, Length / Pitch / Area / Notes diagrams, per-structure
    Report Summary with areas-per-pitch, complexity, waste table, lengths)
    plus a Markdown version. Returns JSON with file paths and a summary.

    Args:
        address: Full property address.
        company: Company name on the report (Stronghouse logo when Stronghouse).
        contact: Contact person shown on the cover / prepared-for block.
        phone: Contact phone.
        email: Contact email.
        company_address: Company mailing address (comma separated).
        claim_number: Insurance claim number to print in the header.
        include_outbuildings: Include detached roofs (garage/shed) as extra
            structures with their own summary pages.
    """
    m, detail, rgb = pipeline.measure_address_full(address, include_outbuildings=include_outbuildings)
    md = report.render_markdown(m, company=company, contact=contact, phone=phone)
    out = os.path.join(_out_dir(), f"Roof_Report_{_slug(address)}.md")
    with open(out, "w") as f:
        f.write(md)
    pdf_path = ""
    if detail is not None:
        from .diagrams import render_pdf_report

        pdf_path = os.path.join(_out_dir(), f"Roof_Report_{_slug(address)}.pdf")
        render_pdf_report(m, detail, pdf_path, rgb=rgb, company=company,
                          contact=contact, phone=phone, email=email,
                          company_address=company_address, claim_number=claim_number)
    return json.dumps({
        "report_pdf": pdf_path or "unavailable (segment-statistics fallback; no raster geometry)",
        "report_pdf_url": _public_url(pdf_path),
        "report_file": out,
        "report_file_url": _public_url(out),
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
            "rakes_measured_ft": m.edges.rakes_measured_ft,
            "eaves_measured_ft": m.edges.eaves_measured_ft,
            "eave_rake_safety_factor": m.edges.safety_factor,
            "suggested_waste_pct": m.suggested_waste_pct(),
            "complexity": m.complexity(),
            "structures": max(len(m.structures), 1),
            "nearby_structures_not_included": m.nearby_structures,
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

    detail = rgb = None
    if measurements_json:
        m = _measurements_from_json(measurements_json, address)
    else:
        m, detail, rgb = pipeline.measure_address_full(address)

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

    diagram_image_path = ""
    if detail is not None:
        try:
            from .diagrams import render_diagram_png

            diagram_image_path = os.path.join(_out_dir(), f"_diagram_{_slug(address)}.png")
            render_diagram_png(m, detail, diagram_image_path, rgb=rgb)
        except Exception as exc:
            diagram_image_path = ""
            m.notes.append(f"Roof diagram image unavailable ({type(exc).__name__}: {exc}).")

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
        diagram_image_path=diagram_image_path,
        imagery_date=m.imagery_date,
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
    totals["estimate_url"] = _public_url(cfg.out_file)
    totals["squares"] = m.squares
    totals["suggested_waste_pct"] = m.suggested_waste_pct()
    totals["notes"] = m.notes
    if tax_rate == 0.0825:
        totals["tax_note"] = ("Default 8.25% tax rate used. Verify the property's combined "
                              "local sales tax rate and re-run with tax_rate if different.")
    return json.dumps(totals, indent=2)


# ---------------------------------------------------------------- LotMeasure

CACHE_TTL_S = 24 * 3600

import threading as _threading  # noqa: E402

# A 100 m lot tile peaks near 0.9 GB; the Fly machine has 2 GB, so lot runs
# are serialized (a second request waits instead of running the box out of memory).
_LOT_LOCK = _threading.Lock()


def _cache_path(kind: str, address: str, variant: str = "") -> str:
    import hashlib

    norm = re.sub(r"\s+", " ", address.strip().lower())
    h = hashlib.sha256(f"{kind}|{norm}|{variant}".encode()).hexdigest()[:24]
    d = os.path.join(_out_dir(), "api_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{kind}_{h}.json")


def _cache_get(kind: str, address: str, variant: str = ""):
    import time

    p = _cache_path(kind, address, variant)
    try:
        if time.time() - os.path.getmtime(p) < CACHE_TTL_S:
            with open(p) as f:
                d = json.load(f)
            d["cached"] = True
            return d
    except (OSError, ValueError):
        pass
    return None


def _cache_put(kind: str, address: str, data: dict, variant: str = "") -> None:
    try:
        with open(_cache_path(kind, address, variant), "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def _lot_with_plan(address: str, include_sun: bool = True, brand: str = "Stronghouse Solutions",
                   prepared_for: str = "", make_plan: bool = True, make_map: bool = False) -> dict:
    from . import lot

    with _LOT_LOCK:
        res = lot.measure_lot(address, include_sun=include_sun, keep_rasters=make_plan or make_map)
        if make_map and "_plan" in res:
            from .scopemap import scope_map, save_aerial
            res["scope_map"] = scope_map(res)
            # Hash avoids address-bearing filenames in the landscape integration.
            import hashlib
            digest = hashlib.sha256(address.strip().lower().encode()).hexdigest()[:24]
            aerial = os.path.join(_out_dir(), f"Scope_Aerial_{digest}.png")
            if save_aerial(res, aerial):
                res["aerial_png_url"] = _public_url(aerial)
        return _finish_lot(res, address, brand, prepared_for, make_plan)


def _finish_lot(res, address, brand, prepared_for, make_plan):
    if make_plan and "_plan" in res:
        from .lotplan import render_lot_plan

        slug = _slug(address)
        btag = re.sub(r"[^A-Za-z0-9]+", "", brand)[:20] or "plan"
        png = os.path.join(_out_dir(), f"Lot_Plan_{slug}_{btag}.png")
        pdf = os.path.join(_out_dir(), f"Lot_Plan_{slug}_{btag}.pdf")
        try:
            render_lot_plan(res, png, pdf_path=pdf, brand=brand, prepared_for=prepared_for)
            res["plan_png"] = png
            res["plan_pdf"] = pdf
            res["plan_png_url"] = _public_url(png)
            res["plan_pdf_url"] = _public_url(pdf)
        except Exception as exc:  # the numbers stand even if the drawing fails
            res["confidence"]["notes"].append(f"Plan drawing unavailable ({type(exc).__name__}: {exc}).")
    res.pop("_plan", None)
    return res


@mcp.tool()
def measure_lot(address: str, include_sun: bool = True) -> str:
    """Measure a property lot for landscape design from a street address.

    Returns JSON: parcel (lot line ring in lat/lng, area sq ft and acres,
    source, county-recorded area and the difference), house footprint (plan
    sq ft and outline, from RoofMeasure), yard zones front/back/left/right
    (area, surface breakdown: lawn_planting, hardscape, tree_canopy,
    pool_water, bare_soil_or_mulch, detached_structure, unknown; setback from
    the house wall to the lot line; largest clear plantable rectangle; sun
    classes full/part/shade from June hours of direct sun), trees (location,
    canopy diameter, height, within 3 m of the house), totals (plantable and
    hardscape), confidence notes, imagery date.

    lot_source is "county_arcgis", "regrid" or "none". When it is "none" there
    is NO lot size: never state one; the yard figures then cover a 15 m band
    around the house. Lot lines come from county GIS, not a survey.

    Args:
        address: Full property address (street, city, state, zip).
        include_sun: Include sun exposure from the Solar API hourly shade layers.
    """
    hit = _cache_get("lot", address, f"sun={int(include_sun)}")
    if hit:
        return json.dumps(hit, indent=1)
    res = _lot_with_plan(address, include_sun=include_sun, make_plan=False)
    _cache_put("lot", address, res, f"sun={int(include_sun)}")
    return json.dumps(res, indent=1)


@mcp.tool()
def generate_lot_plan(address: str, brand: str = "Stronghouse Solutions", prepared_for: str = "",
                      include_sun: bool = True) -> str:
    """Draw the lot plan (PNG + PDF): parcel outline, rectilinear house,
    yard surfaces shaded by class, sun hatching on plantable ground, trees,
    setback dimensions, largest clear rectangles, north arrow, scale bar,
    zone table and confidence notes. Returns JSON with plan_png_url /
    plan_pdf_url download links and the full measure_lot result.

    Args:
        address: Full property address.
        brand: "Stronghouse Solutions" (logo) or "Dream Landscaper" (neutral
            DL header), or any company name for a plain header.
        prepared_for: Optional homeowner / client name for the header.
        include_sun: Include sun exposure hatching and sun classes.
    """
    res = _lot_with_plan(address, include_sun=include_sun, brand=brand, prepared_for=prepared_for)
    out = {k: res.get(k, "") for k in ("plan_png", "plan_png_url", "plan_pdf", "plan_pdf_url")}
    out["lot"] = res
    return json.dumps(out, indent=1)


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


def main_http():
    """Run as a remote MCP server (Streamable HTTP) so it can be added to
    claude.ai as a custom connector and used from any chat.

    Environment:
      PORT                     listen port (default 8080)
      GOOGLE_MAPS_API_KEY      required
      ROOFMEASURE_PATH_SECRET  unguessable path segment; the MCP endpoint is
                               /<secret>/mcp and files are /<secret>/files/<name>
      ROOFMEASURE_PUBLIC_URL   https://your-app.example.com (for download links)
      ROOFMEASURE_OUT_DIR      where reports are written (default /data/reports)
      LOTMEASURE_API_KEY       key the website sends as X-API-Key for /<secret>/api/*
      REGRID_API_KEY           optional: Regrid parcel API fallback for counties without a free layer
    """
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, PlainTextResponse

    secret = os.environ.get("ROOFMEASURE_PATH_SECRET", "").strip("/")
    prefix = f"/{secret}" if secret else ""
    os.environ.setdefault("ROOFMEASURE_OUT_DIR", "/data/reports")
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("PORT", "8080"))
    mcp.settings.streamable_http_path = f"{prefix}/mcp"
    mcp.settings.stateless_http = True
    mcp.settings.json_response = True
    # The SDK's DNS-rebinding guard answers 421 to any Host header that is
    # not on its allow-list, and the list is empty by default, so every
    # request arriving via the public hostname (*.fly.dev) was refused.
    # Allow the public host (from ROOFMEASURE_PUBLIC_URL, or
    # ROOFMEASURE_ALLOWED_HOSTS as a comma list) plus localhost for dev.
    from urllib.parse import urlparse
    from mcp.server.transport_security import TransportSecuritySettings
    hosts = {"localhost:*", "127.0.0.1:*"}
    origins = {"http://localhost:*", "http://127.0.0.1:*"}
    pub = urlparse(os.environ.get("ROOFMEASURE_PUBLIC_URL", ""))
    if pub.hostname:
        hosts.add(pub.netloc)
        hosts.add(pub.hostname)
        origins.add(f"{pub.scheme or 'https'}://{pub.netloc}")
    for h in os.environ.get("ROOFMEASURE_ALLOWED_HOSTS", "").split(","):
        h = h.strip()
        if h:
            hosts.add(h)
            origins.add(f"https://{h}")
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(hosts),
        allowed_origins=sorted(origins),
    )

    @mcp.custom_route(f"{prefix}/files/{{name}}", methods=["GET"])
    async def _serve_file(request: Request):
        name = os.path.basename(request.path_params["name"])
        path = os.path.join(_out_dir(), name)
        if not os.path.isfile(path):
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(path, filename=name)

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(_request: Request):
        return JSONResponse({"ok": True, "service": "roofmeasure-mcp"})

    # ---- JSON API for websites (Dream Landscaper): X-API-Key: <LOTMEASURE_API_KEY>
    import hmac

    import anyio

    from . import __version__

    def _authorized(request: Request) -> bool:
        want = os.environ.get("LOTMEASURE_API_KEY", "").strip()
        got = request.headers.get("x-api-key", "").strip()
        return bool(want) and hmac.compare_digest(want.encode(), got.encode())

    def _deny():
        return JSONResponse({"ok": False, "error": "missing or invalid X-API-Key"}, status_code=401)

    def _flag(request: Request, name: str, default: bool) -> bool:
        v = request.query_params.get(name)
        return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

    @mcp.custom_route(f"{prefix}/api/health", methods=["GET"])
    async def _api_health(request: Request):
        return JSONResponse({"ok": True, "service": "roofmeasure-mcp", "version": __version__,
                             "authorized": _authorized(request),
                             "api_key_configured": bool(os.environ.get("LOTMEASURE_API_KEY", "").strip())})

    @mcp.custom_route(f"{prefix}/api/lot", methods=["GET"])
    async def _api_lot(request: Request):
        if not _authorized(request):
            return _deny()
        address = (request.query_params.get("address") or "").strip()
        if len(address) < 6:
            return JSONResponse({"ok": False, "error": "address query parameter is required"}, status_code=400)
        sun = _flag(request, "sun", True)
        plan = _flag(request, "plan", True)
        make_map = _flag(request, "map", False)
        brand = request.query_params.get("brand") or "Dream Landscaper"
        variant = f"sun={int(sun)}|plan={int(plan)}|brand={brand}"
        if make_map:
            variant += "|map=1"
        if not _flag(request, "refresh", False):
            hit = _cache_get("lot", address, variant)
            if hit:
                return JSONResponse({"ok": True, **hit})
        try:
            res = await anyio.to_thread.run_sync(
                lambda: _lot_with_plan(address, include_sun=sun, brand=brand, make_plan=plan, make_map=make_map))
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=502)
        for k in ("plan_png", "plan_pdf"):
            res.pop(k, None)               # server paths stay private; the *_url fields are the links
        res["cached"] = False
        _cache_put("lot", address, res, variant)
        return JSONResponse({"ok": True, **res})

    @mcp.custom_route(f"{prefix}/api/roof", methods=["GET"])
    async def _api_roof(request: Request):
        if not _authorized(request):
            return _deny()
        address = (request.query_params.get("address") or "").strip()
        if len(address) < 6:
            return JSONResponse({"ok": False, "error": "address query parameter is required"}, status_code=400)
        if not _flag(request, "refresh", False):
            hit = _cache_get("roof", address)
            if hit:
                return JSONResponse({"ok": True, **hit})
        try:
            raw = await anyio.to_thread.run_sync(lambda: measure_roof(address))
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=502)
        res = json.loads(raw)
        res["cached"] = False
        _cache_put("roof", address, res)
        return JSONResponse({"ok": True, **res})

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
