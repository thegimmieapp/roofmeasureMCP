"""Auto-build Xactimate line items from roof measurements.

Encodes the Stronghouse Solutions standing pricing rules (all overridable):
  - Synthetic 30# felt underlayment + architectural laminated shingles only.
  - 1:1 replacement of existing components.
  - Blended target ~$550/SQ across core roofing components (tear-off,
    underlayment, shingles, hip/ridge cap, starter, valley, step flashing);
    the shingle unit price is back-solved so the blend hits the target.
  - Steep-roof labor surcharge line items for 8/12-9/12 and >9/12 areas.
  - O&P off by default.
Pricing leans to the higher end of the region (insurance claims).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import RoofMeasurements


@dataclass
class PricingRules:
    blended_target_per_sq: float = 550.0
    tearoff_per_sq: float = 110.0
    underlayment_per_sq: float = 75.0
    shingle_per_sq_default: float = 268.0   # back-solved toward blend target
    hip_ridge_cap_per_lf: float = 11.0
    starter_per_lf: float = 3.50
    valley_metal_per_lf: float = 6.50
    step_flashing_per_lf: float = 22.0
    drip_edge_per_lf: float = 3.25
    steep_8_9_per_sq: float = 65.0
    steep_gt9_per_sq: float = 95.0
    # material fraction per component (drives sales tax)
    matl_frac: dict = field(default_factory=lambda: {
        "tearoff": 0.0, "underlayment": 0.60, "shingles": 0.65,
        "hip_ridge": 0.55, "starter": 0.55, "valley": 0.55,
        "step_flashing": 0.50, "drip_edge": 0.65,
    })
    # specialty items (EA)
    pipe_jack_ea: float = 135.0
    turtle_vent_ea: float = 150.0
    ridge_vent_per_lf: float = 14.0
    exhaust_cap_ea: float = 150.0
    skylight_flash_kit_ea: float = 1100.0
    chimney_flashing_ea: float = 550.0


def build_roof_line_items(
    m: RoofMeasurements,
    waste_pct: float | None = None,
    rules: PricingRules | None = None,
    components: dict | None = None,
) -> list[tuple]:
    """Build the dwelling-roof `items` list for xactimate.build_estimate.

    components: optional counts, e.g. {"pipe_jacks": 3, "turtle_vents": 4,
    "ridge_vent_lf": 40, "exhaust_caps": 1, "skylights_flash": 1, "chimneys": 1}
    """
    rules = rules or PricingRules()
    waste = waste_pct if waste_pct is not None else m.suggested_waste_pct()
    sq = m.total_area_sqft / 100.0
    sq_waste = m.area_with_waste(waste) / 100.0
    e = m.edges
    steep = m.steep_area_split()

    est_note = ""
    if m.method == "segments":
        est_note = (" Linear footage estimated from satellite segment statistics; "
                    "field verification recommended prior to material order.")

    # --- back-solve shingle price toward the blended target ---
    hip_ridge_lf = e.ridges_hips_ft
    starter_lf = e.eaves_ft
    valley_lf = e.valleys_ft
    step_lf = e.step_flashing_ft
    fixed = (
        sq * rules.tearoff_per_sq
        + sq_waste * rules.underlayment_per_sq
        + hip_ridge_lf * rules.hip_ridge_cap_per_lf
        + starter_lf * rules.starter_per_lf
        + valley_lf * rules.valley_metal_per_lf
        + step_lf * rules.step_flashing_per_lf
    )
    target_total = rules.blended_target_per_sq * sq
    shingle_price = (target_total - fixed) / sq_waste if sq_waste > 0 else rules.shingle_per_sq_default
    shingle_price = max(min(shingle_price, rules.shingle_per_sq_default * 1.35),
                        rules.shingle_per_sq_default * 0.75)
    shingle_price = round(shingle_price, 2)

    mf = rules.matl_frac
    items: list[tuple] = [
        ("Tear off, haul and dispose of comp. shingles - Laminated",
         round(sq, 2), "SQ", rules.tearoff_per_sq, mf["tearoff"]),
        ("Roofing felt - synthetic underlayment, 30# equivalent",
         round(sq_waste, 2), "SQ", rules.underlayment_per_sq, mf["underlayment"]),
        ("Laminated - comp. shingle rfg. - w/out felt",
         round(sq_waste, 2), "SQ", shingle_price, mf["shingles"],
         f"{waste}% waste factor applied for roof complexity ({m.facet_count} facets, "
         f"predominant pitch {m.predominant_pitch})."),
    ]
    if hip_ridge_lf > 0:
        items.append(("Hip / Ridge cap - High profile - composition shingles",
                      round(hip_ridge_lf, 2), "LF", rules.hip_ridge_cap_per_lf, mf["hip_ridge"],
                      f"Measured: {e.ridges_ft:.0f} LF ridge + {e.hips_ft:.0f} LF hip.{est_note}"))
    if starter_lf > 0:
        items.append(("Asphalt starter - universal starter course",
                      round(starter_lf, 2), "LF", rules.starter_per_lf, mf["starter"],
                      f"Eave length.{est_note}"))
    if e.drip_edge_ft > 0:
        items.append(("Drip edge",
                      round(e.drip_edge_ft, 2), "LF", rules.drip_edge_per_lf, mf["drip_edge"],
                      f"Eaves ({e.eaves_ft:.0f} LF) + rakes ({e.rakes_ft:.0f} LF).{est_note}"))
    if valley_lf > 0:
        items.append(("Valley metal - self-adhering waterproof underlayment",
                      round(valley_lf, 2), "LF", rules.valley_metal_per_lf, mf["valley"],
                      est_note.strip() or None))
    if step_lf > 0:
        items.append(("Flashing - step/counter, remove & replace",
                      round(step_lf, 2), "LF", rules.step_flashing_per_lf, mf["step_flashing"]))

    # steep charges (labor only, no material, no tax)
    if steep["steep_8_9"] > 0:
        items.append(("Additional charge for steep roof - 8/12 to 9/12 slope",
                      round(steep["steep_8_9"] / 100.0, 2), "SQ", rules.steep_8_9_per_sq, 0.0,
                      "Labor-only surcharge, no material component. Applied to measured "
                      "8/12-9/12 pitch area from the satellite pitch table."))
    if steep["steep_gt9"] > 0:
        items.append(("Additional charge for steep roof - greater than 9/12 slope",
                      round(steep["steep_gt9"] / 100.0, 2), "SQ", rules.steep_gt9_per_sq, 0.0,
                      "Labor-only surcharge, no material component. Applied to measured "
                      "10/12+ pitch area from the satellite pitch table."))

    # specialty components
    c = components or {}
    if c.get("pipe_jacks"):
        items.append(('Flashing - pipe jack, 1" to 3" - remove & replace',
                      float(c["pipe_jacks"]), "EA", rules.pipe_jack_ea, 0.50))
    if c.get("turtle_vents"):
        items.append(("Roof vent - turtle/box type, passive, static - remove & replace",
                      float(c["turtle_vents"]), "EA", rules.turtle_vent_ea, 0.55))
    if c.get("ridge_vent_lf"):
        items.append(("Continuous ridge vent - shingle-over style - remove & replace",
                      float(c["ridge_vent_lf"]), "LF", rules.ridge_vent_per_lf, 0.55))
    if c.get("exhaust_caps"):
        items.append(('Exhaust cap - Type B gas appliance vent, 3" to 5" - remove & replace',
                      float(c["exhaust_caps"]), "EA", rules.exhaust_cap_ea, 0.55))
    if c.get("skylights_flash"):
        items.append(("Skylight - flashing kit remove & replace only, existing unit reused",
                      float(c["skylights_flash"]), "EA", rules.skylight_flash_kit_ea, 0.50))
    if c.get("chimneys"):
        items.append(("Chimney flashing - average (32\" x 36\") - remove & replace",
                      float(c["chimneys"]), "EA", rules.chimney_flashing_ea, 0.50))

    # strip trailing None notes
    return [tuple(x for x in it if x is not None) for it in items]


def build_measurement_block(m: RoofMeasurements) -> list[tuple[str, str]]:
    e = m.edges
    return [
        (f"{m.total_area_sqft:,.2f}", "Surface Area (sq ft)"),
        (f"{m.squares:.2f}", "Number of Squares"),
        (f"{e.eaves_ft:,.0f}", "Total Eave Length (LF)"),
        (f"{e.rakes_ft:,.0f}", "Total Rake Length (LF)"),
        (f"{e.ridges_ft:,.0f}", "Total Ridge Length (LF)"),
        (f"{e.hips_ft:,.0f}", "Total Hip Length (LF)"),
        (f"{e.valleys_ft:,.0f}", "Total Valley Length (LF)"),
        (f"{m.predominant_pitch}", "Predominant Pitch"),
    ]
