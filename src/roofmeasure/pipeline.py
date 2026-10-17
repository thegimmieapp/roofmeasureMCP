"""End-to-end measurement pipeline: address -> RoofMeasurements."""

from __future__ import annotations

from collections import deque

import math

import numpy as np

from . import geometry, solar
from .pipeline_util import box_sum
from .model import RoofMeasurements


def measure_address(address: str, quality: str = "LOW", radius_m: float = 60.0,
                    include_outbuildings: bool = False) -> RoofMeasurements:
    """Geocode an address and produce full roof measurements."""
    m, _detail, _rgb = measure_address_full(address, quality=quality, radius_m=radius_m, want_rgb=False,
                                            include_outbuildings=include_outbuildings)
    return m


def measure_address_full(address: str, quality: str = "LOW", radius_m: float = 60.0,
                         want_rgb: bool = True, include_outbuildings: bool = False):
    """Full pipeline returning (RoofMeasurements, MeasureDetail | None, rgb | None).

    detail carries raster geometry for diagram rendering; rgb is the aerial
    image aligned to the same raster. Both are None when the DSM pipeline is
    unavailable and the segment-statistics fallback was used.
    """
    lat, lng, formatted = solar.geocode(address)
    insights = solar.building_insights(lat, lng, quality=quality)

    try:
        layers = solar.data_layers(lat, lng, radius_m=radius_m, quality=quality)
        dsm_url = layers.get("dsmUrl")
        mask_url = layers.get("maskUrl")
        if not dsm_url or not mask_url:
            raise solar.SolarAPIError("dataLayers response missing dsmUrl/maskUrl")
        dsm, px_x, px_y = solar.fetch_geotiff(dsm_url)
        mask, _, _ = solar.fetch_geotiff(mask_url)
        m, detail = measure_from_rasters(dsm, mask > 0, px_x, px_y, formatted, lat, lng,
                                         insights=insights, include_outbuildings=include_outbuildings)
        m.imagery_date = geometry._fmt_date(layers.get("imageryDate"))
        m.imagery_quality = layers.get("imageryQuality", insights.get("imageryQuality", ""))
        rgb = solar.fetch_geotiff_rgb(layers["rgbUrl"]) if (want_rgb and layers.get("rgbUrl")) else None
        return m, detail, crop_rgb(rgb, detail)
    except Exception as exc:  # graceful fallback
        m = geometry.estimate_from_segments(insights, formatted, lat, lng)
        m.notes.append(f"DSM pipeline unavailable ({type(exc).__name__}: {exc}); used segment-statistics fallback.")
        return m, None, None


def measure_from_rasters(dsm: np.ndarray, mask: np.ndarray, px_x: float, px_y: float,
                         address: str, lat: float, lng: float, insights: dict | None = None,
                         include_outbuildings: bool = False):
    """Measure from already-downloaded rasters (used by the live pipeline and
    by the offline calibration harness). Returns (RoofMeasurements, detail).

    When detached roofs (garage, shed) are found near the primary structure
    they are measured separately and attached as m.structures; the returned
    m then carries the All-Structures totals and detail.structures the
    per-structure rasters for diagrams.
    """
    seed_yx = _building_center_px(dsm.shape, lat, lng, px_x, px_y, insights)
    roof, grown_pct, cand = recover_footprint(dsm, mask, px_x, px_y, seed_yx=seed_yx,
                                              return_candidates=True)
    # Crop every later stage to the roof neighborhood (+25 m, enough to catch
    # a detached garage): plane fitting and flood fills are O(pixels).
    ys, xs = np.where(roof)
    if len(ys) == 0:
        raise RuntimeError("no roof pixels found at this location")
    margin = int(25.0 / px_x)
    h, w = dsm.shape
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, h)
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, w)
    crop = (y0, y1, x0, x1)
    dsm, mask, roof, cand = dsm[y0:y1, x0:x1], mask[y0:y1, x0:x1], roof[y0:y1, x0:x1], cand[y0:y1, x0:x1]
    ground = ground_model(dsm, px_x, px_y)
    main, main_detail = _measure_structure(dsm, roof, ground, px_x, px_y, address, lat, lng)
    main_detail.crop = crop
    if grown_pct > 10:
        main.notes.append(
            f"About {grown_pct:.0f}% of the roof footprint was recovered from the "
            "elevation model beyond Google's solar mask (tree-shaded or obstructed "
            "areas). Edge lengths in those areas are approximate; field verify "
            "before material order."
        )
    _assess_confidence(main, dsm, roof, ground, px_x, px_y, insights)
    structures = [(main, main_detail)]
    blobs = find_outbuildings(roof, cand, px_x, px_y)
    if blobs and not include_outbuildings:
        # Parcel boundaries are unknown, so nearby detached roofs may belong to a
        # neighbor: report them, do not add them to the totals.
        descr = []
        for blob in blobs:
            plan_sqft = float(blob.sum()) * px_x * px_y * 10.76391
            dist_m = _blob_distance_m(roof, blob, px_x)
            descr.append(f"~{plan_sqft:.0f} sq ft plan, {dist_m * 3.28084:.0f} ft away")
        main.nearby_structures = descr
        main.notes.append(
            f"{len(blobs)} detached roof(s) detected near the primary structure "
            f"({'; '.join(descr)}). Not included in totals: re-run with "
            "include_outbuildings=true if they are on this property (detached garage, shed)."
        )
    elif blobs:
        for k, blob in enumerate(blobs, start=2):
            try:
                sm, sd = _measure_structure(dsm, blob, ground, px_x, px_y, address, lat, lng)
            except Exception as exc:  # never let a shed break the main report
                main.notes.append(f"Secondary structure {k} skipped ({type(exc).__name__}: {exc}).")
                continue
            if sm.total_area_sqft < 60:
                continue
            sm.structure_index = k
            structures.append((sm, sd))
    if len(structures) == 1:
        return main, main_detail
    combined = combine_structures([s for s, _ in structures])
    combined_detail = main_detail
    combined_detail.structures = [d for _, d in structures]
    return combined, combined_detail


def crop_rgb(rgb, detail):
    """Crop an aerial image aligned to the full raster to detail's crop window."""
    if rgb is None:
        return None
    c = getattr(detail, "crop", None)
    if not c:
        return rgb
    y0, y1, x0, x1 = c
    return rgb[y0:y1, x0:x1]


def _assess_confidence(m: RoofMeasurements, dsm, roof, ground, px_x, px_y, insights) -> None:
    """Flag roofs the satellite data cannot fully see (tree canopy).

    Canopy over or beside the roof shows up as rough, non-planar elevation
    within a short halo of the footprint. When a large share of that halo is
    rough, the footprint itself is likely clipped and every measurement is
    a lower bound. Google's own roof segment total is reported alongside as
    an independent cross-check.
    """
    halo = roof.copy()
    for _ in range(int(2.0 / px_x)):
        halo = _dilate4(halo)
    ring = halo & ~roof
    if ring.any():
        resid = local_plane_residual(dsm, px_x, px_y, k=4)
        ndsm = dsm - ground
        rough_elev = (resid > 0.12) & (ndsm > 2.0)
        canopy_frac = float((rough_elev & ring).sum()) / float(ring.sum())
    else:
        canopy_frac = 0.0
    m.canopy_fraction = round(canopy_frac, 2)
    google_sqft = 0.0
    try:
        google_sqft = float(insights["solarPotential"]["wholeRoofStats"]["areaMeters2"]) * 10.76391
    except (KeyError, TypeError, ValueError):
        pass
    m.google_roof_area_sqft = round(google_sqft)
    if canopy_frac >= 0.30:
        m.confidence = "LOW"
        m.notes.append(
            f"LOW CONFIDENCE: about {canopy_frac:.0%} of the roof perimeter is bordered by tree "
            "canopy, so parts of the roof are hidden from the satellite elevation model. Treat "
            "these measurements as a LOWER BOUND and field measure before ordering material."
        )
    elif canopy_frac >= 0.20:
        m.confidence = "MEDIUM"
        m.notes.append(
            f"Tree canopy borders about {canopy_frac:.0%} of the roof perimeter; edges and areas "
            "near the trees are approximate. Field verify before material order."
        )
    else:
        m.confidence = "HIGH"


def _measure_structure(dsm, roof, ground, px_x, px_y, address, lat, lng):
    m, detail = geometry.measure_from_dsm(dsm, roof, px_x, px_y, address, lat, lng,
                                          return_detail=True)
    detail.dsm = dsm
    detail.ground = ground
    detail.roof_mask = roof
    _attach_skeleton(m, detail, px_x)
    _apply_eave_rake_margin(m)
    return m, detail


def _apply_eave_rake_margin(m: RoofMeasurements) -> None:
    """Report eaves and rakes high on purpose (see model.EAVE_RAKE_SAFETY)."""
    from .model import EAVE_RAKE_SAFETY

    e = m.edges
    e.apply_safety(EAVE_RAKE_SAFETY)
    pct = round((EAVE_RAKE_SAFETY - 1.0) * 100)
    m.notes.append(
        f"Eave and rake totals include a +{pct}% ordering margin: measured eaves "
        f"{e.eaves_measured_ft:,.0f} ft and rakes {e.rakes_measured_ft:,.0f} ft are reported as "
        f"{e.eaves_ft:,.0f} ft and {e.rakes_ft:,.0f} ft. Outline edges are the least certain lines "
        "in satellite elevation data and drip edge / starter should err long, not short. "
        "Diagram segment labels show the measured lengths."
    )


def find_outbuildings(roof: np.ndarray, cand: np.ndarray, px_x: float, px_y: float,
                      max_dist_m: float = 20.0, min_area_m2: float = 9.0,
                      max_rel_area: float = 0.35, max_area_m2: float = 200.0) -> list[np.ndarray]:
    """Detached roof blobs near the primary structure (garage, shed, barn).

    Candidates are elevated+planar components that do not touch the main
    roof, lie within max_dist_m of it, and are smaller than max_rel_area of
    it (anything comparable in size is treated as a neighbor's house).
    """
    h, w = roof.shape
    px_area = px_x * px_y
    main_px = float(roof.sum())
    # exclusion halo around the main roof so eave-edge debris is not a shed
    halo = roof.copy()
    for _ in range(int(1.5 / px_x)):
        halo = _dilate4(halo)
    free = cand & ~halo
    ys, xs = np.where(roof)
    out = []
    comp, n = _label_components(free)
    ids, counts = np.unique(comp[comp > 0], return_counts=True)
    for cid, cnt in zip(ids.tolist(), counts.tolist()):
        n_px = float(cnt)
        if n_px * px_area < min_area_m2 or n_px > max_rel_area * main_px or n_px * px_area > max_area_m2:
            continue
        blob = comp == cid
        by, bx = np.where(blob)
        step = max(int(len(by) / 400), 1)
        dmin = np.inf
        for yy_, xx_ in zip(by[::step], bx[::step]):
            d = np.min((ys[::7] - yy_) ** 2 + (xs[::7] - xx_) ** 2)
            dmin = min(dmin, d)
        if math.sqrt(dmin) * px_x > max_dist_m:
            continue
        out.append(_fill_holes(blob))
    out.sort(key=lambda b: -float(b.sum()))
    return out[:4]


def _blob_distance_m(roof: np.ndarray, blob: np.ndarray, px: float) -> float:
    ys, xs = np.where(roof)
    by, bx = np.where(blob)
    step = max(int(len(by) / 300), 1)
    dmin = np.inf
    for yy_, xx_ in zip(by[::step], bx[::step]):
        dmin = min(dmin, float(np.min((ys[::7] - yy_) ** 2 + (xs[::7] - xx_) ** 2)))
    return math.sqrt(dmin) * px


def _point_mask(h, w, y, x):
    m = np.zeros((h, w), dtype=bool)
    m[y, x] = True
    return m


def _dilate4(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def combine_structures(structures: list) -> RoofMeasurements:
    """All-Structures totals from per-structure measurements."""
    from .model import EdgeSummary, Facet

    main = structures[0]
    edges = EdgeSummary()
    facets: list[Facet] = []
    per_pitch: dict[str, float] = {}
    notes: list[str] = []
    for sm in structures:
        for k in ("ridges_ft", "hips_ft", "valleys_ft", "rakes_ft", "eaves_ft", "flashing_ft",
                  "step_flashing_ft", "eaves_measured_ft", "rakes_measured_ft",
                  "ridge_count", "hip_count", "valley_count", "rake_count", "eave_count"):
            setattr(edges, k, getattr(edges, k) + getattr(sm.edges, k))
        edges.safety_factor = sm.edges.safety_factor
        facets.extend(sm.facets)
        for p, a in sm.areas_per_pitch.items():
            per_pitch[p] = per_pitch.get(p, 0.0) + a
        for n in sm.notes:
            if n not in notes:
                notes.append(n)
    facets = sorted(facets, key=lambda f: f.surface_area_sqft)
    for rank, f in enumerate(facets):
        f.label = geometry._facet_label(rank)
    total = round(sum(f.surface_area_sqft for f in facets))
    combined = RoofMeasurements(
        address=main.address, latitude=main.latitude, longitude=main.longitude,
        total_area_sqft=total, facets=facets, edges=edges,
        predominant_pitch=max(per_pitch.items(), key=lambda kv: kv[1])[0] if per_pitch else main.predominant_pitch,
        areas_per_pitch={k: round(v, 1) for k, v in sorted(per_pitch.items(), key=lambda kv: int(kv[0].split("/")[0]))},
        imagery_date=main.imagery_date, imagery_quality=main.imagery_quality, method=main.method,
        notes=notes, structure_index=0, structures=list(structures),
    )
    combined.confidence = main.confidence
    combined.canopy_fraction = main.canopy_fraction
    combined.google_roof_area_sqft = main.google_roof_area_sqft
    combined.notes.insert(0, f"{len(structures)} structures measured (primary plus "
                             f"{len(structures) - 1} detached); totals are for all structures.")
    return combined


def _building_center_px(shape, lat, lng, px_x, px_y, insights):
    """Pixel (y, x) of Google's building center; raster center if unknown.

    dataLayers rasters are centered on the requested location, so the
    buildingInsights center maps to an offset from the raster center."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    try:
        c = insights["center"]
        dy = -(float(c["latitude"]) - lat) * 111_320.0 / px_y
        dx = (float(c["longitude"]) - lng) * 111_320.0 * float(np.cos(np.radians(lat))) / px_x
        if abs(dy) < h / 2 and abs(dx) < w / 2:
            return (cy + dy, cx + dx)
    except (KeyError, TypeError, ValueError):
        pass
    return (cy, cx)


FRAMING_FIT_P90_M = 0.60     # framing model accepted when the DSM agrees this closely (90th pct)
FRAMING_FIT_RMS_M = 0.45
FRAMING_FIT_BIAS_M = 0.30


def _attach_skeleton(m: RoofMeasurements, detail, px_m: float) -> None:
    """Build the plane-graph wireframe and adopt its linear totals and faces.

    Creases are analytic intersections of the fitted facet planes, so ridges,
    hips and valleys are straight, meet at nodes, and carry true 3D lengths.
    Tree-occluded outline corners are squared off and the facet raster is
    extended under them so areas include the hidden corner. A sanity gate
    keeps the raster-derived numbers if the wireframe disagrees badly with
    the raster evidence.
    """
    from . import wireframe as wfm
    from .model import Facet
    from .vectorize import trace_boundary

    try:
        chain = trace_boundary(detail.labels > 0)
        axis = wfm.dominant_axis_from_planes(detail.labels, detail.planes)
        outline = wfm.regularize_outline(chain, px_m, axis_hint=axis)
        if len(outline) < 3:
            m.notes.append("Roof wireframe unavailable (outline degenerate); edge lengths are raster-derived.")
            return
        restored = 0
        dsm = getattr(detail, "dsm", None)
        ground = getattr(detail, "ground", None)
        if dsm is not None and ground is not None:
            outline, restored = wfm.complete_occluded_corners(outline, dsm, ground, px_m)
            if restored:
                detail.labels = wfm.extend_labels_to_outline(detail.labels, outline)
        wf = wfm.build_wireframe(detail.labels, detail.planes, outline, detail.px_x, detail.px_y)
        # ---- framing model first: the graph derived from the eave line the way
        # the roof is framed, kept when it fits the elevation model ----
        framing_note = None
        try:
            from . import roofgraph as rgm
            foot = (detail.labels > 0)
            if getattr(detail, "roof_mask", None) is not None:
                foot = foot | detail.roof_mask        # the footprint, including facets segmentation dropped
            g = rgm.generate_roof(detail.dsm.astype(float), foot, detail.px_x, detail.px_y,
                                  axis_hint=axis, labels=detail.labels, planes=detail.planes,
                                  ground=getattr(detail, "ground", None))
            fit = g.info if g.ok else {}
            p90 = fit.get("p90_m", 9.9); rms = fit.get("fit_rms_m", 9.9); bias = abs(fit.get("bias_m", 9.9))
            mask_px = float((detail.labels > 0).sum())
            area_ratio = g.totals.get("plan_area_sqft", 0) / max(mask_px * detail.px_x * detail.px_y * 10.76391, 1.0)
            if g.ok and p90 <= FRAMING_FIT_P90_M and rms <= FRAMING_FIT_RMS_M and bias <= FRAMING_FIT_BIAS_M \
                    and 0.85 <= area_ratio <= 1.15 and g.totals.get("eaves_ft"):
                g.outline = outline
                g.polys_ready = True
                wf = g
                framing_note = (
                    "Roof model: framing model. Ridges, hips and valleys are generated from the eave "
                    "line the way the roof is framed (hips from outside corners, valleys from inside "
                    f"corners to the eaves, one pitch of {fit.get('rise')}/12, gable ends where the "
                    "elevation data shows them); the elevation model confirms it within "
                    f"{p90 * 3.28084 * 12:.0f} in (90th percentile)."
                )
            else:
                m.notes.append(
                    "Framing model did not fit this roof (90th percentile residual "
                    f"{p90 * 3.28084 * 12:.0f} in, rms {rms * 3.28084 * 12:.0f} in); lines come from the "
                    "measured plane intersections instead. Expect more line segments than a framed plan."
                )
        except Exception as exc:  # the framing model must never break a report
            m.notes.append(f"Framing model unavailable ({type(exc).__name__}: {exc}); lines come from measured plane intersections.")
        if not wf.ok or not wf.totals.get("eaves_ft"):
            m.notes.append("Roof wireframe unavailable; edge lengths are raster-derived.")
            return
        e = m.edges
        t = wf.totals
        oy, ox = outline[:, 0], outline[:, 1]
        poly_area_px = abs(float(np.sum(ox * np.roll(oy, -1) - np.roll(ox, -1) * oy))) / 2.0
        mask_px = float((detail.labels > 0).sum())
        raster_creases = e.ridges_ft + e.hips_ft + e.valleys_ft
        wf_creases = t["ridges_ft"] + t["hips_ft"] + t["valleys_ft"]
        reasons = []
        if mask_px > 0 and not (0.85 <= poly_area_px / mask_px <= 1.15):
            reasons.append(f"outline area {poly_area_px / mask_px:.2f}x raster footprint")
        if raster_creases > 40 and wf_creases < 0.4 * raster_creases:
            reasons.append(f"creases {wf_creases:.0f} ft vs raster {raster_creases:.0f} ft")
        if reasons:
            m.notes.append("Roof wireframe rejected (" + "; ".join(reasons) +
                           "); linear measurements are raster-derived.")
            return
        detail.skeleton = wf
        e.ridges_ft, e.hips_ft = t["ridges_ft"], t["hips_ft"]
        e.valleys_ft, e.rakes_ft, e.eaves_ft = t["valleys_ft"], t["rakes_ft"], t["eaves_ft"]
        e.flashing_ft = t.get("flashing_ft", 0.0)
        e.step_flashing_ft = t.get("step_flashing_ft", 0.0)
        e.ridge_count, e.hip_count = t["ridge_count"], t["hip_count"]
        e.valley_count, e.rake_count, e.eave_count = t["valley_count"], t["rake_count"], t["eave_count"]
        # faces: rebuild the facet list from the wireframe (areas include any
        # restored corners); letters run smallest -> largest like EagleView
        order = sorted(wf.face_area_sqft, key=lambda i: wf.face_area_sqft[i])
        facets = []
        areas_per_pitch: dict[str, float] = {}
        detail.facet_centroids, detail.facet_ids, detail.facet_azimuths = {}, {}, {}
        for rank, fid in enumerate(order):
            letter = geometry._facet_label(rank)
            pitch = wf.face_pitch[fid]
            facets.append(Facet(label=letter, plan_area_sqft=round(wf.face_plan_sqft[fid], 1),
                                surface_area_sqft=round(wf.face_area_sqft[fid], 1), pitch=pitch,
                                slope_deg=round(wf.face_slope_deg[fid], 1),
                                azimuth_deg=round(wf.face_azimuth[fid], 1)))
            areas_per_pitch[pitch] = areas_per_pitch.get(pitch, 0.0) + wf.face_area_sqft[fid]
            detail.facet_centroids[letter] = wf.face_centroids[fid]
            detail.facet_ids[letter] = fid
            detail.facet_azimuths[letter] = wf.face_azimuth[fid]
        m.facets = facets
        m.total_area_sqft = round(sum(f.surface_area_sqft for f in facets))
        m.areas_per_pitch = {k: round(v, 1) for k, v in sorted(areas_per_pitch.items(), key=lambda kv: int(kv[0].split("/")[0]))}
        if areas_per_pitch:
            m.predominant_pitch = max(areas_per_pitch.items(), key=lambda kv: kv[1])[0]
        if restored:
            m.notes.append(f"{restored} roof corner(s) hidden by tree canopy were squared off from the "
                           "adjacent roof edges; verify those corners in the field.")
        if framing_note:
            m.notes.append(framing_note)
        else:
            m.notes.append(
                "Linear measurements (ridges/hips/valleys/rakes/eaves) come from a roof "
                "wireframe built from the measured facet planes: each crease is the "
                "intersection of two fitted roof planes with its true sloped length."
            )
    except Exception as exc:
        m.notes.append(f"Roof wireframe unavailable ({type(exc).__name__}: {exc}); "
                       "edge lengths are raster-derived.")


def local_plane_residual(dsm: np.ndarray, px_x: float, px_y: float, k: int = 4) -> np.ndarray:
    """RMS residual of a least-squares plane fitted in a (2k+1)^2 window around
    every pixel. Roof surfaces are planar to a few centimetres; tree canopy,
    cars, and raster edges are not. Closed form via window moments."""
    h, w = dsm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx * px_x
    Y = yy * px_y
    Z = dsm.astype(np.float64)
    n = float((2 * k + 1) ** 2)
    S = lambda a: box_sum(a, k)  # noqa: E731
    sx, sy, sz = S(X), S(Y), S(Z)
    sxx, syy, sxy = S(X * X), S(Y * Y), S(X * Y)
    sxz, syz, szz = S(X * Z), S(Y * Z), S(Z * Z)
    mx, my, mz = sx / n, sy / n, sz / n
    cxx = sxx / n - mx * mx
    cyy = syy / n - my * my
    cxy = sxy / n - mx * my
    cxz = sxz / n - mx * mz
    cyz = syz / n - my * mz
    czz = szz / n - mz * mz
    det = cxx * cyy - cxy * cxy
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (cxz * cyy - cyz * cxy) / det
        b = (cyz * cxx - cxz * cxy) / det
        var = czz - (a * cxz + b * cyz)
    var = np.where(np.isfinite(var), np.maximum(var, 0.0), czz)
    return np.sqrt(var)


def _open(m: np.ndarray, r: int) -> np.ndarray:
    """Binary opening with a (2r+1) square: erode then dilate."""
    e = m.copy()
    for _ in range(r):
        s = e.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s &= np.roll(np.roll(e, dy, 0), dx, 1)
        e = s
    d = e
    for _ in range(r):
        s = d.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s |= np.roll(np.roll(d, dy, 0), dx, 1)
        d = s
    return d


def _label_components(allowed: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels via run-length union-find (numpy, fast)."""
    h, w = allowed.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]
    # label runs per row
    run_id = 0
    row_runs: list[list[tuple[int, int, int]]] = []
    for y in range(h):
        row = allowed[y]
        runs = []
        if row.any():
            d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            for s_, e_ in zip(starts, ends):
                run_id += 1
                parent.append(run_id)
                labels[y, s_:e_] = run_id
                runs.append((int(s_), int(e_), run_id))
        row_runs.append(runs)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(1, h):
        up, cur = row_runs[y - 1], row_runs[y]
        i = j = 0
        while i < len(up) and j < len(cur):
            s0, e0, r0 = up[i]
            s1, e1, r1 = cur[j]
            if s0 < e1 and s1 < e0:
                ra, rb = find(r0), find(r1)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            if e0 < e1:
                i += 1
            else:
                j += 1
    lut = np.zeros(len(parent), dtype=np.int32)
    roots = {}
    for k in range(1, len(parent)):
        r = find(k)
        if r not in roots:
            roots[r] = len(roots) + 1
        lut[k] = roots[r]
    return lut[labels], len(roots)


def _flood(seed_mask: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Union of connected components of `allowed` that contain a seed pixel."""
    comp, n = _label_components(allowed)
    hit = np.unique(comp[seed_mask & allowed])
    hit = hit[hit > 0]
    if len(hit) == 0:
        return np.zeros_like(allowed, dtype=bool)
    return np.isin(comp, hit)


def _fill_holes(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg = _flood(border, ~lab)
    return lab | ~bg


def ground_model(dsm: np.ndarray, px_x: float, px_y: float) -> np.ndarray:
    """Ground surface under the DSM.

    Blockwise low percentile (20 m blocks) followed by a 3x3 block minimum,
    so a block that happens to be mostly roof still inherits ground level
    from its neighbors, then bilinear upsampling. The result is at or below
    true ground almost everywhere and independent of where the block grid
    falls relative to the building (which a plain block percentile is not).
    """
    h, w = dsm.shape
    B = max(int(20.0 / max(px_x, px_y)), 20)
    gh, gw = (h + B - 1) // B, (w + B - 1) // B
    gblk = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            gblk[i, j] = np.percentile(dsm[i * B:(i + 1) * B, j * B:(j + 1) * B], 5)
    # A block that is mostly roof reads far above its neighbors: replace such
    # outliers by the neighborhood median (terrain slope is preserved because
    # the median follows it; a true 2 m terrain jump over 20 m is rare).
    pad = np.pad(gblk, 1, mode="edge")
    neigh = np.stack([pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    med = np.median(neigh, axis=0)
    gmin = np.where(gblk > med + 1.5, med, gblk)
    # bilinear upsample from block centers
    cy = (np.arange(gh) + 0.5) * B
    cx = (np.arange(gw) + 0.5) * B
    yy = np.arange(h); xx = np.arange(w)
    iy = np.clip(np.searchsorted(cy, yy) - 1, 0, max(gh - 2, 0))
    ix = np.clip(np.searchsorted(cx, xx) - 1, 0, max(gw - 2, 0))
    if gh > 1:
        ty = np.clip((yy - cy[iy]) / (cy[iy + 1] - cy[iy]), 0.0, 1.0)
    else:
        ty = np.zeros(h)
    if gw > 1:
        tx = np.clip((xx - cx[ix]) / (cx[ix + 1] - cx[ix]), 0.0, 1.0)
    else:
        tx = np.zeros(w)
    iy1 = np.minimum(iy + 1, gh - 1); ix1 = np.minimum(ix + 1, gw - 1)
    g00 = gmin[np.ix_(iy, ix)]; g01 = gmin[np.ix_(iy, ix1)]
    g10 = gmin[np.ix_(iy1, ix)]; g11 = gmin[np.ix_(iy1, ix1)]
    TY = ty[:, None]; TX = tx[None, :]
    return (g00 * (1 - TY) * (1 - TX) + g01 * (1 - TY) * TX + g10 * TY * (1 - TX) + g11 * TY * TX)


def recover_footprint(dsm: np.ndarray, mask: np.ndarray, px_x: float, px_y: float,
                      seed_yx: tuple | None = None, planar_rms_m: float = 0.06,
                      min_height_m: float = 2.0, return_candidates: bool = False):
    """Recover the full roof footprint of the target building.

    Google's solar mask excludes tree-shaded / unsuitable roof area (and is
    sometimes missing entirely), which under-measures a roof; growing naively
    over "elevated" pixels leaks into adjacent tree canopy, which
    over-measures it. Rule used here:

      roof candidate = above local ground AND locally planar (plane-fit RMS
                       within a ~0.9 m window below `planar_rms_m`)

    The candidate raster is opened (removes sub-metre bridges to canopy),
    the component containing the building seed is taken, Google's mask
    pixels for that building are unioned back in, and enclosed holes
    (chimneys, vents) are filled.

    Returns (roof_mask, percent_of_final_area_beyond_google_mask).
    """
    h, w = dsm.shape
    ndsm = dsm - ground_model(dsm, px_x, px_y)
    resid = local_plane_residual(dsm, px_x, px_y, k=4)
    elevated = ndsm > min_height_m
    planar = resid < planar_rms_m
    cand = _open(elevated & planar, 3)  # ~0.3 m opening at 0.1 m/px

    # ---- seed: Google building center, else nearest mask pixel to center ----
    if seed_yx is None:
        seed_yx = (h / 2.0, w / 2.0)
    sy, sx = int(round(seed_yx[0])), int(round(seed_yx[1]))
    seed = np.zeros((h, w), dtype=bool)
    r = int(3.0 / px_x)  # 3 m search radius around the seed
    y0, y1 = max(sy - r, 0), min(sy + r + 1, h)
    x0, x1 = max(sx - r, 0), min(sx + r + 1, w)
    seed[y0:y1, x0:x1] = True
    seed_px = seed & (cand | (mask & elevated))
    if not seed_px.any():
        # fall back: nearest elevated/planar pixel to the seed anywhere
        ys, xs = np.where(cand)
        if len(ys) == 0:
            return (mask.astype(bool), 0.0, cand) if return_candidates else (mask.astype(bool), 0.0)
        i = int(np.argmin((ys - sy) ** 2 + (xs - sx) ** 2))
        seed_px = np.zeros((h, w), dtype=bool)
        seed_px[ys[i], xs[i]] = True

    lab = _flood(seed_px, cand)
    # union the Google mask component(s) touching the grown roof: the mask is
    # conservative on area but reliable on "this is roof", and it covers
    # planar pixels the opening may have nibbled at the eaves.
    mask_touch = _flood(lab, mask.astype(bool) & elevated)
    lab |= mask_touch
    # re-grow once through planar pixels adjacent to the union (eave edges)
    lab = _flood(lab, lab | (elevated & planar))
    lab = _fill_holes(lab)

    total = int(lab.sum())
    in_mask = int((lab & mask.astype(bool)).sum())
    grown_pct = 100.0 * (total - in_mask) / max(total, 1)
    if return_candidates:
        return lab, grown_pct, cand
    return lab, grown_pct
