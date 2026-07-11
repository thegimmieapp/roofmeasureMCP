"""End-to-end measurement pipeline: address -> RoofMeasurements."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import geometry, solar
from .model import RoofMeasurements


def measure_address(address: str, quality: str = "LOW", radius_m: float = 40.0) -> RoofMeasurements:
    """Geocode an address and produce full roof measurements."""
    m, _detail, _rgb = measure_address_full(address, quality=quality, radius_m=radius_m, want_rgb=False)
    return m


def measure_address_full(address: str, quality: str = "LOW", radius_m: float = 40.0,
                         want_rgb: bool = True):
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
        roof, grown_pct = recover_footprint(dsm, mask > 0, px_x, px_y)
        m, detail = geometry.measure_from_dsm(dsm, roof, px_x, px_y, formatted, lat, lng,
                                              return_detail=True)
        m.imagery_date = geometry._fmt_date(layers.get("imageryDate"))
        m.imagery_quality = layers.get("imageryQuality", insights.get("imageryQuality", ""))
        if grown_pct > 10:
            m.notes.append(
                f"About {grown_pct:.0f}% of the roof footprint was recovered from the "
                "elevation model beyond Google's solar mask (tree-shaded or obstructed "
                "areas). Edge lengths in those areas are approximate; field verify "
                "before material order."
            )
        _attach_skeleton(m, detail, px_x)
        rgb = solar.fetch_geotiff_rgb(layers["rgbUrl"]) if (want_rgb and layers.get("rgbUrl")) else None
        return m, detail, rgb
    except Exception as exc:  # graceful fallback
        m = geometry.estimate_from_segments(insights, formatted, lat, lng)
        m.notes.append(f"DSM pipeline unavailable ({type(exc).__name__}: {exc}); used segment-statistics fallback.")
        return m, None, None