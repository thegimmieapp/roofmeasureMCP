"""End-to-end measurement pipeline: address -> RoofMeasurements."""

from __future__ import annotations

import numpy as np

from . import geometry, solar
from .model import RoofMeasurements


def measure_address(address: str, quality: str = "LOW", radius_m: float = 30.0) -> RoofMeasurements:
    """Geocode an address and produce full roof measurements.

    Tries the high-resolution DSM pipeline first (measured edges); falls back
    to buildingInsights segment statistics (estimated edges) if data layers
    are unavailable for the location.
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
        mask_b = mask > 0
        # Keep only the building containing/nearest the geocoded point: the
        # mask can include neighboring structures within the radius.
        mask_b = _select_center_building(mask_b)
        m = geometry.measure_from_dsm(dsm, mask_b, px_x, px_y, formatted, lat, lng)
        m.imagery_date = geometry._fmt_date(layers.get("imageryDate"))
        m.imagery_quality = layers.get("imageryQuality", insights.get("imageryQuality", ""))
        return m
    except Exception as exc:  # graceful fallback
        m = geometry.estimate_from_segments(insights, formatted, lat, lng)
        m.notes.append(f"DSM pipeline unavailable ({type(exc).__name__}: {exc}); used segment-statistics fallback.")
        return m


def _select_center_building(mask: np.ndarray) -> np.ndarray:
    """Keep only the connected mask component nearest the raster center."""
    from collections import deque

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    comps = {}
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            q = deque([(sy, sx)])
            labels[sy, sx] = next_label
            pts = []
            while q:
                y, x = q.popleft()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = next_label
                        q.append((ny, nx))
            comps[next_label] = pts
            next_label += 1
    if not comps:
        return mask
    cy, cx = h / 2.0, w / 2.0

    def score(pts: list) -> float:
        arr = np.array(pts)
        d = np.hypot(arr[:, 0] - cy, arr[:, 1] - cx).min()
        return d - 0.001 * len(pts)  # prefer near-center, break ties by size

    best = min(comps, key=lambda k: score(comps[k]))
    return labels == best
