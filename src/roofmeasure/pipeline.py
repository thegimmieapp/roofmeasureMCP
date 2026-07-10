"""End-to-end measurement pipeline: address -> RoofMeasurements."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import geometry, solar
from .model import RoofMeasurements


def measure_address(address: str, quality: str = "LOW", radius_m: float = 40.0) -> RoofMeasurements:
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
        roof, grown_pct = recover_footprint(dsm, mask > 0, px_x, px_y)
        m = geometry.measure_from_dsm(dsm, roof, px_x, px_y, formatted, lat, lng)
        m.imagery_date = geometry._fmt_date(layers.get("imageryDate"))
        m.imagery_quality = layers.get("imageryQuality", insights.get("imageryQuality", ""))
        if grown_pct > 10:
            m.notes.append(
                f"About {grown_pct:.0f}% of the roof footprint was recovered from the "
                "elevation model beyond Google's solar mask (tree-shaded or obstructed "
                "areas). Edge lengths in those areas are approximate; field verify "
                "before material order."
            )
        return m
    except Exception as exc:  # graceful fallback
        m = geometry.estimate_from_segments(insights, formatted, lat, lng)
        m.notes.append(f"DSM pipeline unavailable ({type(exc).__name__}: {exc}); used segment-statistics fallback.")
        return m


def recover_footprint(dsm: np.ndarray, mask: np.ndarray, px_x: float, px_y: float
                      ) -> tuple[np.ndarray, float]:
    """Recover the full roof footprint of the center building.

    Google's solar mask excludes tree-shaded / unsuitable roof area, which can
    badly under-measure a roof. Starting from the mask component nearest the
    raster center, grow over pixels that are (a) well above local ground level
    and (b) locally planar (roofs are smooth; tree canopies are rough), then
    fill enclosed holes.

    Returns (roof_mask, percent_of_final_area_recovered_beyond_google_mask).
    """
    h, w = dsm.shape

    # ---- local ground model: blockwise low percentile, lightly smoothed ----
    B = max(int(12.0 / max(px_x, px_y)), 20)  # ~12 m blocks
    gh, gw = (h + B - 1) // B, (w + B - 1) // B
    gblk = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            gblk[i, j] = np.percentile(dsm[i * B:(i + 1) * B, j * B:(j + 1) * B], 8)
    ground = np.kron(gblk, np.ones((B, B)))[:h, :w]
    for _ in range(2):
        g2 = ground.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g2 += np.roll(np.roll(ground, dy, 0), dx, 1)
        ground = g2 / 5
    ndsm = dsm - ground

    # ---- roughness: local gradient variance (trees rough, roofs smooth) ----
    gy, gx = np.gradient(dsm, px_y, px_x)

    def box(a: np.ndarray, k: int = 2) -> np.ndarray:
        out = np.zeros_like(a)
        n = 0
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                out += np.roll(np.roll(a, dy, 0), dx, 1)
                n += 1
        return out / n

    gxm, gym = box(gx), box(gy)
    rough = np.sqrt(box((gx - gxm) ** 2 + (gy - gym) ** 2))
    grow_ok = (ndsm > 2.0) & (rough < 0.45)

    # ---- grow from the mask component nearest the raster center ----
    if not mask.any():
        return mask, 0.0
    ys, xs = np.where(mask)
    ci = int(np.argmin((ys - h / 2) ** 2 + (xs - w / 2) ** 2))
    lab = np.zeros((h, w), dtype=bool)
    q = deque([(int(ys[ci]), int(xs[ci]))])
    lab[ys[ci], xs[ci]] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not lab[ny, nx] and (mask[ny, nx] or grow_ok[ny, nx]):
                lab[ny, nx] = True
                q.append((ny, nx))

    # ---- fill enclosed holes (chimneys, skylight dropouts, canopy gaps) ----
    bg = np.zeros((h, w), dtype=bool)
    q = deque()
    for y in range(h):
        for x in (0, w - 1):
            if not lab[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if not lab[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not lab[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    lab |= ~lab & ~bg

    seed_and_grown = int(lab.sum())
    seed_only = int((lab & mask).sum())
    grown_pct = 100.0 * (seed_and_grown - seed_only) / max(seed_and_grown, 1)
    return lab, grown_pct
