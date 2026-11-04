"""Google Geocoding + Solar API client.

Requires env var GOOGLE_MAPS_API_KEY with Geocoding API and Solar API enabled.
Free tier: $200/month credit covers roughly 1,000 buildingInsights+dataLayers pulls.
"""

from __future__ import annotations

import io
import os

import numpy as np
import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
INSIGHTS_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
LAYERS_URL = "https://solar.googleapis.com/v1/dataLayers:get"

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671


class SolarAPIError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise SolarAPIError(
            "GOOGLE_MAPS_API_KEY environment variable is not set. "
            "Create a free Google Cloud API key with the Geocoding API and Solar API "
            "enabled, then set GOOGLE_MAPS_API_KEY."
        )
    return key


def geocode(address: str) -> tuple[float, float, str]:
    """Return (lat, lng, formatted_address)."""
    r = requests.get(GEOCODE_URL, params={"address": address, "key": _key()}, timeout=30)
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise SolarAPIError(f"Could not geocode address '{address}': {data.get('status')} {data.get('error_message', '')}")
    res = data["results"][0]
    loc = res["geometry"]["location"]
    return loc["lat"], loc["lng"], res.get("formatted_address", address)


def building_insights(lat: float, lng: float, quality: str = "LOW") -> dict:
    """Fetch buildingInsights (roof segment stats, pitch, azimuth, areas)."""
    r = requests.get(
        INSIGHTS_URL,
        params={
            "location.latitude": f"{lat:.7f}",
            "location.longitude": f"{lng:.7f}",
            "requiredQuality": quality,
            "key": _key(),
        },
        timeout=60,
    )
    data = r.json()
    if "error" in data:
        raise SolarAPIError(f"Solar API buildingInsights error: {data['error'].get('message')}")
    return data


def data_layers(lat: float, lng: float, radius_m: float = 30.0, quality: str = "LOW") -> dict:
    """Fetch dataLayers metadata (URLs for DSM and mask GeoTIFFs)."""
    r = requests.get(
        LAYERS_URL,
        params={
            "location.latitude": f"{lat:.7f}",
            "location.longitude": f"{lng:.7f}",
            "radiusMeters": str(radius_m),
            "view": "FULL_LAYERS",
            "requiredQuality": quality,
            "key": _key(),
        },
        timeout=60,
    )
    data = r.json()
    if "error" in data:
        raise SolarAPIError(f"Solar API dataLayers error: {data['error'].get('message')}")
    return data


def fetch_geotiff(url: str) -> tuple[np.ndarray, float, float]:
    """Download a Solar API GeoTIFF. Returns (array, px_size_x_m, px_size_y_m).

    Solar API rasters are in EPSG:4326; pixel scale tags are in degrees, so we
    convert to meters at the tile's latitude.
    """
    import tifffile

    r = requests.get(url, params={"key": _key()}, timeout=120)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    with tifffile.TiffFile(buf) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        tags = page.tags
        sx = sy = None
        lat0 = None
        xform = tags.get(34264)   # ModelTransformationTag (projected, meters)
        scale = tags.get(33550)   # ModelPixelScaleTag
        tie = tags.get(33922)     # ModelTiepointTag
        if xform is not None:
            t = xform.value
            sx, sy = abs(float(t[0])), abs(float(t[5]))
            if abs(float(t[3])) <= 360.0:  # geographic transform (degrees)
                lat0 = float(t[7])
        elif scale is not None:
            sx, sy = abs(float(scale.value[0])), abs(float(scale.value[1]))
            if sx < 0.01:  # degrees, not meters
                lat0 = float(tie.value[4]) if tie is not None else 0.0
        if sx is None:
            raise SolarAPIError("GeoTIFF missing georeferencing tags")
        if lat0 is not None:  # convert degrees -> meters
            sy *= 111_320.0
            sx *= 111_320.0 * float(np.cos(np.radians(lat0)))
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr[0]
    return arr.astype(np.float32), sx, sy


def fetch_geotiff_rgb(url: str) -> np.ndarray | None:
    """Download a Solar API RGB GeoTIFF as an HxWx3 uint8 array (or None)."""
    import tifffile

    try:
        r = requests.get(url, params={"key": _key()}, timeout=120)
        r.raise_for_status()
        with tifffile.TiffFile(io.BytesIO(r.content)) as tf:
            arr = tf.pages[0].asarray()
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3:
            return None
        return arr[..., :3].astype(np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------- v0.6.0: georeferenced layers (LotMeasure)


def geocode_full(address: str) -> dict:
    """Geocode with the parts LotMeasure needs: lat, lng, formatted address,
    county, state, route (street name) and location_type (ROOFTOP etc.)."""
    r = requests.get(GEOCODE_URL, params={"address": address, "key": _key()}, timeout=30)
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise SolarAPIError(f"Could not geocode address '{address}': {data.get('status')} {data.get('error_message', '')}")
    return parse_geocode_result(data["results"][0], address)


def parse_geocode_result(res: dict, address: str = "") -> dict:
    comps = {c["types"][0]: c for c in res.get("address_components", []) if c.get("types")}
    loc = res["geometry"]["location"]

    def name(t, short=False):
        c = comps.get(t)
        return (c["short_name"] if short else c["long_name"]) if c else ""

    return {
        "lat": loc["lat"], "lng": loc["lng"],
        "formatted": res.get("formatted_address", address),
        "county": name("administrative_area_level_2"),
        "state": name("administrative_area_level_1", short=True),
        "route": name("route"),
        "locality": name("locality"),
        "location_type": res["geometry"].get("location_type", ""),
    }


def _download(url: str) -> bytes:
    r = requests.get(url, params={"key": _key()}, timeout=180)
    r.raise_for_status()
    return r.content


def read_geotiff_geo(content: bytes, bands: bool = False):
    """Parse GeoTIFF bytes -> (array, px_x_m, px_y_m, TileGeo).

    Unlike fetch_geotiff this keeps the tile origin, so a lat/lng polygon can
    be mapped into pixels and back (geo.TileGeo). With bands=True a
    multi-band raster (hourly shade: 24 bands) is returned as H x W x B in its
    native dtype; otherwise band 0 as float32 (RGB: H x W x 3 uint8 via
    read_geotiff_rgb_geo)."""
    import logging
    import tifffile

    from .geo import geo_from_tiff_tags

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    with tifffile.TiffFile(io.BytesIO(content)) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        geo = geo_from_tiff_tags(page.tags, page.shape)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 32 and arr.shape[1] == geo.shape[1]:
        arr = np.moveaxis(arr, 0, -1)            # band-first -> band-last
    h, w = arr.shape[:2]
    geo.shape = (h, w)
    lat_c, _ = geo.px_to_latlng(h / 2.0, w / 2.0)
    px_x, px_y = geo.px_size_m(float(lat_c))
    if bands:
        return arr, px_x, px_y, geo
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32), px_x, px_y, geo


def read_geotiff_rgb_geo(content: bytes):
    arr, px_x, px_y, geo = read_geotiff_geo(content, bands=True)
    if arr.ndim != 3:
        return None, geo
    return arr[..., :3].astype(np.uint8), geo


def fetch_geotiff_geo(url: str, bands: bool = False):
    """Download a Solar API GeoTIFF keeping its georeference:
    (array, px_x_m, px_y_m, TileGeo)."""
    return read_geotiff_geo(_download(url), bands=bands)
