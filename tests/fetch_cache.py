"""Fetch Solar API rasters once and cache them as .npz for offline calibration.

Usage: GOOGLE_MAPS_API_KEY=... python tests/fetch_cache.py <cache_dir>
"""
import json
import os
import sys

import numpy as np

from roofmeasure import solar

ADDRESSES = {
    "andy": "2046 West Nottingham St, Springfield, MO 65810",
    "joe": "4934 South Franwood Place, Springfield, MO 65810",
    "tony": "3638 Breech Avenue, Springfield, MO 65807",
    "lynda": "2022 North Benton Avenue, Springfield, MO 65803",
    "fred": "4834 East Kingwood Court, Springfield, MO 65809",
    "jenny": "3287 West Farm Road 94, Springfield, MO 65803",
}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
for key, addr in ADDRESSES.items():
    path = os.path.join(out, f"{key}.npz")
    if os.path.exists(path):
        print("cached", key)
        continue
    lat, lng, formatted = solar.geocode(addr)
    insights = solar.building_insights(lat, lng, quality="LOW")
    layers = solar.data_layers(lat, lng, radius_m=radius, quality="LOW")
    dsm, px_x, px_y = solar.fetch_geotiff(layers["dsmUrl"])
    mask, _, _ = solar.fetch_geotiff(layers["maskUrl"])
    rgb = solar.fetch_geotiff_rgb(layers["rgbUrl"]) if layers.get("rgbUrl") else None
    np.savez_compressed(path, dsm=dsm, mask=mask, rgb=rgb if rgb is not None else np.zeros((1, 1, 3), np.uint8),
                        px_x=px_x, px_y=px_y, lat=lat, lng=lng)
    meta = {"address": formatted, "lat": lat, "lng": lng, "px_x": px_x, "px_y": px_y,
            "imageryDate": layers.get("imageryDate"), "imageryQuality": layers.get("imageryQuality"),
            "dsm_shape": list(dsm.shape), "insights": insights}
    with open(os.path.join(out, f"{key}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(key, formatted, dsm.shape, px_x, px_y, layers.get("imageryQuality"), layers.get("imageryDate"))
