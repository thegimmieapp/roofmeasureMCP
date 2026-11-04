"""Fetch raw Solar API GeoTIFFs for the lot calibration set (radius 100 m).

Saves, per address key: dsm.tif, mask.tif, rgb.tif, flux.tif, shade06.tif,
shade12.tif and meta.json (geocode result with address components,
buildingInsights, imagery date). Raw TIFFs keep the georeference, so the
lot pipeline can map parcel polygons into pixels offline.

Usage: GOOGLE_MAPS_API_KEY=... python tests/fetch_lot_cache.py <cache_dir> [keys...]
"""
import json
import os
import sys

import requests

ADDRESSES = {
    "andy": "2046 West Nottingham St, Springfield, MO 65810",
    "joe": "4934 South Franwood Place, Springfield, MO 65810",
    "tony": "3638 Breech Avenue, Springfield, MO 65807",
    "lynda": "2022 North Benton Avenue, Springfield, MO 65803",
    "fred": "4834 East Kingwood Court, Springfield, MO 65809",
    "jenny": "3287 West Farm Road 94, Springfield, MO 65803",
    "garland": "1747 E Garland Ct, Republic, MO 65738",
    "maplewood": "3669 W Maplewood St, Springfield, MO 65807",
}


def main():
    key = os.environ["GOOGLE_MAPS_API_KEY"].strip()
    out = sys.argv[1]
    keys = sys.argv[2:] or list(ADDRESSES)
    for k in keys:
        d = os.path.join(out, k)
        if os.path.exists(os.path.join(d, "meta.json")):
            print("cached", k)
            continue
        os.makedirs(d, exist_ok=True)
        g = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                         params={"address": ADDRESSES[k], "key": key}, timeout=30).json()
        res = g["results"][0]
        lat, lng = res["geometry"]["location"]["lat"], res["geometry"]["location"]["lng"]
        ins = requests.get("https://solar.googleapis.com/v1/buildingInsights:findClosest",
                           params={"location.latitude": f"{lat:.7f}", "location.longitude": f"{lng:.7f}",
                                   "requiredQuality": "LOW", "key": key}, timeout=60).json()
        lay = requests.get("https://solar.googleapis.com/v1/dataLayers:get",
                           params={"location.latitude": f"{lat:.7f}", "location.longitude": f"{lng:.7f}",
                                   "radiusMeters": "100", "view": "FULL_LAYERS",
                                   "requiredQuality": "LOW", "key": key}, timeout=60).json()
        if "error" in lay:
            print(k, "dataLayers error", lay["error"].get("message"))
            continue
        urls = {"dsm": lay.get("dsmUrl"), "mask": lay.get("maskUrl"), "rgb": lay.get("rgbUrl"),
                "flux": lay.get("annualFluxUrl")}
        sh = lay.get("hourlyShadeUrls") or []
        if len(sh) == 12:
            urls["shade06"], urls["shade12"] = sh[5], sh[11]
        for name, u in urls.items():
            if not u:
                continue
            r = requests.get(u, params={"key": key}, timeout=180)
            r.raise_for_status()
            with open(os.path.join(d, f"{name}.tif"), "wb") as f:
                f.write(r.content)
        meta = {"address": res.get("formatted_address"), "lat": lat, "lng": lng,
                "address_components": res.get("address_components"),
                "location_type": res["geometry"].get("location_type"),
                "imageryDate": lay.get("imageryDate"), "imageryQuality": lay.get("imageryQuality"),
                "insights": ins}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(k, meta["address"], lay.get("imageryQuality"), lay.get("imageryDate"),
              {n: os.path.getsize(os.path.join(d, f"{n}.tif")) for n in urls if urls[n]})


if __name__ == "__main__":
    main()
