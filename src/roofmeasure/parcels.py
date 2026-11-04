"""Parcel (lot line) lookup for LotMeasure.

The Solar API has no lot lines. Provider chain, first hit wins:

1. County / state ArcGIS parcel layers (free). Seeded in parcels_registry.json
   (verified layers for the counties Austin works in); other counties are
   discovered through the ArcGIS Online search API and cached in
   <ROOFMEASURE_OUT_DIR>/parcels_registry_discovered.json, labelled
   "discovered, unverified".
2. Regrid parcel API when REGRID_API_KEY is set (paid, nationwide).
3. None: lot_source "none". A lot polygon is never invented.

OpenStreetMap (Overpass) is used only for the road frontage (lot.py).

Privacy: owner, mailing and valuation attributes are never returned; only the
parcel id and the recorded lot-size fields leave this module.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import requests

from .geo import ring_area_m2

SQM_TO_SQFT = 10.76391041671
SQFT_PER_ACRE = 43560.0
TIMEOUT = 25
UA = {"User-Agent": "LotMeasure/0.6 (roofmeasure-mcp)"}

_REG_PATH = os.path.join(os.path.dirname(__file__), "parcels_registry.json")
AREA_FIELD_GUESSES = [("ACRES", "acres"), ("ACREAGE", "acres"), ("GIS_ACRES", "acres"), ("DEEDACRES", "acres"),
                      ("LOT_SIZE", "sqft"), ("LOTSIZE", "sqft"), ("LAND_SQFT", "sqft"), ("LND_SQFOOT", "sqft"),
                      ("SQFT", "sqft"), ("Shape_Area", "native"), ("Shape__Area", "native")]


@dataclass
class Parcel:
    ring: list                      # [(lat, lng), ...] closed exterior ring
    source: str                     # county_arcgis | regrid
    provider: str                   # layer / service name
    layer_url: str = ""
    parcel_id: str = ""
    verified_layer: bool = True
    area_m2: float = 0.0
    county_recorded: dict | None = None
    notes: list = field(default_factory=list)

    @property
    def area_sqft(self) -> float:
        return self.area_m2 * SQM_TO_SQFT

    @property
    def area_acres(self) -> float:
        return self.area_sqft / SQFT_PER_ACRE

    def to_dict(self) -> dict:
        d = {
            "ring_latlng": [[round(a, 7), round(b, 7)] for a, b in self.ring],
            "area_sqft": round(self.area_sqft),
            "area_acres": round(self.area_acres, 3),
            "source": self.source,
            "provider": self.provider,
            "layer_url": self.layer_url,
            "layer_verified": self.verified_layer,
            "parcel_id": self.parcel_id,
            "county_recorded": self.county_recorded,
        }
        return d


# ---------------------------------------------------------------- registry

def _discovered_path() -> str:
    d = os.environ.get("ROOFMEASURE_OUT_DIR", os.path.join(os.path.expanduser("~"), "RoofMeasure Reports"))
    return os.path.join(d, "parcels_registry_discovered.json")


def load_registry() -> dict:
    with open(_REG_PATH) as f:
        reg = json.load(f)
    reg.pop("_about", None)
    try:
        with open(_discovered_path()) as f:
            for k, v in json.load(f).items():
                reg.setdefault(k, [])
                reg[k].extend(x for x in v if x["url"] not in {y["url"] for y in reg[k]})
    except (OSError, ValueError):
        pass
    return reg


def _save_discovered(key: str, entry: dict) -> None:
    p = _discovered_path()
    try:
        data = json.load(open(p)) if os.path.exists(p) else {}
    except ValueError:
        data = {}
    data.setdefault(key, [])
    if entry["url"] not in {x["url"] for x in data[key]}:
        data[key].append(entry)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


def registry_key(state: str, county: str) -> str:
    return f"{state.upper()}|{county.strip()}"


# ---------------------------------------------------------------- ArcGIS

def query_point(layer_url: str, lat: float, lng: float) -> list[dict]:
    """GeoJSON features of a parcel layer that contain the point."""
    params = dict(geometry=f"{lng:.7f},{lat:.7f}", geometryType="esriGeometryPoint", inSR=4326,
                  spatialRel="esriSpatialRelIntersects", outFields="*", outSR=4326, f="geojson",
                  returnGeometry="true")
    r = requests.get(layer_url.rstrip("/") + "/query", params=params, timeout=TIMEOUT, headers=UA)
    d = r.json()
    if "error" in d:
        raise RuntimeError(f"parcel layer error: {d['error'].get('message', d['error'])}")
    return d.get("features", []) or []


def query_envelope(layer_url: str, lat0: float, lng0: float, lat1: float, lng1: float) -> list[dict]:
    """Parcels intersecting a lat/lng box (neighbours for the road-frontage test)."""
    params = dict(geometry=f"{min(lng0, lng1):.7f},{min(lat0, lat1):.7f},{max(lng0, lng1):.7f},{max(lat0, lat1):.7f}",
                  geometryType="esriGeometryEnvelope", inSR=4326, spatialRel="esriSpatialRelIntersects",
                  outFields="OBJECTID", outSR=4326, f="geojson", returnGeometry="true", resultRecordCount=200)
    r = requests.get(layer_url.rstrip("/") + "/query", params=params, timeout=TIMEOUT, headers=UA)
    d = r.json()
    if "error" in d:
        return []
    return d.get("features", []) or []


def feature_rings(feat: dict) -> list[list[tuple]]:
    """Exterior rings of a GeoJSON Polygon/MultiPolygon as [(lat, lng)] lists."""
    g = feat.get("geometry") or {}
    t = g.get("type")
    polys = [g["coordinates"]] if t == "Polygon" else (g.get("coordinates", []) if t == "MultiPolygon" else [])
    out = []
    for p in polys:
        if p and len(p[0]) >= 4:
            out.append([(float(c[1]), float(c[0])) for c in p[0]])
    return out


def _contains(ring, lat, lng) -> bool:
    from shapely.geometry import Point, Polygon

    return Polygon([(b, a) for a, b in ring]).buffer(0).contains(Point(lng, lat))


def _pick_ring(feats: list[dict], lat: float, lng: float):
    """The polygon containing the point; smallest if several (condo/common-area overlaps)."""
    best = None
    for f in feats:
        for ring in feature_rings(f):
            if not _contains(ring, lat, lng):
                continue
            a = ring_area_m2(ring)
            if best is None or a < best[1]:
                best = (ring, a, f)
    return best


def _recorded_area(props: dict, area_fields: list) -> dict | None:
    """County-recorded lot size from the attributes (first usable field)."""
    upper = {k.upper(): k for k in props}
    for fname, units in area_fields:
        k = upper.get(fname.upper())
        if k is None:
            continue
        v = props.get(k)
        try:
            v = float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        if units == "acres":
            sqft = v * SQFT_PER_ACRE
        elif units == "sqft":
            sqft = v
        else:
            continue          # native-unit Shape_Area depends on the layer projection: not trusted
        return {"field": k, "value": v, "units": units, "sqft": round(sqft), "acres": round(sqft / SQFT_PER_ACRE, 4)}
    return None


def _make_parcel(ring, area_m2, feat, entry, source="county_arcgis") -> Parcel:
    props = feat.get("properties") or {}
    pid = ""
    idf = entry.get("id_field")
    if idf:
        up = {k.upper(): k for k in props}
        if idf.upper() in up:
            pid = str(props[up[idf.upper()]] or "")
    rec = _recorded_area(props, entry.get("area_fields") or AREA_FIELD_GUESSES)
    p = Parcel(ring=ring, source=source, provider=entry.get("name", ""), layer_url=entry.get("url", ""),
               parcel_id=pid, verified_layer=bool(entry.get("verified")), area_m2=area_m2)
    if rec:
        diff = 100.0 * (p.area_sqft - rec["sqft"]) / rec["sqft"]
        rec["difference_pct"] = round(diff, 1)
        rec["note"] = ("county record and measured polygon agree within 5%" if abs(diff) <= 5 else
                       "county record differs from the mapped polygon by more than 5%: the recorded "
                       "(deeded) size and the GIS drawing disagree; verify with the plat or survey")
        p.county_recorded = rec
    gis = _recorded_area(props, entry.get("gis_area_fields") or [])
    if gis:
        # the county's own area of its drawn polygon (state-plane units): an
        # independent check of our geodesic area of the same polygon
        gis_diff = 100.0 * (p.area_sqft - gis["sqft"]) / gis["sqft"]
        if p.county_recorded is None:
            p.county_recorded = {}
        p.county_recorded["county_gis_sqft"] = gis["sqft"]
        p.county_recorded["county_gis_difference_pct"] = round(gis_diff, 2)
    return p


def lookup_registry(lat, lng, county, state) -> Parcel | None:
    reg = load_registry()
    keys = [registry_key(state, county), registry_key(state, "*")] if county else [registry_key(state, "*")]
    tried = set()
    for k in keys:
        for entry in reg.get(k, []):
            if entry["url"] in tried:
                continue
            tried.add(entry["url"])
            try:
                feats = query_point(entry["url"], lat, lng)
            except Exception:
                continue
            hit = _pick_ring(feats, lat, lng)
            if hit:
                p = _make_parcel(hit[0], hit[1], hit[2], entry)
                if not entry.get("verified"):
                    p.notes.append("Parcel layer was discovered automatically and has not been verified "
                                   "by hand; compare the lot line with the county map.")
                return p
    return None


def discover_layers(county: str, state: str, max_items: int = 8) -> list[dict]:
    """Candidate parcel layers for a county from ArcGIS Online search."""
    cname = county.replace(" County", "").replace(" Parish", "")
    q = f'parcels "{cname}" {state} type:"Feature Service"'
    try:
        r = requests.get("https://www.arcgis.com/sharing/rest/search",
                         params={"q": q, "f": "json", "num": 20, "sortField": "numviews", "sortOrder": "desc"},
                         timeout=TIMEOUT, headers=UA).json()
    except Exception:
        return []
    out = []
    for it in r.get("results", []):
        url = it.get("url") or ""
        title = (it.get("title") or "")
        if not url or "parcel" not in (title + " " + url).lower():
            continue
        out.append({"url": url, "title": title, "owner": it.get("owner", "")})
        if len(out) >= max_items:
            break
    return out


def _layer_candidates(service_url: str) -> list[str]:
    if re.search(r"/(FeatureServer|MapServer)/\d+$", service_url):
        return [service_url]
    try:
        d = requests.get(service_url.rstrip("/"), params={"f": "json"}, timeout=TIMEOUT, headers=UA).json()
    except Exception:
        return []
    out = []
    for l in d.get("layers", []):
        if l.get("geometryType") in (None, "esriGeometryPolygon") and "parcel" in (l.get("name") or "").lower():
            out.append(f"{service_url.rstrip('/')}/{l['id']}")
    if not out:
        out = [f"{service_url.rstrip('/')}/{l['id']}" for l in d.get("layers", [])
               if l.get("geometryType") == "esriGeometryPolygon"][:2]
    return out


def lookup_discovered(lat, lng, county, state) -> Parcel | None:
    if not county:
        return None
    for cand in discover_layers(county, state):
        for layer in _layer_candidates(cand["url"]):
            try:
                feats = query_point(layer, lat, lng)
            except Exception:
                continue
            hit = _pick_ring(feats, lat, lng)
            if not hit:
                continue
            area_sqft = hit[1] * SQM_TO_SQFT
            if not (300 <= area_sqft <= 300 * SQFT_PER_ACRE):
                continue
            entry = {"url": layer, "name": f"{cand['title']} (ArcGIS Online, owner {cand['owner']})",
                     "area_fields": AREA_FIELD_GUESSES, "verified": "", "discovered": time.strftime("%Y-%m-%d"),
                     "coverage": "discovered automatically"}
            _save_discovered(registry_key(state, county), entry)
            p = _make_parcel(hit[0], hit[1], hit[2], entry)
            p.verified_layer = False
            p.notes.append("Parcel layer was discovered automatically on ArcGIS Online and has not been "
                           "verified by hand; compare the lot line with the county map.")
            return p
    return None


# ---------------------------------------------------------------- Regrid

def lookup_regrid(lat, lng) -> Parcel | None:
    token = os.environ.get("REGRID_API_KEY", "").strip()
    if not token:
        return None
    try:
        r = requests.get("https://app.regrid.com/api/v2/parcels/point",
                         params={"lat": f"{lat:.7f}", "lon": f"{lng:.7f}", "token": token, "limit": 1},
                         timeout=TIMEOUT, headers=UA)
        d = r.json()
    except Exception:
        return None
    feats = ((d.get("parcels") or {}).get("features")) or d.get("features") or []
    hit = _pick_ring(feats, lat, lng)
    if not hit:
        return None
    props = (hit[2].get("properties") or {})
    fields = props.get("fields", props)
    entry = {"name": "Regrid parcel API", "url": "https://app.regrid.com/api/v2/parcels/point",
             "area_fields": [("ll_gissqft", "sqft"), ("ll_gisacre", "acres"), ("deeded_acres", "acres")],
             "id_field": "parcelnumb", "verified": "Regrid nationwide"}
    return _make_parcel(hit[0], hit[1], {"properties": fields}, entry, source="regrid")


# ---------------------------------------------------------------- chain

def find_parcel(lat: float, lng: float, county: str = "", state: str = "") -> tuple[Parcel | None, list[str]]:
    """Run the provider chain. Returns (parcel or None, provider log)."""
    log = []
    p = lookup_registry(lat, lng, county, state)
    if p:
        log.append(f"county layer hit: {p.provider}")
        return p, log
    log.append(f"no registered layer for {county}, {state}" if county else "county unknown")
    p = lookup_discovered(lat, lng, county, state)
    if p:
        log.append(f"discovered layer hit: {p.provider}")
        return p, log
    log.append("no ArcGIS Online parcel layer found")
    p = lookup_regrid(lat, lng)
    if p:
        log.append("Regrid hit")
        return p, log
    log.append("Regrid not configured" if not os.environ.get("REGRID_API_KEY") else "Regrid: no parcel")
    return None, log


def neighbour_rings(parcel: Parcel, pad_m: float = 25.0) -> list[list[tuple]]:
    """Other parcels around this one (same layer), for the road-gap frontage rule."""
    if parcel.source != "county_arcgis" or not parcel.layer_url:
        return []
    r = np.asarray(parcel.ring)
    dlat = pad_m / 111_000.0
    dlng = pad_m / (111_000.0 * np.cos(np.radians(r[:, 0].mean())))
    try:
        feats = query_envelope(parcel.layer_url, r[:, 0].min() - dlat, r[:, 1].min() - dlng,
                               r[:, 0].max() + dlat, r[:, 1].max() + dlng)
    except Exception:
        return []
    own = ring_area_m2(parcel.ring)
    c = r.mean(axis=0)
    out = []
    for f in feats:
        for ring in feature_rings(f):
            if abs(ring_area_m2(ring) - own) < 0.5 and _contains(ring, c[0], c[1]):
                continue
            out.append(ring)
    return out


# ---------------------------------------------------------------- OSM (frontage only)

def osm_roads(lat: float, lng: float, radius_m: float = 80.0) -> list[dict]:
    """Named road centrelines near a point from Overpass: [{name, coords [(lat, lng)]}].
    Empty on any failure (Overpass is a best-effort confirmation source)."""
    q = (f"[out:json][timeout:10];way(around:{radius_m:.0f},{lat:.7f},{lng:.7f})[highway]"
         f"[highway!~\"footway|path|cycleway|steps|service\"];out geom;")
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, timeout=12, headers=UA)
            d = r.json()
        except Exception:
            continue
        out = []
        for el in d.get("elements", []):
            g = el.get("geometry") or []
            if len(g) >= 2:
                out.append({"name": (el.get("tags") or {}).get("name", ""),
                            "coords": [(p["lat"], p["lon"]) for p in g]})
        return out
    return []
