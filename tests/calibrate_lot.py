"""Offline LotMeasure calibration on raw GeoTIFFs cached by fetch_lot_cache.py.

For each address: county parcel (live ArcGIS query), lot area vs the county
record, surface breakdown, zones, setbacks, trees, sun; optional plan PNGs.

Usage: python tests/calibrate_lot.py <cache_dir> [plan_dir] [keys...]
"""
import json
import os
import sys
import time

from roofmeasure import lot, parcels, solar


def load_tiles(d: str) -> tuple:
    meta = json.load(open(os.path.join(d, "meta.json")))
    rd = lambda n: open(os.path.join(d, n), "rb").read()  # noqa: E731
    dsm, px_x, px_y, geo = solar.read_geotiff_geo(rd("dsm.tif"))
    mask, _, _, _ = solar.read_geotiff_geo(rd("mask.tif"))
    t = lot.LotTiles(dsm=dsm, mask=mask, geo=geo, px_x=px_x, px_y=px_y)
    if os.path.exists(os.path.join(d, "rgb.tif")):
        t.rgb, _ = solar.read_geotiff_rgb_geo(rd("rgb.tif"))
    if os.path.exists(os.path.join(d, "flux.tif")):
        t.flux, _, _, t.flux_geo = solar.read_geotiff_geo(rd("flux.tif"))
    for mon in (6, 12):
        p = os.path.join(d, f"shade{mon:02d}.tif")
        if os.path.exists(p):
            arr, _, _, g = solar.read_geotiff_geo(rd(f"shade{mon:02d}.tif"), bands=True)
            t.shade[mon] = (arr, g)
    from roofmeasure import geometry
    t.imagery_date = geometry._fmt_date(meta.get("imageryDate"))
    t.imagery_quality = meta.get("imageryQuality", "")
    gc = solar.parse_geocode_result({"address_components": meta["address_components"],
                                     "geometry": {"location": {"lat": meta["lat"], "lng": meta["lng"]},
                                                  "location_type": meta.get("location_type", "")},
                                     "formatted_address": meta["address"]})
    return t, gc, meta.get("insights") or {}


def main():
    cache = sys.argv[1]
    plan_dir = sys.argv[2] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else None
    keys = [a for a in sys.argv[2:] if a != plan_dir] or sorted(os.listdir(cache))
    rows = []
    for k in keys:
        d = os.path.join(cache, k)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        t0 = time.time()
        tiles, gc, ins = load_tiles(d)
        parcel, log = parcels.find_parcel(gc["lat"], gc["lng"], gc["county"], gc["state"])
        nb = parcels.neighbour_rings(parcel) if parcel else []
        roads = parcels.osm_roads(gc["lat"], gc["lng"]) if parcel else []
        res = lot.measure_lot_from_tiles(tiles, gc, ins, parcel=parcel, neighbours=nb, roads=roads,
                                         provider_log=log, keep_rasters=bool(plan_dir))
        dt = time.time() - t0
        p = res["parcel"]
        rec = (p or {}).get("county_recorded") or {}
        print(f"== {k} ({dt:.0f}s) {res['address']}")
        if p:
            print(f"   lot {p['area_sqft']:,} sq ft ({p['area_acres']:.3f} ac) vs county {rec.get('sqft', 0):,} "
                  f"({rec.get('difference_pct', 'n/a')}%)  frontage {res['frontage']['rule']}: {res['frontage']['detail']}")
        print(f"   house {(res['house'] or {}).get('plan_area_sqft', 'not reported')} sq ft; totals {res['totals']}")
        for z, v in res["zones"].items():
            print(f"   {z:6s} {v['area_sqft']:6,} sq ft lawn {v.get('lawn_planting_pct', 0)}% setback {v['setback_ft']} ft "
                  f"rect {(v['largest_clear_rectangle'] or {}).get('area_sqft')} sun {v.get('sun')}")
        print(f"   trees {res['tree_count']} ({res['trees_within_3m_of_house']} within 3 m of house); "
              f"canopy {res['confidence']['canopy_fraction']:.0%}; coverage {res['confidence']['tile_coverage']:.0%}")
        if plan_dir:
            from roofmeasure.lotplan import render_lot_plan
            render_lot_plan(res, os.path.join(plan_dir, f"{k}_lot_plan.png"), pdf_path=None)
        if p and rec.get("sqft"):
            rows.append((k, p["area_sqft"], rec["sqft"], rec["difference_pct"], rec.get("county_gis_sqft"),
                         rec.get("county_gis_difference_pct")))
        res.pop("_plan", None)
        with open(os.path.join(d, "lot_result.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
    print("\nlot area: ours vs county recorded (deed/assessor field) vs county GIS polygon area")
    for r in rows:
        g = f"{r[4]:9,} {r[5]:+6.2f}%" if r[4] else "      n/a"
        print(f"  {r[0]:10s} {r[1]:9,} {r[2]:9,} {r[3]:+6.1f}%  {g}")


if __name__ == "__main__":
    main()
