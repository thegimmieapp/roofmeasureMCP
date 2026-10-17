"""Offline calibration: run the pipeline on cached rasters and compare with
EagleView ground truth (main structure only; other structures are separate
buildings that the pipeline does not yet report).

Usage: python tests/calibrate.py <cache_dir> [viz_dir] [keys...]
"""
import json
import sys
import time

import numpy as np

from roofmeasure import pipeline

# EagleView truth for the primary structure at each address.
TRUTH = {
    "andy": dict(area=3546, facets=16, pitch="8/12", ridges=60, hips=189, valleys=103, rakes=46, eaves=232),
    "joe": dict(area=6912, facets=26, pitch="8/12", ridges=110, hips=401, valleys=239, rakes=18, eaves=432),
    "tony": dict(area=3924, facets=15, pitch="10/12", ridges=111, hips=65, valleys=129, rakes=185, eaves=155),
    "lynda": dict(area=1598, facets=16, pitch="9/12", ridges=48, hips=44, valleys=54, rakes=117, eaves=144),
    "fred": dict(area=6802, facets=22, pitch="8/12", ridges=139, hips=216, valleys=136, rakes=137, eaves=389),
    "jenny": dict(area=6348, facets=6, pitch="4/12", ridges=142, hips=0, valleys=41, rakes=160, eaves=262),
}

cache = sys.argv[1]
viz = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2] in TRUTH else None
keys = [a for a in sys.argv[2:] if a in TRUTH] or list(TRUTH)


def pct(a, b):
    return f"{100.0 * (a - b) / b:+.0f}%" if b else "n/a"


rows = []
for k in keys:
    d = np.load(f"{cache}/{k}.npz")
    meta = json.load(open(f"{cache}/{k}.json"))
    t0 = time.time()
    m, detail = pipeline.measure_from_rasters(d["dsm"], d["mask"] > 0, float(d["px_x"]), float(d["px_y"]),
                                              meta["address"], meta["lat"], meta["lng"],
                                              insights=meta.get("insights"))
    dt = time.time() - t0
    e, T = (m.structures[0].edges if m.structures else m.edges), TRUTH[k]
    m_main = m.structures[0] if m.structures else m
    print(f"== {k:6s} ({dt:.0f}s) area {m_main.total_area_sqft:5.0f} vs {T['area']:5d} ({pct(m_main.total_area_sqft, T['area'])})"
          f"  facets {m_main.facet_count:2d} vs {T['facets']:2d}  pitch {m_main.predominant_pitch} vs {T['pitch']}")
    print(f"   ridges {e.ridges_ft:4.0f} vs {T['ridges']:4d} ({pct(e.ridges_ft, T['ridges'])})"
          f"  hips {e.hips_ft:4.0f} vs {T['hips']:4d} ({pct(e.hips_ft, T['hips'])})"
          f"  valleys {e.valleys_ft:4.0f} vs {T['valleys']:4d} ({pct(e.valleys_ft, T['valleys'])})")
    rk = e.rakes_measured_ft or e.rakes_ft
    ev = e.eaves_measured_ft or e.eaves_ft
    print(f"   rakes  {rk:4.0f} vs {T['rakes']:4d} ({pct(rk, T['rakes'])})"
          f"  eaves {ev:4.0f} vs {T['eaves']:4d} ({pct(ev, T['eaves'])})"
          f"  R+H {e.ridges_hips_ft:4.0f} vs {T['ridges'] + T['hips']:4d}  perim {rk + ev:4.0f} vs {T['rakes'] + T['eaves']:4d}"
          f"   [reported with margin: rakes {e.rakes_ft:.0f}, eaves {e.eaves_ft:.0f}]")
    for n in m.notes:
        if "rejected" in n or "unavailable" in n or "structures" in n:
            print("   NOTE:", n[:160])
    for sm in m.structures[1:]:
        print(f"   structure {sm.structure_index}: {sm.total_area_sqft:.0f} sqft, {sm.facet_count} facets, {sm.predominant_pitch}, R{sm.edges.ridges_ft:.0f} H{sm.edges.hips_ft:.0f} V{sm.edges.valleys_ft:.0f} Rk{sm.edges.rakes_ft:.0f} E{sm.edges.eaves_ft:.0f}")
    print("   pitches:", m_main.areas_per_pitch)
    if viz:
        from roofmeasure.diagrams import render_pdf_report
        rgb = pipeline.crop_rgb(d["rgb"], detail) if d["rgb"].shape[0] > 1 else None
        render_pdf_report(m, detail, f"{viz}/{k}_report.pdf", rgb=rgb)
    rows.append((k, m_main.total_area_sqft / T["area"], (e.ridges_ft + e.hips_ft) / max(T["ridges"] + T["hips"], 1),
                 e.valleys_ft / max(T["valleys"], 1), (rk + ev) / (T["rakes"] + T["eaves"])))

print("\nratio ours/EV:  area  ridge+hip  valley  perimeter")
for r in rows:
    print(f"  {r[0]:6s} {r[1]:6.2f} {r[2]:9.2f} {r[3]:7.2f} {r[4]:9.2f}")
