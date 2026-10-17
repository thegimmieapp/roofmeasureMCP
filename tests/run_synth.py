"""Quick harness: run both engines on each synthetic roof and print vs truth."""
import sys

sys.path.insert(0, "tests")
import numpy as np  # noqa: E402

from roofmeasure import pipeline  # noqa: E402
from roofmeasure.geometry import measure_from_dsm  # noqa: E402
from synth import gable_roof, hip_roof, l_hip_roof, t_gable_roof  # noqa: E402

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
for name, fn in [("hip", hip_roof), ("gable", gable_roof), ("t_gable", t_gable_roof), ("l_hip", l_hip_roof)]:
    z, mask, px, truth = fn(noise=noise)
    m, detail = measure_from_dsm(z, mask, px, px, name, 0, 0, return_detail=True)
    raster = (m.edges.ridges_ft, m.edges.hips_ft, m.edges.valleys_ft, m.edges.rakes_ft, m.edges.eaves_ft)
    pipeline._attach_skeleton(m, detail, px)
    e = m.edges
    print(f"== {name}: facets {m.facet_count} (truth {truth['facets']}), area {m.total_area_sqft:.0f} (truth {truth['surface_sqft']:.0f}), pitch {m.predominant_pitch}")
    print(f"   raster   R/H/V/Rk/E = {raster}")
    print(f"   skeleton R/H/V/Rk/E = {(e.ridges_ft, e.hips_ft, e.valleys_ft, e.rakes_ft, e.eaves_ft)}")
    tr = {k: round(truth.get(k, -1)) for k in ("ridges", "hips", "valleys", "rakes", "eaves")}
    print(f"   truth    R/H/V/Rk/E = {tuple(tr.values())}")
    if m.notes and "unavailable" in m.notes[-1]:
        print("   NOTE:", m.notes[-1])
