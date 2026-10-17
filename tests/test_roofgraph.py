"""Framing-model generator: exact answers on textbook footprints."""
import math

import numpy as np

from roofmeasure import roofgraph as rg

FT = rg.M_TO_FT


def _rect(w_m, h_m, px=0.1):
    # (y, x) px outline of a w x h metre rectangle
    return np.array([(0, 0), (0, w_m / px), (h_m / px, w_m / px), (h_m / px, 0)], dtype=float)


def test_rectangle_hip_roof():
    # 10 x 6 m at 6/12: ridge = 10 - 6 = 4 m; four hips, each run 3 m diag = 4.243 m plan, rise 1.5 m
    g = rg.build_roof_graph(_rect(10, 6), 0.1, 0.1, 6)
    t = g.totals
    assert abs(t["ridges_ft"] - 4 * FT) <= 1
    hip = math.hypot(math.hypot(3, 3), 1.5)
    assert abs(t["hips_ft"] - 4 * hip * FT) <= 2
    assert t["valleys_ft"] == 0 and t["rakes_ft"] == 0
    assert abs(t["eaves_ft"] - 32 * FT) <= 1
    assert len(g.faces) == 4
    # surface area = plan / cos(slope)
    assert abs(t["surface_area_sqft"] - 60 * rg.SQM_TO_SQFT / math.cos(math.atan(0.5))) <= 3


def test_rectangle_gable_roof():
    # gable ends on both short sides: ridge runs the full 10 m, rakes 4 x hypot(3, 1.5)
    g = rg.build_roof_graph(_rect(10, 6), 0.1, 0.1, 6, gables={1, 3})
    t = g.totals
    assert abs(t["ridges_ft"] - 10 * FT) <= 1, t
    assert t["hips_ft"] == 0 and t["valleys_ft"] == 0
    assert abs(t["rakes_ft"] - 4 * math.hypot(3, 1.5) * FT) <= 2, t
    assert abs(t["eaves_ft"] - 20 * FT) <= 1
    assert len(g.faces) == 2


def test_l_shape_wing_ties_into_main():
    # 12 x 5 main run with a 6 m wide wing: valleys from the inside corner, the
    # wing ridge lower than the main and connected to it; every line meets another
    poly = np.array([(0, 0), (0, 120), (50, 120), (50, 60), (100, 60), (100, 0)], dtype=float)
    g = rg.build_roof_graph(poly, 0.1, 0.1, 8)
    t = g.totals
    assert t["ridge_count"] >= 2 and t["valley_count"] >= 1 and t["hip_count"] >= 5
    # connectivity: every ridge/hip/valley endpoint touches another edge
    pts = [np.asarray(p) for e in g.edges for p in (e.p0, e.p1)]
    for e in g.edges:
        if e.kind in ("ridge", "hip", "valley"):
            for p in (e.p0, e.p1):
                others = [q for q in pts if np.hypot(*(q - np.asarray(p))) < 1.0]
                assert len(others) >= 2, (e.kind, p)


def test_simplify_outline_removes_small_jogs():
    poly = np.array([(0, 0), (0, 100), (3, 100), (3, 50), (0, 50), (0, 50.0001), (0, 0)], dtype=float)
    poly = np.array([(0, 0), (0, 100), (60, 100), (60, 52), (57, 52), (57, 0)], dtype=float)  # 0.3 m jog
    out = rg.simplify_outline(poly, 0.1, 1.0)
    assert len(out) == 4
