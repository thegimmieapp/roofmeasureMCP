"""Verify the geometry engine against synthetic roofs with known dimensions."""

import math

import numpy as np
import pytest

from roofmeasure.geometry import M_TO_FT, measure_from_dsm
from roofmeasure.model import pitch_to_ratio


def build_hip_roof(length_m=16.0, width_m=10.0, pitch_ratio=8 / 12, px=0.1, pad=8):
    """Classic hip roof: rectangular footprint, ridge along the long axis.

    Known geometry:
      ridge = length - width          (hip ends rise at 45 deg in plan)
      hips  = 4 * (width/2) * sqrt(2 + pitch^2) / ... computed below
      eaves = 2*(length+width), rakes = 0, valleys = 0
    """
    W = int(width_m / px) + 2 * pad
    L = int(length_m / px) + 2 * pad
    z = np.zeros((W, L), dtype=np.float32)
    mask = np.zeros((W, L), dtype=bool)
    base = 3.0
    for i in range(W):
        for j in range(L):
            x = (j - pad) * px   # along length
            y = (i - pad) * px   # along width
            if 0 <= x <= length_m and 0 <= y <= width_m:
                mask[i, j] = True
                # hip roof height: min distance to nearest eave edge
                d = min(x, length_m - x, y, width_m - y)
                z[i, j] = base + d * pitch_ratio
    return z, mask, px


def test_hip_roof_measurements():
    length_m, width_m, pitch = 16.0, 10.0, 8 / 12
    z, mask, px = build_hip_roof(length_m, width_m, pitch)
    m = measure_from_dsm(z, mask, px, px, "synthetic hip", 0.0, 0.0)

    # ---- areas ----
    slope = math.atan(pitch)
    plan_area_sqft = length_m * width_m * 10.7639
    expected_surface = plan_area_sqft / math.cos(slope)
    assert m.total_area_sqft == pytest.approx(expected_surface, rel=0.08)

    # ---- pitch ----
    assert m.predominant_pitch == "8/12"

    # ---- facet count: 4 (two trapezoids + two triangles) ----
    assert m.facet_count == 4

    # ---- ridge ----
    expected_ridge_ft = (length_m - width_m) * M_TO_FT  # 6 m
    assert m.edges.ridges_ft == pytest.approx(expected_ridge_ft, rel=0.20)

    # ---- hips: 4 hips, plan length (w/2)*sqrt(2), slope-corrected ----
    half = width_m / 2
    hip_plan = half * math.sqrt(2)
    hip_rise = half * pitch
    hip_true = math.hypot(hip_plan, hip_rise)
    expected_hips_ft = 4 * hip_true * M_TO_FT
    assert m.edges.hips_ft == pytest.approx(expected_hips_ft, rel=0.20)

    # ---- eaves = full perimeter, no rakes ----
    expected_eaves_ft = 2 * (length_m + width_m) * M_TO_FT
    assert m.edges.eaves_ft == pytest.approx(expected_eaves_ft, rel=0.12)
    assert m.edges.rakes_ft < 0.25 * expected_eaves_ft

    # ---- no valleys on a plain hip roof ----
    assert m.edges.valleys_ft < 20


def test_gable_roof_measurements():
    """Gable roof: two facets, ridge full length, rakes at both ends."""
    length_m, width_m, pitch = 14.0, 9.0, 6 / 12
    px, pad = 0.1, 8
    W = int(width_m / px) + 2 * pad
    L = int(length_m / px) + 2 * pad
    z = np.zeros((W, L), dtype=np.float32)
    mask = np.zeros((W, L), dtype=bool)
    for i in range(W):
        for j in range(L):
            x = (j - pad) * px
            y = (i - pad) * px
            if 0 <= x <= length_m and 0 <= y <= width_m:
                mask[i, j] = True
                d = min(y, width_m - y)
                z[i, j] = 3.0 + d * pitch

    m = measure_from_dsm(z, mask, px, px, "synthetic gable", 0.0, 0.0)

    assert m.facet_count == 2
    assert m.predominant_pitch == "6/12"

    expected_ridge_ft = length_m * M_TO_FT
    assert m.edges.ridges_ft == pytest.approx(expected_ridge_ft, rel=0.15)

    # eaves: the two long sides
    expected_eaves_ft = 2 * length_m * M_TO_FT
    assert m.edges.eaves_ft == pytest.approx(expected_eaves_ft, rel=0.15)

    # rakes: 4 sloped end edges, slope-corrected
    half = width_m / 2
    rake_true = math.hypot(half, half * pitch)
    expected_rakes_ft = 4 * rake_true * M_TO_FT
    assert m.edges.rakes_ft == pytest.approx(expected_rakes_ft, rel=0.25)

    assert m.edges.hips_ft < 10
    assert m.edges.valleys_ft < 10


def test_waste_table_and_squares():
    z, mask, px = build_hip_roof()
    m = measure_from_dsm(z, mask, px, px, "synthetic", 0.0, 0.0)
    wt = m.waste_table()
    assert wt[0]["waste_pct"] == 0
    assert wt[-1]["waste_pct"] == 32
    assert wt[3]["area_sqft"] > wt[0]["area_sqft"]
    # squares round UP to 1/3
    assert m.squares >= m.total_area_sqft / 100.0
