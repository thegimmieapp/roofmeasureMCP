"""Verify the measurement engine against synthetic roofs with known dimensions."""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import gable_roof, hip_roof, l_hip_roof, t_gable_roof  # noqa: E402

from roofmeasure import pipeline  # noqa: E402
from roofmeasure.geometry import measure_from_dsm  # noqa: E402


def _measure(builder, noise=0.03):
    z, mask, px, truth = builder(noise=noise)
    m, detail = measure_from_dsm(z, mask, px, px, "synthetic", 0.0, 0.0, return_detail=True)
    pipeline._attach_skeleton(m, detail, px)
    assert detail.skeleton is not None, m.notes
    return m, truth


def test_hip_roof():
    m, t = _measure(hip_roof)
    e = m.edges
    assert m.facet_count == 4
    assert m.predominant_pitch == "8/12"
    assert m.total_area_sqft == pytest.approx(t["surface_sqft"], rel=0.03)
    assert e.ridges_ft == pytest.approx(t["ridges"], abs=3)
    assert e.hips_ft == pytest.approx(t["hips"], rel=0.05)
    assert e.eaves_ft == pytest.approx(t["eaves"], rel=0.05)
    assert e.valleys_ft == 0 and e.rakes_ft <= 12


def test_gable_roof():
    m, t = _measure(gable_roof)
    e = m.edges
    assert m.facet_count == 2
    assert m.total_area_sqft == pytest.approx(t["surface_sqft"], rel=0.03)
    assert e.ridges_ft == pytest.approx(t["ridges"], abs=3)
    assert e.rakes_ft == pytest.approx(t["rakes"], rel=0.06)
    assert e.eaves_ft == pytest.approx(t["eaves"], rel=0.05)
    assert e.hips_ft == 0 and e.valleys_ft == 0


def test_t_gable_roof():
    m, t = _measure(t_gable_roof)
    e = m.edges
    assert m.total_area_sqft == pytest.approx(t["surface_sqft"], rel=0.04)
    assert e.ridges_ft == pytest.approx(t["ridges"], rel=0.10)
    assert e.rakes_ft == pytest.approx(t["rakes"], rel=0.10)
    assert e.eaves_ft == pytest.approx(t["eaves"], rel=0.06)
    assert e.valleys_ft == pytest.approx(t["valleys"], rel=0.10) and e.hips_ft == 0


def test_l_hip_roof():
    m, t = _measure(l_hip_roof)
    e = m.edges
    assert m.total_area_sqft == pytest.approx(t["surface_sqft"], rel=0.04)
    assert e.hips_ft == pytest.approx(t["hips"], rel=0.08)
    assert e.eaves_ft == pytest.approx(t["eaves"], rel=0.05)
    assert e.ridges_ft == pytest.approx(t["ridges"], rel=0.08)
    assert e.valleys_ft == pytest.approx(t["valleys"], rel=0.08)
    assert e.rakes_ft == pytest.approx(t["rakes"], rel=0.20)


def test_bad_wireframe_never_zeroes_edges():
    """A degenerate outline must not replace raster-derived edges with zeros."""
    z, mask, px, _ = hip_roof(noise=0.03)
    m, detail = measure_from_dsm(z, mask, px, px, "synthetic", 0.0, 0.0, return_detail=True)
    before = m.edges.eaves_ft
    detail.planes = {}  # force wireframe failure
    pipeline._attach_skeleton(m, detail, px)
    assert m.edges.eaves_ft == before and m.edges.eaves_ft > 100
