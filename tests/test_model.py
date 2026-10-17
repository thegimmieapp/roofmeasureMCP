"""EagleView conventions in the data model (waste table, complexity)."""

from roofmeasure.model import EdgeSummary, Facet, RoofMeasurements


def _m(area, facets, hips, valleys, pitches):
    return RoofMeasurements("x", 0, 0, area, facets=[Facet("A", 1, 1, "8/12", 33, 0)] * facets,
                            edges=EdgeSummary(hips_ft=hips, valleys_ft=valleys), areas_per_pitch=pitches)


def test_waste_table_matches_eagleview_andy_sallee():
    m = _m(3546, 16, 189, 103, {"1/12": 249.7, "8/12": 3295.9})
    assert m.suggested_waste_pct() == 19
    assert m.complexity() == "Normal"
    assert m.waste_columns() == [0, 4, 9, 14, 17, 19, 21, 24, 29]
    wt = {r["waste_pct"]: r for r in m.waste_table()}
    assert wt[0]["area_sqft"] == 3296 and wt[0]["squares"] == 33.0   # 3/12+ area only
    assert wt[19]["squares"] == 39.33


def test_waste_table_tony_russell():
    m = _m(3924, 15, 65, 129, {"7/12": 389.1, "10/12": 3534.2})
    assert m.suggested_waste_pct() in (12, 13)
    assert m.waste_table()[0]["squares"] == 39.33


def test_small_simple_structure():
    m = _m(146, 2, 0, 0, {"4/12": 145.6})
    assert m.complexity() == "Simple"
    assert m.suggested_waste_pct() >= 30
