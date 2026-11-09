"""Map/raster contract: no imagery API, addresses, or remote measurements."""
import json
import numpy as np
from PIL import Image
from shapely.geometry import box, Polygon
from roofmeasure.scopemap import scope_map, save_aerial
from roofmeasure.lot import M_TO_FT


def result():
    return {"house": {"plan_area_sqft": 1}, "_plan": {
        "win": (20, 10), "shape": (40, 30), "px": (0.5, 0.25),
        "lot_poly": box(5, -14.75, 19.5, -5),
        "region": box(5, -14.75, 19.5, -5), "house_poly": box(8, -10, 12, -6),
        "zones": {"back": {"poly": box(5, -14.75, 19.5, -10), "rect": {"corners_xy": [(6,-11),(10,-11),(10,-13),(6,-13)]}}},
        "trees": [{"_xy": (6,-12), "_r": 2, "height_ft": 30, "within_3m_of_house": True}],
        "rgb": np.full((40,30,3), 120, dtype=np.uint8),
    }}


def test_map_origin_and_scale():
    data = scope_map(result())
    assert data["width_ft"] == round(15 * M_TO_FT, 3)
    assert data["height_ft"] == round(10 * M_TO_FT, 3)
    first = data["trees"][0]["center"]
    assert first == [round((6 - 4.75) * M_TO_FT,3), round((-4.875 + 12) * M_TO_FT,3)]
    assert data["origin"] == "raster_top_left"
    assert data["y_axis"] == "south"
    json.dumps(data)  # no numpy, shapely or raster objects escape into JSON


def test_withheld_house_is_not_reintroduced():
    res = result()
    res["house"] = None
    res["_plan"]["lot_poly"] = None
    data = scope_map(res)
    assert data["house"] == []
    assert data["parcel"] == []
    assert data["region"]


def test_holes_and_rectangles_are_preserved():
    res = result()
    res["_plan"]["region"] = Polygon([(5,-5),(19,-5),(19,-14),(5,-14)], holes=[[(7,-7),(8,-7),(8,-8),(7,-8)]])
    data = scope_map(res)
    assert len(data["region"][0]) == 2
    assert len(data["zones"]["back"]["clear_rectangle"]) == 4


def test_aerial_has_no_plot_margins_and_handles_float_rgb(tmp_path):
    res = result()
    path = tmp_path / "aerial.png"
    assert save_aerial(res, str(path))
    assert Image.open(path).size == (30,40)
    res["_plan"]["rgb"] = np.full((40,30,3),0.5)
    assert save_aerial(res,str(path))
    assert Image.open(path).getpixel((0,0)) == (127,127,127)
    res["_plan"]["rgb"] = None
    assert not save_aerial(res,str(path))


def test_server_map_contract_is_opt_in_and_drops_private_rasters(tmp_path, monkeypatch):
    from roofmeasure import lot, server
    monkeypatch.setenv("ROOFMEASURE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("ROOFMEASURE_PUBLIC_URL", "https://example.test")
    calls = []
    def measured(_address, include_sun=True, keep_rasters=False):
        calls.append(keep_rasters)
        return result() if keep_rasters else {"house": None}
    monkeypatch.setattr(lot, "measure_lot", measured)
    original = server._lot_with_plan("synthetic", make_plan=False)
    assert "scope_map" not in original
    res = server._lot_with_plan("synthetic", make_plan=False, make_map=True)
    assert "scope_map" in res and "_plan" not in res
    assert res["aerial_png_url"].endswith(".png")
    assert calls == [False, True]
    json.dumps(res)
