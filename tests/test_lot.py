"""LotMeasure geometry: exact answers on a synthetic rectangular lot and house."""
import math

import numpy as np
from shapely.geometry import box

from roofmeasure import geo, lot

FT = lot.M_TO_FT

# 20 m wide x 40 m deep lot, street on the north side (Y = 40).
# House 11 x 12 m: x 4..15, y 15..27.
LOT = box(0, 0, 20, 40)
HOUSE = box(4, 15, 15, 27)
U = np.array([0.0, 1.0])        # towards the street


def test_zone_split_areas():
    z = lot.split_zones(LOT, HOUSE, U)
    assert math.isclose(z["front"].area, 20 * 13)          # y 27..40
    assert math.isclose(z["back"].area, 20 * 15)           # y 0..15
    # viewer on the street faces south: right is west (x < 4), left is east (x > 15)
    assert math.isclose(z["right"].area, 4 * 12)
    assert math.isclose(z["left"].area, 5 * 12)
    total = sum(z[k].area for k in ("front", "back", "left", "right"))
    assert math.isclose(total, LOT.area - HOUSE.area)


def test_setbacks_exact():
    walls = lot.wall_setbacks(HOUSE, LOT)
    assert len(walls) == 4 and all(w["parallel_lot_line"] for w in walls)
    z = lot.split_zones(LOT, HOUSE, U)
    sb = lot.zone_setbacks(walls, U, z["_right"])
    assert math.isclose(sb["front"]["m"], 13.0, abs_tol=0.02)
    assert math.isclose(sb["back"]["m"], 15.0, abs_tol=0.02)
    assert math.isclose(sb["right"]["m"], 4.0, abs_tol=0.02)
    assert math.isclose(sb["left"]["m"], 5.0, abs_tol=0.02)
    assert sb["front"]["ft"] == round(13.0 * FT, 1)


def test_rotated_lot_setbacks():
    from shapely import affinity
    lp = affinity.rotate(LOT, 30, origin=(0, 0))
    hp = affinity.rotate(HOUSE, 30, origin=(0, 0))
    u = np.array([-math.sin(math.radians(30)), math.cos(math.radians(30))])
    walls = lot.wall_setbacks(hp, lp)
    z = lot.split_zones(lp, hp, u)
    sb = lot.zone_setbacks(walls, u, z["_right"])
    assert math.isclose(sb["front"]["m"], 13.0, abs_tol=0.02)
    assert math.isclose(sb["left"]["m"], 5.0, abs_tol=0.02)
    assert math.isclose(z["front"].area, 260.0, rel_tol=1e-9)


def test_merge_sides_and_frontage():
    sides = lot.merge_sides(LOT)
    assert len(sides) == 4
    north = max(sides, key=lambda s: s.mid[1])
    assert math.isclose(north.length, 20.0)
    n = lot.frontage_outward(north, LOT)
    assert np.allclose(n, [0, 1])
    axes = lot.house_axes(HOUSE)
    assert np.allclose(np.abs(lot.snap_to_axis(np.array([0.1, 0.99]), axes)), [0, 1], atol=1e-9)


def test_largest_rectangle():
    m = np.zeros((10, 12), bool)
    m[2:7, 3:11] = True            # 5 x 8
    m[0:10, 0:2] = True            # 10 x 2 = 20 < 40
    area, r0, c0, nr, nc = lot.largest_rectangle(m)
    assert (area, r0, c0, nr, nc) == (40, 2, 3, 5, 8)


def test_clear_rectangle_in_back_yard():
    z = lot.split_zones(LOT, HOUSE, U)
    back = z["back"]
    rect = lot.clear_rectangle(lambda X, Y: (X > 0) & (X < 20) & (Y > 0) & (Y < 15), back, lot.house_axes(HOUSE))
    assert math.isclose(rect["area_m2"], 300.0, rel_tol=0.04)


def test_rasterize_polygon_area():
    px = 0.1
    poly = box(1.03, -6.03, 5.03, -1.03)   # 4 x 5 m, edges between pixel centres
    m = lot.rasterize_polygon(poly, (80, 80), px, px)
    assert m.sum() == 40 * 50


def test_sun_hours_popcount():
    shade = np.zeros((2, 2, 24), np.int32)
    shade[0, 0, 10:16] = (1 << 30) - 1          # 30 days sunny, hours 10..15
    shade[0, 1, 12] = np.int32(-2147483648)     # bit 31 only: day 32, never counted as invalid
    shade[1, 1, :] = -9999
    h = lot.sun_hours(shade, 30)
    assert math.isclose(h[0, 0], 6.0)
    assert math.isclose(h[0, 1], 1 / 30)
    assert np.isnan(h[1, 1])


def test_utm_round_trip_and_parcel_area():
    E, N = geo.utm_forward(37.1232533, -93.4540679, 15)
    la, lo = geo.utm_inverse(E, N, 15)
    assert abs(la - 37.1232533) < 1e-8 and abs(lo - (-93.4540679)) < 1e-8
    # 100 m x 100 m square built in local metres has area 10,000 m2 within 0.01%
    lat0, lng0 = 37.12, -93.45
    mx, my = geo.metres_per_degree(lat0)
    ring = [(lat0, lng0), (lat0 + 100 / my, lng0), (lat0 + 100 / my, lng0 + 100 / mx), (lat0, lng0 + 100 / mx)]
    assert abs(geo.ring_area_m2(ring) - 10_000.0) < 1.0


def test_tile_geo_pixel_mapping():
    g = geo.TileGeo(32615, 459564.5, 4108741.7, 0.1, 0.1, (1997, 1994))
    lat, lng = g.px_to_latlng(997.0, 994.0)
    r, c = g.latlng_to_px(lat, lng)
    assert abs(r - 997.0) < 0.01 and abs(c - 994.0) < 0.01   # 1 mm
