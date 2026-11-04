"""Georeferencing for Solar API rasters: tile pixels <-> latitude/longitude.

Solar API GeoTIFFs are delivered in the UTM zone of the tile (EPSG:326xx /
327xx, metres) with a ModelTransformation tag; some products use plain
EPSG:4326 (degrees) with ModelPixelScale + ModelTiepoint. TileGeo handles
both. UTM is computed with the Krueger series (sub-millimetre inside a zone),
pure Python, no pyproj, because the Mac and the Fly image have no compiler.

Pixel convention used everywhere in roofmeasure: (row, col) index space,
integer values are pixel CENTRES (row 0 is the top row). GeoTIFF raster
space has (0, 0) at the upper-left CORNER of the upper-left pixel
(PixelIsArea), hence the 0.5 shifts below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16,
          13 * _N ** 2 / 48 - 3 * _N ** 3 / 5,
          61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96,
         _N ** 2 / 48 + _N ** 3 / 15,
         17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3,
          7 * _N ** 2 / 3 - 8 * _N ** 3 / 5,
          56 * _N ** 3 / 15)

M_PER_DEG_LAT = 111_320.0   # same equirectangular scale solar.py uses


def utm_forward(lat, lon, zone: int, north: bool = True):
    """WGS84 lat/lon (degrees, scalars or arrays) -> UTM easting, northing (m)."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lon0 = math.radians(zone * 6 - 183)
    e = 2 * math.sqrt(_N) / (1 + _N)
    t = np.sinh(np.arctanh(np.sin(lat)) - e * np.arctanh(e * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(lon - lon0))
    eta_p = np.arctanh(np.sin(lon - lon0) / np.sqrt(1 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    E = 500000.0 + _K0 * _AA * eta
    Nn = _K0 * _AA * xi + (0.0 if north else 10_000_000.0)
    return E, Nn


def utm_inverse(E, Nn, zone: int, north: bool = True):
    """UTM easting/northing (m) -> WGS84 lat, lon (degrees)."""
    E = np.asarray(E, dtype=float)
    Nn = np.asarray(Nn, dtype=float) - (0.0 if north else 10_000_000.0)
    xi = Nn / (_K0 * _AA)
    eta = (E - 500000.0) / (_K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    lat = chi.copy()
    for j, d in enumerate(_DELTA, start=1):
        lat = lat + d * np.sin(2 * j * chi)
    lon0 = math.radians(zone * 6 - 183)
    lon = lon0 + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lat), np.degrees(lon)


@dataclass
class TileGeo:
    """Affine georeference of a raster: X = x0 + col_corner * sx, Y = y0 - row_corner * sy.

    epsg 4326: X = longitude, Y = latitude (degrees). epsg 326zz / 327zz: UTM metres.
    """
    epsg: int
    x0: float
    y0: float
    sx: float
    sy: float
    shape: tuple = (0, 0)

    @property
    def is_utm(self) -> bool:
        return 32601 <= self.epsg <= 32760

    def _zone(self):
        return (self.epsg - 32600, True) if self.epsg < 32700 else (self.epsg - 32700, False)

    def px_size_m(self, lat: float | None = None) -> tuple[float, float]:
        if self.is_utm:
            return self.sx, self.sy
        lat = self.y0 if lat is None else lat
        return self.sx * M_PER_DEG_LAT * math.cos(math.radians(lat)), self.sy * M_PER_DEG_LAT

    def latlng_to_px(self, lat, lng):
        """-> (row, col) float index coordinates (pixel centres at integers)."""
        if self.is_utm:
            z, north = self._zone()
            X, Y = utm_forward(lat, lng, z, north)
        elif self.epsg == 4326:
            X, Y = np.asarray(lng, float), np.asarray(lat, float)
        else:
            raise ValueError(f"unsupported raster CRS EPSG:{self.epsg}")
        col = (X - self.x0) / self.sx - 0.5
        row = (self.y0 - Y) / self.sy - 0.5
        return row, col

    def px_to_latlng(self, row, col):
        X = self.x0 + (np.asarray(col, float) + 0.5) * self.sx
        Y = self.y0 - (np.asarray(row, float) + 0.5) * self.sy
        if self.is_utm:
            z, north = self._zone()
            return utm_inverse(X, Y, z, north)
        return Y, X

    def window(self, r0: int, c0: int, shape=None) -> "TileGeo":
        """Georeference of a sub-window starting at pixel (r0, c0)."""
        return TileGeo(self.epsg, self.x0 + c0 * self.sx, self.y0 - r0 * self.sy, self.sx, self.sy,
                       tuple(shape) if shape is not None else self.shape)

    def to_dict(self) -> dict:
        return {"epsg": self.epsg, "x0": self.x0, "y0": self.y0, "sx": self.sx, "sy": self.sy,
                "shape": list(self.shape)}


def geo_from_tiff_tags(tags, shape) -> TileGeo:
    """Build a TileGeo from tifffile page tags (ModelTransformation or
    ModelPixelScale + ModelTiepoint, and the GeoKeyDirectory for the CRS)."""
    epsg = 4326
    gk = tags.get(34735)
    if gk is not None:
        v = list(gk.value)
        for i in range(4, len(v) - 3, 4):
            key, loc, _cnt, val = v[i:i + 4]
            if key in (3072, 2048) and loc == 0:     # ProjectedCSType / GeographicType
                epsg = int(val)
                if key == 3072:
                    break
    xform = tags.get(34264)
    if xform is not None:
        t = [float(a) for a in xform.value]
        if abs(t[1]) > 1e-12 or abs(t[4]) > 1e-12:
            raise ValueError("rotated GeoTIFF transforms are not supported")
        return TileGeo(epsg, t[3], t[7], abs(t[0]), abs(t[5]), tuple(shape[:2]))
    scale, tie = tags.get(33550), tags.get(33922)
    if scale is None or tie is None:
        raise ValueError("GeoTIFF missing georeferencing tags")
    s = [float(a) for a in scale.value]
    tp = [float(a) for a in tie.value]
    i, j, X, Y = tp[0], tp[1], tp[3], tp[4]
    return TileGeo(epsg, X - i * s[0], Y + j * s[1], abs(s[0]), abs(s[1]), tuple(shape[:2]))


def metres_per_degree(lat0: float) -> tuple[float, float]:
    """(m per degree longitude, m per degree latitude) at lat0 on the WGS84
    ellipsoid (meridional and prime-vertical radii). solar.py's flat 111,320
    m/deg is the equatorial value and reads north-south distances 0.2 to 0.3%
    long at US latitudes; parcel areas use the ellipsoid radii instead so lot
    sizes are not padded."""
    e2 = _F * (2 - _F)
    s = math.sin(math.radians(lat0))
    w = math.sqrt(1 - e2 * s * s)
    M = _A * (1 - e2) / w ** 3
    Nr = _A / w
    return math.radians(1) * Nr * math.cos(math.radians(lat0)), math.radians(1) * M


def local_metres(lats, lngs, lat0: float, lng0: float):
    """Local metres (x east, y north) around (lat0, lng0), equirectangular with
    ellipsoid radii at lat0 (error far below a centimetre over a lot)."""
    lats = np.asarray(lats, float)
    lngs = np.asarray(lngs, float)
    mx, my = metres_per_degree(lat0)
    return (lngs - lng0) * mx, (lats - lat0) * my


def ring_area_m2(ring_latlng) -> float:
    """Area of a lat/lng ring [(lat, lng), ...] on a local metre projection."""
    from shapely.geometry import Polygon

    r = np.asarray(ring_latlng, float)
    lat0, lng0 = float(r[:, 0].mean()), float(r[:, 1].mean())
    x, y = local_metres(r[:, 0], r[:, 1], lat0, lng0)
    return float(Polygon(np.column_stack([x, y])).buffer(0).area)
