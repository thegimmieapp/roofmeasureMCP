"""Small JSON map contract for landscape scope overlays, in local feet.

Raster edges and vector coordinates share the same origin. No assessor
attributes, provider logs, source URLs or credentials are included here.
"""
from __future__ import annotations

import numpy as np
from .lot import M_TO_FT, polygons_of


def scope_map(result: dict) -> dict:
    p = result["_plan"]
    r0, c0 = p["win"]
    h, w = p["shape"]
    sx, sy = p["px"]
    left, top = (c0 - 0.5) * sx, -(r0 - 0.5) * sy

    def point(x, y):
        return [round((float(x) - left) * M_TO_FT, 3),
                round((top - float(y)) * M_TO_FT, 3)]

    def polygons(geometry):
        if geometry is None or geometry.is_empty:
            return []
        return [[ [point(x, y) for x, y in ring.coords]
                  for ring in [poly.exterior, *poly.interiors]]
                for poly in polygons_of(geometry)]

    return {
        "version": 1, "units": "ft", "origin": "raster_top_left", "y_axis": "south",
        "north": "UTM grid north; approximately true north",
        "width_ft": round(w * sx * M_TO_FT, 3),
        "height_ft": round(h * sy * M_TO_FT, 3),
        "parcel": polygons(p["lot_poly"]), "region": polygons(p["region"]),
        "house": polygons(p["house_poly"]) if result.get("house") else [],
        "zones": {name: {"polygons": polygons(z["poly"]),
                          "clear_rectangle": [point(x, y) for x, y in z["rect"]["corners_xy"]]
                          if z.get("rect") else []}
                  for name, z in p["zones"].items()},
        "trees": [{"center": point(*t["_xy"]), "radius_ft": round(t["_r"] * M_TO_FT, 2),
                   "height_ft": t["height_ft"], "near_house": t["within_3m_of_house"]}
                  for t in p["trees"]],
    }


def save_aerial(result: dict, path: str) -> bool:
    """Export the cropped raster without plot margins, preserving map alignment."""
    from PIL import Image
    rgb = result["_plan"].get("rgb")
    if rgb is None:
        return False
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1 else arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr[:, :, :3]).save(path)
    return True
