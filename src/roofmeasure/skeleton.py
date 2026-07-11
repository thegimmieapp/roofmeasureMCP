"""Straight-skeleton roof wireframe.

Builds a geometrically consistent roof model from the regularized footprint:
each eave edge carries a plane rising inward at its measured pitch; the roof
surface is the lower envelope of those planes; the creases where planes meet
are the ridges, hips, and valleys. By construction every ridge is parallel to
its eaves, hips bisect outside corners, valleys sit at inside corners, and all
lines meet at shared nodes, exactly like an EagleView wireframe.

Rake (gable) edges carry no plane: the neighboring eave planes extend to the
rake, so the ridge runs out to the gable end automatically.

The DSM contributes only: the footprint, eave-vs-rake classification, and the
pitch of each eave edge. All wireframe geometry is analytic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

from .geometry import MeasureDetail

M_TO_FT = 3.280839895
NODE_SNAP_M = 1.2         # merge nodes within this distance
MIN_CREASE_FT = 2.0       # drop creases shorter than this
GRID_RES_M = 0.15         # skeleton raster resolution


@dataclass
class SkelEdge:
    kind: str                 # ridge | hip | valley | eave | rake
    p0: tuple                 # (y, x) raster px coords
    p1: tuple
    length_ft: float          # slope-corrected where applicable
    plan_ft: float = 0.0


@dataclass
class RoofSkeleton:
    outline: np.ndarray                     # Nx2 (y, x) px
    edges: list = field(default_factory=list)          # [SkelEdge] perimeter + creases
    face_regions: np.ndarray | None = None  # region raster (edge index per px, -1 outside)
    face_pitch: dict = field(default_factory=dict)     # edge idx -> pitch str
    face_area_sqft: dict = field(default_factory=dict)  # edge idx -> surface sqft
    face_centroids: dict = field(default_factory=dict)  # edge idx -> (y, x) px
    face_downslope: dict = field(default_factory=dict)   # edge idx -> (dy, dx) unit
    grid_origin: tuple = (0.0, 0.0)         # (y0, x0) px offset of region raster
    grid_step_px: float = 1.0
    totals: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.edges) > 0