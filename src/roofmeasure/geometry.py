"""Roof geometry engine.

Extracts roof facets from a DSM (digital surface model) raster and classifies
every edge as ridge / hip / valley / eave / rake, with slope-corrected lengths.

Pipeline:
  1. Smooth the masked DSM lightly.
  2. Compute per-pixel slope (deg) and aspect (downslope azimuth).
  3. Region-grow pixels into facets where slope and aspect are locally similar.
  4. Merge slivers into their largest neighbor.
  5. Walk facet boundaries:
       - shared boundary between two facets -> ridge / hip / valley
       - boundary against non-roof -> eave / rake (by edge direction vs aspect)
  6. Straight-segment lengths via PCA extent of boundary pixel clusters,
     slope-corrected using elevation change along the segment.

Everything is numpy-only (no scipy / rasterio / shapely).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .model import EdgeSummary, Facet, RoofMeasurements, slope_deg_to_pitch


@dataclass
class EdgeSegment:
    """One classified edge segment, with pixel geometry for diagram rendering."""
    kind: str                  # ridge | hip | valley | eave | rake
    length_ft: float           # slope-corrected where applicable
    pixels: list               # [(y, x), ...]
    mid_yx: tuple              # label anchor
    plan_length_ft: float = 0.0  # plan-view length (for diagram scaling)


@dataclass
class MeasureDetail:
    """Raster-space geometry for diagram rendering."""
    labels: np.ndarray         # facet label raster (0 = background)
    px_x: float
    px_y: float
    edges: list = field(default_factory=list)        # [EdgeSegment]
    facet_centroids: dict = field(default_factory=dict)  # facet letter -> (y, x)
    facet_ids: dict = field(default_factory=dict)         # facet letter -> raster id
    facet_azimuths: dict = field(default_factory=dict)    # facet letter -> azimuth deg
    skeleton: object = None    # RoofSkeleton once built (see skeleton.py)

M_TO_FT = 3.280839895
SQM_TO_SQFT = 10.76391041671