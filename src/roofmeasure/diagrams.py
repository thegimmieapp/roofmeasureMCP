"""EagleView-style PDF measurement report with vector (straight-line) diagrams.

Roofs are combinations of straight-edged shapes; all diagrams here are drawn
from vectorized facet polygons (see vectorize.py), never raster pixels.

Pages:
  1. Cover: summary table + aerial image
  2. Length Diagram: color-coded ridges/hips/valleys/rakes/eaves, labeled
  3. Pitch Diagram: facets shaded, pitch labels + slope-direction arrows
  4. Area Diagram: facet letters + areas
  5. Tables: areas per pitch, waste calculation, facet detail
  6. Notes & disclaimer
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from .geometry import MeasureDetail  # noqa: E402
from .model import RoofMeasurements  # noqa: E402
from .vectorize import (classify_outline_edges, classify_polygon_edges,  # noqa: E402
                        facet_polygons, internal_lines, outline_polygon)

EDGE_COLORS = {
    "ridge": "#d62728",   # red
    "hip": "#ff7f0e",     # orange
    "valley": "#1f77b4",  # blue
    "eave": "#2ca02c",    # green
    "rake": "#9467bd",    # purple
}
UNMATCHED_COLOR = "#777777"
MIN_LABEL_FT = 5.0