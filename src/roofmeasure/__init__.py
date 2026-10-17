"""RoofMeasure MCP - free satellite roof measurements + Xactimate-style estimating."""

__version__ = "0.4.1"

import sys as _sys

# bpypolyskel (the straight-skeleton engine behind roofgraph) does
# ``import mathutils``, Blender's C extension. It has no wheels for macOS
# arm64 or slim Linux images and broke both the local plugin and the Fly
# build. roofmeasure ships the small pure-Python subset it needs
# (_mathutils.py). Register it under that name unless a real mathutils is
# already loaded in this process (for example inside Blender).
if "mathutils" not in _sys.modules:
    from . import _mathutils as _mu
    _sys.modules["mathutils"] = _mu

from .model import RoofMeasurements, Facet, EdgeSummary  # noqa: E402,F401
