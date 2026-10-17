"""Data model for roof measurements (EagleView-style)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict


# EagleView waste table: 0% plus an 8-column window of offsets from the
# suggested factor (negative offsets dropped when they would go <= 0).
WASTE_LADDER = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
WASTE_MIN_PITCH_RISE = 3   # waste table counts only roof area of 3/12 and steeper
# Eaves and rakes are the least certain lines (outline edges under trees and
# overhangs, rake vs eave classification), and under-ordering drip edge and
# starter costs more than over-ordering. Reported eave and rake totals are
# the measured lengths scaled up by this factor; the measured lengths are
# kept alongside and the margin is disclosed in the report notes.
EAVE_RAKE_SAFETY = 1.15


def round_up_third_square(sqft: float) -> float:
    """Squares rounded UP to the nearest 1/3 square (EagleView convention)."""
    return math.ceil((sqft / 100.0) * 3.0) / 3.0


def pitch_to_ratio(pitch_str: str) -> float:
    """'8/12' -> 8/12 as float rise/run."""
    rise, run = pitch_str.split("/")
    return float(rise) / float(run)


def slope_deg_to_pitch(slope_deg: float) -> str:
    """Slope in degrees -> nearest x/12 pitch string."""
    rise = round(math.tan(math.radians(slope_deg)) * 12)
    rise = max(0, min(rise, 24))
    return f"{rise}/12"


@dataclass
class Facet:
    label: str
    plan_area_sqft: float          # footprint (plan-view) area
    surface_area_sqft: float       # true sloped surface area
    pitch: str                     # e.g. "8/12"
    slope_deg: float
    azimuth_deg: float             # downslope compass direction
    mean_height_ft: float = 0.0


@dataclass
class EdgeSummary:
    ridges_ft: float = 0.0
    hips_ft: float = 0.0
    valleys_ft: float = 0.0
    rakes_ft: float = 0.0
    eaves_ft: float = 0.0
    flashing_ft: float = 0.0        # wall flashing (horizontal wall junctions)
    step_flashing_ft: float = 0.0   # sloped wall junctions
    eaves_measured_ft: float = 0.0  # before the ordering margin (0 = margin not applied)
    rakes_measured_ft: float = 0.0
    safety_factor: float = 1.0      # factor applied to eaves_ft / rakes_ft
    ridge_count: int = 0
    hip_count: int = 0
    valley_count: int = 0
    rake_count: int = 0
    eave_count: int = 0

    @property
    def ridges_hips_ft(self) -> float:
        return self.ridges_ft + self.hips_ft

    @property
    def drip_edge_ft(self) -> float:
        return self.eaves_ft + self.rakes_ft

    @property
    def perimeter_ft(self) -> float:
        return self.eaves_ft + self.rakes_ft

    def apply_safety(self, factor: float = EAVE_RAKE_SAFETY) -> None:
        """Scale eaves and rakes up by `factor` once, keeping the measured values."""
        if self.safety_factor != 1.0:
            return
        self.eaves_measured_ft = self.eaves_ft
        self.rakes_measured_ft = self.rakes_ft
        self.eaves_ft = round(self.eaves_ft * factor)
        self.rakes_ft = round(self.rakes_ft * factor)
        self.safety_factor = factor


@dataclass
class RoofMeasurements:
    address: str
    latitude: float
    longitude: float
    total_area_sqft: float          # sloped surface area, all facets
    facets: list[Facet] = field(default_factory=list)
    edges: EdgeSummary = field(default_factory=EdgeSummary)
    predominant_pitch: str = "0/12"
    areas_per_pitch: dict[str, float] = field(default_factory=dict)  # pitch -> sqft
    imagery_date: str = ""
    imagery_quality: str = ""       # HIGH / MEDIUM / LOW / BASE
    method: str = "dsm"             # "dsm" (measured) or "segments" (estimated)
    notes: list[str] = field(default_factory=list)
    structure_index: int = 1        # 1 = primary structure
    structures: list = field(default_factory=list)  # per-structure RoofMeasurements (when > 1)
    nearby_structures: list = field(default_factory=list)  # descriptions of detached roofs not included
    confidence: str = ""            # HIGH / MEDIUM / LOW (tree canopy exposure)
    canopy_fraction: float = 0.0    # share of roof perimeter bordered by canopy
    google_roof_area_sqft: float = 0.0  # Google buildingInsights whole-roof area (cross-check)

    # ---- derived ----
    @property
    def facet_count(self) -> int:
        return len(self.facets)

    @property
    def squares(self) -> float:
        return round_up_third_square(self.total_area_sqft)

    def steep_area_sqft(self) -> float:
        """Roof area of 3/12 pitch and steeper (what the waste table is based on)."""
        tot = 0.0
        for pitch, area in self.areas_per_pitch.items():
            if int(pitch.split("/")[0]) >= WASTE_MIN_PITCH_RISE:
                tot += area
        return tot if self.areas_per_pitch else self.total_area_sqft

    def low_slope_area_sqft(self) -> float:
        return max(self.total_area_sqft - self.steep_area_sqft(), 0.0)

    def area_with_waste(self, waste_pct: float) -> float:
        return self.steep_area_sqft() * (1.0 + waste_pct / 100.0)

    def squares_with_waste(self, waste_pct: float) -> float:
        return round_up_third_square(self.area_with_waste(waste_pct))

    def waste_columns(self) -> list[int]:
        sug = self.suggested_waste_pct()
        vals = [sug + off for off in WASTE_LADDER if sug + off > 0]
        return [0] + vals[:8]

    def waste_table(self) -> list[dict]:
        sug = self.suggested_waste_pct()
        return [
            {
                "waste_pct": w,
                "area_sqft": round(self.area_with_waste(w)),
                "squares": round(self.squares_with_waste(w), 2),
                "suggested": w == sug,
                "measured": w == 0,
            }
            for w in self.waste_columns()
        ]

    def suggested_waste_pct(self) -> int:
        """Suggested shingle waste factor.

        Linear model fitted to EagleView suggested-waste values on ten
        structures (cut-up length per square, size, and facets per square):
        reproduces EagleView within 1-4 points on the calibration set.
        """
        sq = max(self.steep_area_sqft() / 100.0, 0.5)
        cut_up = (self.edges.hips_ft + self.edges.valleys_ft) / sq
        w = 2.86 + 1.338 * cut_up + 0.365 * (100.0 / sq) + 6.574 * (self.facet_count / sq)
        return int(max(1, min(round(w), 60)))

    def complexity(self) -> str:
        """Structure complexity as EagleView rates it: Simple / Normal / Complex."""
        if self.facet_count <= 3:
            return "Simple"
        if self.facet_count >= 20:
            return "Complex"
        return "Normal"

    def steep_area_split(self) -> dict[str, float]:
        """Surface sqft at <8/12, 8/12-9/12, and >9/12 (for steep charges)."""
        out = {"normal": 0.0, "steep_8_9": 0.0, "steep_gt9": 0.0}
        for pitch, area in self.areas_per_pitch.items():
            rise = int(pitch.split("/")[0])
            if rise >= 10:
                out["steep_gt9"] += area
            elif rise >= 8:
                out["steep_8_9"] += area
            else:
                out["normal"] += area
        return out

    def to_dict(self) -> dict:
        subs = self.structures
        self.structures = []
        try:
            d = asdict(self)
        finally:
            self.structures = subs
        if subs:
            d["structures"] = [sm.to_dict() for sm in subs]
        d["facet_count"] = self.facet_count
        d["squares"] = self.squares
        d["suggested_waste_pct"] = self.suggested_waste_pct()
        d["waste_table"] = self.waste_table()
        d["complexity"] = self.complexity()
        d["steep_area_sqft"] = round(self.steep_area_sqft())
        d["steep_area_split"] = self.steep_area_split()
        d["edges"]["ridges_hips_ft"] = self.edges.ridges_hips_ft
        d["edges"]["drip_edge_ft"] = self.edges.drip_edge_ft
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
