"""Data model for roof measurements (EagleView-style)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict


# EagleView-style waste columns
WASTE_COLUMNS = [0, 7, 12, 17, 20, 22, 24, 27, 32]


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

    # ---- derived ----
    @property
    def facet_count(self) -> int:
        return len(self.facets)

    @property
    def squares(self) -> float:
        return round_up_third_square(self.total_area_sqft)

    def area_with_waste(self, waste_pct: float) -> float:
        return self.total_area_sqft * (1.0 + waste_pct / 100.0)

    def squares_with_waste(self, waste_pct: float) -> float:
        return round_up_third_square(self.area_with_waste(waste_pct))

    def waste_table(self) -> list[dict]:
        return [
            {
                "waste_pct": w,
                "area_sqft": round(self.area_with_waste(w)),
                "squares": round(self.squares_with_waste(w), 2),
            }
            for w in WASTE_COLUMNS
        ]

    def suggested_waste_pct(self) -> int:
        """Suggested waste factor from complexity (facets, hips/valleys per square)."""
        sq = max(self.total_area_sqft / 100.0, 1.0)
        cut_up = (self.edges.hips_ft + self.edges.valleys_ft) / sq
        if self.facet_count <= 4 and cut_up < 4:
            return 7
        if self.facet_count <= 10 and cut_up < 8:
            return 12
        if self.facet_count <= 20 and cut_up < 14:
            return 15
        return 17

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
        d = asdict(self)
        d["facet_count"] = self.facet_count
        d["squares"] = self.squares
        d["suggested_waste_pct"] = self.suggested_waste_pct()
        d["waste_table"] = self.waste_table()
        d["steep_area_split"] = self.steep_area_split()
        d["edges"]["ridges_hips_ft"] = self.edges.ridges_hips_ft
        d["edges"]["drip_edge_ft"] = self.edges.drip_edge_ft
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
