"""Extract EagleView length-diagram vectors (eaves, rakes, ridges/hips, valleys)
from a Premium Report PDF, with the page scale derived from segment labels.

Returns segments in feet (plan view, north up) so they can be overlaid on our
measured outline. EagleView draws: black = eave, green = rake, red = ridge/hip,
blue dashed = valley, brown dashed = flashing.
"""
from __future__ import annotations

import math

import numpy as np
import pdfplumber

COLORS = {
    "0.0": "eave", "(0, 0, 0)": "eave", "(0.0, 0.0, 0.0)": "eave",
    "(0.10196079, 0.60000002, 0.10196079)": "rake",
    "(1.0, 0.0, 0.0)": "ridgehip",
    "(0.0, 0.0, 1.0)": "valley",
    "(0.627451, 0.3764706, 0.0)": "flashing",
}


def _kind(c):
    s = str(c)
    if s in COLORS:
        return COLORS[s]
    try:
        r, g, b = c if len(c) == 3 else (c[0], c[0], c[0])
    except Exception:
        return None
    if r > 0.8 and g < 0.2 and b < 0.2:
        return "ridgehip"
    if g > 0.5 and r < 0.3 and b < 0.3:
        return "rake"
    if b > 0.8 and r < 0.3 and g < 0.3:
        return "valley"
    if r < 0.1 and g < 0.1 and b < 0.1:
        return "eave"
    if r > 0.5 and g > 0.3 and b < 0.1:
        return "flashing"
    return None


def extract(pdf_path: str, page_index: int):
    pdf = pdfplumber.open(pdf_path)
    p = pdf.pages[page_index]
    segs = []
    for l in p.lines:
        k = _kind(l.get("stroking_color"))
        if k is None or l.get("linewidth", 0) < 0.4:
            continue
        if l["top"] < 180 or l["bottom"] > 680:   # diagram band
            continue
        segs.append(dict(kind=k, x0=l["x0"], y0=l["top"], x1=l["x1"], y1=l["bottom"],
                         dashed=bool(l.get("dash")), lw=l.get("linewidth", 0)))
    for c in p.curves:  # some diagonal strokes are stored as curves (2-point)
        k = _kind(c.get("stroking_color"))
        pts = c.get("pts") or c.get("points")
        if k is None or not pts or len(pts) < 2:
            continue
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        if y0 < 180 or y1 > 680:
            continue
        segs.append(dict(kind=k, x0=x0, y0=y0, x1=x1, y1=y1, dashed=bool(c.get("dash")), lw=c.get("linewidth", 0)))
    # labels: integer words inside the band
    labels = []
    for w in p.extract_words():
        t = w["text"].lstrip("+")
        if t.isdigit() and 180 < w["top"] < 680:
            labels.append((int(t), (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2))
    return segs, labels


def scale_pt_per_ft(segs, labels):
    """Match each label to the nearest EAVE segment (level, so plan length =
    labeled length) and take the median points-per-foot."""
    ratios = []
    for val, lx, ly in labels:
        best = None
        for s in segs:
            if s["kind"] != "eave":
                continue
            mx, my = (s["x0"] + s["x1"]) / 2, (s["y0"] + s["y1"]) / 2
            L = math.hypot(s["x1"] - s["x0"], s["y1"] - s["y0"])
            if L < 4:
                continue
            # distance from label to segment line, and along-segment overlap
            ux, uy = (s["x1"] - s["x0"]) / L, (s["y1"] - s["y0"]) / L
            dx, dy = lx - s["x0"], ly - s["y0"]
            t = dx * ux + dy * uy
            d = abs(dx * uy - dy * ux)
            if -3 < t < L + 3 and d < 10:
                if best is None or d < best[0]:
                    best = (d, L)
        if best and val >= 5:
            ratios.append(best[1] / val)
    return float(np.median(ratios)) if ratios else None, len(ratios)


def outline_segments_ft(pdf_path: str, page_index: int):
    segs, labels = extract(pdf_path, page_index)
    s, n = scale_pt_per_ft(segs, labels)
    out = []
    for g in segs:
        if g["kind"] in ("eave", "rake"):
            out.append(dict(kind=g["kind"], x0=g["x0"] / s, y0=g["y0"] / s, x1=g["x1"] / s, y1=g["y1"] / s))
    return out, s, n, segs, labels


if __name__ == "__main__":
    import sys
    segs, labels = extract(sys.argv[1], int(sys.argv[2]))
    s, n = scale_pt_per_ft(segs, labels)
    from collections import Counter
    print("segments by kind:", Counter(g["kind"] for g in segs))
    print(f"scale {s:.3f} pt/ft from {n} eave labels")
    tot = Counter()
    for g in segs:
        tot[g["kind"]] += math.hypot(g["x1"] - g["x0"], g["y1"] - g["y0"]) / s
    print("plan totals ft:", {k: round(v) for k, v in tot.items()})
