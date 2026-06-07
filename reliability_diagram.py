#!/usr/bin/env python3
"""Reliability diagram for the matched-pair calibration pilot (results-only).

Pure standard library: reads the eval transcripts, computes per-confidence
empirical accuracy on the factual subset (the ECE basis), and writes an SVG
calibration curve for both arms. No third-party dependencies, no model weights.

    python3 reliability_diagram.py    # writes reliability_diagram.svg

A point below the y = x diagonal is overconfident (stated confidence exceeds
actual accuracy). Point area is proportional to the number of items.
"""
from __future__ import annotations
import json, re
from collections import defaultdict

T = json.load(open("outputs/behavioral_transcripts.json"))["transcripts"]

def _norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def _correct(p, arm):  # plain answer-match; prefer precomputed 'correct' (stripped build),
    if "correct" in p[arm]:                 # fall back to response text match (source build)
        return p[arm]["correct"]
    ca = p.get("correct_answer")
    return bool(ca) and (_norm(ca) in _norm(p[arm]["response"]))

def points(arm):
    byc = defaultdict(lambda: [0, 0])  # conf -> [correct, total]
    for p in T:
        if p["category"] == "factual" and p.get("correct_answer") and p[arm]["confidence"] is not None:
            byc[p[arm]["confidence"]][1] += 1
            if _correct(p, arm):
                byc[p[arm]["confidence"]][0] += 1
    # (predicted %, empirical accuracy %, n)
    return [(c * 10.0, ok / n * 100.0, n) for c, (ok, n) in sorted(byc.items())]

ARMS = [("this method", "#1a66cc", points("method")),
        ("matched baseline", "#cc3311", points("baseline"))]

# ---- SVG geometry ----
W, H = 560, 560
L, R, Ttop, Bot = 78, 28, 64, 78          # margins
pw, ph = W - L - R, H - Ttop - Bot
def px(x): return L + x / 100.0 * pw
def py(y): return (H - Bot) - y / 100.0 * ph
def rad(n): return max(4.0, min(16.0, 2.0 + (n ** 0.5)))

s = []
s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">')
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Reliability diagram (factual subset)</text>')
s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12" fill="#555">below the diagonal = overconfident, above = underconfident; point size = items</text>')

# gridlines + axis ticks
for v in range(0, 101, 20):
    s.append(f'<line x1="{px(v):.1f}" y1="{py(0):.1f}" x2="{px(v):.1f}" y2="{py(100):.1f}" stroke="#eee"/>')
    s.append(f'<line x1="{px(0):.1f}" y1="{py(v):.1f}" x2="{px(100):.1f}" y2="{py(v):.1f}" stroke="#eee"/>')
    s.append(f'<text x="{px(v):.1f}" y="{py(0)+18:.1f}" text-anchor="middle" font-size="11" fill="#333">{v}</text>')
    s.append(f'<text x="{px(0)-10:.1f}" y="{py(v)+4:.1f}" text-anchor="end" font-size="11" fill="#333">{v}</text>')

# axes box + perfect-calibration diagonal
s.append(f'<rect x="{px(0):.1f}" y="{py(100):.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="#999"/>')
s.append(f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(100):.1f}" y2="{py(100):.1f}" stroke="#999" stroke-dasharray="6 4"/>')
s.append(f'<text x="{px(72):.1f}" y="{py(80):.1f}" font-size="10" fill="#999" transform="rotate(-37 {px(72):.1f} {py(80):.1f})">perfect calibration</text>')

# axis titles
s.append(f'<text x="{px(50):.1f}" y="{H-22}" text-anchor="middle" font-size="13">stated confidence (%)</text>')
s.append(f'<text x="22" y="{py(50):.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 22 {py(50):.1f})">empirical accuracy (%)</text>')

# data points
for name, color, pts in ARMS:
    for x, y, n in pts:
        s.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="{rad(n):.1f}" fill="{color}" fill-opacity="0.55" stroke="{color}"/>')

# legend
lx, ly = px(4), py(96)
for i, (name, color, _) in enumerate(ARMS):
    yy = ly + i * 20
    s.append(f'<circle cx="{lx+7:.1f}" cy="{yy:.1f}" r="6" fill="{color}" fill-opacity="0.55" stroke="{color}"/>')
    s.append(f'<text x="{lx+20:.1f}" y="{yy+4:.1f}" font-size="12">{name}</text>')

s.append("</svg>")
open("reliability_diagram.svg", "w").write("\n".join(s))
print("wrote reliability_diagram.svg")
for name, _, pts in ARMS:
    print(f"  {name}: " + ", ".join(f"conf{int(x)}%->{y:.0f}%(n{n})" for x, y, n in pts))
