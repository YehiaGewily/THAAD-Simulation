"""Colour-vision and contrast checks for the viewer and dashboard palettes.

    python scripts/palette_check.py

Reads the colours from their sources (viewer/src/replay/palette.js, the CSS
tokens in viewer/src/replay/styles.css and viewer/src/dashboard/styles.css)
and checks, for each group of colours that must be told apart:

* CVD separation: the OKLab distance (x100) between every pair after
  simulating protanopia and deuteranopia with the Machado, Oliveira and
  Fernandes (2009) severity-1.0 matrices, against a target of 8 and a floor
  of 6 (tritanopia is reported). Every pair counts, since the scene shows the
  colours together rather than as neighbours in a legend.
* Normal-vision separation: the same distance without simulation, floor 15.
* Contrast: the WCAG ratio of each mark against every background it is drawn
  on (3:1 for marks), and of each text token against its surfaces (4.5:1 for
  body text).

Backgrounds are the colours the scene actually renders behind the marks in
each style, sampled from screenshots of the viewer (sky at the horizon, the
terrain and the HUD panels). Exits 1 if a hard check fails.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PALETTE_JS = ROOT / "viewer" / "src" / "replay" / "palette.js"
VIEWER_CSS = ROOT / "viewer" / "src" / "replay" / "styles.css"
DASHBOARD_CSS = ROOT / "viewer" / "src" / "dashboard" / "styles.css"

CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR = 8.0, 6.0, 15.0
MARK_CONTRAST, TEXT_CONTRAST = 3.0, 4.5

# Machado, Oliveira & Fernandes (2009), severity 1.0, linear RGB
MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}

# Rendered backgrounds behind the scene's marks: medians of screenshot regions
# of the viewer (engineering seed 7, T+60 s, cinematic and top-down cameras):
# sky near the horizon, terrain seen from above, the defended area (dome over
# terrain), the brighter half of the cloud deck (cinematic only) and the HUD
# glass.
SCENE_BACKGROUNDS = {
    "scientific": {"sky": "#060f1f", "terrain": "#2a2d32", "defended area": "#354948", "panel": "#111419"},
    "cinematic": {"sky": "#070f20", "terrain": "#2a2e33", "defended area": "#354948",
                  "cloud deck": "#434345", "panel": "#111419"},
}


def hex_rgb(h):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def linear(h):
    c = hex_rgb(h)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(lin_rgb):
    r, g, b = np.clip(lin_rgb, 0, 1)
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def delta_e(a, b, kind=None):
    la, lb = linear(a), linear(b)
    if kind:
        la, lb = MACHADO[kind] @ la, MACHADO[kind] @ lb
    return 100 * float(np.linalg.norm(oklab(la) - oklab(lb)))


def luminance(h):
    return float(np.dot([0.2126, 0.7152, 0.0722], linear(h)))


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def js_object(src, name):
    body = re.search(rf"export const {name} = \{{(.*?)\}};", src, re.S).group(1)
    return dict(re.findall(r"(\w+):\s*'(#[0-9a-fA-F]{6})'", body))


def css_tokens(src, selector_start=0, selector_end=None):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", src[selector_start:selector_end]))


class Report:
    def __init__(self):
        self.failures = 0

    def section(self, title):
        print(f"\n## {title}\n")

    def pairs(self, colors):
        """All-pairs CVD and normal-vision separation for {name: hex}."""
        print("| pair | normal | protan | deutan | tritan | result |")
        print("| --- | ---: | ---: | ---: | ---: | --- |")
        for (na, a), (nb, b) in itertools.combinations(colors.items(), 2):
            n = delta_e(a, b)
            p, d, t = (delta_e(a, b, k) for k in ("protan", "deutan", "tritan"))
            worst = min(p, d)
            if n < NORMAL_FLOOR or worst < CVD_FLOOR:
                verdict = "FAIL"
                self.failures += 1
            elif worst < CVD_TARGET:
                verdict = "warn: needs a second cue"
            else:
                verdict = "pass"
            print(f"| {na} {a} / {nb} {b} | {n:.1f} | {p:.1f} | {d:.1f} | {t:.1f} | {verdict} |")

    def contrasts(self, colors, backgrounds, minimum):
        names = list(backgrounds)
        print("| colour | " + " | ".join(f"vs {n} {backgrounds[n]}" for n in names) + " | result |")
        print("| --- | " + " | ".join("---:" for _ in names) + " | --- |")
        for name, c in colors.items():
            ratios = [contrast(c, backgrounds[n]) for n in names]
            ok = min(ratios) >= minimum
            self.failures += not ok
            print(f"| {name} {c} | " + " | ".join(f"{r:.2f}" for r in ratios)
                  + f" | {'pass' if ok else f'FAIL (< {minimum}:1)'} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pal_src = PALETTE_JS.read_text(encoding="utf-8")
    palette, series = js_object(pal_src, "PALETTE"), js_object(pal_src, "SERIES")
    viewer_css = css_tokens(VIEWER_CSS.read_text(encoding="utf-8"))
    dash_src = DASHBOARD_CSS.read_text(encoding="utf-8")
    dark_at = dash_src.index("@media (prefers-color-scheme: dark)")
    dash = {"light": css_tokens(dash_src, 0, dark_at), "dark": css_tokens(dash_src, dark_at)}

    r = Report()
    print("# Palette check")
    print(f"\nCVD: OKLab ΔE×100 under Machado (2009) severity 1.0; target ≥ {CVD_TARGET:g}, "
          f"floor {CVD_FLOOR:g} (worst of protan/deutan). Normal vision floor {NORMAL_FLOOR:g}. "
          f"Contrast: WCAG, marks ≥ {MARK_CONTRAST:g}:1, text ≥ {TEXT_CONTRAST:g}:1.")

    scene = {k: palette[k] for k in ("threat", "kv", "accent", "track", "radar", "cov")}
    r.section("Replay scene: identity colours, all pairs")
    r.pairs(scene)
    for style, bgs in SCENE_BACKGROUNDS.items():
        r.section(f"Replay scene, {style} style: marks against the rendered background")
        r.contrasts(scene, bgs, MARK_CONTRAST)

    # Threat and kv share figures (altitude, speed, filtered vs raw error); eng
    # is the hue of the single-series figures and never shares a plot
    r.section("Replay analysis figures: series that share a figure")
    r.pairs({k: series[k] for k in ("threat", "kv")})
    r.section("Replay analysis figures: series against the drawer surface")
    r.contrasts(series, {"drawer": viewer_css["surface-solid"] if "surface-solid" in viewer_css else "#0f141b"},
                MARK_CONTRAST)

    r.section("Replay HUD: text tokens against the panels")
    hud_text = {k: viewer_css[k] for k in ("text", "muted", "faint")}
    hud_text.update({f"status {k}": palette[k] for k in ("threat", "kv", "accent", "safe")})
    r.contrasts(hud_text, {"panel": SCENE_BACKGROUNDS["scientific"]["panel"], "page": viewer_css["bg"]},
                TEXT_CONTRAST)

    for mode in ("light", "dark"):
        t = dash[mode]
        r.section(f"Dashboard, {mode}: series")
        r.pairs({"series 1": t["s1"], "series 2": t["s2"]})
        r.contrasts({"series 1": t["s1"], "series 2": t["s2"]}, {"surface": t["surface"]}, MARK_CONTRAST)
        r.section(f"Dashboard, {mode}: text")
        r.contrasts({k: t[k] for k in ("text", "text-2", "muted")},
                    {"surface": t["surface"], "page": t["page"]}, TEXT_CONTRAST)

    print(f"\n{'All hard checks pass.' if not r.failures else f'{r.failures} hard check(s) FAIL.'}")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
