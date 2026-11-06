"""Defensive Battlespace & Engagement Envelope Study.

Sweeps Threat Initial Altitude (45–75 km) vs. Initial Ground Range (25–65 km)
for a threat aimed at the defended origin and maps the KV divert delta-v
required and the successful intercept boundary under Baseline (KF6 + PN) and
Advanced (EKF7 + APN) guidance. Writes docs/figures/engagement_envelope.png.

    python scripts/envelope_study.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.engagement import simulate  # noqa: E402

OUT_FIG = ROOT / "docs" / "figures" / "engagement_envelope.png"
INK, INK_2, INK_3, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"


def _eval_grid(base_cfg, ranges_km, alts_km, seed=7):
    quiet = Console(quiet=True)
    dv_grid = np.zeros((len(alts_km), len(ranges_km)))
    kill_grid = np.zeros((len(alts_km), len(ranges_km)), dtype=bool)
    for ia, alt_km in enumerate(alts_km):
        for ir, rng_km in enumerate(ranges_km):
            sc = replace(
                base_cfg.scenario,
                altitude=(alt_km * 1000.0, alt_km * 1000.0),
                range=(rng_km * 1000.0, rng_km * 1000.0),
                aim_radius=(22000.0, 22000.0),
                impact_time=(85.0, 85.0),
            )
            eng = replace(base_cfg.engagement, launch_rule="fire_on_track", min_track_time=5.0)
            cfg = replace(base_cfg, scenario=sc, engagement=eng)
            h = simulate(cfg, seed, console=quiet)
            m = h["meta"]
            dv_grid[ia, ir] = float(m["divert_dv_used"])
            kill_grid[ia, ir] = m["outcome"] == "KILL"
    return dv_grid, kill_grid


def main():
    ranges_km = np.linspace(25.0, 62.0, 8)
    alts_km = np.linspace(45.0, 75.0, 7)
    cfg_base = load_config(ROOT / "configs" / "engineering.yaml")
    cfg_adv = load_config(ROOT / "configs" / "advanced.yaml")

    dv_base, kill_base = _eval_grid(cfg_base, ranges_km, alts_km)
    dv_adv, kill_adv = _eval_grid(cfg_adv, ranges_km, alts_km)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), facecolor=SURFACE, sharey=True)
    R, A = np.meshgrid(ranges_km, alts_km)

    for ax, dv, kill, title in (
        (axes[0], dv_base, kill_base, "a) Baseline (KF6 + Classical PN) — Divert Δv & Kill Zone"),
        (axes[1], dv_adv, kill_adv, "b) Advanced (EKF7 + Augmented PN) — Expanded Battlespace"),
    ):
        ax.set_facecolor(SURFACE)
        cf = ax.contourf(R, A, dv, levels=np.linspace(80, 440, 19), cmap="YlOrRd", extend="both")
        # Mark every grid point with a hit (circle) or divert-exhaustion miss (x)
        for ia in range(len(alts_km)):
            for ir in range(len(ranges_km)):
                if kill[ia, ir]:
                    ax.plot(ranges_km[ir], alts_km[ia], "o", color="#1b7837", ms=4.5, alpha=0.85)
                else:
                    ax.plot(ranges_km[ir], alts_km[ia], "x", color="#b2182b", ms=5.5, mew=1.4)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=9, fontweight="semibold")
        ax.set_xlabel("threat initial ground range  [km]", color=INK_2, fontsize=9.5)
        ax.tick_params(colors=INK_2, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("threat initial altitude  [km]", color=INK_2, fontsize=9.5)
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("KV divert Δv consumed  [m/s]  (budget = 435 m/s)", color=INK_2, fontsize=9)
    cbar.ax.tick_params( labelsize=8, colors=INK_2)

    fig.suptitle(
        f"Defensive battlespace footprint: Baseline kills {kill_base.sum()}/{kill_base.size} grid points vs. "
        f"Advanced kills {kill_adv.sum()}/{kill_adv.size} (green ● = KILL, red × = MISS)",
        x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold",
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {OUT_FIG} (Baseline {kill_base.sum()}/{kill_base.size} kills, Advanced {kill_adv.sum()}/{kill_adv.size} kills)")


if __name__ == "__main__":
    main()
