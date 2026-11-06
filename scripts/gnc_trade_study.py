"""GNC Trade Study: Baseline (KF6 + Vacuum PIP + PN) vs. Advanced (EKF7 + Drag PIP + RAE + APN).

Evaluates N paired seeds (default 200) across both GNC configurations on the
same physical interceptor (150 kN boost for 5 s, 30 g limit, 435 m/s KV divert
budget, 10 m kill radius) and writes docs/figures/gnc_trade_study.png and
results/gnc_trade_study.json.

    python scripts/gnc_trade_study.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.consistency import monte_carlo, pip_bias  # noqa: E402
from thaad_sim.engagement import simulate  # noqa: E402
from thaad_sim.stats import wilson_interval  # noqa: E402

OUT_FIG = ROOT / "docs" / "figures" / "gnc_trade_study.png"
OUT_JSON = ROOT / "results" / "gnc_trade_study.json"

INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
C_BASE, C_ADV = "#eb6834", "#2a78d6"


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=9, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200, help="paired seeds 0..N-1 (default: 200)")
    parser.add_argument("--out", type=Path, default=OUT_FIG)
    args = parser.parse_args(argv)

    cfg_base = load_config(ROOT / "configs" / "engineering.yaml")
    cfg_adv = load_config(ROOT / "configs" / "advanced.yaml")
    quiet = Console(quiet=True)

    seeds = range(args.runs)
    res = {"base": [], "adv": []}
    for s in seeds:
        for key, cfg in (("base", cfg_base), ("adv", cfg_adv)):
            h = simulate(cfg, s, console=quiet)
            m = h["meta"]
            res[key].append({
                "seed": s,
                "outcome": m["outcome"],
                "launched": m["launch_time"] is not None,
                "dv": float(m["divert_dv_used"]) if m["launch_time"] is not None else np.nan,
                "miss": float(m["min_separation"]) if m["min_separation"] is not None else np.nan,
            })

    summary = {}
    for key, label in (("base", "Baseline (KF6 + Vacuum PIP + PN)"),
                       ("adv", "Advanced (EKF7 + Drag PIP + APN)")):
        rows = res[key]
        eng = [r for r in rows if r["launched"]]
        kills = [r for r in eng if r["outcome"] == "KILL"]
        k, n = len(kills), len(eng)
        p = k / n if n else 0.0
        lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
        dvs = np.array([r["dv"] for r in eng if np.isfinite(r["dv"])])
        summary[key] = {
            "label": label,
            "runs": args.runs,
            "engaged": n,
            "kills": k,
            "kill_rate": round(p * 100, 2),
            "ci95": [round(lo * 100, 2), round(hi * 100, 2)],
            "median_dv_mps": round(float(np.median(dvs)), 1),
            "mean_dv_mps": round(float(np.mean(dvs)), 1),
        }
        print(f"{label}:")
        print(f"  Kill | engaged: {k}/{n} = {p*100:.1f}% (95% CI {lo*100:.1f}–{hi*100:.1f}%)")
        print(f"  Divert delta-v: median {np.median(dvs):.1f} m/s, mean {np.mean(dvs):.1f} m/s")

    # PIP bias comparison over 15 tracking runs
    tgo_bins = np.arange(0.0, 140.0, 5.0)
    tr_base = pip_bias(monte_carlo(cfg_base, range(min(args.runs, 15))), tgo_bins)
    tr_adv = pip_bias(monte_carlo(cfg_adv, range(min(args.runs, 15))), tgo_bins)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), facecolor=SURFACE)

    # Panel 1: Engaged kill rate & 95% Wilson CI
    ax = axes[0]
    _style_ax(ax, "a) Intercept rate among engaged threats", "GNC architecture", "kill probability  [%]")
    x = [0, 1]
    rates = [summary["base"]["kill_rate"], summary["adv"]["kill_rate"]]
    yerr = [
        [rates[0] - summary["base"]["ci95"][0], rates[1] - summary["adv"]["ci95"][0]],
        [summary["base"]["ci95"][1] - rates[0], summary["adv"]["ci95"][1] - rates[1]],
    ]
    bars = ax.bar(x, rates, width=0.48, color=[C_BASE, C_ADV], alpha=0.88, zorder=3)
    ax.errorbar(x, rates, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.5, capsize=5, zorder=4)
    ax.set_xticks(x, ["Baseline\n(KF6 + Vac PIP + PN)", "Advanced\n(EKF7 + Drag PIP + APN)"])
    ax.set_ylim(0, 108)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 5.5, f"{r:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)

    # Panel 2: Kill-vehicle divert delta-v distribution
    ax = axes[1]
    _style_ax(ax, "b) Kill-vehicle divert Δv consumption", "divert Δv spent  [m/s]", "engaged runs")
    dv_base = [r["dv"] for r in res["base"] if r["launched"]]
    dv_adv = [r["dv"] for r in res["adv"] if r["launched"]]
    bins = np.linspace(0, 450, 24)
    ax.hist(dv_base, bins=bins, color=C_BASE, alpha=0.55, label=f"Baseline (med {summary['base']['median_dv_mps']:.0f} m/s)", zorder=2)
    ax.hist(dv_adv, bins=bins, color=C_ADV, alpha=0.65, label=f"Advanced (med {summary['adv']['median_dv_mps']:.0f} m/s)", zorder=3)
    ax.axvline(435.0, color=INK, ls="--", lw=1.1, label="435 m/s divert budget")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")

    # Panel 3: PIP error vs true time-to-go
    ax = axes[2]
    _style_ax(ax, "c) Predicted impact point (PIP) bias", "true time to impact  [s]", "median |PIP error|  [m]")
    tgo = tr_base.tgo
    med_b = np.nanmedian(np.hypot(tr_base.along, tr_base.cross), axis=1)
    med_a = np.nanmedian(np.hypot(tr_adv.along, tr_adv.cross), axis=1)
    valid = np.isfinite(med_b) & np.isfinite(med_a)
    ax.plot(tgo[valid], med_b[valid], color=C_BASE, lw=2.0, label="Vacuum PIP (KF6)")
    ax.plot(tgo[valid], med_a[valid], color=C_ADV, lw=2.0, label="Drag-integrated PIP (EKF7)")
    ax.set_yscale("log")
    ax.set_xlim(tgo[valid].max() + 2, 0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")

    fig.suptitle(f"Closing the GNC model-mismatch loop ({args.runs} paired seeds, identical 435 m/s KV divert budget)",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} and {OUT_JSON}")


if __name__ == "__main__":
    main()
