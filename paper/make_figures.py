"""Make the paper's data figures from the committed results.

Reads only the JSON summaries in results/ and writes vector PDFs to
paper/figures/. The palette, markers and fonts come from the study scripts
(scripts/ablation_study.py and scripts/sensitivity_study.py), so the paper
figures match the repository figures. Unlike those, the paper figures carry no
suptitle: that text is in the LaTeX captions.

Usage (from the repository root or from paper/):
    python paper/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RESULTS = ROOT / "results"
OUT = PAPER / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

import ablation_study as ablation  # noqa: E402
import sensitivity_study as sensitivity  # noqa: E402

INK, INK_2, INK_3, GRID, SURFACE = ablation.INK, ablation.INK_2, ablation.INK_3, ablation.GRID, ablation.SURFACE
BUDGET_RED = "#b2182b"          # the 435 m/s budget line in the repository figures
BAND = "#ecebe7"                # scripts/ekf7_consistency.py
SERIES, TRUTH = "#2a78d6", "#e8622c"

# Per-configuration colours, as in ablation_study.plot_ablation
CONFIG_IDS = ["C1_base_cart", "C2_base_rae", "C3_ekf7_vac_pn", "C4_ekf7_drag_pn", "C5_kf6_vac_apn",
              "C6_ekf7_vac_apn", "C7_ekf7_drag_apn", "C8_ekf7_drag_zem", "C9_adv_rae"]
CONFIG_COLORS = [ablation.C_BASE, ablation.C_BASE, ablation.C_MID, ablation.C_MID, "#7b68ee",
                 ablation.C_ADV, ablation.C_ADV, ablation.C_ZEM, "#184f95"]
CONFIG_LABELS = [
    "C1  KF6 + vac + PN (baseline)",
    "C2  KF6 + vac + PN, RAE radar",
    "C3  EKF7 + vac + PN",
    "C4  EKF7 + drag + PN",
    "C5  KF6 + vac + PN, no coast g",
    "C6  EKF7 + vac + APN",
    "C7  EKF7 + drag + APN",
    "C8  EKF7 + drag + ZEM",
    "C9  EKF7 + drag + APN, RAE",
]

# The four configurations of the sweeps (scripts/sensitivity_study.py)
SWEEP_IDS = ["C1_base_cart", "C5_kf6_vac_apn", "C7_ekf7_drag_apn", "C9_adv_rae"]
SWEEP_LABELS = {
    "C1_base_cart": "C1 baseline",
    "C5_kf6_vac_apn": "C5 no coast g",
    "C7_ekf7_drag_apn": "C7 EKF7 + drag PIP + APN",
    "C9_adv_rae": "C9 as C7, RAE radar",
}
COLORS, MARKERS = sensitivity.COLORS, sensitivity.MARKERS

COL_W, TEXT_W = 3.5, 7.16      # IEEEtran column and text width, inches

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": SURFACE,
})


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def style(ax, title="", xlabel="", ylabel=""):
    """The repository's axis style (ablation_study._style_ax), at paper font sizes."""
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=4, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=2.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.6)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- Fig. 2
def fig_ablation():
    const = {c["config_id"]: c for c in load("ablation_summary.json")["configs"]}
    expo = {c["config_id"]: c for c in load("ablation_summary_exp.json")["configs"]}
    n = len(CONFIG_IDS)
    y = np.arange(n)[::-1].astype(float)
    h = 0.36
    y_c, y_e = y + h / 2, y - h / 2       # constant density above, exponential below

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.75), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.75]})
    err_kw = dict(fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=4)

    def bars(ax, key_c, key_e):
        ax.barh(y_c, key_c, height=h, color=CONFIG_COLORS, alpha=0.38, edgecolor=CONFIG_COLORS,
                lw=0.6, zorder=3)
        ax.barh(y_e, key_e, height=h, color=CONFIG_COLORS, alpha=0.92, zorder=3)

    # (a) protection rate with Wilson 95 % intervals
    ax = axes[0]
    style(ax, "(a) Protection rate", "threatening threats killed [%]")
    for src, yy in ((const, y_c), (expo, y_e)):
        r = np.array([src[c]["kill_rate_threatening"] for c in CONFIG_IDS])
        lo = np.array([src[c]["ci95_threatening"][0] for c in CONFIG_IDS])
        hi = np.array([src[c]["ci95_threatening"][1] for c in CONFIG_IDS])
        ax.errorbar(r, yy, xerr=[r - lo, hi - r], **err_kw)
    bars(ax, [const[c]["kill_rate_threatening"] for c in CONFIG_IDS],
         [expo[c]["kill_rate_threatening"] for c in CONFIG_IDS])
    for yi, c in zip(y_c, CONFIG_IDS):
        r, hi = const[c]["kill_rate_threatening"], const[c]["ci95_threatening"][1]
        if r < 99.5:
            ax.text(hi + 2.0, yi, f"{r:.1f}", va="center", ha="left", fontsize=6.2, color=INK)
    ax.set_xlim(0, 105)
    ax.set_yticks(y, CONFIG_LABELS, color=INK)
    ax.tick_params(axis="y", length=0)

    # (b) median divert with interquartile range, engaged runs
    ax = axes[1]
    style(ax, "(b) Divert used (median, IQR)", "kill-vehicle divert $\\Delta v$ [m/s]")
    for src, yy in ((const, y_c), (expo, y_e)):
        m = np.array([src[c]["median_dv_mps"] for c in CONFIG_IDS])
        q1 = np.array([src[c]["iqr_dv_mps"][0] for c in CONFIG_IDS])
        q3 = np.array([src[c]["iqr_dv_mps"][1] for c in CONFIG_IDS])
        ax.errorbar(m, yy, xerr=[m - q1, q3 - m], **err_kw)
    bars(ax, [const[c]["median_dv_mps"] for c in CONFIG_IDS], [expo[c]["median_dv_mps"] for c in CONFIG_IDS])
    ax.axvline(435.0, color=BUDGET_RED, ls="--", lw=0.9, zorder=2)
    ax.text(430.0, y[0] + 0.62, "435 m/s budget", ha="right", va="bottom", fontsize=6.2, color=BUDGET_RED)
    for src, yy in ((const, y_c), (expo, y_e)):
        for yi, c in zip(yy, CONFIG_IDS):
            m, q3 = src[c]["median_dv_mps"], src[c]["iqr_dv_mps"][1]
            if m < 300:
                ax.text(q3 + 8.0, yi, f"{m:.0f}", va="center", ha="left", fontsize=5.8, color=INK_2)
            elif m < 430:     # long solid bars: label inside, near the axis
                ax.text(8.0, yi, f"{m:.0f}", va="center", ha="left", fontsize=5.8, color="white",
                        fontweight="semibold", zorder=5)
    ax.set_xlim(0, 470)
    ax.set_ylim(-0.75, n + 0.15)
    ax.set_yticks(y, [""] * n)
    ax.tick_params(axis="y", length=0)

    # (c) false engagements (launches at threats that would land outside the defended area)
    ax = axes[2]
    style(ax, "(c) False engagements", "launches [count]")
    bars(ax, [const[c]["false_engages"] for c in CONFIG_IDS], [expo[c]["false_engages"] for c in CONFIG_IDS])
    for src, yy in ((const, y_c), (expo, y_e)):
        for yi, c in zip(yy, CONFIG_IDS):
            v = src[c]["false_engages"]
            ax.text(v + 2.0, yi, f"{v}", va="center", ha="left", fontsize=5.8, color=INK_2)
    ax.set_xlim(0, 140)
    ax.set_yticks(y, [""] * n)
    ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(axes[1].get_ylim())
    axes[2].set_ylim(axes[1].get_ylim())

    handles = [Patch(facecolor=INK_3, alpha=0.38, edgecolor=INK_3, lw=0.6, label="constant density (upper bar)"),
               Patch(facecolor=INK_3, alpha=0.92, label="exponential atmosphere (lower bar)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.6, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    save(fig, "ablation.pdf")


# ---------------------------------------------------------------- Fig. 3
def sweep_series(entries, cid, key):
    return np.array([e["configs"][cid][key] for e in entries], dtype=float)


def fig_sweeps():
    const, expo = load("sensitivity_summary.json"), load("sensitivity_summary_exp.json")
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 4.0), facecolor=SURFACE, sharey="row")
    titles = {("divert", 0): "(a) Constant density", ("divert", 1): "(b) Exponential atmosphere",
              ("drag", 0): "(c) Constant density", ("drag", 1): "(d) Exponential atmosphere"}

    for col, data in enumerate((const, expo)):
        # protection rate vs divert budget, nominal threat drag
        ax = axes[0, col]
        style(ax, titles[("divert", col)], "kill-vehicle divert budget [m/s]",
              "threatening threats killed [%]" if col == 0 else "")
        entries = data["divert"]
        x = np.array([e["value"] for e in entries])
        for cid in SWEEP_IDS:
            r = sweep_series(entries, cid, "kill_rate_threatening")
            lo = np.array([e["configs"][cid]["ci95_threatening"][0] for e in entries])
            hi = np.array([e["configs"][cid]["ci95_threatening"][1] for e in entries])
            ax.fill_between(x, lo, hi, color=COLORS[cid], alpha=0.14, lw=0)
            ax.plot(x, r, color=COLORS[cid], marker=MARKERS[cid], ms=3.2, lw=1.2, label=SWEEP_LABELS[cid])
        ax.axvline(435.0, color=INK_2, ls="--", lw=0.8)
        ax.axhline(99.0, color=INK_3, ls=":", lw=0.8)
        ax.set_xticks([100, 200, 300, 435, 550])
        ax.set_ylim(-3, 104)

        # false engagements vs true threat drag, nominal budget, as a share of non-threatening seeds
        ax = axes[1, col]
        style(ax, titles[("drag", col)], "true threat drag $k_T$ [$10^{-5}$ m$^{-1}$]",
              "non-threatening threats engaged [%]" if col == 0 else "")
        entries = data["drag"]
        x = np.array([e["value"] for e in entries]) * 1e5
        n_safe = np.array([data["runs_per_point"] - e["threatening"] for e in entries], dtype=float)
        for cid in SWEEP_IDS:
            if cid == "C1_base_cart":
                continue        # C1 uses the same vacuum PIP as C5 and launches on the same seeds
            fe = sweep_series(entries, cid, "false_engages")
            lab = "C1 = C5 (vacuum PIP)" if cid == "C5_kf6_vac_apn" else SWEEP_LABELS[cid]
            ax.plot(x, 100.0 * fe / n_safe, color=COLORS[cid], marker=MARKERS[cid], ms=3.2, lw=1.2, label=lab)
        ax.axvline(4.0, color=INK_2, ls="--", lw=0.8)
        ax.set_xticks([0, 1, 2, 4, 6, 8])
        ax.set_ylim(-3, 100)

    axes[0, 1].legend(frameon=False, labelcolor=INK, loc="lower right")
    axes[1, 0].legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, "sweeps.pdf")


# ---------------------------------------------------------------- Fig. 4
def fig_noise():
    data = load("sensitivity_noise_summary_exp.json")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), facecolor=SURFACE)
    panels = [("noise_435", "kill_rate_threatening", "(a) Protection, 435 m/s budget",
               "threatening threats killed [%]"),
              ("noise_100", "kill_rate_threatening", "(b) Protection, 100 m/s budget",
               "threatening threats killed [%]"),
              ("noise_435", "median_dv_mps", "(c) Divert used, 435 m/s budget",
               "median divert $\\Delta v$ [m/s]")]
    for ax, (sweep, key, title, ylabel) in zip(axes, panels):
        entries = data[sweep]
        k = np.array([e["value"] for e in entries])
        style(ax, title, "radar noise factor", ylabel)
        for cid in SWEEP_IDS:
            yv = sweep_series(entries, cid, key)
            if key == "kill_rate_threatening":
                lo = np.array([e["configs"][cid]["ci95_threatening"][0] for e in entries])
                hi = np.array([e["configs"][cid]["ci95_threatening"][1] for e in entries])
                ax.fill_between(k, lo, hi, color=COLORS[cid], alpha=0.14, lw=0)
            ax.plot(k, yv, color=COLORS[cid], marker=MARKERS[cid], ms=3.2, lw=1.2, label=SWEEP_LABELS[cid])
        ax.set_xscale("log", base=2)
        ax.set_xticks(k, ["0.25×", "0.5×", "1×", "2×", "4×"])
        ax.minorticks_off()
        ax.axvline(1.0, color=INK_2, ls="--", lw=0.8)
        if key == "median_dv_mps":
            ax.axhline(435.0, color=BUDGET_RED, ls="--", lw=0.9)
            ax.set_ylim(0, 460)
        else:
            ax.set_ylim(-3, 104)
    axes[0].legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "noise.pdf")


# ---------------------------------------------------------------- Fig. 5
def fig_drag_estimate():
    data = load("consistency_ekf7.json")
    t_launch = data["launch_time_s"]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.0), facecolor=SURFACE, sharey=True)
    for ax, atm, title in zip(axes, ("constant", "exponential"),
                              ("(a) Constant density", "(b) Exponential atmosphere")):
        case = data["cases"][f"configured_{atm}_rae"]
        s = case["series"]
        t = np.array(s["t"])
        est, sig, truth = (1e5 * np.array(s[k]) for k in ("est_median", "sigma_median", "truth_median"))
        t_int = data["intercept_time_s"][atm]
        style(ax, title, "time since first detection [s]" if atm == "exponential" else "",
              "$\\hat k_T$ [$10^{-5}$ m$^{-1}$]")
        ax.fill_between(t, est - sig, est + sig, color=SERIES, alpha=0.18, lw=0,
                        label="median estimate ± filter 1σ")
        ax.plot(t, est, color=SERIES, lw=1.2, label="median estimate")
        ax.plot(t, truth, color=TRUTH, lw=1.0, ls=(0, (4, 2)), label="truth")
        ax.axvline(t_launch, color=INK_2, ls=":", lw=0.9)
        ax.axvline(t_int, color=INK_2, ls="--", lw=0.9)
        ax.text(t_launch + 1.0, 6.6, "launch\ndecision", fontsize=6.0, color=INK_2, va="top")
        ax.text(t_int + 1.0, 6.6, "median\nintercept", fontsize=6.0, color=INK_2, va="top")
        ax.set_xlim(0, t_int + 20.0)
        ax.set_ylim(0, 7)
    axes[0].legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9, labelcolor=INK,
                   loc="lower right", fontsize=6.0)
    fig.tight_layout(h_pad=0.6)
    save(fig, "drag_estimate.pdf")


if __name__ == "__main__":
    fig_ablation()
    fig_sweeps()
    fig_noise()
    fig_drag_estimate()
