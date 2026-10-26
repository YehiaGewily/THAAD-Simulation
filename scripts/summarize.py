"""Outcome rates with 95 % Wilson intervals, and summary figures, for a batch CSV.

    python -m thaad_sim.batch --config configs/engineering.yaml --n 1000 --seed 0
    pip install -e ".[plot]"
    python scripts/summarize.py results/engineering.csv

Prints the share of runs ending in each outcome, the share engaged (an
interceptor was launched) and the kill rate among engaged threats, each with
k / n and its 95 % Wilson score interval, then the median and interquartile
range of the key metrics. Writes to docs/figures/:

* <name>_outcomes.png   outcome rates with their 95 % intervals as error bars
* <name>_metrics.png    histograms of the engagement metrics
* <name>_kf_rms.png     distribution of the filter's per-run RMS errors
* <name>_sweep.png      sweep CSVs only: filter RMS error and kill rate against
                        the swept value (with sensor.pos_noise, the filter's
                        RMS error against the measurement noise level)

For a sweep CSV every printed number is per swept value, and the metric and
RMS histograms are left out, since they would pool different configs.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim.stats import wilson_interval  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "docs" / "figures"

OUTCOMES = ("KILL", "GROUND", "NO ENGAGE", "TIMEOUT")

# Categorical slot 1 of the reference palette, sequential blue steps for swept
# values, and text, grid and surface inks
SERIES = "#2a78d6"
RAMP = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b")
INK, INK_2, INK_3, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#ecebe7"

# Axis labels for parameters that are commonly swept; others show as section.field
PARAM_LABELS = {
    "sensor.pos_noise": "radar position noise, 1σ per axis  [m]",
    "sensor.vel_noise": "radar velocity noise, 1σ per axis  [m/s]",
    "kalman.q": "filter process noise q  [m²/s³]",
}

# (column, label, scale, which runs) for the metric histograms
METRICS = (
    ("kill_alt_m", "kill altitude  [km]", 1e-3, "kills"),
    ("closing_velocity_mps", "closing velocity at kill  [m/s]", 1.0, "kills"),
    ("time_to_kill_s", "time from launch to kill  [s]", 1.0, "kills"),
    ("miss_distance_m", "closest approach  [m]  (log scale)", 1.0, "engaged"),
    ("max_g", "peak commanded acceleration  [g]", 1.0, "engaged"),
    ("divert_dv_used_mps", "kill-vehicle divert Δv used  [m/s]", 1.0, "engaged"),
)


# --- data --------------------------------------------------------------------

class Batch:
    """The rows of one batch CSV (or of one swept value), as numpy columns."""

    def __init__(self, rows):
        self.rows = rows
        self.n = len(rows)
        self.outcome = np.array([r['outcome'] for r in rows])
        self.launched = self.col('launched') == 1
        self.kill = self.outcome == "KILL"

    def col(self, key):
        if key == "time_to_kill_s":
            return self.col("kill_time_s") - self.col("launch_time_s")
        return np.array([float(r[key]) if r[key] != "" else np.nan for r in self.rows])

    def seeds(self):
        s = self.col('seed').astype(int)
        return f"{s.min()}-{s.max()}"

    def rates(self):
        """(label, k, n) for each outcome over all runs, the engaged share and kills | engaged."""
        out = [(o, int(np.sum(self.outcome == o)), self.n) for o in OUTCOMES]
        out += [(o, int(np.sum(self.outcome == o)), self.n)
                for o in sorted(set(self.outcome) - set(OUTCOMES))]
        out.append(("engaged", int(np.sum(self.launched)), self.n))
        out.append(("KILL | engaged", int(np.sum(self.kill & self.launched)), int(np.sum(self.launched))))
        return out


def load(path):
    """{group label: Batch}; one group per swept value in file order, or one group (None)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        sweep = 'sweep_param' in reader.fieldnames
    if not rows:
        raise SystemExit(f"{path} has no rows")
    if not sweep:
        return None, {None: Batch(rows)}
    groups = {}
    for r in rows:
        groups.setdefault(r['sweep_value'], []).append(r)
    return rows[0]['sweep_param'], {v: Batch(rs) for v, rs in groups.items()}


# --- printing ----------------------------------------------------------------

def _pct(x):
    return f"{100 * x:5.1f}"


def print_rates(batch, title):
    print(title)
    print(f"  {'':16s} {'k':>5s} / {'n':<5s} {'rate':>6s}   95 % Wilson CI")
    for label, k, n in batch.rates():
        if n == 0:
            print(f"  {label:16s} {k:5d} / {n:<5d}      -")
            continue
        lo, hi = wilson_interval(k, n)
        print(f"  {label:16s} {k:5d} / {n:<5d} {_pct(k / n)} %   [{_pct(lo)}, {_pct(hi)}] %")


def print_metrics(batch):
    print(f"  {'metric':28s} {'runs':>5s} {'median':>10s}   interquartile range")
    for key, sel in (("kf_rms_pos_m", "all"), ("kf_rms_vel_mps", "all"),
                     ("kill_alt_m", "kills"), ("closing_velocity_mps", "kills"),
                     ("time_to_kill_s", "kills"), ("miss_distance_m", "misses"),
                     ("divert_dv_used_mps", "engaged")):
        x = batch.col(key)
        mask = {"all": np.ones(batch.n, bool), "kills": batch.kill,
                "engaged": batch.launched, "misses": batch.launched & ~batch.kill}[sel]
        x = x[mask & np.isfinite(x)]
        if x.size == 0:
            continue
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        print(f"  {key + ' (' + sel + ')':28s} {x.size:5d} {med:10.4g}   [{q1:.4g}, {q3:.4g}]")


# --- figures -----------------------------------------------------------------

def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)


def _save(fig, title, out):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


def _ramp(m):
    """m sequential blues spread over the ramp, light to dark."""
    idx = np.linspace(0, len(RAMP) - 1, m).round().astype(int) if m > 1 else [3]
    return [RAMP[i] for i in idx]


def outcome_figure(groups, param, name, out):
    # Outcomes that never occur are left out of the figure (they are still printed)
    first = next(iter(groups.values()))
    labels = [lab for lab, _, _ in first.rates()
              if lab in ("engaged", "KILL | engaged")
              or any(k for g in groups.values() for l2, k, _ in g.rates() if l2 == lab)]
    m = len(groups)
    width = 0.8 / m
    colors = [SERIES] if m == 1 else _ramp(m)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * len(labels) + 0.25 * m * len(labels)), 4.6),
                           facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(labels))
    for j, (value, batch) in enumerate(groups.items()):
        rates = {lab: (k, n) for lab, k, n in batch.rates()}
        p, lo, hi = [], [], []
        for lab in labels:
            k, n = rates[lab]
            a, b = wilson_interval(k, n) if n else (np.nan, np.nan)
            p.append(k / n if n else np.nan)
            lo.append(a)
            hi.append(b)
        p, lo, hi = (100 * np.array(v) for v in (p, lo, hi))
        xs = x - 0.4 + width * (j + 0.5)
        label = None if param is None else f"{param} = {value}"
        ax.bar(xs, p, width * 0.92, color=colors[j], zorder=2, label=label)
        ax.errorbar(xs, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, elinewidth=1.2,
                    capsize=3, zorder=3)
        if m == 1:
            for xi, lab, pi, hii in zip(xs, labels, p, hi):
                k, n = rates[lab]
                ax.text(xi, hii + 2, f"{pi:.1f} %\n{k}/{n}", ha="center", va="bottom",
                        fontsize=8.5, color=INK_2)
    # Rates over all runs on the left, the conditional kill rate on the right
    ax.axvline(len(labels) - 1.5, color=INK_3, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(x, [lab.replace(" | ", "\n| ") for lab in labels], color=INK)
    ax.set_ylim(0, 115 if m == 1 else 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("share of runs  [%]   (error bars: 95 % Wilson CI)", color=INK_2)
    ax.text(0.0, 1.0, "share of all runs", transform=ax.transAxes, fontsize=9, color=INK_3,
            va="bottom")
    ax.text(1.0, 1.0, "share of engaged runs", transform=ax.transAxes, fontsize=9, color=INK_3,
            va="bottom", ha="right")
    if m > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=min(m, 5),
                  loc="upper center", bbox_to_anchor=(0.5, -0.17))
    n = first.n
    _save(fig, f"{name}: outcomes of {n} seeded runs{'' if m == 1 else ' per value'} "
               f"(seeds {first.seeds()})", out)


def _hist(ax, x, xlabel, log=False, note=None):
    _style(ax)
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 40)
        ax.set_xscale("log")
    else:
        bins = 40
    ax.hist(x, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    med = np.median(x)
    ax.axvline(med, color=INK, lw=1.2, zorder=3, label=f"median {med:.4g}")
    if note:
        ax.plot([], [], " ", label=note)   # legend-only line of context
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel("runs", color=INK_2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="best")
    return x.size


def metrics_figure(batch, name, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE)
    for ax, (key, label, scale, sel) in zip(axes.flat, METRICS):
        mask = batch.kill if sel == "kills" else batch.launched
        x = batch.col(key)[mask] * scale
        if not np.any(np.isfinite(x)):
            ax.set_visible(False)
            continue
        size = _hist(ax, x, label, log=key == "miss_distance_m")
        ax.set_title(f"{sel}, n = {size}", loc="left", fontsize=10, color=INK_2)
    _save(fig, f"{name}: engagement metrics over {batch.n} seeded runs (seeds {batch.seeds()})", out)


def kf_rms_figure(batch, name, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, key, noise, unit, what in ((axes[0], "kf_rms_pos_m", "pos_noise_m", "m", "position"),
                                       (axes[1], "kf_rms_vel_mps", "vel_noise_mps", "m/s", "velocity")):
        # The raw measurement error is about ten times larger, so it is quoted, not drawn
        x = batch.col(key)
        raw = np.sqrt(3) * np.median(batch.col(noise))
        size = _hist(ax, x, f"RMS 3D {what} error of the Kalman estimate  [{unit}]",
                     note=f"raw radar RMS √3·σ = {raw:.3g} {unit}  "
                          f"({raw / np.nanmedian(x):.1f}× the median)")
        ax.set_title(f"all runs, n = {size}", loc="left", fontsize=10, color=INK_2)
    _save(fig, f"{name}: Kalman filter RMS error per run, over {batch.n} seeded runs "
               f"(seeds {batch.seeds()})", out)


def sweep_figure(groups, param, name, out):
    values = list(groups)
    try:
        xs = np.array([float(v) for v in values])
        numeric = True
    except ValueError:
        xs, numeric = np.arange(len(values), dtype=float), False
    log_x = numeric and xs.min() > 0 and xs.max() / xs.min() >= 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, key, noise, unit, what in ((axes[0], "kf_rms_pos_m", "pos_noise_m", "m", "position"),
                                       (axes[1], "kf_rms_vel_mps", "vel_noise_mps", "m/s", "velocity")):
        _style(ax, "both")
        q = np.array([np.nanpercentile(g.col(key), [25, 50, 75]) for g in groups.values()])
        raw = np.array([np.sqrt(3) * np.median(g.col(noise)) for g in groups.values()])
        ax.fill_between(xs, q[:, 0], q[:, 2], color=BAND, lw=0, zorder=1,
                        label="interquartile range over runs")
        ax.plot(xs, raw, color=INK_3, lw=1.2, ls=(0, (4, 3)), zorder=2,
                label="raw radar RMS, √3·σ")
        ax.plot(xs, q[:, 1], color=SERIES, lw=2, marker="o", ms=6, zorder=3,
                label="Kalman estimate, median per-run RMS")
        ax.set_ylabel(f"RMS 3D {what} error  [{unit}]", color=INK_2)
        if raw.max() >= 10 * raw.min():
            # The swept value scales this noise: show both curves on log axes
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.minorticks_off()
        else:
            ax.set_ylim(0, 1.15 * max(raw.max(), q[:, 2].max()))
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="best")

    ax = axes[2]
    _style(ax, "both")
    k = [int(np.sum(g.kill & g.launched)) for g in groups.values()]
    n = [int(np.sum(g.launched)) for g in groups.values()]
    ci = np.array([wilson_interval(a, b) if b else (np.nan, np.nan) for a, b in zip(k, n)])
    p = np.array([a / b if b else np.nan for a, b in zip(k, n)])
    ax.errorbar(xs, 100 * p, yerr=[100 * (p - ci[:, 0]), 100 * (ci[:, 1] - p)], color=SERIES,
                ecolor=INK, elinewidth=1.2, capsize=3, lw=2, marker="o", ms=6, zorder=3,
                label="kill rate among engaged, 95 % Wilson CI")
    ax.set_ylabel("kills / engaged  [%]", color=INK_2)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")

    for ax in axes:
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(xs, [str(v) for v in values])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(PARAM_LABELS.get(param, param), color=INK_2)
    first = next(iter(groups.values()))
    _save(fig, f"{name}: sweep of {param}, {first.n} seeded runs per value "
               f"(seeds {first.seeds()}, the same at every value)", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="batch CSV written by python -m thaad_sim.batch")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    param, groups = load(args.csv)
    name = args.csv.stem
    for value, batch in groups.items():
        where = "" if param is None else f", {param} = {value}"
        print_rates(batch, f"{args.csv.as_posix()}{where}: {batch.n} runs, seeds {batch.seeds()}")
        print_metrics(batch)
        print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcome_figure(groups, param, name, args.out_dir / f"{name}_outcomes.png")
    if param is None:
        batch = groups[None]
        metrics_figure(batch, name, args.out_dir / f"{name}_metrics.png")
        kf_rms_figure(batch, name, args.out_dir / f"{name}_kf_rms.png")
    else:
        sweep_figure(groups, param, name, args.out_dir / f"{name}_sweep.png")


if __name__ == "__main__":
    main()
