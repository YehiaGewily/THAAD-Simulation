"""Convergence study for the integrators: error vs dt on a log-log plot.

Runs the reference problems in thaad_sim.verification (a dragged ballistic
threat, and a vertical interceptor launch through burnout) at dt, dt/2, dt/4
and dt/8 with each integrator, measures the final-position error against a
fine RK4 run, and plots it against slope-1 and slope-4 reference lines.

It also plots the analytic check behind tests/test_integrators.py: with drag
off the threat has a closed-form solution, which RK4 reproduces to rounding
and from which forward Euler's altitude departs by exactly g t dt / 2.

    pip install -e ".[plot]"
    python scripts/convergence_study.py
    # writes docs/figures/convergence.png, docs/figures/analytic.png and results/convergence.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.verification import (  # noqa: E402
    G, INTERCEPTOR_DTS, THREAT_DTS, THREAT_P0, THREAT_V0, ballistic_closed_form,
    interceptor_convergence, observed_orders, propagate_threat, threat_convergence,
)

DEFAULT_OUT = ROOT / "docs" / "figures" / "convergence.png"
DEFAULT_ANALYTIC_OUT = ROOT / "docs" / "figures" / "analytic.png"
DEFAULT_JSON = ROOT / "results" / "convergence.json"
ANALYTIC_DT = 0.1
DEFAULT_CONFIG = ROOT / "configs" / "demo.yaml"

# Categorical slots 1 and 2 of the reference palette, with text and grid inks
COLORS = {"euler": "#2a78d6", "rk4": "#eb6834"}
LABELS = {"euler": "Forward Euler", "rk4": "RK4"}
SLOPES = {"euler": 1, "rk4": 4}
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
REF_OFFSET = 0.2


def _panel(ax, dts, errors, title):
    dts = np.asarray(dts)
    for method in ("euler", "rk4"):
        e = errors[method]
        order = observed_orders(e, dts)[-1]
        ax.loglog(dts, e, color=COLORS[method], lw=2, marker="o", ms=7,
                  mec=SURFACE, mew=1.5, label=f"{LABELS[method]} (observed order {order:.2f})",
                  zorder=3)
        # Reference slope through the coarsest point, shifted down so it does not hide the data
        ref = REF_OFFSET * e[0] * (dts / dts[0]) ** SLOPES[method]
        ax.loglog(dts, ref, color=INK_3, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"slope {SLOPES[method]}", (dts[-1], ref[-1]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color=INK_2)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("time step  dt  [s]", color=INK_2)
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.minorticks_off()
    ax.set_xlim(dts[-1] / 1.4, dts[0] * 1.4)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center")


def analytic_errors(dt=ANALYTIC_DT):
    """Drag-free threat: altitude error of each integrator against the closed form."""
    out = {}
    for method in ("euler", "rk4"):
        t, states = propagate_threat(method, dt, drag=0.0)
        p, _ = ballistic_closed_form(THREAT_P0, THREAT_V0, G, t)
        out[method] = states[:, 2] - p[:, 2]
    return t, out


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)


def analytic_figure(t, err, dt, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ax.plot(t, err["euler"], color=COLORS["euler"], lw=2, label="Forward Euler, simulated", zorder=3)
    ax.plot(t, 0.5 * G * dt * t, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4,
            label="predicted truncation error  g·t·dt/2")
    ax.set_title(f"Forward Euler, dt = {dt:g} s: altitude error", loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel("altitude minus closed form  [m]", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax = axes[1]
    _style(ax)
    e = np.abs(err["rk4"])
    ax.semilogy(t[1:], np.maximum(e[1:], 1e-16), color=COLORS["rk4"], lw=1.2, zorder=3,
                label=f"RK4, |error| (max {e.max():.1e} m)")
    ax.axhline(np.finfo(float).eps * np.abs(THREAT_P0[2]), color=INK_3, lw=1, ls=(0, (4, 3)),
               label="machine epsilon × 60 km")
    ax.set_ylim(1e-13, 1e-3)
    ax.set_title(f"RK4, dt = {dt:g} s: altitude error", loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    for ax in axes:
        ax.set_xlabel("time  [s]", color=INK_2)
        ax.set_xlim(t[0], t[-1])
    fig.suptitle("Analytic check: drag-free ballistic threat against its closed-form solution",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out}")


def write_json(path, studies, t, err, dt):
    """Errors and observed orders behind the figures, read by python -m thaad_sim.dashboard."""
    data = {"convergence": [], "analytic": {
        "dt": dt, "t": np.round(t, 3).tolist(),
        "euler": [float(f"{v:.6g}") for v in err["euler"]],
        "euler_predicted": [float(f"{v:.6g}") for v in 0.5 * G * dt * t],
        "euler_max_dev": float(np.max(np.abs(err["euler"] - 0.5 * G * dt * t))),
        "rk4_max_abs": float(np.max(np.abs(err["rk4"]))),
    }}
    for key, (dts, errors, title) in zip(("threat", "interceptor"), studies):
        data["convergence"].append({
            "problem": key, "title": title, "dts": list(dts),
            **{m: {"errors": [float(f"{v:.6g}") for v in errors[m]],
                   "orders": [round(float(o), 3) for o in observed_orders(errors[m], dts)]}
               for m in ("euler", "rk4")},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG path")
    parser.add_argument("--analytic-out", type=Path, default=DEFAULT_ANALYTIC_OUT,
                        help="output PNG path of the analytic check")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON,
                        help="where to write the errors for the dashboard")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="config whose interceptor flies the burnout problem")
    args = parser.parse_args(argv)

    spec = load_config(args.config).interceptor
    studies = [
        (THREAT_DTS, threat_convergence(), "Ballistic threat with drag, 40 s"),
        (INTERCEPTOR_DTS, interceptor_convergence(spec),
         f"Interceptor boost through burnout ({spec.burn_time:.2f} s), 12.8 s"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (dts, errors, title) in zip(axes, studies):
        ax.set_facecolor(SURFACE)
        _panel(ax, dts, errors, title)
        for method in ("euler", "rk4"):
            orders = ", ".join(f"{p:.2f}" for p in observed_orders(errors[method], dts))
            print(f"{title}: {method:5s} errors {np.array2string(errors[method], precision=3)}"
                  f"  orders [{orders}]")
    axes[0].set_ylabel("final position error  [m]", color=INK_2)
    fig.suptitle("Integrator convergence: error vs dt (reference: RK4 at dt/128)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")

    t, err = analytic_errors()
    print(f"Analytic check, dt={ANALYTIC_DT}: Euler max |error - g t dt/2| "
          f"{np.max(np.abs(err['euler'] - 0.5 * G * ANALYTIC_DT * t)):.2e} m, "
          f"RK4 max |error| {np.max(np.abs(err['rk4'])):.2e} m")
    analytic_figure(t, err, ANALYTIC_DT, args.analytic_out)
    write_json(args.json, studies, t, err, ANALYTIC_DT)


if __name__ == "__main__":
    main()
