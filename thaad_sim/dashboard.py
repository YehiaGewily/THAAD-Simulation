"""Batch results dashboard: one self-contained HTML page from a batch CSV.

    python -m thaad_sim.batch --config configs/engineering.yaml --n 1000 --seed 0
    python -m thaad_sim.dashboard results/engineering.csv

The page shows the outcome rates with their 95 % Wilson intervals, histograms
of the engagement metrics and of the Kalman filter's per-run RMS error, and,
when their data files exist, the filter consistency study (average NEES and
NIS against chi-square bounds, results/consistency.json from
scripts/filter_consistency.py) and the integrator convergence study
(results/convergence.json from scripts/convergence_study.py).

Every number is computed here; the template (thaad_sim/viewer/
dashboard_template.html, built from viewer/) only draws it. The page needs no
server and no Python to view.
"""
import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np

from .stats import wilson_interval
from .summary import load

TEMPLATE = Path(__file__).resolve().parent / "viewer" / "dashboard_template.html"
DEFAULT_OUT = Path("site") / "dashboard.html"
RESULTS = Path("results")
BINS = 30

# (column, title, unit, scale, which runs, log x) for the metric histograms
METRICS = (
    ("kill_alt_m", "Kill altitude", "km", 1e-3, "kills", False),
    ("closing_velocity_mps", "Closing velocity at kill", "m/s", 1.0, "kills", False),
    ("time_to_kill_s", "Time from launch to kill", "s", 1.0, "kills", False),
    ("miss_distance_m", "Closest approach", "m", 1.0, "engaged", True),
    ("max_g", "Peak commanded acceleration", "g", 1.0, "engaged", False),
    ("divert_dv_used_mps", "Kill-vehicle divert Δv used", "m/s", 1.0, "engaged", False),
    ("kf_rms_pos_m", "Filter RMS position error", "m", 1.0, "all", False),
    ("kf_rms_vel_mps", "Filter RMS velocity error", "m/s", 1.0, "all", False),
)
NOISE_COLUMN = {"kf_rms_pos_m": "pos_noise_m", "kf_rms_vel_mps": "vel_noise_mps"}
SUBSET_NAMES = {"kills": "kills", "engaged": "engaged runs", "all": "all runs"}


def _num(x, digits=6):
    return float(f"{float(x):.{digits}g}")


def rate_rows(batch):
    rows = []
    for label, k, n in batch.rates():
        if n == 0:
            continue
        lo, hi = wilson_interval(k, n)
        rows.append({"label": label, "k": k, "n": n, "p": _num(k / n), "lo": _num(lo), "hi": _num(hi),
                     "of": "engaged" if label == "KILL | engaged" else "all"})
    return rows


def histogram(x, log, bins=BINS):
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
    if x.size == 0:
        return None
    if log:
        lo, hi = np.log10(x.min()), np.log10(x.max())
        edges = np.logspace(lo, hi if hi > lo else lo + 1, bins + 1)
    else:
        lo, hi = x.min(), x.max()
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    counts, edges = np.histogram(x, edges)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"edges": [_num(e) for e in edges], "counts": counts.tolist(), "n": int(x.size),
            "median": _num(med), "q1": _num(q1), "q3": _num(q3)}


def metric_histograms(batch):
    out = []
    for key, title, unit, scale, which, log in METRICS:
        h = histogram(batch.col(key)[batch.subset(which)] * scale, log)
        if h is None:
            continue
        h.update(key=key, title=title, unit=unit, subset=SUBSET_NAMES[which], log=log)
        if key in NOISE_COLUMN:
            # A single raw radar return has RMS 3D error sqrt(3) sigma
            h["reference"] = {"value": _num(np.sqrt(3) * np.nanmedian(batch.col(NOISE_COLUMN[key]))),
                              "label": "raw radar √3·σ"}
        out.append(h)
    return out


def _read_json(path):
    path = Path(path) if path else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_data(csv_path, consistency=None, convergence=None):
    param, groups = load(csv_path)
    if param is not None:
        raise SystemExit(f"{csv_path} is a sweep over {param}; the dashboard takes a single-config "
                         "batch CSV (summarize sweeps with scripts/summarize.py)")
    batch = groups[None]
    seeds = batch.seed_range()
    return {
        "name": Path(csv_path).stem,
        "source": Path(csv_path).as_posix(),
        "runs": batch.n,
        "seeds": list(seeds),
        "rates": rate_rows(batch),
        "histograms": metric_histograms(batch),
        "consistency": _read_json(consistency),
        "convergence": _read_json(convergence),
        "commands": {
            "batch": f"python -m thaad_sim.batch --config configs/{Path(csv_path).stem}.yaml "
                     f"--n {batch.n} --seed {seeds[0]}",
            "dashboard": f"python -m thaad_sim.dashboard {Path(csv_path).as_posix()}",
            "consistency": "python scripts/filter_consistency.py",
            "convergence": "python scripts/convergence_study.py",
        },
    }


def write_dashboard(data, out):
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DASHBOARD_DATA__", payload)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m thaad_sim.dashboard",
                                     description="Write an HTML dashboard of a batch CSV.")
    parser.add_argument("csv", type=Path, help="batch CSV written by python -m thaad_sim.batch")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"HTML file to write (default: {DEFAULT_OUT.as_posix()})")
    parser.add_argument("--consistency", type=Path, default=RESULTS / "consistency.json",
                        help="NEES/NIS study from scripts/filter_consistency.py (skipped if missing)")
    parser.add_argument("--convergence", type=Path, default=RESULTS / "convergence.json",
                        help="integrator study from scripts/convergence_study.py (skipped if missing)")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)

    data = build_data(args.csv, args.consistency, args.convergence)
    path = write_dashboard(data, args.out)
    missing = [n for n in ("consistency", "convergence") if data[n] is None]
    print(f"Wrote {path}" + (f" (without {' and '.join(missing)}: file not found)" if missing else ""))
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
