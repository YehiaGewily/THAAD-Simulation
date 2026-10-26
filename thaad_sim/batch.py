"""Batch runs: many seeded engagements in parallel, one CSV row per run.

    python -m thaad_sim.batch --config configs/engineering.yaml --n 1000 --seed 0
    python -m thaad_sim.batch --config configs/engineering.yaml --n 500 --seed 0 \\
        --sweep sensor.pos_noise=5,10,25,50,100

Run k uses seed ``seed + k``, the seed ``python -m thaad_sim --seed`` takes, so
any row can be replayed in the browser. No replays are written. A sweep runs
the same seeds at every value, so the scenarios are paired across values.

The CSV is a function of the config, the seeds and the sweep only: rows come
back in seed order whatever the number of worker processes.
Summarize it with ``python scripts/summarize.py results/<name>.csv``.
"""
import argparse
import csv
import multiprocessing
import os
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.progress import track

from .config import SECTIONS, _coerce, load_config
from .engagement import simulate

DEFAULT_CONFIG = "configs/engineering.yaml"
DEFAULT_OUT_DIR = "results"

# One row per run. Columns that do not apply to a run (kill altitude of a miss,
# divert use of a threat never engaged) are left empty.
COLUMNS = (
    'seed', 'outcome', 'launched',
    'launch_time_s', 'launch_pip_range_m',
    'kill_time_s', 'kill_alt_m', 'closing_velocity_mps',
    'miss_distance_m',       # closest approach since launch (the kill distance for a KILL)
    'max_g', 'divert_dv_used_mps',
    'impact_range_m',        # GROUND / NO ENGAGE: impact point distance from the origin
    'end_time_s',
    'kf_rms_pos_m', 'kf_rms_vel_mps',   # RMS of the 3D filter error over the whole run
    'pos_noise_m', 'vel_noise_mps',     # radar 1-sigma per axis, for comparison with the RMS
)
SWEEP_COLUMNS = ('sweep_param', 'sweep_value')


def _rms(err):
    """RMS of the 3D error norm over the rows of ``err``."""
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def _range(point):
    return None if point is None else float(np.linalg.norm(point[:2]))


def run_metrics(cfg, seed):
    """Run one seeded engagement silently and return its CSV row as a dict."""
    hist = simulate(cfg, seed, console=Console(quiet=True))
    meta = hist['meta']
    kill = meta['kill'] or {}
    launched = meta['launch_time'] is not None
    err = np.asarray(hist['kf_err'])
    return {
        'seed': seed,
        'outcome': meta['outcome'],
        'launched': int(launched),
        'launch_time_s': meta['launch_time'],
        'launch_pip_range_m': _range(meta['launch_pip']),
        'kill_time_s': kill.get('time'),
        'kill_alt_m': kill.get('alt'),
        'closing_velocity_mps': kill.get('closing'),
        'miss_distance_m': meta['min_separation'],
        'max_g': max(meta['g_hist']) if meta['g_hist'] else None,
        'divert_dv_used_mps': meta['divert_dv_used'] if launched else None,
        'impact_range_m': _range(meta['impact']),
        'end_time_s': hist['t'][-1],
        'kf_rms_pos_m': _rms(err[:, :3]),
        'kf_rms_vel_mps': _rms(err[:, 3:]),
        'pos_noise_m': cfg.sensor.pos_noise,
        'vel_noise_mps': cfg.sensor.vel_noise,
    }


def parse_sweep(spec):
    """``'section.field=v1,v2,...'`` -> (param, [values]), each value read as a YAML scalar."""
    param, sep, values = spec.partition("=")
    if not sep or not values.strip():
        raise argparse.ArgumentTypeError(
            f"expected section.field=v1,v2,..., e.g. sensor.pos_noise=5,10,25; got {spec!r}")
    return param.strip(), [yaml.safe_load(v) for v in values.split(",")]


def with_param(cfg, param, value):
    """``cfg`` with one scalar parameter, named ``section.field``, set to ``value``."""
    section, _, name = param.partition(".")
    cls = SECTIONS.get(section)
    if cls is None or name not in {f.name for f in fields(cls)}:
        raise ValueError(f"unknown config parameter {param!r}; "
                         f"use section.field, e.g. sensor.pos_noise")
    return replace(cfg, **{section: replace(getattr(cfg, section), **{name: _coerce(value)})})


def _job(job):
    cfg, seed, extra = job
    return {**extra, **run_metrics(cfg, seed)}


def run_jobs(jobs, workers=None, progress=False):
    """CSV rows for ``(cfg, seed, extra_columns)`` jobs, in job order.

    ``workers`` processes (default: one per CPU) share the jobs; ``workers=1``
    runs them in this process.
    """
    jobs = list(jobs)
    workers = min(workers or os.cpu_count() or 1, len(jobs))

    def show(rows):
        return track(rows, total=len(jobs), description=f"{len(jobs)} runs",
                     console=Console(stderr=True), disable=not progress)

    if workers <= 1:
        return list(show(map(_job, jobs)))
    with multiprocessing.Pool(workers) as pool:
        # imap returns results in job order, so the CSV does not depend on scheduling
        return list(show(pool.imap(_job, jobs, chunksize=2)))


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.9g}"
    return str(value)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row[c]) for c in columns])
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m thaad_sim.batch",
        description="Run N seeded engagements without replays and write one CSV row per run.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"YAML config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--n", type=int, default=1000, help="number of runs (default: 1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="first seed; run k uses seed + k (default: 0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: one per CPU; 1 runs in-process)")
    parser.add_argument("--sweep", type=parse_sweep, metavar="SECTION.FIELD=V1,V2,...",
                        help="repeat the N runs for each value of one config parameter, "
                             "e.g. sensor.pos_noise=5,10,25,50,100")
    parser.add_argument("--name", default=None,
                        help="output name (default: the config's file name, plus _sweep_<param>)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"output folder (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")

    cfg = load_config(args.config)
    name = args.name or Path(args.config).stem
    columns = COLUMNS
    cases = [(cfg, {})]
    if args.sweep:
        param, values = args.sweep
        try:
            cases = [(with_param(cfg, param, v), {'sweep_param': param, 'sweep_value': v})
                     for v in values]
        except (ValueError, TypeError) as e:
            parser.error(str(e))
        columns = SWEEP_COLUMNS + COLUMNS
        if args.name is None:
            name += "_sweep_" + param.replace(".", "_")

    seeds = range(args.seed, args.seed + args.n)
    jobs = [(c, seed, extra) for c, extra in cases for seed in seeds]
    rows = run_jobs(jobs, args.workers, progress=True)
    path = write_csv(Path(args.out_dir) / f"{name}.csv", rows, columns)

    counts = Counter(r['outcome'] for r in rows)
    print(f"Wrote {path.as_posix()}: {len(rows)} runs (seeds {seeds[0]}-{seeds[-1]}"
          f"{f' at {len(cases)} values of {args.sweep[0]}' if args.sweep else ''}), "
          + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"Summarize with: python scripts/summarize.py {path.as_posix()}")


if __name__ == "__main__":
    main()
