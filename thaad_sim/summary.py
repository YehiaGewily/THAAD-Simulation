"""Reading batch CSVs: outcome counts and metric columns.

Shared by scripts/summarize.py (printed tables and matplotlib figures) and
thaad_sim/dashboard.py (the HTML dashboard). No plotting dependencies.
"""
import csv

import numpy as np

OUTCOMES = ("KILL", "GROUND", "NO ENGAGE", "TIMEOUT")


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

    def seed_range(self):
        s = self.col('seed').astype(int)
        return int(s.min()), int(s.max())

    def seeds(self):
        lo, hi = self.seed_range()
        return f"{lo}-{hi}"

    def subset(self, which):
        """Boolean mask of the runs in ``which``: all, kills, engaged or misses (engaged, not killed)."""
        return {"all": np.ones(self.n, bool), "kills": self.kill, "engaged": self.launched,
                "misses": self.launched & ~self.kill}[which]

    def rates(self):
        """(label, k, n) for each outcome over all runs, the engaged share and kills | engaged."""
        out = [(o, int(np.sum(self.outcome == o)), self.n) for o in OUTCOMES]
        out += [(o, int(np.sum(self.outcome == o)), self.n)
                for o in sorted(set(self.outcome) - set(OUTCOMES))]
        out.append(("engaged", int(np.sum(self.launched)), self.n))
        out.append(("KILL | engaged", int(np.sum(self.kill & self.launched)), int(np.sum(self.launched))))
        return out


def load(path):
    """(sweep param or None, {group label: Batch}); one group per swept value in file order."""
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
