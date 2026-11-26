"""Check every traced number in paper/numbers.md against its source.

Each table row in numbers.md names a source (a JSON result file, a YAML
config, docs/methods.md, or a named computation on the per-seed CSVs) and a
field. This script reads that field and compares it with the row's "Raw"
value (within 0.1 %), or, for docs/methods.md, checks that the quoted snippet
appears in the file word for word. It exits with status 1 if any row fails.

Usage (from the repository root or from paper/):
    python paper/check_numbers.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

try:
    import yaml
    _load_yaml = yaml.safe_load
except ModuleNotFoundError:
    def _load_yaml(stream):
        """Minimal fallback loader for the two-level configs/*.yaml files when PyYAML is absent."""
        out, section = {}, None
        for raw in (stream.read() if hasattr(stream, "read") else str(stream)).splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = out.setdefault(line[:-1].strip(), {})
            elif ":" in line and section is not None:
                k, v = (p.strip() for p in line.split(":", 1))
                try:
                    section[k] = json.loads(v)
                except Exception:
                    section[k] = v.strip("\"'")
        return out

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RESULTS = ROOT / "results"


# ------------------------------------------------------------------ field paths
def _match(item, selector):
    """True if a list item matches 'key=value[,key=value]'."""
    for part in selector.split(","):
        key, value = part.split("=", 1)
        if key not in item:
            return False
        have = item[key]
        if isinstance(have, (int, float)):
            if not math.isclose(float(have), float(value), rel_tol=1e-9, abs_tol=1e-15):
                return False
        elif str(have) != value:
            return False
    return True


def resolve(data, path):
    """Follow a path such as 'configs[config_id=C1]/kill_rate_threatening' or 'divert[0]/value'."""
    for step in path.split("/"):
        m = re.fullmatch(r"([^\[\]]*)((?:\[[^\]]+\])*)", step)
        name, brackets = m.group(1), m.group(2)
        if name:
            data = data[name]
        for sel in re.findall(r"\[([^\]]+)\]", brackets):
            if re.fullmatch(r"-?\d+", sel):
                data = data[int(sel)]
            else:
                hits = [x for x in data if _match(x, sel)]
                if len(hits) != 1:
                    raise KeyError(f"{sel!r} matched {len(hits)} items")
                data = hits[0]
    return data


def same(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
    a, b = float(a), float(b)
    if b == 0.0:
        return abs(a) < 1e-12
    return math.isclose(a, b, rel_tol=1e-3)


# ------------------------------------------------------------------ CSV computations
def _rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _by_config(name):
    out = {}
    for r in _rows(name):
        out.setdefault(r["config_id"], {})[int(r["seed"])] = r
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _both_kill():
    a = _by_config("ablation_1000.csv")
    c1, c5 = a["C1_base_cart"], a["C5_kf6_vac_apn"]
    seeds = [s for s in c1 if c1[s]["outcome"] == "KILL" and c5[s]["outcome"] == "KILL"]
    return c1, c5, seeds


def _coast(row):
    """Kill-vehicle coast: kill time minus launch time minus the 5 s boost."""
    return float(row["kill_time_s"]) - float(row["launch_time_s"]) - 5.0


def both_kill_seeds():
    return len(_both_kill()[2])


def median_extra_divert():
    c1, c5, seeds = _both_kill()
    return _median([float(c1[s]["divert_dv_used_mps"]) - float(c5[s]["divert_dv_used_mps"]) for s in seeds])


def median_coast_c5():
    _, c5, seeds = _both_kill()
    return _median([_coast(c5[s]) for s in seeds])


def median_coast_c1():
    c1, _, seeds = _both_kill()
    return _median([_coast(c1[s]) for s in seeds])


def cost_per_second_c5():
    return median_extra_divert() / median_coast_c5()


def cost_per_second_c1():
    return median_extra_divert() / median_coast_c1()


def _kill_alts_c7_exp():
    rows = _by_config("ablation_1000_exp.csv")["C7_ekf7_drag_apn"].values()
    return [float(r["kill_alt_m"]) for r in rows if r["outcome"] == "KILL"]


def kill_alt_median_c7_exp():
    return _median(_kill_alts_c7_exp())


def kill_alt_min_c7_exp():
    return min(_kill_alts_c7_exp())


def kill_alt_max_c7_exp():
    return max(_kill_alts_c7_exp())


def density_ratio_at_34p5km():
    return math.exp(-34.5 / 7.0)


def noise_sweep_runs():
    return sum(len(_rows(f"sensitivity_noise_{b}_exp.csv")) for b in (435, 150, 100))


def _json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def vacuum_false_share_at_8e5():
    point = resolve(_json("sensitivity_summary.json"), "drag[value=8e-05]")
    n_safe = _json("sensitivity_summary.json")["runs_per_point"] - point["threatening"]
    return 100.0 * point["configs"]["C5_kf6_vac_apn"]["false_engages"] / n_safe


def min_false_engage_p_exp():
    return min(c["false_engage_mcnemar_p"] for c in _json("ablation_summary_exp.json")["contrasts"])


def min_protection_drag_sweep_exp():
    return min(c["kill_rate_threatening"] for e in _json("sensitivity_summary_exp.json")["drag"]
               for c in e["configs"].values())


def max_divert_spread_drag_sweep_exp():
    entries = _json("sensitivity_summary_exp.json")["drag"]
    spreads = []
    for cid in entries[0]["configs"]:
        meds = [e["configs"][cid]["median_dv_mps"] for e in entries]
        spreads.append(max(meds) - min(meds))
    return round(max(spreads), 3)


def min_c5_protection_drag_sweep_const():
    return min(e["configs"]["C5_kf6_vac_apn"]["kill_rate_threatening"]
               for e in _json("sensitivity_summary.json")["drag"])


def noise_misses_not_exhausted():
    data = _json("sensitivity_noise_summary_exp.json")
    return sum(c["misses"] - c["misses_dv_exhausted"] for key, entries in data.items()
               if key.startswith("noise_") for e in entries for c in e["configs"].values())


COMPUTED = {name: fn for name, fn in globals().items() if callable(fn) and not name.startswith("_")
            and name not in {"resolve", "same", "main"}}


# ------------------------------------------------------------------ main
def main():
    text = (PAPER / "numbers.md").read_text(encoding="utf-8")
    methods = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    cache = {}
    checked = failed = 0
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not cells[0].isdigit():
            continue
        num, _, _, source, field, raw = cells
        try:
            if source == "docs/methods.md":
                snippet = field.strip("`")
                ok = snippet in methods
                got = "found" if ok else "missing"
            elif source.startswith("computed:"):
                got = COMPUTED[source.split(":", 1)[1].strip()]()
                ok = same(got, json.loads(raw))
            else:
                if source not in cache:
                    with open(ROOT / source, encoding="utf-8") as f:
                        cache[source] = _load_yaml(f) if source.endswith(".yaml") else json.load(f)
                got = resolve(cache[source], field)
                ok = same(got, json.loads(raw))
        except Exception as exc:          # a broken path counts as a failure
            ok, got = False, f"error: {exc}"
        checked += 1
        if not ok:
            failed += 1
            print(f"FAIL row {num}: {source} {field} -> {got!r}, numbers.md says {raw}")
    print(f"{checked} rows checked, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
