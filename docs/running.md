# How to Run

All simulation modes, CLI runs, Monte Carlo campaigns, and live workstation sessions execute the same Python engine (`thaad_sim/`).

---

## 1. Installation

Requires **Python 3.10 or newer** (no Node.js is required to run simulations, batches, or the compiled static replays).

```bash
pip install -e ".[dev]"                  # Core simulator + pytest suite
pip install -e ".[dev,plot]"             # + matplotlib for scripts/*.py analysis figures
pip install -e ".[dev,plot,workstation]" # + FastAPI & uvicorn for the live workstation (--serve)
```

*(On Windows, if `python` is not on your `PATH`, use `py -m pip ...` and `py -m thaad_sim ...`.)*

---

## 2. Single Deterministic Engagement (CLI)

Running the package prints the live event log and Battle Damage Assessment (BDA) in the terminal and writes a portable offline HTML replay to `replays/` (pass `--no-browser` for headless runs):

```bash
# Default engineering engagement (prints its random seed for deterministic replay)
python -m thaad_sim --config configs/engineering.yaml

# Direct kinetic hit at 37.9 km altitude (1.79 m miss):
python -m thaad_sim --config configs/engineering.yaml --seed 913

# Divert-budget exhaustion & close miss on Seed 13 (16.7 m CPA miss):
python -m thaad_sim --config configs/engineering.yaml --seed 13

# Advanced GNC rescue of Seed 13 (EKF7 + RAE + Drag PIP + APN -> 1.96 m hit, 137 m/s Δv):
python -m thaad_sim --config configs/advanced.yaml --seed 13

# High-performance demo mode:
python -m thaad_sim --config configs/demo.yaml --seed 42
```

### CLI Options (`python -m thaad_sim`)

| Option | Effect |
| --- | --- |
| `--config PATH` | Scenario configuration (`configs/engineering.yaml`, `configs/advanced.yaml`, `configs/demo.yaml`) |
| `--seed N` | Deterministic scenario seed (every run is a pure function of config + seed) |
| `--out PATH` | Output path for the HTML replay (default: `replays/replay_<timestamp>.html`) |
| `--no-browser` | Write the replay file without opening a browser tab |
| `--legacy-rng` | Match legacy `np.random.seed` stream for older prototype seeds |
| `--serve` | Launch the FastAPI Live Workstation instead of a single CLI run |
| `--port N` | Bind port for `--serve` (default: `8765`) |

---

## 3. Monte Carlo Batches & Parameter Sweeps

`thaad_sim.batch` runs $N$ seeded engagements across all CPU cores without writing replay HTMLs and outputs deterministic CSVs to `results/<config>.csv` (byte-identical regardless of worker count):

```bash
# 1000-seed Monte Carlo campaign (seeds 0..999)
python -m thaad_sim.batch --config configs/engineering.yaml --n 1000 --seed 0

# Sweep a YAML parameter across values
python -m thaad_sim.batch --config configs/engineering.yaml --n 200 --sweep sensor.pos_noise=5,25,50,100

# Summarize CSV results & write matplotlib figures to docs/figures/ (requires [plot])
python scripts/summarize.py results/engineering.csv

# Build a static HTML batch dashboard (site/dashboard.html)
python -m thaad_sim.dashboard results/engineering.csv --open
```

---

## 4. Live Interactive Workstation (`--serve`)

To explore runs interactively in the browser on top of the live Python engine (varying filter, radar model, guidance law, gain, actuator lag $\tau$, divert budget, 3-law overlays, and multi-core live batches):

```bash
python -m thaad_sim --serve
```

This opens <http://127.0.0.1:8765>. See [Viewer Characteristics](visualizers.md) for all tabs, controls, and API endpoints.

---

## 5. Verification Suite & Study Scripts

```bash
python -m pytest                     # Physics, GNC, verification, API, telemetry & template tests
python scripts/convergence_study.py  # Euler vs. RK4 order & analytic checks -> results/convergence.json
python scripts/filter_consistency.py # 50-run 6-DOF NEES & NIS study -> results/consistency.json
python scripts/drag_mismatch.py      # Vacuum vs. drag PIP overshoot study
python scripts/gnc_trade_study.py    # KF6 vs. EKF7 & PN vs. APN vs. ZEM trade study
python scripts/envelope_study.py     # Defended area & intercept altitude envelope
python scripts/palette_check.py      # Color-blindness & contrast checks
```

