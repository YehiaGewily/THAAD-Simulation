# THAAD-Style Terminal Intercept Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24208791.svg)](https://doi.org/10.5281/zenodo.24208791)

A deterministic Python simulation of **terminal ballistic missile defense**—modeling radar sensing, Kalman state estimation, impact-point fire control, proportional/optimal guidance, and hit-to-kill two-stage vehicle dynamics.

> ⚠️ An educational and hobby simulation. It models no real weapon system, and all vehicle, sensor, and engagement numbers are illustrative.

---

## Documentation Map

| Guide | What's Inside |
| --- | --- |
| **Overview (this page)** | Simulation scope, closed-loop architecture, YAML modes, headline results, and repository map |
| **[How to Run](docs/running.md)** | Installation, CLI deterministic runs, multi-core Monte Carlo & `--sweep` campaigns, `--serve`, and `pytest` |
| **[Mathematical Model & Verification](docs/methods.md)** | Equations of motion, RK4 burnout splitting, KF6/EKF7, RAE covariance, PN/APN/ZEM laws, and NEES/NIS proofs |
| **[Viewer Characteristics](docs/visualizers.md)** | Live FastAPI Workstation (`--serve`), offline 3D HTML replays, batch dashboard, controls, and frontend architecture |

---

## What the Simulation Models

Every engagement runs a closed six-stage GNC loop over flat-Earth East–North–Up (ENU) coordinates, uniquely determined by a YAML config and a random seed:

1. **Detection** — Ballistic threat under quadratic drag; radar returns noisy measurements in `cartesian` coordinates or polar Range–Azimuth–Elevation (`rae`) with Doppler range-rate.
2. **Estimation** — Tracks the threat using a **6-state linear Kalman filter** (`kf6`, assuming vacuum free-fall) or a **7-state Extended Kalman Filter** (`ekf7`) that estimates the threat's ballistic drag coefficient $k_T$ in flight.
3. **Fire Control** — Predicts the threat's impact point (`vacuum` parabola or RK4 `drag` integration) and gates launch (`pip_gated` vs. `fire_on_track`) against the defended perimeter.
4. **Guidance & Control** — Commands lateral acceleration via **Classical PN** (`pn`), **Augmented PN** (`apn` with EKF drag feed-forward), or **Optimal ZEM** (`zem`), constrained by lateral $g$-limits, 1st-order actuator lag $\tau$, and a finite kill-vehicle divert $\Delta v$ budget.
5. **Physics & Integration** — Two-stage interceptor (booster with mass flow $F / (I_{sp} g_0)$, separation at burnout, kinetic kill vehicle) integrated with **RK4** (with exact burnout sub-step splitting) or Forward Euler.
6. **Kill Assessment** — Continuous chord closest-point-of-approach (CPA) intersection across kill radius $R_k$ and terminal Battle Damage Assessment.

See **[docs/methods.md](docs/methods.md)** for all governing equations, Jacobians, assumptions, and verification details.

---

## Modes & Key Engineering Insight

All scenarios run the exact same simulation code; only YAML settings change (`configs/`). The presets use the default constant-density drag model, whose conclusions the next section qualifies:

| Mode | Radar, Filter & Guidance | Interceptor Limits | Outcome (constant-density drag) |
| --- | --- | --- | --- |
| [`configs/engineering.yaml`](configs/engineering.yaml) | Cartesian radar, `kf6`, vacuum PIP, classical `pn` | 150 kN (5 s), 30 g, 435 m/s divert budget, 10 m $R_k$ | **56.8%** of threatening threats killed (480 / 845, 95% CI 53.4–60.1%); all 385 misses exhaust divert $\Delta v$ |
| [`configs/advanced.yaml`](configs/advanced.yaml) | Polar `rae` radar, `ekf7`, drag PIP, `apn` | Identical 150 kN (5 s), 30 g, 435 m/s divert, 10 m $R_k$ | **100%** of threatening threats killed (845 / 845, 95% CI 99.5–100%) at a median 132 m/s divert $\Delta v$ |
| [`configs/demo.yaml`](configs/demo.yaml) | Cartesian radar, `kf6`, vacuum PIP, classical `pn` | 220 kN (8 s), 50 g, unlimited divert, 30 m $R_k$ | **100%** (1000 / 1000, 95% CI 99.6–100%) |

"Threatening" means the unintercepted threat would land inside the 30 km defended radius (845 of seeds 0–999 with constant-density drag, 830 in the exponential atmosphere). This denominator is the same for every configuration, unlike the kill rate among engaged threats, which depends on each configuration's own launch decisions.

**Main finding: classical PN wastes divert during the coast** (1000-seed paired ablation, [`scripts/ablation_study.py`](scripts/ablation_study.py), and budget sweeps, [`scripts/sensitivity_study.py`](scripts/sensitivity_study.py)). Classical PN commands $+g$ throughout the kill vehicle's coast, although gravity cancels between two free-falling bodies, so the term pushes the vehicle off course and then spends divert $\Delta v$ correcting it. Changing one GNC component at a time on the same seeds, dropping that term after burnout is the only change with a large effect under both drag models:

| Baseline → baseline without the coast $+g$ term | Constant-density drag (presets) | Exponential atmosphere ($H$ = 7 km) |
| --- | --- | --- |
| Protection at the 435 m/s budget | 56.8% → 100% (365 rescued, none lost) | 100% → 100% |
| Median divert $\Delta v$ used | 435 (whole budget) → 271 m/s | 390 → 146 m/s |
| Smallest tested budget for ≥ 99% protection | 550 → 435 m/s | 435 → 150 m/s |

**What depends on the drag model.** With the presets' constant-density drag, the drag-aware software looks valuable on top of that. EKF7 drag feed-forward halves divert use again (271 → 132 m/s) and lowers the budget needed for 99% protection from 435 to 200 m/s, and the drag-integrated PIP cuts launches at threats that would land outside the defended area from 115 to 5 (of 155). A bundled comparison of the two presets (56.8% → 100%) would credit the EKF7, the drag PIP and the polar radar, although changing them one at a time adds at most 1.2 points of protection at the nominal budget.

With density falling off as $e^{-z/7\,\mathrm{km}}$, intercepts happen at a median 34.5 km, where the air is under 1% of its sea-level density. Drag is then too weak to matter or to be estimated before intercept, and those benefits disappear: false engagements are 1–6 of 170 for every configuration, threat drag from 0 to 8 × 10⁻⁵ 1/m changes nothing, and the EKF7 saves 15 m/s. That saving still shows at a tight budget: at 100 m/s, adding the EKF7 and drag PIP to PN without the term rescues 34 seeds and loses 3 (McNemar $p \approx 10^{-7}$). An altitude-independent drag coefficient therefore changes the conclusion of the trade study, not just its numbers. Of the two, the exponential profile is the closer to the real atmosphere at these altitudes.

In both models the RAE radar and ZEM vs. APN change neither protection nor false engagements measurably at the nominal radar noise.

**Radar noise is the second budget driver** (exponential atmosphere, all radar noise values × 0.25–4, 60,000 paired runs). Divert use grows with noise in every configuration, and above the nominal noise even the 435 m/s budget stops protecting every threat: at twice the nominal noise 4–12% of threatening threats get through, at four times 41–54%, all of them divert-exhausted misses. Dropping the coast gravity term still saves 192–253 m/s across the range. The EKF7 stack's edge grows with noise (at 100 m/s and 4× noise, 36% against 21% protection), though that contrast also lowers the filter's process noise, which is the more likely cause. At 2× and 4× noise the RAE radar does worse than the Cartesian one, because these settings give it larger cross-range noise at long range. Full tables, sweeps and figures: [docs/methods.md](docs/methods.md#component-ablation-1000-paired-seeds) (constant density) and [its exponential-atmosphere rerun](docs/methods.md#the-same-studies-in-an-exponential-atmosphere), with the [radar-noise sweep](docs/methods.md#sensitivity-to-radar-noise-exponential-atmosphere-1000-paired-seeds-per-point).

![GNC ablation, exponential atmosphere](docs/figures/ablation_study_exp.png)

![GNC ablation, constant-density drag](docs/figures/ablation_study.png)

![GNC trade study](docs/figures/gnc_trade_study.png)

![Engineering-mode outcome rates](docs/figures/engineering_outcomes.png)

---

## Verification Snapshot

- **Integrator order** (`scripts/convergence_study.py`): Step-halving confirms order **1.00** (Euler) and **4.01** (RK4) on dragged threats, and **0.99** / **4.00** across interceptor booster burnout. With drag off, RK4 matches the closed-form solution to $6 \times 10^{-11}\text{ m}$.
- **Kalman consistency** (`scripts/filter_consistency.py`): Over 50 seeded runs, 6-DOF NEES/NIS stay inside the 95% $\chi^2$ interval $[5.08, 7.00]$ when matched (mean NEES **6.11**, **94.3%** inside; NIS **5.99**, **95.1%** inside), while `kf6` on a dragged threat drops to **58.9%** NEES coverage—quantifying the drag mismatch solved by `ekf7`. The 7-state EKF (`scripts/ekf7_consistency.py`, 200 runs) is consistent with its own model (mean 7-state NEES **7.08**, **94.0%** inside) and conservative as tuned (mean NEES 4.55). With constant density it learns the drag to about 1% (median error) by the 5 s launch decision; in the exponential atmosphere its estimate is still the prior at intercept.

---

## Quick Start

```bash
pip install -e ".[dev,plot,workstation]"

python -m thaad_sim --config configs/engineering.yaml --seed 13   # Single run -> offline 3D HTML replay
python -m thaad_sim --serve                                       # Live FastAPI GNC Workstation
python -m thaad_sim.batch --config configs/engineering.yaml --n 1000
python -m pytest
```

Full commands, sweep options, and study scripts: **[docs/running.md](docs/running.md)**.

---

## Visualization Layers

Two browser tools sit on top of `thaad_sim/` without altering the physics core (details in **[docs/visualizers.md](docs/visualizers.md)**):

- **Live Workstation (`--serve`)** — FastAPI server (`server.py`, `telemetry.py`, `workstation/`) running the real Python engine on demand for 3D inspection, optical seeker inset, 14 synchronized GNC/filter plots, 3-law comparisons, and multi-core live Monte Carlo batches.
- **Offline Replays & Batch Dashboard (`viewer/`)** — Zero-dependency single-file HTML exports (`replays/*.html`, `site/dashboard.html`) and the [GitHub Pages live demo](https://yehiagewily.github.io/THAAD-Simulation/).

![The live workstation playing Seed 13 with EKF7 and Augmented PN](docs/figures/workstation.gif)

---

## Project Structure

```text
.
├── thaad_sim/                  # Core Python simulation & GNC package
│   ├── __main__.py             # CLI entry point (--config, --seed, --serve)
│   ├── config.py               # YAML dataclasses (SimConfig)
│   ├── dynamics.py             # Ballistic threat & 2-stage interceptor ODEs
│   ├── integrators.py          # Forward Euler & RK4 (burnout sub-step splitting)
│   ├── estimation.py           # KF6, 7-state drag EKF7, RAE radar, vacuum/drag PIP
│   ├── guidance.py             # PN, APN, Optimal ZEM, g-limit, divert Δv budget
│   ├── scenario.py             # Seeded threat scenario generator
│   ├── engagement.py           # Closed-loop engagement runner, launch gate, CPA, BDA
│   ├── batch.py                # Multi-core Monte Carlo & --sweep runner -> CSV
│   ├── summary.py, stats.py    # Batch aggregation & Wilson confidence intervals
│   ├── consistency.py          # NEES, NIS & PIP bias diagnostics
│   ├── verification.py         # Step-halving & closed-form verification problems
│   ├── telemetry.py            # Vectorized telemetry builder for the workstation
│   ├── server.py               # FastAPI backend for the Live Workstation
│   ├── replay_viewer.py        # Self-contained HTML replay packer
│   ├── dashboard.py            # Self-contained HTML batch dashboard packer
│   ├── workstation/            # Live workstation frontend (HTML/CSS/ES modules)
│   └── viewer/                 # Compiled single-file HTML templates
├── configs/                    # engineering.yaml, advanced.yaml, demo.yaml
├── scripts/                    # Verification, GNC trade-off, envelope, & summary scripts
├── results/                    # Committed 1000-seed CSVs & verification JSONs
├── docs/                       # methods.md, running.md, visualizers.md, figures/, site/
├── viewer/                     # Source + Vite bundler for standalone HTML exports
├── tests/                      # pytest suite (physics, GNC, verification, API, templates)
└── older_attempts/             # Early prototype scripts (unused by thaad_sim)
```

---

## Limitations & Possible Extensions

Analytical point-mass sandbox: flat Earth with constant gravity, constant quadratic drag coefficient by default (an exponential atmosphere $\rho(z)$ is available as `sim.atmosphere: exponential` and in the study scripts; no Mach $C_D(M)$ table), and no 6-DOF attitude or physical seeker gimbal dynamics ([docs/methods.md §8](docs/methods.md#8-limitations)).

**Possible extensions:** multi-target salvos (shoot-look-shoot fire control), US Standard Atmosphere $\rho(z)$ and Mach drag curves $C_D(M)$, and 6-DOF rigid-body dynamics with strapdown/gimballed terminal IR seeker tracking.

---

## License & Citation

Released under the [MIT License](LICENSE). If you use this software in research or teaching, please cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).
