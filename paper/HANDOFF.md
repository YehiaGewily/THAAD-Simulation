# Paper handoff brief

This brief is for whoever writes the arXiv paper from this repository. The experiments are finished. Every number below comes from a result file in `results/`, and each figure is in `docs/figures/`. `docs/methods.md` has the full equations, tables and readings; treat it as the source of truth if anything here disagrees.

## Target

- **Venue:** arXiv, eess.SY. About 8 pages, LaTeX, in this `paper/` folder.
- **Working title:** "What a Bundled GNC Upgrade Hides: A Paired Ablation of a Divert-Limited Hit-to-Kill Interceptor". It can change, but the paper must not claim that drag-model mismatch limits performance. The ablation contradicts that claim.
- **Parameters:** every parameter is illustrative and unclassified. Say so in the abstract and in the model section.
- **Naming:** keep "THAAD" out of the title and abstract. Describe the model as a terminal hit-to-kill interceptor, and mention THAAD only as the repository's name (THAAD-Simulation) in the code-availability statement.
- **References:** 15–25. Each one must be checked to exist, with the authors, title, venue, year and DOI confirmed against the publisher or a library index. Invented or unverified citations are not acceptable.
- **Figures:** all in one consistent style. The repository figures share a palette and fonts (`scripts/ablation_study.py`, `INK`/`SURFACE` and the per-configuration colours), so regenerate from the scripts rather than restyling by hand. A block diagram of the GNC loop still has to be made.
- **Code and data citation:** cite the v1.1.0 archive, which holds the exact code and data behind every number here: **10.5281/zenodo.24311162** (GitHub release v1.1.0, commit 0caac9f). The concept DOI, which always resolves to the latest version, is 10.5281/zenodo.24208791. The README badge and `CITATION.cff` carry the concept DOI; 10.5281/zenodo.24208792 is the v1.0.0 version DOI.

## Model in one paragraph

3D flat-Earth point-mass simulation, RK4 at 20 ms. A ballistic threat has quadratic drag $k_T = 4 \times 10^{-5}\ \mathrm{m^{-1}}$ (sea-level value in the exponential atmosphere). A two-stage interceptor has a 150 kN, 5 s boost, a 30 g lateral limit and a kill vehicle with a 435 m/s divert budget; a kill is a closest approach under 10 m. Radar: Cartesian (25 m, 8 m/s) or range–azimuth–elevation with Doppler. Filters: 6-state linear KF (no drag) or 7-state EKF estimating $k_T$. Launch is gated on the predicted impact point (vacuum closed form, or RK4 drag-integrated) falling inside a 30 km defended radius. Guidance: PN with the classical $+g$ term, APN (PN plus estimated target acceleration, without the coast $+g$ term), or ZEM. Atmosphere: constant density (the presets' default) or exponential with $H$ = 7 km (`sim.atmosphere`). Details: `docs/methods.md` §1–6.

## Design of the experiments

- **Ablation:** nine configurations C1–C9 change one GNC component at a time, on the same 1000 seeds (paired). Script `scripts/ablation_study.py`.
- **Metrics:** protection rate = kills of *threatening* threats (unintercepted impact inside the defended radius) over a denominator shared by every configuration; false engagements = launches at non-threatening threats; median divert used. Exact McNemar tests on discordant pairs; paired bootstrap intervals for divert differences.
- **Sweeps** (`scripts/sensitivity_study.py`): divert budget 100–550 m/s, true threat drag 0–8e-5, and radar noise × 0.25–4 at 435, 150 and 100 m/s, on four configurations (C1 baseline, C5 PN without the coast $+g$, C7 EKF7 + drag PIP + APN, C9 advanced preset with RAE radar).
- **Filter consistency** (`scripts/ekf7_consistency.py`): 7-state NEES, NIS and drag-estimate checks over 200 tracking runs.

## Framing

Experts already know that PN should not add $+g$ while both bodies are falling freely. Finding 1 therefore measures the cost of a known pitfall under a fixed divert budget; it is not a discovery, and the paper must not present it as one. The contributions are:

- **The paired ablation method:** one component changed at a time on matched seeds, with a protection rate whose denominator is shared by every configuration, exact McNemar tests and paired divert differences.
- **The bundled-comparison lesson:** comparing two complete GNC stacks credits components that do nothing. It holds in both atmospheres. With constant density, C1 → C9 goes from 56.8 % to 100 % protection, almost all of it from the gravity term. In the exponential atmosphere, C1 → C9 cuts median divert from 390 to 130 m/s, and 243 m/s of that comes from dropping the gravity term alone.
- **The atmosphere-model caution:** an altitude-independent drag coefficient changes which components appear to matter (finding 2).
- **Radar noise as the second budget driver:** with the gravity term removed, measurement noise sets how much divert is left over, and above the nominal noise it costs kills even at the full budget (finding 3).

Do not describe the gravity term's cost as $g$ (9.8 m/s) per second of coast. PN's line-of-sight term partly works against it, so the measured cost is lower: about 3.6 m/s per second of coast at the nominal constant-density drag (`docs/methods.md`, "Component ablation", Reading).

## Findings to report

**1. The robust finding: classical PN wastes divert during the coast.** It commands $+g$ throughout the kill vehicle's coast, although gravity cancels between two free-falling bodies. Dropping that term after burnout is the only change with a large effect under both drag models.

| Baseline (C1) → without the coast $+g$ term (C5) | Constant density | Exponential atmosphere |
| --- | --- | --- |
| Protection at 435 m/s | 56.8 % → 100 % (365 rescued, 0 lost, $p < 10^{-100}$) | 100 % → 100 % |
| Median divert used | 435 (whole budget) → 271 m/s | 390 → 146 m/s (paired −243 m/s, 95 % CI −245 to −240) |
| Smallest tested budget for ≥ 99 % protection | 550 → 435 m/s | 435 → 150 m/s |

Sources: `results/ablation_summary.json`, `results/ablation_summary_exp.json`, `results/sensitivity_summary.json`, `results/sensitivity_summary_exp.json`.

**2. The cautionary finding: constant-density drag changes the conclusion, not just the numbers.**

- With constant density the drag-aware stack looks valuable: EKF7 drag feed-forward halves divert again (271 → 132 m/s) and lowers the 99 % budget to 200 m/s, and the drag PIP cuts false launches from 115 to 5 of 155.
- A bundled comparison of the two presets (56.8 % → 100 %) would credit the EKF7, the drag PIP and the polar radar, although changed one at a time they add at most 1.2 points of protection.
- In the exponential atmosphere, intercepts happen at a median 34.5 km (25–43 km), where density is under 1 % of sea level. False engagements fall to 1–6 of 170 in every configuration, threat drag from 0 to 8e-5 changes nothing, and the EKF7 saves only 15 m/s. That saving still decides kills at 100 m/s: C5 → C7 rescues 34 seeds and loses 3 (McNemar $p = 1.2 \times 10^{-7}$).
- Mechanism, from the consistency study (`results/consistency_ekf7.json`): with constant density the EKF7 learns $k_T$ to about 1 % (median error) before the 5 s launch decision; in the exponential atmosphere its estimate is still the prior (3.51 vs. 3.5 × 10⁻⁵, truth 4.0) at the median intercept, and its 1σ has not shrunk.

**3. Radar noise is the second budget driver** (exponential atmosphere; every radar noise value × 0.25, 0.5, 1, 2, 4; divert budgets 435, 150 and 100 m/s; C1, C5, C7, C9; 60,000 paired runs). Sources: `results/sensitivity_noise_summary_exp.json`, `docs/figures/sensitivity_noise_exp.png`, `docs/methods.md` "Sensitivity to radar noise". The 1× runs reproduce the exponential ablation and divert sweep exactly.

- Divert use grows with noise in every configuration (C7 at 435 m/s: 104, 130, 152, 197 m/s at 0.25×, 1×, 2×, 4×). Above the nominal noise even 435 m/s stops protecting every threat: 4–12 % of threatening threats get through at 2×, 41–54 % at 4×, and every miss is divert-exhausted. Below 1× little changes, so the nominal noise sits just below where it starts costing kills.
- The gravity term stays the largest divert cost (it saves 253 m/s at 0.25×, 192 m/s at 4×), but at 4× noise and 435 m/s C1 and C5 protect alike (52 % vs. 55 %, $p$ = 0.29) because noise now limits both. At 2× C1 kills slightly more than C5 (46 vs. 27 discordant seeds, $p$ = 0.034); the reason is not established, so report it without a story.
- The EKF7 stack's edge (C5 → C7) grows with noise: 4 points at 1× to 16 points at 4× (100 m/s budget, 36 % vs. 21 %), and it keeps false engagements at 4–5 of 170 where the vacuum PIP rises to 13–19. **Caveat:** this contrast also lowers the filter's process noise ($q$ = 5 vs. 30 m²/s³). The drag estimate barely leaves its prior in this atmosphere, so the lower $q$ is the more likely cause. Do not credit drag estimation for it; it has not been tested separately.

**4. Components that make no measurable difference at the nominal noise.** ZEM vs. APN (C7 → C8) changes neither protection nor false engagements measurably in either atmosphere (divert difference under 3 m/s). The same holds for the RAE radar vs. the Cartesian radar (C7 → C9) at 1× noise and below. At 2× and 4× the RAE radar loses 6.5 and 13 points at 435 m/s ($p = 10^{-8}$ and $5 	imes 10^{-10}$), but that is a property of these noise settings: at the 72–86 km slant range where tracking starts, its 1.2 mrad angle noise is already 85–105 m of cross-range noise against the Cartesian radar's 25 m, and the factor multiplies both. Do not generalise it to range–angle radars.

**5. Filter consistency.** The EKF7 is consistent with its own model (matched case: mean 7-state NEES 7.08, 94 % of steps inside the 95 % bounds; NIS 6.00) and conservative as tuned (NEES about 4.5 against 7), because it assumes process noise the nominal threat does not have. It was not retuned, so that the ablation's settings stand.

## Limitations to state

- Flat Earth, constant gravity, point masses, no seeker or attitude dynamics; a kill is purely geometric.
- No Mach-dependent drag; isothermal exponential atmosphere only.
- Filters were not retuned for the sweeps, including the radar-noise sweep, so the high-noise points mix sensor and tuning effects.
- The radar-noise sweep was run in the exponential atmosphere only.
- With constant density and a drag-free threat, the drag PIP launches at more non-threatening threats than the vacuum PIP (36 vs. 7 of 173). The measured cause is a one-sided spread of short predictions, consistent with the EKF7 clipping $\hat k_T$ at zero; no run with the clip removed has tested that explanation.
- Results come from one scenario distribution (`scenario` section of the configs).

## Figures available

| Figure | File |
| --- | --- |
| Ablation, exponential (lead figure) | `docs/figures/ablation_study_exp.png` |
| Ablation, constant density | `docs/figures/ablation_study.png` |
| Divert and drag sweeps | `docs/figures/sensitivity_study_exp.png`, `docs/figures/sensitivity_study.png` |
| Radar-noise sweep | `docs/figures/sensitivity_noise_exp.png` |
| EKF7 consistency | `docs/figures/ekf7_consistency.png` |
| Earlier studies | `nees.png`, `nis.png`, `pip_bias.png`, `convergence.png`, `engagement_envelope.png`, `gnc_trade_study.png` |

The figure titles are long dashboard-style titles. For the paper, strip the suptitles and move that text into captions.

## Reproducing

```bash
pip install -e ".[dev,plot]"
python -m pytest
python scripts/ablation_study.py --atmosphere exponential
python scripts/sensitivity_study.py --atmosphere exponential
python scripts/sensitivity_study.py --sweeps noise --atmosphere exponential
python scripts/ekf7_consistency.py
```

Each 1000-seed sweep takes one to two hours on 18 cores. Every script also takes `--from-csv` (where offered) to re-analyse the saved CSVs without re-running.

## Repository rules

- Commits carry no Claude co-author trailer.
- Commit or push only when the owner asks.
