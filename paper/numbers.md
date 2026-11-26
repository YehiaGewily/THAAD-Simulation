# Where every number in the paper comes from

This table traces each number in `main.tex` to the file and field it was taken from. It covers the text, the captions and the tables. `python paper/check_numbers.py` re-reads every row from its source and fails if any value differs by more than 0.1 %.

How to read the **Source** and **Field** columns:

- **JSON files** in `results/`: the field is a path. `/` separates keys. `[key=value]` picks the list item with that key, for example `configs[config_id=C1_base_cart]`, or `contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]` for a paired contrast. `[0]` and `[-1]` are list positions.
- **YAML configs** in `configs/`: the same path syntax.
- **`docs/methods.md`**: the field is a text snippet that appears in that file word for word.
- **`computed:`** rows are recomputed from the per-seed CSVs (`results/ablation_1000*.csv`) by the checker. The function with that name in `check_numbers.py` shows the calculation.

**Raw** is the stored value. The paper shows it rounded, and no rounding changes the meaning. Two values are rounded half-up: 145.5 → 146 and 130.5 → 130/131. The paper follows `docs/methods.md`, which writes them as 146 and 130.

Short names: A = `results/ablation_summary.json` (constant density), AE = `results/ablation_summary_exp.json`, S = `results/sensitivity_summary.json`, SE = `results/sensitivity_summary_exp.json`, N = `results/sensitivity_noise_summary_exp.json`, K = `results/consistency_ekf7.json`. The Source column always gives the full path.

## Abstract and introduction

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 1 | 1000 random threats (seeds 0–999) | Abstract, §I, §IV-B, captions | results/ablation_summary.json | runs_per_config | 1000 |
| 2 | saves a median 243 m/s (exponential) | Abstract, §V-A, §V-B, §VI, Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/median_dv_diff_mps | -242.7 |
| 3 | 56.8 % protection, baseline C1, constant density | Abstract, §V-A, §V-B, Fig. 2 | results/ablation_summary.json | configs[config_id=C1_base_cart]/kill_rate_threatening | 56.8 |
| 4 | 100 % protection, C5, constant density | Abstract, §V-A | results/ablation_summary.json | configs[config_id=C5_kf6_vac_apn]/kill_rate_threatening | 100.0 |
| 5 | "at most 1.2 percentage points": the largest one-factor protection gain from the EKF7 (C1→C3) | Abstract, §V-B | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/protection_diff_pts | 1.18 |
| 6 | the other one-factor steps that upgrade the filter, drag PIP or radar: RAE radar, C1→C2 | Abstract, §V-B | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/protection_diff_pts | -0.47 |
| 7 | drag PIP with PN, C3→C4 | Abstract, §V-B | results/ablation_summary.json | contrasts[ref=C3_ekf7_vac_pn,cmp=C4_ekf7_drag_pn]/protection_diff_pts | -0.24 |
| 8 | drag PIP with APN, C6→C7 | Abstract, §V-B | results/ablation_summary.json | contrasts[ref=C6_ekf7_vac_apn,cmp=C7_ekf7_drag_apn]/protection_diff_pts | 0.0 |
| 9 | RAE radar on the full stack, C7→C9 | Abstract, §V-B | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/protection_diff_pts | 0.0 |

## Model (Section III, Table I)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 10 | RK4 step 20 ms | Table I, §III-A | configs/engineering.yaml | sim/dt | 0.02 |
| 11 | g = 9.81 m/s² | Table I | configs/engineering.yaml | sim/g | 9.81 |
| 12 | starting altitude 55–75 km | Table I | configs/engineering.yaml | scenario/altitude | [55000.0, 75000.0] |
| 13 | ground range 35–55 km | Table I | configs/engineering.yaml | scenario/range | [35000.0, 55000.0] |
| 14 | 70–100 s to impact | Table I | configs/engineering.yaml | scenario/impact_time | [70.0, 100.0] |
| 15 | aim point within 35 km | Table I | configs/engineering.yaml | scenario/aim_radius | [0.0, 35000.0] |
| 16 | threat drag k_T = 4×10⁻⁵ m⁻¹ | Table I, Fig. 5 | configs/engineering.yaml | threat/drag | 4e-05 |
| 17 | scale height H = 7 km | Table I, §III-A, Appendix | results/ablation_summary_exp.json | scale_height_m | 7000.0 |
| 18 | sea-level drag "even at 70 km" (constant model) | §III-A | docs/methods.md | `it acts at full strength at 70 km` | — |
| 19 | booster 150 kN | Table I, §III-B | configs/engineering.yaml | interceptor/boost_thrust | 150000.0 |
| 20 | burn time 5 s | Table I, §III-B | docs/methods.md | `= 5.0\ \mathrm{s}` | — |
| 21 | kill vehicle 200 kg | Table I | configs/engineering.yaml | interceptor/kv_mass | 200.0 |
| 22 | 30 g lateral limit | Table I, §III-B, Fig. 1, Appendix | configs/engineering.yaml | interceptor/max_g | 30.0 |
| 23 | 435 m/s divert budget | everywhere | configs/engineering.yaml | interceptor/kv_divert_dv | 435.0 |
| 24 | kill: closest approach under 10 m | Table I, Fig. 1, Appendix | configs/engineering.yaml | engagement/kill_radius | 10.0 |
| 25 | navigation constant N = 4 | Table I | configs/engineering.yaml | interceptor/pn_gain | 4.0 |
| 26 | Cartesian radar 25 m (1σ, per axis) | Table I, §V-D, Fig. 4 | configs/engineering.yaml | sensor/pos_noise | 25.0 |
| 27 | Cartesian radar 8 m/s | Table I | configs/engineering.yaml | sensor/vel_noise | 8.0 |
| 28 | RAE range noise 10 m | Table I | configs/advanced.yaml | sensor/range_noise | 10.0 |
| 29 | RAE angle noise 1.2 mrad | Table I, §V-D | configs/advanced.yaml | sensor/angle_noise | 0.0012 |
| 30 | RAE Doppler 3 m/s | Table I | configs/advanced.yaml | sensor/range_rate_noise | 3.0 |
| 31 | KF6 q = 30 m²/s³ | Table I, §III-B, §V-D | configs/engineering.yaml | kalman/q | 30.0 |
| 32 | EKF7 q = 5 m²/s³ | Table I, §III-B, §V-D | configs/advanced.yaml | kalman/q | 5.0 |
| 33 | EKF7 prior mean 3.5×10⁻⁵ m⁻¹ | Table I, §III-B, §V-C | configs/advanced.yaml | kalman/drag_init | 3.5e-05 |
| 34 | EKF7 prior 1σ 2.5×10⁻⁵ m⁻¹ | Table I, §III-B, §V-C | configs/advanced.yaml | kalman/drag_std | 2.5e-05 |
| 35 | launch gate: track ≥ 5 s | Table I, §III-C | configs/engineering.yaml | engagement/min_track_time | 5.0 |
| 36 | defended radius 30 km | Table I, §III-A, §III-C | configs/engineering.yaml | engagement/defended_radius | 30000.0 |
| 211 | drag compensation factor 1.03 | §III-A | configs/engineering.yaml | scenario/drag_comp | 1.03 |
| 212 | booster tube-exit kick 80 m/s | §III-B | docs/methods.md | `= 5.0\ \mathrm{s}` | — |
| 213 | booster Isp 170 s | §III-B | configs/engineering.yaml | interceptor/isp | 170.0 |
| 214 | booster wet mass 900 kg | §III-B | configs/engineering.yaml | interceptor/mass_wet | 900.0 |
| 215 | booster dry mass 450 kg | §III-B | configs/engineering.yaml | interceptor/mass_dry | 450.0 |
| 216 | booster drag 10⁻⁴ m⁻¹ | §III-B | configs/engineering.yaml | interceptor/booster_drag | 0.0001 |
| 217 | boost lateral command share 10 % (η = 0.1) | §III-B | docs/methods.md | `\eta = \begin{cases}0.1` | — |
| 218 | kill vehicle drag 8×10⁻⁵ m⁻¹ | §III-B | configs/engineering.yaml | interceptor/kv_drag | 8e-05 |
| 219 | commands cease below 20 m range | §III-B | docs/methods.md | `Below 20 m range no command is issued.` | — |
| 220 | EKF7 q_drag = 10⁻¹² m⁻²s⁻¹ | Appendix | configs/advanced.yaml | kalman/q_drag | 1e-12 |

## Experiment design (Section IV)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 37 | 845 threatening (constant) | §IV-B, Fig. 2 | results/ablation_summary.json | threatening_scenarios | 845 |
| 38 | 155 not threatening (constant) | §IV-B, Fig. 2, Table III | results/ablation_summary.json | non_threatening_scenarios | 155 |
| 39 | 830 threatening (exponential) | §IV-B, Fig. 2 | results/ablation_summary_exp.json | threatening_scenarios | 830 |
| 40 | 170 not threatening (exponential) | §IV-B, §V-C, §V-D, Table III | results/ablation_summary_exp.json | non_threatening_scenarios | 170 |
| 41 | C1 launches at 960 threats | §IV-B | results/ablation_summary.json | configs[config_id=C1_base_cart]/engaged | 960 |
| 42 | C7 launches at 850 | §IV-B | results/ablation_summary.json | configs[config_id=C7_ekf7_drag_apn]/engaged | 850 |
| 43 | divert sweep from 100 m/s | §IV-C | results/sensitivity_summary.json | divert[0]/value | 100.0 |
| 44 | divert sweep to 550 m/s | §IV-C | results/sensitivity_summary.json | divert[-1]/value | 550.0 |
| 45 | drag sweep to 8×10⁻⁵ m⁻¹ | §IV-C, §V-A, §V-C | results/sensitivity_summary.json | drag[-1]/value | 8e-05 |
| 46 | noise factor 0.25 | §IV-C, Fig. 4 | results/sensitivity_noise_summary_exp.json | noise_435[0]/value | 0.25 |
| 47 | noise factor 4 | §IV-C, Fig. 4 | results/sensitivity_noise_summary_exp.json | noise_435[-1]/value | 4.0 |
| 48 | 60,000 runs (5 factors × 3 budgets × 4 configurations × 1000 seeds) | §IV-C | computed: noise_sweep_runs | — | 60000 |
| 49 | 200 tracking runs | §IV-C, Fig. 5 | results/consistency_ekf7.json | runs | 200 |
| 50 | RK4 order 4.01 (dragged threat) | §IV-D | results/convergence.json | convergence[problem=threat]/rk4/orders[-1] | 4.009 |
| 51 | RK4 order 4.00 (through burnout) | §IV-D | results/convergence.json | convergence[problem=interceptor]/rk4/orders[-1] | 4.0 |
| 52 | every 1× noise run matches the ablation | §IV-D | docs/methods.md | `At 1× every one of the 12,000 runs matches the exponential ablation and divert sweep exactly` | — |
| 53 | bootstrap with 4000 resamples | §II-D | docs/methods.md | `4000-resample percentile bootstrap` | — |

## Results: the coast gravity term (Section V-A)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 54 | 95 % interval 53.4–60.1 % | §V-A | results/ablation_summary.json | configs[config_id=C1_base_cart]/ci95_threatening | [53.44, 60.11] |
| 55 | all 385 misses … | §V-A | results/ablation_summary.json | configs[config_id=C1_base_cart]/misses | 385 |
| 56 | … are divert-exhausted | §V-A | results/ablation_summary.json | configs[config_id=C1_base_cart]/misses_dv_exhausted | 385 |
| 57 | rescues 365 seeds | §V-A, §V-B, Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/discordant_only_cmp | 365 |
| 58 | loses none | §V-A, Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/discordant_only_ref | 0 |
| 59 | p < 10⁻¹⁰⁰ | §V-A, Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/mcnemar_p | 2.6612e-110 |
| 60 | C5 median divert 271 m/s | §V-A, Fig. 2 | results/ablation_summary.json | configs[config_id=C5_kf6_vac_apn]/median_dv_mps | 271.0 |
| 61 | C1 uses the full 435 m/s | §V-A, Fig. 2 | results/ablation_summary.json | configs[config_id=C1_base_cart]/median_dv_mps | 435.0 |
| 62 | exponential: 390 m/s (C1) | §V-A, §V-B, Fig. 2 | results/ablation_summary_exp.json | configs[config_id=C1_base_cart]/median_dv_mps | 390.1 |
| 63 | exponential: 146 m/s (C5) | §V-A, Fig. 2 | results/ablation_summary_exp.json | configs[config_id=C5_kf6_vac_apn]/median_dv_mps | 145.5 |
| 64 | 95 % interval 240–245 m/s | §V-A, §VI (±2 m/s), Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/median_dv_diff_ci95 | [-244.7, -240.3] |
| 65 | "stays at 242–243 m/s" across drag: k_T = 0 | §V-A | results/sensitivity_summary_exp.json | drag[value=0.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -243.0 |
| 66 | k_T = 1×10⁻⁵ | §V-A | results/sensitivity_summary_exp.json | drag[value=1e-05]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -243.2 |
| 67 | k_T = 2×10⁻⁵ | §V-A | results/sensitivity_summary_exp.json | drag[value=2e-05]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -242.4 |
| 68 | k_T = 6×10⁻⁵ | §V-A | results/sensitivity_summary_exp.json | drag[value=6e-05]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -242.1 |
| 69 | k_T = 8×10⁻⁵ | §V-A | results/sensitivity_summary_exp.json | drag[value=8e-05]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -242.1 |
| 70 | 99 % budget, constant, C1: 550 m/s (99.05 %) | §V-A, Fig. 3a | results/sensitivity_summary.json | divert[value=550.0]/configs/C1_base_cart/kill_rate_threatening | 99.05 |
| 71 | … C1 below 99 % at 435 m/s | §V-A | results/sensitivity_summary.json | divert[value=435.0]/configs/C1_base_cart/kill_rate_threatening | 56.8 |
| 72 | 99 % budget, constant, C5: 435 m/s | §V-A | results/sensitivity_summary.json | divert[value=435.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 100.0 |
| 73 | … C5 below 99 % at 350 m/s | §V-A | results/sensitivity_summary.json | divert[value=350.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 97.4 |
| 74 | 99 % budget, exponential, C1: 435 m/s | §V-A | results/sensitivity_summary_exp.json | divert[value=435.0]/configs/C1_base_cart/kill_rate_threatening | 100.0 |
| 75 | … C1 below 99 % at 350 m/s | §V-A | results/sensitivity_summary_exp.json | divert[value=350.0]/configs/C1_base_cart/kill_rate_threatening | 68.31 |
| 76 | 99 % budget, exponential, C5: 150 m/s | §V-A | results/sensitivity_summary_exp.json | divert[value=150.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 100.0 |
| 77 | … C5 below 99 % at 100 m/s | §V-A | results/sensitivity_summary_exp.json | divert[value=100.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 88.8 |
| 78 | 575 seeds where both C1 and C5 kill | §V-A | computed: both_kill_seeds | — | 575 |
| 79 | C1 spends a median 183 m/s more | §V-A | computed: median_extra_divert | — | 183.4 |
| 80 | (same, as written in methods) | §V-A | docs/methods.md | `C1 spends a median 183 m/s more divert, about 3.6 m/s per second of coast` | — |
| 81 | median coast of about 51 s (C5) | §V-A | computed: median_coast_c5 | — | 51.39 |
| 82 | about 3.6 m/s per second of coast | §V-A | computed: cost_per_second_c5 | — | 3.57 |
| 83 | footnote: C1 median coast 44 s | §V-A footnote | computed: median_coast_c1 | — | 44.46 |
| 84 | footnote: 4.1 m/s per second | §V-A footnote | computed: cost_per_second_c1 | — | 4.13 |

## Results: the bundled comparison (Section V-B)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 85 | bundle rescues the same 365 seeds | §V-B, Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/discordant_only_cmp | 365 |
| 86 | C1→C3 p = 0.006 | §V-B, Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/mcnemar_p | 0.00635 |
| 87 | exponential bundle ends at 130 m/s (C9) | §V-B | results/ablation_summary_exp.json | configs[config_id=C9_adv_rae]/median_dv_mps | 130.5 |
| 88 | the EKF7 adds 15 m/s | §V-B, §V-C, Table III | results/ablation_summary_exp.json | contrasts[ref=C5_kf6_vac_apn,cmp=C6_ekf7_vac_apn]/median_dv_diff_mps | -15.2 |
| 89 | drag PIP < 1 m/s | §V-B, Table III | results/ablation_summary_exp.json | contrasts[ref=C6_ekf7_vac_apn,cmp=C7_ekf7_drag_apn]/median_dv_diff_mps | 0.0 |
| 90 | ZEM < 1 m/s | §V-B, Table III | results/ablation_summary_exp.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C8_ekf7_drag_zem]/median_dv_diff_mps | 0.6 |
| 91 | RAE radar < 1 m/s | §V-B, Table III | results/ablation_summary_exp.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/median_dv_diff_mps | 0.0 |

## Results: the atmosphere model (Section V-C, Fig. 5)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 92 | drag feed-forward saves another 137 m/s | §V-C, Table III | results/ablation_summary.json | contrasts[ref=C5_kf6_vac_apn,cmp=C6_ekf7_vac_apn]/median_dv_diff_mps | -137.1 |
| 93 | C7 needs a median 132 m/s | §V-C, Fig. 2 | results/ablation_summary.json | configs[config_id=C7_ekf7_drag_apn]/median_dv_mps | 131.5 |
| 94 | 99 % budget for C7: 200 m/s | §V-C | results/sensitivity_summary.json | divert[value=200.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 99.64 |
| 95 | … C7 below 99 % at 150 m/s | §V-C | results/sensitivity_summary.json | divert[value=150.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 93.14 |
| 96 | at 200 m/s C5→C7 rescues 514 | §V-C | results/sensitivity_summary.json | divert[value=200.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/discordant_only_cmp | 514 |
| 97 | … and loses none | §V-C | results/sensitivity_summary.json | divert[value=200.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/discordant_only_ref | 0 |
| 98 | false engagements 113 → … | §V-C, Table III | results/ablation_summary.json | contrasts[ref=C6_ekf7_vac_apn,cmp=C7_ekf7_drag_apn]/false_engage_ref | 113 |
| 99 | … → 5 of 155 | §V-C, Table III | results/ablation_summary.json | contrasts[ref=C6_ekf7_vac_apn,cmp=C7_ekf7_drag_apn]/false_engage_cmp | 5 |
| 100 | p = 6×10⁻³³ | §V-C | results/ablation_summary.json | contrasts[ref=C6_ekf7_vac_apn,cmp=C7_ekf7_drag_apn]/false_engage_mcnemar_p | 6.1630e-33 |
| 101 | 91 % of non-threatening threats engaged at 8×10⁻⁵ (465 of 1000 − 490 = 510) | §V-C, Fig. 3c | computed: vacuum_false_share_at_8e5 | — | 91.18 |
| 102 | C7 median intercept altitude 34.5 km | §V-C | computed: kill_alt_median_c7_exp | — | 34551 |
| 103 | … range from 25 km | §V-C | computed: kill_alt_min_c7_exp | — | 24795 |
| 104 | … to 43 km | §V-C | computed: kill_alt_max_c7_exp | — | 42714 |
| 105 | (same, as written in methods) | §V-C | docs/methods.md | `Kills happen at a median 34.5 km altitude (range 25–43 km), where density is under 1 % of its sea-level value` | — |
| 106 | density under 1 % of sea level at 34.5 km: e^(−34.5/7) | §V-C | computed: density_ratio_at_34p5km | — | 0.007237 |
| 107 | false engagements 1–6 of 170: smallest (C3, C6) | §V-C | results/ablation_summary_exp.json | configs[config_id=C3_ekf7_vac_pn]/false_engages | 1 |
| 108 | … largest (C2) | §V-C | results/ablation_summary_exp.json | configs[config_id=C2_base_rae]/false_engages | 6 |
| 109 | no one-factor change moves them significantly (smallest false-engagement p) | §V-C | computed: min_false_engage_p_exp | — | 0.25 |
| 110 | protection stays at 99.9–100 % across threat drag (smallest value) | §V-C | computed: min_protection_drag_sweep_exp | — | 99.88 |
| 111 | median divert moves by about 1 m/s at most across threat drag (largest spread of one configuration's median, C1: 389.0–390.1) | §V-C | computed: max_divert_spread_drag_sweep_exp | — | 1.1 |
| 112 | velocity error 2.0 m/s (EKF7, C7) | §V-C | results/ablation_summary_exp.json | configs[config_id=C7_ekf7_drag_apn]/median_kf_rms_vel_mps | 1.98 |
| 113 | velocity error 3.1 m/s (KF6, C5) | §V-C | results/ablation_summary_exp.json | configs[config_id=C5_kf6_vac_apn]/median_kf_rms_vel_mps | 3.09 |
| 114 | at 100 m/s C7 protects 92.5 % | §V-C, §V-D | results/sensitivity_summary_exp.json | divert[value=100.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 92.53 |
| 115 | … C5 88.8 % | §V-C | results/sensitivity_summary_exp.json | divert[value=100.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 88.8 |
| 116 | C5→C7 rescues 34 | §V-C | results/sensitivity_summary_exp.json | divert[value=100.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/discordant_only_cmp | 34 |
| 117 | … loses 3 | §V-C | results/sensitivity_summary_exp.json | divert[value=100.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/discordant_only_ref | 3 |
| 118 | p = 1.2×10⁻⁷ | §V-C | results/sensitivity_summary_exp.json | divert[value=100.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/mcnemar_p | 1.2331e-07 |
| 119 | launch decision at 5 s | §V-C, Fig. 5 | results/consistency_ekf7.json | launch_time_s | 5.0 |
| 120 | median intercept time, constant (Fig. 5a mark) | Fig. 5 | results/consistency_ekf7.json | intercept_time_s/constant | 58.5 |
| 121 | median intercept time, exponential (Fig. 5b mark) | Fig. 5 | results/consistency_ekf7.json | intercept_time_s/exponential | 50.7 |
| 122 | at 5 s the median estimate is 3.99×10⁻⁵ | §V-C | results/consistency_ekf7.json | cases/configured_constant_rae/drag_at_launch/estimate_median | 3.9885e-05 |
| 123 | truth 4.0×10⁻⁵ | §V-C | results/consistency_ekf7.json | cases/configured_constant_rae/drag_at_launch/truth_median | 4e-05 |
| 124 | median absolute error about 1 % (0.044 of 4.0) | §V-C | results/consistency_ekf7.json | cases/configured_constant_rae/drag_at_launch/abs_error_median | 4.4117e-07 |
| 125 | exponential, at intercept: 3.51×10⁻⁵ | §V-C | results/consistency_ekf7.json | cases/configured_exponential_rae/drag_at_intercept/estimate_median | 3.5146e-05 |
| 126 | its 1σ 2.58×10⁻⁵ (prior 2.5) | §V-C | results/consistency_ekf7.json | cases/configured_exponential_rae/drag_at_intercept/sigma_median | 2.5786e-05 |

## Results: radar noise (Section V-D, Fig. 4)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 127 | C7 at 435 m/s: 104 m/s at 0.25× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=0.25]/configs/C7_ekf7_drag_apn/median_dv_mps | 103.6 |
| 128 | 130 m/s at 1× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=1.0]/configs/C7_ekf7_drag_apn/median_dv_mps | 130.2 |
| 129 | 152 m/s at 2× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C7_ekf7_drag_apn/median_dv_mps | 152.2 |
| 130 | 197 m/s at 4× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C7_ekf7_drag_apn/median_dv_mps | 197.0 |
| 131 | 2×: 4 % get through (C1 protects 96.0 %) | §V-D, §VI | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C1_base_cart/kill_rate_threatening | 96.02 |
| 132 | 2×: 12 % get through (C9 protects 88.4 %) | §V-D, §VI | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C9_adv_rae/kill_rate_threatening | 88.43 |
| 133 | 2×, C5 (inside the range) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 93.73 |
| 134 | 2×, C7 (inside the range) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 94.94 |
| 135 | 4×: 41 % get through (C7 protects 59.2 %) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 59.16 |
| 136 | 4×: 54 % get through (C9 protects 46.1 %) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C9_adv_rae/kill_rate_threatening | 46.14 |
| 137 | every miss used up its divert (misses − divert-exhausted misses, summed over the 435 m/s noise points) | §V-D | computed: noise_misses_not_exhausted | — | 0 |
| 138 | at 100 m/s C7 rises to 94.7 % at 0.25× | §V-D | results/sensitivity_noise_summary_exp.json | noise_100[value=0.25]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 94.7 |
| 139 | gravity term saves 253 m/s at 0.25× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=0.25]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -253.1 |
| 140 | … 192 m/s at 4× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/median_dv_diff_mps | -191.9 |
| 141 | 4×: C1 52.2 % | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C1_base_cart/kill_rate_threatening | 52.17 |
| 142 | 4×: C5 54.8 % | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 54.82 |
| 143 | 4×: p = 0.29 | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/mcnemar_p | 0.29 |
| 144 | 2×: C1 kills 46 seeds that C5 misses | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/discordant_only_ref | 46 |
| 145 | … C5 kills 27 that C1 misses | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/discordant_only_cmp | 27 |
| 146 | … p = 0.034 | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/contrasts/C1_base_cart->C5_kf6_vac_apn/mcnemar_p | 0.0344 |
| 147 | C5→C7 at 100 m/s: 3.7 points at 1× | §V-D | results/sensitivity_noise_summary_exp.json | noise_100[value=1.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/protection_diff_pts | 3.73 |
| 148 | … 15.8 points at 4× | §V-D | results/sensitivity_noise_summary_exp.json | noise_100[value=4.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/protection_diff_pts | 15.78 |
| 149 | … 36.3 % (C7) | §V-D | results/sensitivity_noise_summary_exp.json | noise_100[value=4.0]/configs/C7_ekf7_drag_apn/kill_rate_threatening | 36.27 |
| 150 | … 20.5 % (C5) | §V-D | results/sensitivity_noise_summary_exp.json | noise_100[value=4.0]/configs/C5_kf6_vac_apn/kill_rate_threatening | 20.48 |
| 151 | false engagements 4 of 170 (C7, 2×) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C7_ekf7_drag_apn/false_engages | 4 |
| 152 | … 5 (C7, 4×) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C7_ekf7_drag_apn/false_engages | 5 |
| 153 | vacuum PIP 13 (C5, 2×) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/configs/C5_kf6_vac_apn/false_engages | 13 |
| 154 | … 19 (C5, 4×) | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/configs/C5_kf6_vac_apn/false_engages | 19 |
| 155 | RAE loses 6.5 points at 2× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/contrasts/C7_ekf7_drag_apn->C9_adv_rae/protection_diff_pts | -6.51 |
| 156 | … 13.0 points at 4× | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/contrasts/C7_ekf7_drag_apn->C9_adv_rae/protection_diff_pts | -13.01 |
| 157 | … p = 1.2×10⁻⁸ | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=2.0]/contrasts/C7_ekf7_drag_apn->C9_adv_rae/mcnemar_p | 1.2340e-08 |
| 158 | … p = 4.9×10⁻¹⁰ | §V-D | results/sensitivity_noise_summary_exp.json | noise_435[value=4.0]/contrasts/C7_ekf7_drag_apn->C9_adv_rae/mcnemar_p | 4.8846e-10 |
| 159 | slant range 72–86 km; 85–105 m sideways noise | §V-D | docs/methods.md | `At the 72–86 km slant range where tracking starts, 1.2 mrad of angle noise is already 85–105 m of cross-range noise` | — |

## Results: no measurable effect, and filter honesty (Section V-E)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 160 | ZEM changes divert by < 3 m/s (constant) | §V-E, Table III | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C8_ekf7_drag_zem]/median_dv_diff_mps | -2.8 |
| 161 | matched NEES 7.08 | §V-E | results/consistency_ekf7.json | cases/matched_constant_rae/to_intercept/nees7/mean | 7.084 |
| 162 | 94.0 % of steps inside | §V-E | results/consistency_ekf7.json | cases/matched_constant_rae/to_intercept/nees7/inside | 0.9395 |
| 163 | bounds [6.49, 7.53] | §V-E | results/consistency_ekf7.json | nees7_bounds | [6.491, 7.528] |
| 164 | NIS 6.00 | §V-E | results/consistency_ekf7.json | cases/matched_constant_rae/to_intercept/nis/mean | 5.999 |
| 165 | as configured, constant: 4.55 | §V-E | results/consistency_ekf7.json | cases/configured_constant_rae/to_intercept/nees7/mean | 4.555 |
| 166 | as configured, exponential: 4.57 | §V-E | results/consistency_ekf7.json | cases/configured_exponential_rae/to_intercept/nees7/mean | 4.571 |

## Discussion and limitations (Sections VI–VII)

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 167 | C5 protects at least 99.9 % at every drag value (lowest, exponential, k_T = 6×10⁻⁵) | §VI | results/sensitivity_summary_exp.json | drag[value=6e-05]/configs/C5_kf6_vac_apn/kill_rate_threatening | 99.88 |
| 168 | … and 100 % at every drag value with constant density (lowest of the six) | §VI | computed: min_c5_protection_drag_sweep_const | — | 100.0 |
| 169 | drag PIP 36 false launches (drag-free threat, constant) | §VII | results/sensitivity_summary.json | drag[value=0.0]/configs/C7_ekf7_drag_apn/false_engages | 36 |
| 170 | vacuum PIP 7 | §VII | results/sensitivity_summary.json | drag[value=0.0]/configs/C5_kf6_vac_apn/false_engages | 7 |
| 171 | of 173 (1000 − 827 threatening) | §VII | results/sensitivity_summary.json | drag[value=0.0]/threatening | 827 |
| 172 | p < 10⁻⁸ | §VII | results/sensitivity_summary.json | drag[value=0.0]/contrasts/C5_kf6_vac_apn->C7_ekf7_drag_apn/false_engage_mcnemar_p | 3.7253e-09 |
| 173 | chord error about 1.5 cm at 30 g | Appendix | docs/methods.md | `about 1.5 cm at 30 g` | — |

## Table III (one-factor contrasts)

Each Table III row reads protection change (b/c, p), divert change [CI] and false engagements, for both atmospheres. Rows already listed above are not repeated. "0.0" protection changes come from `protection_diff_pts`, and "1.0" p-values from `mcnemar_p`.

| # | In the paper | Where | Source | Field | Raw |
| --- | --- | --- | --- | --- | --- |
| 174 | C1→C2 constant: b/c 14/10 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/discordant_only_cmp | 10 |
| 175 | C1→C2 constant: b = 14 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/discordant_only_ref | 14 |
| 176 | C1→C2 constant: p = 0.54 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/mcnemar_p | 0.5413 |
| 177 | C1→C2 constant: divert 0.0 (both capped) | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/median_dv_diff_mps | 0.0 |
| 178 | C1→C3 constant: b/c 1/11 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/discordant_only_cmp | 11 |
| 179 | C1→C3 constant: FE 115→113 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/false_engage_cmp | 113 |
| 180 | C3→C4 constant: b/c 2/0, p = 0.50 | Table III | results/ablation_summary.json | contrasts[ref=C3_ekf7_vac_pn,cmp=C4_ekf7_drag_pn]/mcnemar_p | 0.5 |
| 181 | C3→C4 constant: FE 113→5 | Table III | results/ablation_summary.json | contrasts[ref=C3_ekf7_vac_pn,cmp=C4_ekf7_drag_pn]/false_engage_cmp | 5 |
| 182 | C1→C5 constant: −158 m/s | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/median_dv_diff_mps | -157.8 |
| 183 | C1→C5 constant: [−163, −154] | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/median_dv_diff_ci95 | [-163.1, -153.6] |
| 184 | C5→C6 constant: [−141, −134] | Table III | results/ablation_summary.json | contrasts[ref=C5_kf6_vac_apn,cmp=C6_ekf7_vac_apn]/median_dv_diff_ci95 | [-140.6, -133.8] |
| 185 | C7→C8 constant: [−3.6, −2.1] | Table III | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C8_ekf7_drag_zem]/median_dv_diff_ci95 | [-3.6, -2.1] |
| 186 | C7→C9 constant: +0.4 | Table III | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/median_dv_diff_mps | 0.4 |
| 187 | C7→C9 constant: [−0.0, +0.9] | Table III | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/median_dv_diff_ci95 | [-0.0, 0.9] |
| 188 | C7→C9 constant: FE 5→3 | Table III | results/ablation_summary.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/false_engage_cmp | 3 |
| 189 | C1→C9 constant: −297 m/s | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/median_dv_diff_mps | -297.3 |
| 190 | C1→C9 constant: [−299, −294] | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/median_dv_diff_ci95 | [-299.0, -294.3] |
| 191 | C1→C9 constant: FE 115→3 | Table III | results/ablation_summary.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/false_engage_cmp | 3 |
| 192 | C1→C2 exponential: −0.2 points | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/protection_diff_pts | -0.24 |
| 193 | C1→C2 exponential: b = 2, p = 0.50 | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/mcnemar_p | 0.5 |
| 194 | C1→C2 exponential: FE 4→6 | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C2_base_rae]/false_engage_cmp | 6 |
| 195 | C1→C3 exponential: −0.1 points | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/protection_diff_pts | -0.12 |
| 196 | C1→C3 exponential: −7.7 m/s | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/median_dv_diff_mps | -7.7 |
| 197 | C1→C3 exponential: [−7.9, −7.6] | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C3_ekf7_vac_pn]/median_dv_diff_ci95 | [-7.9, -7.6] |
| 198 | C3→C4 exponential: +0.1, b/c 0/1 | Table III | results/ablation_summary_exp.json | contrasts[ref=C3_ekf7_vac_pn,cmp=C4_ekf7_drag_pn]/discordant_only_cmp | 1 |
| 199 | C5→C6 exponential: [−15.4, −15.1] | Table III | results/ablation_summary_exp.json | contrasts[ref=C5_kf6_vac_apn,cmp=C6_ekf7_vac_apn]/median_dv_diff_ci95 | [-15.4, -15.1] |
| 200 | C5→C6 exponential: FE 4→1 | Table III | results/ablation_summary_exp.json | contrasts[ref=C5_kf6_vac_apn,cmp=C6_ekf7_vac_apn]/false_engage_cmp | 1 |
| 201 | C7→C8 exponential: [+0.2, +1.2] | Table III | results/ablation_summary_exp.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C8_ekf7_drag_zem]/median_dv_diff_ci95 | [0.2, 1.2] |
| 202 | C7→C9 exponential: [−0.4, +0.6] | Table III | results/ablation_summary_exp.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/median_dv_diff_ci95 | [-0.4, 0.6] |
| 203 | C7→C9 exponential: FE 2→3 | Table III | results/ablation_summary_exp.json | contrasts[ref=C7_ekf7_drag_apn,cmp=C9_adv_rae]/false_engage_cmp | 3 |
| 204 | C1→C9 exponential: −259 m/s (bundle) | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/median_dv_diff_mps | -258.5 |
| 205 | C1→C9 exponential: [−261, −255] | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/median_dv_diff_ci95 | [-260.7, -255.0] |
| 206 | C1→C9 exponential: protection 0.0 (0/0) | Table III | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C9_adv_rae]/protection_diff_pts | 0.0 |
| 207 | Fig. 2 caption: "about 240 m/s" (the 243 m/s saving, row 2) | Fig. 2 | results/ablation_summary_exp.json | contrasts[ref=C1_base_cart,cmp=C5_kf6_vac_apn]/median_dv_diff_mps | -242.7 |
| 208 | RAE cross-beam velocity noise 8 m/s | Table I, §III-B, §VII, Appendix | configs/advanced.yaml | sensor/vel_noise | 8.0 |
| 209 | C1 threatening kills 480 (of 845; 845 − 480 = 365 missed threatening, plus 20 non-threatening = 385 engaged misses) | §V-A | results/ablation_summary.json | configs[config_id=C1_base_cart]/threatening_kills | 480 |
| 210 | Fig. 2 caption: "99.8–100 %" protection in exponential atmosphere (lowest is C2 at 99.76 %) | Fig. 2 | results/ablation_summary_exp.json | configs[config_id=C2_base_rae]/kill_rate_threatening | 99.76 |

## Code and data archive

| Item | Value | Checked against |
| --- | --- | --- |
| Version DOI | 10.5281/zenodo.24311162 (v1.1.0) | DataCite API: version "v1.1.0", "IsVersionOf 10.5281/zenodo.24208791" |
| Concept DOI | 10.5281/zenodo.24208791 | DataCite API |

## Reference verification

Each entry in `refs.bib` was checked on 2026-11-17. Crossref is `api.crossref.org/works/<DOI>`, which returns the publisher-registered metadata.

| Key | Checked against | Note |
| --- | --- | --- |
| kalman1960 | Crossref, DOI 10.1115/1.3662552 | |
| barshalom2001 | Crossref, DOI 10.1002/0471221279 | Crossref gives the online date 2002-01-04; the print edition is dated 2001 |
| zarchan2009kalman | Crossref, DOI 10.2514/4.867200 (3rd edition) | |
| palumbo2010filter | Article PDF on jhuapl.edu (vol. 29, no. 1, pp. 60–70) and the issue's contents page | No DOI is printed |
| mehra1971 | Crossref, DOI 10.1109/TAC.1971.1099744 | |
| farina2002 | Crossref, DOI 10.1109/TAES.2002.1039404 | Found by a Crossref search; a guessed DOI pointed to a different paper and was discarded |
| li2010survey | Crossref, DOI 10.1109/TAES.2010.5417150 | Found by a Crossref search, as for farina2002 |
| zarchan2012 | Crossref, DOI 10.2514/4.868948 (6th edition) | |
| shneydor1998 | Crossref, DOI 10.1533/9781782420590 | Crossref lists Woodhead Publishing as the publisher |
| murtaugh1966 | Crossref, DOI 10.1109/MSPEC.1966.5217080 | |
| cottrell1971 | Crossref, DOI 10.2514/3.6369 | |
| nesline1981 | Crossref, DOI 10.2514/3.56054 | |
| nesline1983 | Crossref, DOI 10.2514/3.19817 | |
| palumbo2010modern | Article PDF on jhuapl.edu (vol. 29, no. 1, pp. 42–59) and the issue's contents page | No DOI is printed |
| zarchan1999 | Crossref, DOI 10.1080/08929889908426470 | |
| mcnemar1947 | Crossref, DOI 10.1007/BF02295996 | |
| fagerland2013 | Crossref, DOI 10.1186/1471-2288-13-91 | |
| efron1979 | Crossref, DOI 10.1214/aos/1176344552 | |
| wilson1927 | Crossref, DOI 10.1080/01621459.1927.10502953 | |
| glasserman1992 | Crossref, DOI 10.1287/mnsc.38.6.884 | |
| lipton2019 | Crossref, DOI 10.1145/3317287.3328534 | |
| ussa1976 | NASA Technical Reports Server record 19770009539 (NASA-TM-X-74335, NOAA-S/T-76-1562) | No DOI |
