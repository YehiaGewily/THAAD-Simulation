# Methods

A short technical description of what the simulation computes, how it was verified, and where it stops being a good model. Parameter values are those of `configs/engineering.yaml` unless noted; `configs/demo.yaml` uses the same equations with different numbers.

> Educational simulation. The parameters are chosen to make the dynamics visible and are not data for any real system.

## 1. Frame, state and integration

The frame is flat-Earth east–north–up: $x$ east, $y$ north, $z$ up, in SI units, with the defended point at the origin and constant gravity $g = 9.81\ \mathrm{m/s^2}$.

Each body is a point mass whose state is advanced as one first-order system $\dot{\mathbf x} = f(\mathbf x)$:

| Body | State |
| --- | --- |
| Threat | $[\mathbf r, \mathbf v]$, 6 values |
| Interceptor | $[\mathbf r, \mathbf v, m]$, 7 values, so mass is integrated with the motion |

`sim.integrator` selects classical fourth-order Runge–Kutta (default) or forward Euler, at $\Delta t = 0.02$ s:

$$
\mathbf k_1 = f(\mathbf x_n),\quad
\mathbf k_2 = f\!\left(\mathbf x_n + \tfrac{\Delta t}{2}\mathbf k_1\right),\quad
\mathbf k_3 = f\!\left(\mathbf x_n + \tfrac{\Delta t}{2}\mathbf k_2\right),\quad
\mathbf k_4 = f(\mathbf x_n + \Delta t\,\mathbf k_3),
$$

$$
\mathbf x_{n+1} = \mathbf x_n + \tfrac{\Delta t}{6}\left(\mathbf k_1 + 2\mathbf k_2 + 2\mathbf k_3 + \mathbf k_4\right).
$$

Inputs that are not part of the state (the guidance command) are held constant over a step. Burnout is a discontinuity, so the step that contains it is split at the burnout time.

## 2. Threat

A point mass under gravity and quadratic drag with a constant coefficient $k_T = 4\times10^{-5}\ \mathrm{m^{-1}}$:

$$
\ddot{\mathbf r}_T = -g\,\hat{\mathbf z} - k_T\,\lVert\dot{\mathbf r}_T\rVert\,\dot{\mathbf r}_T .
$$

Each scenario draws a nominal time to impact $T$ (70–100 s), an initial altitude $z_0$ (55–75 km), a ground range (35–55 km) at a random azimuth, and an aim point up to 35 km from the origin. The initial velocity is the vacuum ballistic solution that reaches the aim point after $T$,

$$
v_{z0} = \frac{\tfrac12 g T^2 - z_0}{T},\qquad v_{x0} = \frac{x_{aim} - x_0}{T},\qquad v_{y0} = \frac{y_{aim} - y_0}{T},
$$

multiplied by a drag compensation factor of 1.03.

## 3. Interceptor

Two stages. During boost a constant thrust $F_T$ = 150 kN acts along the guidance command; propellant burns at the rate set by thrust and specific impulse until $m_0 - m_b$ is spent:

$$
m\,\ddot{\mathbf r}_I = F_T\,\hat{\mathbf a}_c - m g\,\hat{\mathbf z} - k\,\lVert\dot{\mathbf r}_I\rVert\,\dot{\mathbf r}_I + \eta\, m\,\mathbf a_c,
\qquad \eta = \begin{cases}0.1 & t < t_b\\ 1 & t \ge t_b\end{cases}
$$

$$
\dot m = -\frac{F_T}{I_{sp}\,g_0},\qquad t_b = \frac{(m_0 - m_b)\,I_{sp}\,g_0}{F_T} = 5.0\ \mathrm{s},\qquad g_0 = 9.80665\ \mathrm{m/s^2}.
$$

At burnout the booster separates: the mass state jumps to the kill vehicle's $m_{KV}$ = 200 kg, thrust stops, and the drag coefficient changes from the booster's to the kill vehicle's. The kill vehicle steers with divert thrusters whose total use is capped by a $\Delta v$ budget (435 m/s; unlimited in the demo). Divert propellant is not modeled, so $m_{KV}$ stays constant.

## 4. Guidance

Proportional navigation with a gravity bias, on the relative state from the Kalman estimate of the threat ($\mathbf r = \hat{\mathbf r}_T - \mathbf r_I$, $\mathbf v = \hat{\mathbf v}_T - \mathbf v_I$), with navigation constant $N = 4$:

$$
\boldsymbol\omega = \frac{\mathbf r\times\mathbf v}{\lVert\mathbf r\rVert^2},\qquad
\mathbf a_c = N\,(\mathbf v\times\boldsymbol\omega) + g\,\hat{\mathbf z},\qquad
\lVert\mathbf a_c\rVert \le n_{max}\,g = 30\,g,\qquad
\int_{t_b}^{t}\lVert\mathbf a_c\rVert\,d\tau \le \Delta v_{KV}.
$$

The command is computed once per step and held over it. Below 20 m range no command is issued.

## 5. Sensing and tracking

The radar measures Cartesian position and velocity with independent Gaussian noise, $\sigma_p = 25$ m and $\sigma_v = 8$ m/s per axis. A six-state linear Kalman filter tracks the threat with gravity as a known control input and continuous white-noise acceleration of spectral density $q = 30\ \mathrm{m^2/s^3}$ per axis:

$$
\mathbf x_{k+1} = F\mathbf x_k + B\mathbf u + \mathbf w_k,\qquad \mathbf z_k = H\mathbf x_k + \boldsymbol\nu_k,
$$

$$
F = \begin{bmatrix} I & \Delta t\, I\\ 0 & I\end{bmatrix},\quad
B = \begin{bmatrix} \tfrac12\Delta t^2 I\\ \Delta t\, I\end{bmatrix},\quad
\mathbf u = -g\,\hat{\mathbf z},\quad
H = I,\quad
R = \operatorname{diag}(\sigma_p^2 I,\ \sigma_v^2 I),\quad
Q = q\begin{bmatrix} \tfrac13\Delta t^3 I & \tfrac12\Delta t^2 I\\ \tfrac12\Delta t^2 I & \Delta t\, I\end{bmatrix}.
$$

The update uses a linear solve for the gain and the Joseph form for the covariance:

$$
S = HP^-H^\top + R,\qquad K = P^-H^\top S^{-1},\qquad \mathbf x = \mathbf x^- + K(\mathbf z - H\mathbf x^-),\qquad P = (I - KH)P^-(I - KH)^\top + KRK^\top .
$$

The first measurement initializes the state with $P_0 = R$.

**The track-uncertainty layer in the replay.** The viewer draws the filter's $3\sigma$ position ellipsoid: the position block of $P$ is diagonalized every frame, and the semi-axes are $3\sqrt{\lambda_i}$ along the eigenvectors. The geometry is exaggerated 10× about the true threat position (both the ellipsoid and the estimate's offset from the truth), so the threat is inside the shell exactly when the estimate error is inside the $3\sigma$ ellipsoid. With this sensor model the ellipsoid is always a sphere: $H = I$, the per-axis noise is equal and $P_0 = R$, so $F$, $Q$, $R$ and therefore $P$ have the same block structure on every axis. The position block of $P$ is $\sigma^2(t)\,I$ throughout, and the sphere shrinks from $3\sigma_p = 75$ m as the track converges. A range–angle radar would make it a true ellipsoid, and the viewer handles that case.

## 6. Launch decision and kill assessment

**Predicted impact point (PIP).** The drag-free ground crossing from the current estimate:

$$
t^\ast = \frac{v_z + \sqrt{v_z^2 + 2 g z}}{g},\qquad \mathrm{PIP} = (x + v_x t^\ast,\ y + v_y t^\ast).
$$

In engineering mode the interceptor launches once the track is 5 s old, the estimate is descending and $\lVert\mathrm{PIP}\rVert \le R_D$ = 30 km; otherwise the outcome is `NO ENGAGE`. The demo launches on the first descending estimate after 3 s.

**Kill check.** Within each step, both bodies move along the straight chords between their integrated start and end positions, and the closest point of approach decides the kill:

$$
t_{min} = \operatorname{clamp}\!\left(-\frac{\Delta\mathbf p\cdot\Delta\mathbf v}{\lVert\Delta\mathbf v\rVert^2},\ 0,\ \Delta t\right),\qquad \text{kill if } \lVert\Delta\mathbf p + \Delta\mathbf v\, t_{min}\rVert < R_k = 10\ \mathrm{m}.
$$

The chord departs from the curved path by at most $a\Delta t^2/8$, about 1.5 cm at 30 g.

## 7. Verification

Every number below comes from a script in `scripts/` or a test in `tests/`, and the data behind the figures is committed in `results/`.

### Integrator convergence

`scripts/convergence_study.py` integrates two smooth reference problems at $\Delta t$, $\Delta t/2$, $\Delta t/4$ and $\Delta t/8$ and measures the final-position error against an RK4 run at a 16× smaller step.

| Problem | Δt (s) | Euler error (m) | RK4 error (m) | Observed order, Euler | Observed order, RK4 |
| --- | --- | --- | --- | --- | --- |
| Dragged ballistic threat, 40 s | 0.8 → 0.1 | 65.2 → 8.18 | 3.6×10⁻⁴ → 8.3×10⁻⁸ | 1.00 | 4.01 |
| Interceptor through burnout, 12.8 s | 0.64 → 0.08 | 1373 → 180 | 8.4×10⁻⁴ → 2.4×10⁻⁷ | 0.99 | 4.00 |

The orders are those between the two finest steps. RK4 keeps fourth order through burnout because the step containing it is split.

![Integrator convergence](figures/convergence.png)

### Analytic checks

With drag off the threat has a closed-form solution $\mathbf r(t) = \mathbf r_0 + \mathbf v_0 t - \tfrac12 g t^2\hat{\mathbf z}$. RK4 is exact for this quadratic and reproduces it to $5.8\times10^{-11}$ m over 40 s, a few times machine epsilon at 60 km. Forward Euler gets the velocity exactly, and its altitude lags by exactly the truncation error $g\,t\,\Delta t/2$, to within $4.5\times10^{-11}$ m.

![Analytic check](figures/analytic.png)

`tests/test_integrators.py` also checks energy conservation without drag, that the energy lost with drag equals the drag work $\int k\lVert\mathbf v\rVert^3dt$, the mass flow $F/(I_{sp} g_0)$, booster separation at burnout, and the rocket-equation $\Delta v = I_{sp} g_0 \ln(m_0/m_b)$.

### Kalman filter consistency

For a consistent filter the normalized estimation error squared and the normalized innovation squared,

$$
\epsilon = (\hat{\mathbf x} - \mathbf x)^\top P^{-1}(\hat{\mathbf x} - \mathbf x),\qquad \epsilon_\nu = \boldsymbol\nu^\top S^{-1}\boldsymbol\nu,
$$

are $\chi^2$ with 6 degrees of freedom. Averaged over $N$ = 50 independent runs, $N$ times the average is $\chi^2_{300}$, giving 95 % bounds [5.08, 7.00]. `scripts/filter_consistency.py`:

| Setting | Mean NEES | Steps inside bounds | Mean NIS | Steps inside bounds |
| --- | --- | --- | --- | --- |
| Matched model: drag off, truth driven by the filter's $Q$ | 6.11 | 94.3 % | 5.99 | 95.1 % |
| As configured: threat with drag | 5.35 | 58.9 % | 5.92 | 94.6 % |

With its own model the filter is consistent. Against the dragged threat, NEES is about 8–9 in the first 15 s (overconfident while drag is strongest) and settles near 5 later (mildly conservative). NIS hardly notices, because with $H = I$ the innovations are dominated by $R$.

![Average NEES](figures/nees.png)

### Drag mismatch in the impact prediction

The filter and the PIP ignore drag, while the threat has it. Over 100 seeded threats (`scripts/drag_mismatch.py`) the PIP lies downrange of the true impact point by a median 1.7 km at 80 s to go, 185 m at 40 s and 3 m at 10 s, and the predicted time to impact is too short. In a drag-free control the median along-track error is within ±10 m. At the first launch opportunity, 12 of 100 PIP-gated decisions would launch at a threat that lands outside the defended area, and 1 would hold fire on one that lands inside.

![PIP bias](figures/pip_bias.png)

### Batch statistics

`python -m thaad_sim.batch` runs N seeded engagements; rates are reported with 95 % Wilson score intervals. Over seeds 0–999, engineering mode engages 960 threats and kills 575 of them: 59.9 % (95 % CI 56.8–63.0 %). All 385 misses had spent the full divert budget. The demo kills 1000 of 1000 (95 % CI 99.6–100 %). `python -m thaad_sim.dashboard results/engineering.csv` collects these results with the studies above in one page.

### Colour and contrast

`scripts/palette_check.py` checks the replay and dashboard colours: every pair of scene identity colours (threat, interceptor, accent, Kalman track, radar returns, uncertainty ellipsoid) for separation under simulated protanopia and deuteranopia (Machado et al. 2009, OKLab ΔE ≥ 8 target) and under normal vision (ΔE ≥ 15), every mark against the rendered sky, terrain, defended area, cloud deck and HUD in both rendering styles (WCAG ≥ 3:1), and text tokens (≥ 4.5:1). The audit changed the Kalman track to mint, the radar returns to a neutral slate, the ellipsoid to violet, brightened the faint HUD text and dimmed the cinematic cloud deck. Every check now passes, and `tests/test_presentation.py` keeps it that way.

## 8. Limitations

- **Flat Earth, constant gravity**, no rotation. Acceptable for engagements of tens of kilometres, not for longer flights.
- **Drag** has a constant coefficient with no atmosphere or Mach dependence, so it acts at full strength at 70 km and overstates the slowing of a real high-altitude threat (a dragged threat takes 1.5–2× its nominal flight time).
- **The filter and the predictor do not model drag.** The resulting PIP bias and early overconfidence are measured above, not corrected. An extended Kalman filter with a ballistic coefficient, or a predictor that integrates the dragged dynamics, would fix it.
- **The radar** is an idealized Cartesian position-and-velocity sensor with Gaussian noise at every step: no range–angle geometry, update rate, clutter, dropouts or false tracks.
- **Point masses**: no attitude dynamics, actuator lag, seeker model, or warhead and lethality model. A kill is geometric (closest approach under 10 m).
- **Propulsion** has constant thrust and specific impulse. The divert budget is a Δv limit, not propellant mass.
- **Guidance timing**: the command is held over each step, so the closed loop converges at first order in Δt whatever the integrator. The vehicle dynamics between commands are integrated at the integrator's order.
- **Visualization**: vehicle models are drawn about 8× real size and the uncertainty ellipsoid 10×. The analysis figures compute LOS rate, closing velocity and zero-effort miss from the true states, while guidance uses the estimate.
