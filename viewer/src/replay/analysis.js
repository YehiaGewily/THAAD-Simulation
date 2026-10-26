// The Analysis drawer: per-frame series derived from the replay, the figures
// on the Kinematics, Guidance and Kalman filter tabs, the Methods tab and the
// CSV export.
import { D, MODE_NAMES, N, T, tStart } from './data.js';
import { $, COL, toast } from './dom.js';
import { figure, xMax } from './charts.js';
import { SERIES } from './palette.js';

// ============================================================================
// ANALYSIS – synchronized figures, filter diagnostics and methods
// ============================================================================
// Series hues validated with the dataviz palette check on the dark chart surface
const SER = SERIES;
const PRM = D.params || {};
const dtSim = PRM.dt ?? 0.02;
const hyp = Math.hypot;

const S = {};
['altT', 'altI', 'spdT', 'spdI', 'sep', 'vc', 'los', 'gcmd', 'dv', 'zem', 'ex', 'ey', 'ez', 'sx', 'sy', 'sz', 'ev', 'svn', 'epos', 'eraw']
  .forEach((k) => { S[k] = new Float32Array(N).fill(NaN); });
{
  let dvAcc = 0;
  for (let i = 0; i < N; i++) {
    const a = 3 * i, k = 6 * i, e = D.kfErr, sg = D.kfSig;
    const tx = D.threat[a], ty = D.threat[a + 1], tz = D.threat[a + 2];
    const tvx = D.threatVel[a], tvy = D.threatVel[a + 1], tvz = D.threatVel[a + 2];
    S.altT[i] = tz / 1000;
    S.spdT[i] = hyp(tvx, tvy, tvz);
    S.ex[i] = e[k]; S.ey[i] = e[k + 1]; S.ez[i] = e[k + 2];
    S.sx[i] = 3 * sg[k]; S.sy[i] = 3 * sg[k + 1]; S.sz[i] = 3 * sg[k + 2];
    S.epos[i] = hyp(e[k], e[k + 1], e[k + 2]);
    S.ev[i] = hyp(e[k + 3], e[k + 4], e[k + 5]);
    S.svn[i] = 3 * hyp(sg[k + 3], sg[k + 4], sg[k + 5]);
    // Logged truth is one step after the measurement, so step it back for the raw radar error
    if (D.status[i] !== 2) S.eraw[i] = hyp(D.radar[a] - (tx - tvx * dtSim), D.radar[a + 1] - (ty - tvy * dtSim), D.radar[a + 2] - (tz - tvz * dtSim));
    if (D.status[i] > 0) {
      const ix = D.interceptor[a], iy = D.interceptor[a + 1], iz = D.interceptor[a + 2];
      const ivx = D.interceptorVel[a], ivy = D.interceptorVel[a + 1], ivz = D.interceptorVel[a + 2];
      S.altI[i] = iz / 1000;
      S.spdI[i] = hyp(ivx, ivy, ivz);
      const rx = tx - ix, ry = ty - iy, rz = tz - iz, vx = tvx - ivx, vy = tvy - ivy, vz = tvz - ivz;
      const r = Math.max(hyp(rx, ry, rz), 1e-6), rv = rx * vx + ry * vy + rz * vz, vv = vx * vx + vy * vy + vz * vz;
      S.sep[i] = r / 1000;
      S.vc[i] = -rv / r;
      S.los[i] = (1000 * hyp(ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx)) / (r * r);
      S.gcmd[i] = D.g[i];
      if (D.stage[i] === 2) dvAcc += D.g[i] * 9.81 * (i ? T[i] - T[i - 1] : dtSim);
      S.dv[i] = dvAcc;
      const tgo = vv > 1e-9 ? Math.max(0, -rv / vv) : 0;
      S.zem[i] = hyp(rx + vx * tgo, ry + vy * tgo, rz + vz * tgo);
    }
  }
}

// Filter statistics after the initial transient (t >= 2 s)
const kfStats = (() => {
  let n = 0, sp = 0, sv = 0, sr = 0, nr = 0, inside = 0;
  for (let i = 0; i < N; i++) {
    if (T[i] < tStart + 2 || !Number.isFinite(S.epos[i])) continue;
    n++; sp += S.epos[i] ** 2; sv += S.ev[i] ** 2;
    if (Math.abs(S.ex[i]) <= S.sx[i] && Math.abs(S.ey[i]) <= S.sy[i] && Math.abs(S.ez[i]) <= S.sz[i]) inside++;
    if (Number.isFinite(S.eraw[i])) { nr++; sr += S.eraw[i] ** 2; }
  }
  return n ? { pos: Math.sqrt(sp / n), vel: Math.sqrt(sv / n), raw: nr ? Math.sqrt(sr / nr) : NaN, inside: (100 * inside) / n, n } : null;
})();

// ---- figure definitions ----
const fmtU = (unit, d = 1) => (v) => `${v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d })} ${unit}`;
const fmtSig = (unit) => (v) => `${Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3)} ${unit}`;
let analysisBuilt = false;

function figGrid(id) { const g = document.createElement('div'); g.className = 'figs'; $(id).appendChild(g); return g; }

export function buildAnalysis() {
  if (analysisBuilt) return;
  analysisBuilt = true;

  const kin = figGrid('tab-kin');
  figure(kin, 'kin', {
    title: 'Altitude', unit: 'km', fmt: fmtU('km', 2), yMin: 0,
    series: [{ label: 'Threat', color: SER.threat, data: S.altT }, { label: 'Interceptor', color: SER.kv, data: S.altI }],
    caption: 'Height above the z = 0 ground plane for the threat and the interceptor (true states).',
  });
  figure(kin, 'kin', {
    title: 'Speed', unit: 'm/s', fmt: fmtU('m/s', 0), yMin: 0,
    series: [{ label: 'Threat', color: SER.threat, data: S.spdT }, { label: 'Interceptor', color: SER.kv, data: S.spdI }],
    caption: 'Magnitude of the inertial velocity. The interceptor accelerates only during boost; afterwards gravity and drag slow it.',
  });
  figure(kin, 'kin', {
    title: 'Separation', unit: 'km', fmt: fmtSig('km'), log: true,
    series: [{ label: 'Separation', color: SER.eng, data: S.sep }],
    refs: [{ y: (D.killRadius || 10) / 1000, label: `kill radius ${D.killRadius} m` }],
    caption: 'Distance between interceptor and threat from launch onward, ‖r‖ with r = r<sub>T</sub> − r<sub>I</sub>.',
  });
  figure(kin, 'kin', {
    title: 'Closing velocity', unit: 'm/s', fmt: fmtU('m/s', 0),
    series: [{ label: 'Closing velocity', color: SER.eng, data: S.vc }],
    caption: 'V<sub>c</sub> = −dr/dt = −(r·v)/‖r‖, with v the relative velocity. Positive while the two bodies approach.',
  });

  const gui = figGrid('tab-gui');
  figure(gui, 'gui', {
    title: 'Line-of-sight rate', unit: 'mrad/s', fmt: fmtSig('mrad/s'), yMin: 0,
    series: [{ label: 'LOS rate', color: SER.kv, data: S.los }],
    caption: '‖ω‖ = ‖r × v‖ / ‖r‖² from the true states. Proportional navigation commands acceleration to drive this rate to zero.',
  });
  figure(gui, 'gui', {
    title: 'Commanded acceleration', unit: 'g', fmt: fmtU('g', 2), yMin: 0,
    series: [{ label: 'Command', color: SER.kv, data: S.gcmd }],
    refs: [{ y: D.maxG, label: `limit ${D.maxG} g` }],
    caption: 'Guidance command ‖a<sub>c</sub>‖ after the acceleration limit, including the 1 g gravity bias. During boost only 10 % of it is applied.',
  });
  figure(gui, 'gui', {
    title: 'Divert Δv used', unit: 'm/s', fmt: fmtU('m/s', 1), yMin: 0,
    series: [{ label: 'Δv used', color: SER.kv, data: S.dv }],
    refs: PRM.kv_divert_dv ? [{ y: PRM.kv_divert_dv, label: `budget ${PRM.kv_divert_dv} m/s` }] : undefined,
    caption: PRM.kv_divert_dv
      ? 'Cumulative ∫‖a<sub>c</sub>‖dt spent by the kill vehicle after burnout. Once the budget is reached the kill vehicle can no longer steer.'
      : 'Cumulative ∫‖a<sub>c</sub>‖dt spent by the kill vehicle after burnout. This mode has no divert budget.',
  });
  figure(gui, 'gui', {
    title: 'Zero-effort miss', unit: 'm', fmt: fmtSig('m'), log: true,
    series: [{ label: 'ZEM', color: SER.eng, data: S.zem }],
    refs: [{ y: D.killRadius || 10, label: `kill radius ${D.killRadius} m` }],
    caption: 'Miss distance if neither body accelerated from this instant: ‖r + v·t<sub>go</sub>‖ with t<sub>go</sub> = −(r·v)/‖v‖². The intercept is assured once it stays below the kill radius.',
  });

  const kf = figGrid('tab-kf');
  if (kfStats) {
    const st = document.createElement('dl');
    st.className = 'stats';
    const red = kfStats.raw / kfStats.pos;
    st.innerHTML = [
      ['Position RMS error', `${kfStats.pos.toFixed(2)} m`, `3D, t ≥ 2 s, n = ${kfStats.n.toLocaleString('en-US')}`],
      ['Velocity RMS error', `${kfStats.vel.toFixed(2)} m/s`, '3D, same samples'],
      ['Noise reduction', `${red.toFixed(1)}×`, `raw radar ${kfStats.raw.toFixed(1)} m RMS → filtered ${kfStats.pos.toFixed(2)} m`],
      ['3σ consistency', `${kfStats.inside.toFixed(1)} %`, 'samples inside ±3σ on all three axes; ≈ 99.2 % for a consistent filter'],
    ].map(([k, v, n]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd><small>${n}</small></div>`).join('');
    $('tab-kf').insertBefore(st, kf);
  }
  const axisFig = (label, err, sig) => figure(kf, 'kf', {
    title: `Position error, ${label}`, unit: 'm', fmt: fmtSig('m'), thin: true,
    series: [{ label: 'Error', color: SER.threat, data: err }],
    band: { hi: sig, symmetric: true, color: SER.threat, label: '±3σ' },
    caption: `Kalman estimate minus truth along ${label}, with the filter's ±3σ bound from √P.`,
  });
  axisFig('east (x)', S.ex, S.sx);
  axisFig('north (y)', S.ey, S.sy);
  axisFig('up (z)', S.ez, S.sz);
  figure(kf, 'kf', {
    title: 'Velocity error', unit: 'm/s', fmt: fmtSig('m/s'), thin: true, yMin: 0,
    series: [{ label: '‖v̂ − v‖', color: SER.threat, data: S.ev }],
    band: { hi: S.svn, symmetric: false, color: SER.threat, label: '3σ bound' },
    caption: 'Magnitude of the velocity estimate error against the 3σ bound from the velocity block of P.',
  });
  const kfMag = figure(kf, 'kf', {
    title: 'Filtered vs raw position error', unit: 'm', fmt: fmtSig('m'), log: true, thin: true,
    series: [{ label: 'Kalman estimate', color: SER.threat, data: S.epos }, { label: 'Raw radar', color: SER.kv, data: S.eraw }],
    caption: 'Magnitude of the position error of the raw radar measurements and of the filtered estimate.',
  });
  kfMag.el.style.gridColumn = '1 / -1';

  buildMethods();
}

// ---- methods ----
function tex(src, display = true) {
  return window.katex ? window.katex.renderToString(src, { displayMode: display, throwOnError: false }) : `<code>${src}</code>`;
}
function fmtParam(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-2) { const e = Math.floor(Math.log10(a)); return `${+(v / 10 ** e).toFixed(2)} × 10<sup>${e}</sup>`; }
  return v.toLocaleString('en-US', { maximumSignificantDigits: 6 });
}
const DEFAULT_CONFIGS = { 'ENGINEERING MODE': 'configs/engineering.yaml', 'MAX-PERFORMANCE MODE': 'configs/demo.yaml' };

function buildMethods() {
  const p = PRM;
  const gated = D.engageRule === 'PIP-GATED LAUNCH';
  const PARAMS = [
    ['dt', String.raw`\Delta t`, 'Integration time step', 's'],
    ['g', 'g', 'Gravitational acceleration', 'm/s²'],
    ['pos_noise', String.raw`\sigma_p`, 'Radar position noise, 1σ per axis', 'm'],
    ['vel_noise', String.raw`\sigma_v`, 'Radar velocity noise, 1σ per axis', 'm/s'],
    ['kf_q', 'q', 'White-noise acceleration spectral density, per axis', 'm²/s³'],
    ['threat_drag', 'k_T', 'Threat drag coefficient', '1/m'],
    ['boost_thrust', 'F_T', 'Boost thrust', 'N'],
    ['isp', String.raw`I_{sp}`, 'Booster specific impulse', 's'],
    ['mass_flow', String.raw`\dot m`, 'Propellant mass flow, F_T / (I_sp g₀)', 'kg/s'],
    ['burn_time', 't_b', 'Burn time, (m₀ − m_b) / ṁ', 's'],
    ['mass_wet', 'm_0', 'Launch mass', 'kg'],
    ['mass_dry', 'm_b', 'Mass at burnout', 'kg'],
    ['kv_mass', String.raw`m_{KV}`, 'Kill-vehicle mass', 'kg'],
    ['booster_drag', 'k_B', 'Booster drag coefficient', 'kg/m'],
    ['kv_drag', String.raw`k_{KV}`, 'Kill-vehicle drag coefficient', 'kg/m'],
    ['pn_gain', 'N', 'Navigation constant', '—'],
    ['max_g', String.raw`n_{max}`, 'Acceleration limit', 'g'],
    ['kv_divert_dv', String.raw`\Delta v_{KV}`, 'Kill-vehicle divert budget', 'm/s'],
    ['kill_radius', 'R_k', 'Kill radius', 'm'],
    ['defended_radius', 'R_D', 'Defended radius', 'm'],
    ['min_track_time', String.raw`t_{trk}`, 'Minimum track time before launch', 's'],
    ['t_max', String.raw`t_{max}`, 'Base simulation window', 's'],
  ].filter(([k]) => p[k] != null);

  const config = D.config ?? DEFAULT_CONFIGS[D.mode] ?? 'configs/engineering.yaml';
  const configArg = /\s/.test(config) ? `"${config}"` : config;
  const cmd = D.seed != null ? `python -m thaad_sim --config ${configArg} --seed ${D.seed}${D.legacyRng ? ' --legacy-rng' : ''}` : '';
  const rk4 = D.integrator !== 'euler';
  const sections = [
    ['Frame and integration', `
      <p>Flat-Earth east–north–up frame (x east, y north, z up) in SI units, constant gravity. Each body's state (position and velocity, plus mass for the interceptor) is advanced as one first-order system ẋ = f(x) with ${rk4 ? 'the classical fourth-order Runge–Kutta method (RK4)' : 'forward Euler'} at Δt = ${p.dt} s${rk4 ? '; the global error falls as Δt⁴' : '; the global error falls as Δt'}.</p>
      <div class="eq">${tex(rk4
        ? String.raw`\mathbf k_1 = f(\mathbf x_n),\ \ \mathbf k_2 = f\!\left(\mathbf x_n + \tfrac{\Delta t}{2}\mathbf k_1\right),\ \ \mathbf k_3 = f\!\left(\mathbf x_n + \tfrac{\Delta t}{2}\mathbf k_2\right),\ \ \mathbf k_4 = f(\mathbf x_n + \Delta t\,\mathbf k_3),\qquad \mathbf x_{n+1} = \mathbf x_n + \tfrac{\Delta t}{6}\left(\mathbf k_1 + 2\mathbf k_2 + 2\mathbf k_3 + \mathbf k_4\right)`
        : String.raw`\mathbf x_{n+1} = \mathbf x_n + \Delta t\, f(\mathbf x_n)`)}</div>
      <p>Guidance runs once per step and its command is held over the step. The step that contains burnout is split at t<sub>b</sub>, so thrust cuts off and the booster separates at the right instant. The integrator is set by <code>sim.integrator</code> (<code>rk4</code> or <code>euler</code>); <code>scripts/convergence_study.py</code> checks the observed orders.</p>`],
    ['Threat dynamics', `
      <p>Point mass under gravity and quadratic drag.</p>
      <div class="eq">${tex(String.raw`\ddot{\mathbf r}_T = -g\,\hat{\mathbf z} - k_T\,\lVert\dot{\mathbf r}_T\rVert\,\dot{\mathbf r}_T`)}</div>`],
    ['Interceptor dynamics', `
      <p>Thrust F<sub>T</sub> acts along the guidance command during boost. Mass is part of the integrated state: the motor burns propellant at the rate set by thrust and specific impulse until m<sub>0</sub> − m<sub>b</sub> is spent at t<sub>b</sub>. The booster then separates and the kill vehicle (mass m<sub>KV</sub>) steers with its divert thrusters.</p>
      <div class="eq">${tex(String.raw`m\,\ddot{\mathbf r}_I = F_T\,\hat{\mathbf a}_c - m g\,\hat{\mathbf z} - k\,\lVert\dot{\mathbf r}_I\rVert\,\dot{\mathbf r}_I + \eta\, m\,\mathbf a_c,\qquad \eta = \begin{cases}0.1 & t < t_b\\ 1 & t \ge t_b\end{cases}`)}</div>
      <div class="eq">${tex(String.raw`\dot m = -\frac{F_T}{I_{sp}\,g_0}\ \ (t < t_b),\qquad t_b = \frac{(m_0 - m_b)\,I_{sp}\,g_0}{F_T},\qquad m(t_b^+) = m_{KV},\quad F_T = 0,\ \dot m = 0\ \ (t > t_b)`)}</div>
      <p>With no gravity or drag this gives the rocket equation, Δv = I<sub>sp</sub> g<sub>0</sub> ln(m<sub>0</sub>/m<sub>b</sub>), with g<sub>0</sub> = 9.80665 m/s².</p>`],
    ['Guidance: proportional navigation', `
      <p>Relative state from the Kalman estimate of the threat, r = r̂<sub>T</sub> − r<sub>I</sub> and v = v̂<sub>T</sub> − v<sub>I</sub>. The command includes a gravity bias and is capped at n<sub>max</sub>·g${p.kv_divert_dv ? '. The kill vehicle has a finite divert budget' : ''}.</p>
      <div class="eq">${tex(String.raw`\boldsymbol\omega = \frac{\mathbf r\times\mathbf v}{\lVert\mathbf r\rVert^2},\qquad \mathbf a_c = N\,(\mathbf v\times\boldsymbol\omega) + g\,\hat{\mathbf z},\qquad \lVert\mathbf a_c\rVert \le n_{max}\,g`)}</div>
      ${p.kv_divert_dv ? `<div class="eq">${tex(String.raw`\int_{t_b}^{t}\lVert\mathbf a_c\rVert\,d\tau \;\le\; \Delta v_{KV}`)}</div>` : ''}`],
    ['Tracking: Kalman filter', `
      <p>Six-state linear filter. The threat moves under known gravity, entered as a control input, and is perturbed by continuous white-noise acceleration of spectral density q on each axis. The radar observes position and velocity directly (H = I) with independent Gaussian noise. The first measurement initializes the state, with P₀ = R.</p>
      <div class="eq">${tex(String.raw`\mathbf x = \begin{bmatrix}\mathbf p\\ \mathbf v\end{bmatrix},\quad F = \begin{bmatrix} I & \Delta t\, I\\ 0 & I\end{bmatrix},\quad B = \begin{bmatrix} \tfrac12\Delta t^2 I\\ \Delta t\, I\end{bmatrix},\quad \mathbf u = -g\,\hat{\mathbf z},\quad R = \operatorname{diag}(\sigma_p^2 I,\ \sigma_v^2 I)`)}</div>
      <div class="eq">${tex(String.raw`Q = q\begin{bmatrix} \tfrac13\Delta t^3 I & \tfrac12\Delta t^2 I\\ \tfrac12\Delta t^2 I & \Delta t\, I\end{bmatrix}`)}</div>
      <div class="eq">${tex(String.raw`\mathbf x^- = F\mathbf x + B\mathbf u,\qquad P^- = F P F^\top + Q`)}</div>
      <div class="eq">${tex(String.raw`S = H P^- H^\top + R,\qquad K = P^- H^\top S^{-1},\qquad \mathbf x = \mathbf x^- + K\boldsymbol\nu,\quad \boldsymbol\nu = \mathbf z - H\mathbf x^-`)}</div>
      <div class="eq">${tex(String.raw`P = (I - K H)\,P^-\,(I - K H)^\top + K R K^\top`)}</div>
      <p>Q is the white-noise acceleration integrated over one step. K comes from a linear solve with S rather than an explicit inverse, and the Joseph-form update keeps P symmetric and positive definite for any gain.</p>
      <p>The <b>Track uncertainty</b> layer (<kbd>E</kbd>) draws the 3σ position ellipsoid from the position block of P, with semi-axes 3√λ along its eigenvectors. It is drawn 10× larger about the true position, the estimate's offset included, so the threat is inside the shell exactly when the estimate error is within 3σ. With H = I and equal noise on every axis, P stays isotropic and the ellipsoid is a sphere.</p>
      <p>Consistency is checked with the normalized estimation error squared (NEES) and the normalized innovation squared (NIS). For a consistent filter both are χ² with 6 degrees of freedom; averaged over N independent runs, N times the average is χ² with 6N degrees of freedom.</p>
      <div class="eq">${tex(String.raw`\epsilon = (\hat{\mathbf x} - \mathbf x)^\top P^{-1} (\hat{\mathbf x} - \mathbf x),\qquad \epsilon_\nu = \boldsymbol\nu^\top S^{-1} \boldsymbol\nu,\qquad \mathbb E[\epsilon] = \mathbb E[\epsilon_\nu] = 6`)}</div>
      <p>The filter does not model drag, but the threat has it. With drag off and the truth driven by the same Q, NEES and NIS stay inside their 95 % bounds. Against the dragged threat the filter is overconfident early in the flight, while drag is strongest. <code>scripts/filter_consistency.py</code> plots both cases.</p>`],
    ['Launch decision', gated ? `
      <p>The predicted impact point (PIP) is the ballistic ground crossing from the current estimate, ignoring drag. The interceptor launches once the track is at least t<sub>trk</sub> old, the threat is descending and the PIP is inside the defended radius.</p>
      <p>Because drag is ignored, the PIP lies beyond the true impact point and the predicted time to impact is too short. The bias is kilometres early in flight and shrinks to metres in the last ten seconds; <code>scripts/drag_mismatch.py</code> measures it.</p>
      <div class="eq">${tex(String.raw`t^\ast = \frac{v_z + \sqrt{v_z^2 + 2 g z}}{g},\qquad \mathrm{PIP} = (x + v_x t^\ast,\ y + v_y t^\ast),\qquad \text{launch if } \lVert\mathrm{PIP}\rVert \le R_D`)}</div>` : `
      <p>This mode launches on the first descending track estimate after 3 s and does not gate on the predicted impact point. The PIP shown is for reference only.</p>`],
    ['Kill assessment', `
      <p>Within each step both bodies are taken to move along the straight chords between their integrated start and end positions; the closest point of approach inside the step decides the kill.</p>
      <div class="eq">${tex(String.raw`t^\ast = \operatorname{clamp}\!\left(-\frac{\Delta\mathbf p\cdot\Delta\mathbf v}{\lVert\Delta\mathbf v\rVert^2},\ 0,\ \Delta t\right),\qquad \text{kill if } \lVert\Delta\mathbf p + \Delta\mathbf v\, t^\ast\rVert < R_k`)}</div>`],
    ['Parameters used in this run', `
      <table class="params"><thead><tr><th>Symbol</th><th>Quantity</th><th style="text-align:right">Value</th><th>Unit</th></tr></thead><tbody>
      ${PARAMS.map(([k, sym, desc, unit]) => `<tr><td class="sym">${tex(sym, false)}</td><td>${desc}</td><td class="v">${fmtParam(p[k])}</td><td class="u">${unit}</td></tr>`).join('')}
      </tbody></table>`],
    ['Run', `
      <p>${MODE_NAMES[D.mode] ?? D.mode} mode · outcome <code>${D.outcome}</code> · ${N.toLocaleString('en-US')} frames over ${xMax.toFixed(2)} s${D.seed != null ? ` · seed <code>${D.seed}</code>` : ''}.</p>
      ${cmd ? `<p>Reproduce this exact scenario from the project folder:</p><div class="cmd"><code id="repro">${cmd}</code><button id="copy-cmd">Copy</button></div>` : '<p>This replay was generated without a recorded seed.</p>'}`],
    ['Assumptions and limits', `
      <ul class="assume">
        <li>Flat Earth, constant gravity, no Earth rotation.</li>
        <li>Quadratic drag with constant coefficients: no atmosphere model or Mach dependence.</li>
        <li>The radar measures Cartesian position and velocity directly; real sensors measure range and angles.</li>
        <li>The Kalman filter and the impact predictor assume drag-free flight, while the threat has drag. The mismatch is not corrected: it biases the predicted impact point and makes the filter overconfident.</li>
        <li>Drag uses a constant coefficient with no air-density model, so it acts at full strength even at 70 km altitude and overstates the deceleration of a real high-altitude threat.</li>
        <li>Point-mass vehicles with no attitude dynamics, actuator lag or seeker model.</li>
        <li>Constant thrust and specific impulse during boost. The kill vehicle's mass is held at m<sub>KV</sub>: its divert budget is a Δv limit, and divert propellant is not modeled.</li>
        <li>Analysis quantities (LOS rate, closing velocity, ZEM) are computed from the true states; guidance itself uses the Kalman estimate.</li>
        <li>Parameters are illustrative, not data for any real system.</li>
      </ul>`],
  ];
  $('tab-methods').innerHTML = `<div class="methods">${sections.map(([h, body], i) => `<section><h3><span>${String(i + 1).padStart(2, '0')}</span>${h}</h3>${body}</section>`).join('')}</div>`;
  const copy = $('copy-cmd');
  if (copy) copy.onclick = () => {
    const done = () => toast('Command copied');
    if (navigator.clipboard) navigator.clipboard.writeText(cmd).then(done, () => toast('Copy failed: select the command and copy it'));
    else toast('Copy failed: select the command and copy it');
  };
}

export function exportCsv() {
  const cols = ['t_s', 'status', 'stage',
    'threat_x_m', 'threat_y_m', 'threat_z_m', 'threat_vx_mps', 'threat_vy_mps', 'threat_vz_mps',
    'int_x_m', 'int_y_m', 'int_z_m', 'int_vx_mps', 'int_vy_mps', 'int_vz_mps',
    'est_x_m', 'est_y_m', 'est_z_m', 'radar_x_m', 'radar_y_m', 'radar_z_m',
    'kf_err_x_m', 'kf_err_y_m', 'kf_err_z_m', 'kf_err_vx_mps', 'kf_err_vy_mps', 'kf_err_vz_mps',
    'kf_sigma_x_m', 'kf_sigma_y_m', 'kf_sigma_z_m', 'kf_sigma_vx_mps', 'kf_sigma_vy_mps', 'kf_sigma_vz_mps',
    'accel_cmd_g', 'separation_m', 'closing_velocity_mps', 'los_rate_mradps', 'zero_effort_miss_m', 'divert_dv_mps'];
  const STATUS = ['SEARCH', 'FLYING', 'HIT'], STAGE = ['IDLE', 'BOOST', 'KV_COAST'];
  const num = (v, d) => (Number.isFinite(v) ? +v.toFixed(d) : '');
  const v3 = (arr, i) => [arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]];
  const lines = [cols.join(',')];
  for (let i = 0; i < N; i++) {
    lines.push([
      T[i].toFixed(3), STATUS[D.status[i]], STAGE[D.stage[i]],
      ...v3(D.threat, i), ...v3(D.threatVel, i), ...v3(D.interceptor, i), ...v3(D.interceptorVel, i),
      ...v3(D.track, i), ...v3(D.radar, i),
      ...D.kfErr.slice(6 * i, 6 * i + 6), ...D.kfSig.slice(6 * i, 6 * i + 6),
      D.g[i], num(S.sep[i] * 1000, 2), num(S.vc[i], 2), num(S.los[i], 4), num(S.zem[i], 3), num(S.dv[i], 2),
    ].join(','));
  }
  const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `thaad_${(MODE_NAMES[D.mode] ?? 'replay').toLowerCase().replace(/\s+/g, '_')}_${D.seed != null ? `seed${D.seed}` : 'replay'}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast(`Exported ${N.toLocaleString('en-US')} rows × ${cols.length} columns`);
}
