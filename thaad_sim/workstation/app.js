// THAAD GNC workstation: application controller
import { TacticalViewport3D, locateFrame } from './viewport3d.js';
import { Tactical2DViews } from './tactical2d.js';
import { TelemetryChartsManager } from './charts.js';
import { renderBatchView, renderMethodsView } from './batch_view.js';

const $ = (id) => document.getElementById(id);

const state = {
  data: null,
  compareRuns: [],
  presets: [],
  simTime: 0,
  tStart: 0,
  tEnd: 100,
  playing: true,
  speed: 1,
  slowMo: true,
  workspaceMode: 'split',
};

// Initialize 3D, 2D & Charts controllers
const view3d = new TacticalViewport3D(
  $('viewport-3d'),
  $('overlay-3d-labels'),
  $('seeker-viewport')
);

// OrbitControls fires 'change' from inside view3d.update() (controls.update()).
// Re-rendering synchronously from that event recursed until the stack overflowed,
// which also killed the animation loop. Flag it and redraw on the next frame instead.
let redraw3d = false;
view3d.controls.addEventListener('change', () => { redraw3d = true; });

const view2d = new Tactical2DViews(
  $('canvas-elev'),
  $('canvas-plan'),
  (t) => seekTo(t)
);

const chartsMgr = new TelemetryChartsManager(
  $('charts-grid'),
  (t) => seekTo(t)
);

function seekTo(t) {
  state.simTime = Math.max(state.tStart, Math.min(state.tEnd, t));
  syncFrame(state.simTime);
}

function setPlaying(play) {
  if (play && state.simTime >= state.tEnd - 1e-4) {
    state.simTime = state.tStart;
  }
  state.playing = play;
  $('btn-play').classList.toggle('is-playing', play);
}

function slowDownFactor(time) {
  if (!state.slowMo || !state.data) return 1;
  const cpaT = state.data.cpa?.time ?? state.data.derived?.endEventT;
  if (cpaT == null) return 1;
  const dist = Math.abs(time - cpaT);
  if (dist < 0.8) return 0.12;
  if (dist < 2.5) return 0.12 + ((dist - 0.8) / 1.7) * 0.88;
  return 1;
}

function updateParamSummary() {
  const law = $('ctrl-law').value.toUpperCase();
  const filt = $('ctrl-filter').value.toUpperCase();
  const pip = $('ctrl-pip').value === 'drag' ? 'drag PIP' : 'vacuum PIP';
  const cfgName = $('ctrl-config').selectedOptions[0]?.textContent || '';
  $('param-summary').textContent = `${cfgName} · ${filt} · ${law}`;
  $('param-summary').title = `${filt}, ${pip}, ${$('ctrl-sensor').value.toUpperCase()} radar, ${law}, divert ${$('ctrl-dv').value} m/s`;
}

// Fill the parameter form with a config file's own values
function applyConfigDefaults(path) {
  const c = state.configs?.[path];
  if (!c) return;
  if (c.kalman?.filter_type) $('ctrl-filter').value = c.kalman.filter_type;
  if (c.kalman?.pip_mode) $('ctrl-pip').value = c.kalman.pip_mode;
  if (c.sensor?.model) $('ctrl-sensor').value = c.sensor.model;
  const ic = c.interceptor || {};
  if (ic.guidance_law) $('ctrl-law').value = ic.guidance_law;
  if (ic.pn_gain != null) $('ctrl-pn').value = ic.pn_gain;
  if (ic.tau != null) $('ctrl-tau').value = ic.tau;
  if (ic.max_g != null) $('ctrl-maxg').value = ic.max_g;
  $('ctrl-dv').value = ic.kv_divert_dv == null ? '' : ic.kv_divert_dv;
}

function getFormOverrides() {
  return {
    filter_type: $('ctrl-filter').value,
    pip_mode: $('ctrl-pip').value,
    sensor_model: $('ctrl-sensor').value,
    guidance_law: $('ctrl-law').value,
    ...($('ctrl-dv').value.trim() === ''
      ? { unlimited_dv: true }
      : { kv_divert_dv: +$('ctrl-dv').value }),
    pn_gain: +$('ctrl-pn').value,
    tau: +$('ctrl-tau').value,
    max_g: +$('ctrl-maxg').value,
  };
}

// ---- Busy state ---------------------------------------------------------
// Runs take a couple of seconds on the Python side. Only the newest request
// may update the view, so a slow earlier run can never overwrite a newer one.
let runToken = 0;
let busyTimer = null;

function beginBusy(label) {
  const token = ++runToken;
  const started = performance.now();
  const status = $('run-status');
  document.body.classList.add('is-busy');
  status.className = 'run-status mono';
  const tick = () => {
    status.textContent = `${label} ${((performance.now() - started) / 1000).toFixed(0)} s`;
  };
  clearInterval(busyTimer);
  tick();
  busyTimer = setInterval(tick, 500);
  return token;
}

function endBusy(token, error = null) {
  if (token !== runToken) return; // a newer run owns the status line
  clearInterval(busyTimer);
  document.body.classList.remove('is-busy');
  const status = $('run-status');
  if (error) {
    status.className = 'run-status mono error';
    status.textContent = 'Run failed';
    status.title = String(error.message || error).slice(0, 400);
  } else {
    status.className = 'run-status mono';
    status.textContent = '';
    status.title = '';
  }
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch { /* plain text */ }
    throw new Error(msg);
  }
  return res.json();
}

async function runSimulation() {
  const seed = +$('ctrl-seed').value;
  const token = beginBusy(`Simulating seed ${seed}…`);
  try {
    const payload = await postJson('/api/simulate', {
      config: $('ctrl-config').value,
      seed,
      overrides: getFormOverrides(),
    });
    if (token !== runToken) return;
    state.compareRuns = [];
    $('compare-banner').hidden = true;
    loadPayload(payload, []);
    endBusy(token);
  } catch (err) {
    console.error('Simulation failed:', err);
    endBusy(token, err);
  }
}

async function runCompare3Laws() {
  const seed = +$('ctrl-seed').value;
  const token = beginBusy(`Comparing 3 laws on seed ${seed}…`);
  try {
    const { kv_divert_dv, unlimited_dv, pn_gain, max_g } = getFormOverrides();
    const common = { kv_divert_dv, unlimited_dv, pn_gain, max_g, tau: 0.0 };
    const runsReq = [
      { label: 'EKF7 + APN', color: '#5aa6d6', config: 'configs/advanced.yaml', seed,
        overrides: { filter_type: 'ekf7', pip_mode: 'drag', sensor_model: 'rae', guidance_law: 'apn', ...common } },
      { label: 'KF6 + PN', color: '#e0594a', config: 'configs/engineering.yaml', seed,
        overrides: { filter_type: 'kf6', pip_mode: 'vacuum', sensor_model: 'cartesian', guidance_law: 'pn', ...common } },
      { label: 'EKF7 + ZEM', color: '#6bbd8c', config: 'configs/advanced.yaml', seed,
        overrides: { filter_type: 'ekf7', pip_mode: 'drag', sensor_model: 'rae', guidance_law: 'zem', ...common } },
    ];
    const { runs } = await postJson('/api/compare', { runs: runsReq });
    if (token !== runToken) return;

    $('compare-legend').innerHTML = runs
      .map((r) => {
        const miss = r.cpa ? `${r.cpa.dist.toFixed(1)} m` : '—';
        const dvUsed = Math.round(r.derived.dvUsed[r.derived.dvUsed.length - 1] || 0);
        return `<div class="cmp-item">
          <span class="cmp-swatch" style="background:${r.compareColor}"></span>
          <strong>${r.compareLabel}</strong>
          <span class="badge ${r.outcome === 'KILL' ? 'ok' : 'bad'}">${r.outcome}</span>
          <span>miss ${miss} · Δv ${dvUsed} m/s</span>
        </div>`;
      })
      .join('');
    $('compare-banner').hidden = false;
    loadPayload(runs[0], runs.slice(1));
    endBusy(token);
  } catch (err) {
    console.error('Comparison failed:', err);
    endBusy(token, err);
  }
}

function loadPayload(D, compareRuns = []) {
  state.data = D;
  state.compareRuns = compareRuns;
  state.tStart = D.t[0];
  state.tEnd = D.t[D.t.length - 1];
  state.simTime = state.tStart;

  const slider = $('timeline-slider');
  slider.min = state.tStart;
  slider.max = state.tEnd;
  slider.value = state.tStart;

  // Outcome badge
  const outBadge = $('outcome-badge');
  outBadge.textContent = D.outcome === 'KILL' ? `Kill · ${D.kill.dist.toFixed(2)} m` : D.outcome === 'GROUND' ? 'Miss · ground impact' : D.outcome.toLowerCase();
  outBadge.className = `badge ${D.outcome === 'KILL' ? 'ok' : D.outcome === 'NO ENGAGE' ? 'info' : 'bad'}`;

  $('sim-meta-tag').textContent = `seed ${D.seed ?? '—'} · ${(D.filterType || 'kf6').toUpperCase()} · ${(D.guidanceLaw || 'pn').toUpperCase()}`;

  // Build BDA summary
  buildBdaCard(D);

  // Build Timeline Markers & Phase Jumps
  buildTimelineAndPhases(D);

  // Build Event Log
  buildEventLog(D);

  // Feed 3D, 2D, Charts, and Methods views
  view3d.setData(D, compareRuns);
  view2d.setData(D, compareRuns);
  chartsMgr.setData(D, compareRuns);
  renderMethodsView($('methods-content'), D);

  setPlaying(true);
  syncFrame(state.simTime);
}

function buildBdaCard(D) {
  const card = $('bda-card');
  const head = $('bda-headline');
  const sub = $('bda-subline');
  const metrics = $('bda-metrics');
  const drv = D.derived;
  const prm = D.params || {};

  card.className = `bda-card ${D.outcome === 'KILL' ? 'ok' : D.outcome === 'NO ENGAGE' ? 'info' : 'bad'}`;
  if (D.outcome === 'KILL') {
    head.textContent = 'Hit-to-kill';
    sub.textContent = `T+${D.kill.time.toFixed(2)} s · ${(D.kill.alt / 1000).toFixed(1)} km alt`;
  } else if (D.outcome === 'GROUND') {
    head.textContent = 'Miss, threat reached the ground';
    sub.textContent = `impact T+${state.tEnd.toFixed(1)} s`;
  } else if (D.outcome === 'NO ENGAGE') {
    head.textContent = 'No engagement';
    sub.textContent = 'PIP outside defended area';
  } else {
    head.textContent = D.outcome;
    sub.textContent = '';
  }

  const cpaDist = D.cpa ? `${D.cpa.dist.toFixed(2)} m` : '—';
  const closing = D.cpa ? `${D.cpa.closing.toFixed(0)} m/s` : '—';
  const dvFinal = Math.round(drv.dvUsed[drv.dvUsed.length - 1] || 0);
  const dvBudget = `${dvFinal} m/s`;
  const dvLabel = prm.kv_divert_dv ? `Divert Δv / ${prm.kv_divert_dv}` : 'Divert Δv';
  const maxG = D.g.length ? `${Math.max(...D.g).toFixed(1)} g` : '—';

  metrics.innerHTML = `
    <div class="bda-stat"><small>Miss (CPA)</small><strong>${cpaDist}</strong></div>
    <div class="bda-stat"><small>Closing</small><strong>${closing}</strong></div>
    <div class="bda-stat"><small>${dvLabel}</small><strong>${dvBudget}</strong></div>
    <div class="bda-stat"><small>Peak accel</small><strong>${maxG}</strong></div>
  `;
}

function buildTimelineAndPhases(D) {
  const drv = D.derived;
  const jumps = $('phase-jumps');
  const markers = $('timeline-markers');
  jumps.innerHTML = '';
  markers.innerHTML = '';

  const phases = [{ label: 'Track', t: state.tStart }];
  if (drv.launchT != null) phases.push({ label: 'Boost', t: drv.launchT });
  if (drv.burnT != null) phases.push({ label: 'Midcourse', t: drv.burnT });
  if (drv.terminalT != null) phases.push({ label: 'Terminal', t: drv.terminalT });
  if (drv.endEventT != null) {
    phases.push({
      label: D.outcome === 'KILL' ? 'Kill' : D.outcome === 'NO ENGAGE' ? 'Impact' : 'CPA',
      t: Math.max(state.tStart, (D.cpa?.time ?? drv.endEventT) - 1.5),
      exactT: D.cpa?.time ?? drv.endEventT,
    });
  }

  phases.forEach((p) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = `${p.label} ${(p.exactT ?? p.t).toFixed(1)}`;
    btn.title = `Jump to ${p.label.toLowerCase()} (T+${(p.exactT ?? p.t).toFixed(1)} s)`;
    btn.onclick = () => seekTo(p.t);
    jumps.appendChild(btn);

    if (p.t > state.tStart) {
      const tick = document.createElement('div');
      tick.className = 'tl-tick';
      const pct = (((p.exactT ?? p.t) - state.tStart) / Math.max(state.tEnd - state.tStart, 1e-6)) * 100;
      tick.style.left = `${pct}%`;
      markers.appendChild(tick);
    }
  });
}

function buildEventLog(D) {
  const list = $('event-log-list');
  list.innerHTML = '';
  (D.derived.events || []).forEach((ev) => {
    const li = document.createElement('li');
    li.dataset.t = ev.t;
    li.innerHTML = `
      <span class="ev-time">${ev.t.toFixed(2)}</span>
      <span class="ev-title">${ev.label}</span>
      <span class="ev-detail">${ev.detail}</span>
    `;
    li.onclick = () => seekTo(Math.max(state.tStart, ev.t - 0.5));
    list.appendChild(li);
  });
}

function syncFrame(time) {
  if (!state.data) return;
  const D = state.data;
  const drv = D.derived;
  const prm = D.params || {};
  const [fi] = locateFrame(D.t, time);

  // Clock & timeline slider
  $('mission-clock').textContent = time.toFixed(2).padStart(6, '0');
  $('timeline-slider').value = time;

  // Phase status
  const launched = drv.launchT != null && time >= drv.launchT;
  const killed = drv.killT != null && time >= drv.killT;
  const impacted = drv.impactT != null && time >= drv.impactT;
  const stg = D.stage[fi];

  let phaseText = 'SEARCH / TRACK';
  if (killed) phaseText = 'TARGET DESTROYED';
  else if (impacted) phaseText = D.outcome === 'NO ENGAGE' ? 'WEAPONS HOLD' : 'GROUND IMPACT';
  else if (launched && stg === 1) phaseText = 'STAGE 1 BOOST';
  else if (launched && drv.terminalT != null && time >= drv.terminalT) phaseText = 'TERMINAL HOMING';
  else if (launched) phaseText = 'KV MIDCOURSE COAST';
  $('phase-badge').textContent = phaseText;

  // Threat card
  $('ro-th-status').textContent = killed ? 'DESTROYED' : impacted ? 'IMPACTED' : 'DESCENDING';
  $('ro-th-alt').textContent = `${drv.altT[fi].toFixed(2)} km`;
  $('ro-th-spd').textContent = `${drv.spdT[fi].toFixed(0)} m/s`;
  $('ro-th-rng').textContent = `${drv.rngT[fi].toFixed(2)} km`;
  const a3 = 3 * fi;
  $('ro-th-pos').textContent = `${(D.threat[a3] / 1000).toFixed(1)}, ${(D.threat[a3 + 1] / 1000).toFixed(1)}, ${(D.threat[a3 + 2] / 1000).toFixed(1)}`;
  $('ro-th-drag').textContent = `${((prm.threat_drag || 0.00004) * 1e5).toFixed(2)}×10⁻⁵ m⁻¹`;

  // Interceptor / KV card
  const dvUsed = drv.dvUsed[fi] || 0;
  const dvBudget = prm.kv_divert_dv;
  const gVal = D.g[fi] || 0;
  $('ro-kv-stage').textContent = !launched ? 'ON LAUNCHER' : killed ? 'HIT TARGET' : stg === 1 ? 'BOOST' : 'KV COAST';
  $('ro-kv-alt').textContent = launched && drv.altI[fi] != null ? `${drv.altI[fi].toFixed(2)} km` : '0.00 km';
  $('ro-kv-spd').textContent = launched && drv.spdI[fi] != null ? `${drv.spdI[fi].toFixed(0)} m/s` : '0 m/s';
  $('ro-kv-g').textContent = launched ? `${gVal.toFixed(1)} / ${D.maxG || 30} g` : '0.0 g';
  $('ro-kv-mass').textContent = `${Math.round(drv.mass[fi] || 900)} kg`;
  $('ro-kv-dv').textContent = dvBudget ? `${Math.round(dvUsed)} / ${dvBudget} m/s` : `${Math.round(dvUsed)} m/s`;

  $('bar-g').style.width = `${Math.min(100, (gVal / (D.maxG || 30)) * 100)}%`;
  if (dvBudget) {
    const remPct = Math.max(0, Math.min(100, (1 - dvUsed / dvBudget) * 100));
    $('bar-dv').style.width = `${remPct}%`;
    $('bar-dv').classList.toggle('low', remPct < 20);
  } else {
    $('bar-dv').style.width = '100%';
  }

  // Guidance card
  $('ro-guid-law').textContent = `${(D.guidanceLaw || 'pn').toUpperCase()} · N=${prm.pn_gain || 4}`;
  $('ro-en-sep').textContent = launched && drv.sep[fi] != null ? `${drv.sep[fi].toFixed(3)} km` : '—';
  $('ro-en-vc').textContent = launched && drv.vc[fi] != null ? `${drv.vc[fi].toFixed(0)} m/s` : '—';
  $('ro-en-ttg').textContent = launched && drv.ttg[fi] != null ? `${drv.ttg[fi].toFixed(1)} s` : '—';
  $('ro-en-zem').textContent = launched && drv.zemTrue[fi] != null
    ? (() => { const z = drv.zemTrue[fi], ze = drv.zemEst[fi] ?? 0, d = Math.abs(z) >= 100 ? 0 : 1; return `${z.toFixed(d)} / ${ze.toFixed(d)}`; })()
    : '—';
  $('ro-en-los').textContent = launched && drv.losTrue[fi] != null
    ? `${drv.losTrue[fi].toFixed(2)} / ${(drv.losEst[fi] ?? 0).toFixed(2)}`
    : '—';

  // Kalman Filter card
  $('ro-kf-type').textContent = `${(D.filterType || 'kf6').toUpperCase()} · ${(D.sensorModel || 'cartesian').toUpperCase()}`;
  $('ro-kf-pos').textContent = `${drv.epos[fi].toFixed(1)} m  3σ ${drv.spos3[fi].toFixed(0)}`;
  $('ro-kf-vel').textContent = `${drv.evel[fi].toFixed(1)} m/s  3σ ${drv.svel3[fi].toFixed(0)}`;
  const nees = D.nees?.[fi];
  const nis = D.nis?.[fi];
  $('ro-kf-nees').textContent = `${nees != null ? nees.toFixed(1) : '—'} / ${nis != null ? nis.toFixed(1) : '—'}`;
  $('ro-kf-drag').textContent = D.estDrag ? `${D.estDrag[fi].toFixed(2)}×10⁻⁵ m⁻¹` : 'not estimated';
  const pb = drv.pipBias[fi], pvb = drv.pipVacBias[fi];
  $('ro-kf-pip').textContent = pb != null ? `${Math.round(pb)} m  (${Math.round(pvb ?? 0)})` : '—';

  // Highlight active event in log
  const evItems = document.querySelectorAll('#event-log-list li');
  let activeIdx = -1;
  evItems.forEach((li, i) => {
    const past = time >= +li.dataset.t;
    li.classList.toggle('past', past);
    if (past) activeIdx = i;
  });
  evItems.forEach((li, i) => li.classList.toggle('current', i === activeIdx));

  // Update 3D, 2D, and Charts
  if (state.workspaceMode === 'split') {
    view3d.update(time);
  } else if (state.workspaceMode === 'charts') {
    view2d.update(time);
    chartsMgr.updateCursor(time);
  }
}

// Export full telemetry CSV
function exportTelemetryCsv() {
  if (!state.data) return;
  const D = state.data;
  const drv = D.derived;
  const cols = [
    't_s', 'status', 'stage', 'threat_e_m', 'threat_n_m', 'threat_u_m',
    'kv_e_m', 'kv_n_m', 'kv_u_m', 'sep_km', 'closing_mps', 'g_cmd',
    'divert_dv_mps', 'zem_true_m', 'zem_est_m', 'los_true_mrads',
    'kf_pos_err_m', 'kf_vel_err_mps', 'nees', 'nis', 'pip_bias_m',
  ];
  const lines = [cols.join(',')];
  for (let i = 0; i < D.t.length; i++) {
    const a = 3 * i;
    lines.push([
      D.t[i], D.status[i], D.stage[i],
      D.threat[a], D.threat[a + 1], D.threat[a + 2],
      D.interceptor[a], D.interceptor[a + 1], D.interceptor[a + 2],
      drv.sep[i] ?? '', drv.vc[i] ?? '', D.g[i] ?? 0,
      drv.dvUsed[i] ?? 0, drv.zemTrue[i] ?? '', drv.zemEst[i] ?? '',
      drv.losTrue[i] ?? '', drv.epos[i] ?? '', drv.evel[i] ?? '',
      D.nees?.[i] ?? '', D.nis?.[i] ?? '', drv.pipBias[i] ?? '',
    ].join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `thaad_telemetry_seed_${D.seed ?? 0}.csv`;
  a.click();
}

// Wire UI Controls
function bindUI() {
  $('sim-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runSimulation();
  });

  ['ctrl-config', 'ctrl-filter', 'ctrl-pip', 'ctrl-sensor', 'ctrl-law', 'ctrl-dv', 'ctrl-pn', 'ctrl-tau', 'ctrl-maxg']
    .forEach((id) => $(id).addEventListener('input', updateParamSummary));
  updateParamSummary();

  // Close open dropdowns on outside click or Escape
  document.addEventListener('pointerdown', (e) => {
    document.querySelectorAll('details.dropdown[open]').forEach((d) => {
      if (!d.contains(e.target)) d.open = false;
    });
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') document.querySelectorAll('details.dropdown[open]').forEach((d) => { d.open = false; });
  });

  $('btn-rand-seed').onclick = () => {
    $('ctrl-seed').value = Math.floor(Math.random() * 1000);
    runSimulation();
  };

  $('btn-compare-seed').onclick = () => runCompare3Laws();
  $('btn-clear-compare').onclick = () => {
    state.compareRuns = [];
    $('compare-banner').hidden = true;
    if (state.data) loadPayload(state.data, []);
  };
  $('btn-export-csv').onclick = exportTelemetryCsv;

  // Workspace mode switcher
  document.querySelectorAll('#workspace-tabs button').forEach((btn) => {
    btn.onclick = () => {
      const mode = btn.dataset.ws;
      state.workspaceMode = mode;
      document.querySelectorAll('#workspace-tabs button').forEach((b) => b.classList.toggle('active', b === btn));
      const ws = $('workspace');
      ws.className = `ws-${mode}`;
      $('pane-batch').hidden = mode !== 'batch';
      $('pane-methods').hidden = mode !== 'methods';
      if (mode === 'charts') {
        chartsMgr.setFilterGroup('all');
        document.querySelectorAll('#chart-filter-tabs button').forEach((b) => b.classList.toggle('active', b.dataset.cgrp === 'all'));
      }
      if (mode === 'batch' && !$('batch-content').childElementCount) loadBatchDataset($('batch-dataset-select').value);
      requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
        chartsMgr.redrawAll(true);
        syncFrame(state.simTime);
      });
    };
  });

  // Transport controls
  $('btn-play').onclick = () => setPlaying(!state.playing);
  $('btn-restart').onclick = () => { seekTo(state.tStart); setPlaying(true); };
  $('btn-step-back').onclick = () => { setPlaying(false); seekTo(state.simTime - 0.1); };
  $('btn-step-fwd').onclick = () => { setPlaying(false); seekTo(state.simTime + 0.1); };
  $('timeline-slider').addEventListener('input', (e) => seekTo(+e.target.value));
  $('chk-slowmo').onchange = (e) => { state.slowMo = e.target.checked; };

  $('ctrl-speed').onchange = (e) => { state.speed = +e.target.value; e.target.blur(); };
  $('chk-seeker').onchange = (e) => {
    $('cell-seeker').hidden = !e.target.checked;
    requestAnimationFrame(() => { window.dispatchEvent(new Event('resize')); syncFrame(state.simTime); });
  };

  // 3D Camera & Layer controls
  document.querySelectorAll('#cam-mode-btns button').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('#cam-mode-btns button').forEach((b) => b.classList.toggle('active', b === btn));
      $('cam-label').textContent = btn.textContent;
      view3d.setCameraMode(btn.dataset.cam, true);
      syncFrame(state.simTime);
    };
  });

  $('slider-vscale').addEventListener('input', (e) => {
    view3d.visualScale = +e.target.value;
    $('lbl-vscale').textContent = `${view3d.visualScale}×`;
    syncFrame(state.simTime);
  });

  document.querySelectorAll('.layer-bar input[data-layer]').forEach((chk) => {
    chk.onchange = () => {
      view3d.layers[chk.dataset.layer] = chk.checked;
      syncFrame(state.simTime);
    };
  });

  // 2D Tactical submode buttons
  document.querySelectorAll('#tactical-submode button').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('#tactical-submode button').forEach((b) => b.classList.toggle('active', b === btn));
      $('tactical-grid').className = `tactical-grid mode-${btn.dataset.t2d}`;
      requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
        syncFrame(state.simTime);
      });
    };
  });

  // Chart group filter tabs
  document.querySelectorAll('#chart-filter-tabs button').forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll('#chart-filter-tabs button').forEach((b) => b.classList.toggle('active', b === btn));
      chartsMgr.setFilterGroup(btn.dataset.cgrp);
    };
  });

  // Batch controls
  $('batch-dataset-select').onchange = (e) => loadBatchDataset(e.target.value);
  $('btn-run-live-batch').onclick = runLiveMonteCarlo;

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.code === 'ArrowLeft') { e.preventDefault(); setPlaying(false); seekTo(state.simTime - (e.shiftKey ? 1.0 : 0.1)); }
    else if (e.code === 'ArrowRight') { e.preventDefault(); setPlaying(false); seekTo(state.simTime + (e.shiftKey ? 1.0 : 0.1)); }
    else if (e.code === 'Home') { e.preventDefault(); seekTo(state.tStart); }
    else if (e.code === 'End') { e.preventDefault(); seekTo(state.tEnd); }
  });
}

async function loadBatchDataset(datasetName) {
  try {
    const res = await fetch(`/api/batch?dataset=${encodeURIComponent(datasetName)}`);
    const json = await res.json();
    renderBatchView($('batch-content'), json.data, openSeed);
  } catch (err) {
    console.error('Failed to load batch:', err);
  }
}

async function runLiveMonteCarlo() {
  const btn = $('btn-run-live-batch');
  const n = Math.max(5, Math.min(300, +$('live-batch-n').value || 20));
  const started = performance.now();
  btn.disabled = true;
  const tick = () => { btn.textContent = `Running ${n} seeds… ${((performance.now() - started) / 1000).toFixed(0)} s`; };
  tick();
  const timer = setInterval(tick, 500);
  try {
    const data = await postJson('/api/batch/run', {
      config: $('ctrl-config').value,
      seed_start: 0,
      n,
      overrides: getFormOverrides(),
    });
    renderBatchView($('batch-content'), data, openSeed);
  } catch (err) {
    console.error('Live batch failed:', err);
    $('batch-content').innerHTML = `<div class="batch-section"><h3>Batch failed</h3><p class="sub">${String(err.message || err).slice(0, 300)}</p></div>`;
  } finally {
    clearInterval(timer);
    btn.textContent = 'Run batch';
    btn.disabled = false;
  }
}

function openSeed(seed) {
  $('ctrl-seed').value = seed;
  document.querySelector('#workspace-tabs button[data-ws="split"]').click();
  runSimulation();
}

// Animation loop
let lastNow = performance.now();
function animate(now) {
  const dt = Math.min(0.05, (now - lastNow) / 1000);
  lastNow = now;

  try {
    if (state.playing && state.data) {
      state.simTime += dt * state.speed * slowDownFactor(state.simTime);
      if (state.simTime >= state.tEnd) {
        state.simTime = state.tEnd;
        setPlaying(false);
      }
      redraw3d = false;
      syncFrame(state.simTime);
    } else if (state.data && state.workspaceMode === 'split') {
      // Paused: keep orbit damping and camera moves responsive
      view3d.controls.update();
      if (redraw3d) {
        redraw3d = false;
        view3d.update(state.simTime);
      }
    }
  } catch (err) {
    console.error(err);
  } finally {
    requestAnimationFrame(animate);
  }
}

// Initial bootstrap
async function init() {
  bindUI();
  try {
    const res = await fetch('/api/configs');
    const { presets, configs } = await res.json();
    state.presets = presets || [];
    state.configs = Object.fromEntries((configs || []).map((c) => [c.path, c.settings]));
    $('ctrl-config').onchange = () => { applyConfigDefaults($('ctrl-config').value); updateParamSummary(); };
    const presetSel = $('ctrl-preset');
    presetSel.innerHTML = state.presets
      .map((p) => `<option value="${p.id}">${p.name}</option>`)
      .join('');
    presetSel.onchange = () => {
      const p = state.presets.find((x) => x.id === presetSel.value);
      if (!p) return;
      // Start from the preset's config, then apply its overrides, so nothing
      // leaks over from whatever was selected before.
      $('ctrl-config').value = p.config;
      $('ctrl-seed').value = p.seed;
      applyConfigDefaults(p.config);
      const o = p.overrides || {};
      if (o.filter_type) $('ctrl-filter').value = o.filter_type;
      if (o.pip_mode) $('ctrl-pip').value = o.pip_mode;
      if (o.sensor_model) $('ctrl-sensor').value = o.sensor_model;
      if (o.guidance_law) $('ctrl-law').value = o.guidance_law;
      if (o.unlimited_dv) $('ctrl-dv').value = '';
      else if (o.kv_divert_dv) $('ctrl-dv').value = o.kv_divert_dv;
      if (o.pn_gain) $('ctrl-pn').value = o.pn_gain;
      if (o.tau != null) $('ctrl-tau').value = o.tau;
      if (o.max_g) $('ctrl-maxg').value = o.max_g;
      updateParamSummary();
      runSimulation();
    };
  } catch (e) {
    console.error('Failed to load presets:', e);
  }

  await runSimulation();
  lastNow = performance.now();
  requestAnimationFrame(animate);
}

init();

