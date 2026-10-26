// The HUD: brand and scenario text, event log, timeline ticks, phase stepper,
// debrief card, playback controls, camera and layer switches, keyboard
// shortcuts, the rendering style, the Analysis drawer and the per-frame
// telemetry readouts.
import * as THREE from 'three';
import {
  D, MODE_NAMES, N, RULE_NAMES, T, burnT, defR, endEventT, impactPos, impactT, killT, launchT,
  pipPos, tEnd, terminalT, threatStart, tStart,
} from './data.js';
import { $, COL, fmtKm, fmtSpd, toast } from './dom.js';
import { seg } from './charts.js';
import { buildAnalysis, exportCsv } from './analysis.js';
import { SPEEDS, camLookTarget, hashArgs, seek, state } from './state.js';
import { bloom, camera, clouds, composer, controls, labelRenderer, renderer, terrainMat } from './scene.js';
import { trailMinDefaults, trailUniforms, tubeMeshes } from './effects.js';

$('mode').textContent = MODE_NAMES[D.mode] ?? D.mode;
$('rule').textContent = `${RULE_NAMES[D.engageRule] ?? D.engageRule}${D.seed != null ? ` · seed ${D.seed}` : ''}`;
{
  const alt = threatStart.y, rng = Math.hypot(threatStart.x, threatStart.z);
  $('scenario').innerHTML =
    `Threat appears at <b>${alt.toFixed(1)} km</b> altitude, <b>${rng.toFixed(1)} km</b> out. ` +
    `Defended radius <b>${defR.toFixed(0)} km</b>, kill radius <b>${D.killRadius} m</b>.`;
}

const events = [{ t: tStart, text: 'Radar track initiated', cls: 'info' }];
if (D.outcome === 'NO ENGAGE') events.push({ t: D.minTrackTime ?? tStart, text: 'PIP outside defended area — weapons hold', cls: 'info' });
if (launchT != null) events.push({ t: launchT, text: D.engageRule === 'PIP-GATED LAUNCH' ? 'PIP inside defended area — launch' : 'Track confirmed — launch', cls: 'warn' });
if (burnT != null) events.push({ t: burnT, text: 'Booster burnout, kill vehicle separates', cls: 'kv' });
if (terminalT != null) events.push({ t: terminalT, text: 'Terminal homing, under 15 km', cls: 'kv' });
if (killT != null) events.push({ t: killT, text: `Target destroyed — ${D.kill.dist.toFixed(2)} m miss`, cls: 'ok' });
if (D.outcome === 'GROUND') events.push({ t: impactT, text: 'Threat impact — intercept failed', cls: 'bad' });
if (D.outcome === 'NO ENGAGE') events.push({ t: impactT, text: 'Threat lands outside the defended area', cls: 'info' });
if (D.outcome === 'TIMEOUT') events.push({ t: T[N - 1], text: 'Sim timeout, threat still airborne', cls: 'bad' });
events.sort((a, b) => a.t - b.t);

const logEl = $('log');
events.forEach((e) => {
  const li = document.createElement('li');
  li.className = e.cls;
  li.innerHTML = `<span class="lt">T+${e.t.toFixed(2)}</span><span class="lx">${e.text}</span>`;
  logEl.appendChild(li);
  e.el = li;
});

const tickColors = { info: COL.muted, warn: COL.accent, kv: COL.kv, ok: COL.safe, bad: COL.threat };
const tickLabels = { warn: 'Launch', ok: 'Kill', bad: 'Impact' };
events.slice(1).forEach((e) => {
  const i = document.createElement('i');
  i.style.left = `${((e.t - tStart) / (tEnd - tStart)) * 100}%`;
  i.style.setProperty('--c', tickColors[e.cls]);
  if (tickLabels[e.cls]) i.dataset.label = tickLabels[e.cls];
  $('ticks').appendChild(i);
});

// Phase stepper under the clock; each phase is a jump target
const phases = [{ label: 'Track', t: tStart }];
if (D.outcome === 'NO ENGAGE') {
  phases.push({ label: 'Hold', t: D.minTrackTime ?? tStart });
} else {
  if (launchT != null) phases.push({ label: 'Boost', t: launchT });
  if (burnT != null) phases.push({ label: 'Midcourse', t: burnT });
  if (terminalT != null) phases.push({ label: 'Terminal', t: terminalT });
}
if (killT != null) phases.push({ label: 'Kill', t: killT, end: 'ok' });
else if (impactT != null) phases.push({ label: 'Impact', t: impactT, end: D.outcome === 'GROUND' ? 'bad' : 'info' });
phases.forEach((p) => {
  const li = document.createElement('li');
  const b = document.createElement('button');
  b.textContent = p.label;
  b.title = `Jump to T+${p.t.toFixed(1)} s`;
  // Outcome phases jump a few seconds early so the approach is visible
  b.onclick = () => seek(p.end ? p.t - 3 : p.t);
  li.appendChild(b);
  if (p.end) li.classList.add(p.end);
  $('phases').appendChild(li);
  p.el = li;
});

// Debrief card
{
  const rows = [];
  let title = '', cls = '', sub = '';
  if (D.outcome === 'KILL') {
    title = 'Target destroyed'; cls = 'ok';
    sub = `Hit-to-kill at ${(D.kill.alt / 1000).toFixed(1)} km altitude.`;
    rows.push(['Miss distance', `${D.kill.dist.toFixed(2)} m`], ['Closing speed', fmtSpd(D.kill.closing / 1000)],
      ['Kill time', `T+${D.kill.time.toFixed(2)} s`], ['Kill altitude', `${(D.kill.alt / 1000).toFixed(1)} km`]);
  } else if (D.outcome === 'GROUND') {
    title = 'Intercept failed'; cls = 'bad';
    const r = impactPos.length();
    sub = `The threat reached the ground ${r <= defR ? 'inside' : 'outside'} the defended area.`;
    rows.push(['Impact point', `${r.toFixed(1)} km out`], ['Defended area', r <= defR ? 'Inside' : 'Outside']);
  } else if (D.outcome === 'NO ENGAGE') {
    title = 'Weapons hold'; cls = 'info';
    sub = 'The predicted impact point stayed outside the defended area, so no interceptor was launched.';
    rows.push(['Impact point', `${impactPos.length().toFixed(1)} km out`], ['Defended radius', `${defR.toFixed(0)} km`]);
  } else {
    title = 'Sim timeout'; cls = 'bad';
    sub = 'The simulation ended with the threat still airborne.';
  }
  if (pipPos) rows.push(['PIP at launch', `${pipPos.length().toFixed(1)} km out`]);
  if (launchT != null && D.g.length) rows.push(['Peak acceleration', `${Math.max(...D.g).toFixed(1)} g`]);
  $('bda-title').textContent = title;
  $('bda-title').className = cls;
  $('bda-sub').textContent = sub;
  $('bda-time').textContent = endEventT != null ? `T+${endEventT.toFixed(2)}` : '';
  $('bda-rows').innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}

// ---- scale bar and axis triad (scientific style) ----
const triad = $('triad');
const triadCtx = triad.getContext('2d');
const AXES = [['E', new THREE.Vector3(1, 0, 0)], ['N', new THREE.Vector3(0, 0, -1)], ['U', new THREE.Vector3(0, 1, 0)]];
const qInv = new THREE.Quaternion();
const SCALES = [];
for (let e = -2; e <= 3; e++) for (const m of [1, 2, 5]) SCALES.push(m * 10 ** e);
export function drawRefs() {
  const d = camera.position.distanceTo(camLookTarget);
  const pxPerKm = window.innerHeight / (2 * d * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const L = SCALES.find((v) => v * pxPerKm >= 50) ?? SCALES[SCALES.length - 1];
  $('sb-bar').style.width = `${Math.round(L * pxPerKm)}px`;
  $('sb-label').textContent = L >= 1 ? `${L} km` : `${Math.round(L * 1000)} m`;

  const dpr = Math.min(window.devicePixelRatio, 2), W = 36, C = W / 2;
  if (triad.width !== W * dpr) { triad.width = triad.height = W * dpr; }
  const c = triadCtx;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, W);
  qInv.copy(camera.quaternion).invert();
  const pts = AXES.map(([n, v]) => [n, v.clone().applyQuaternion(qInv)]).sort((a, b) => a[1].z - b[1].z);
  c.font = '600 9px "Geist Mono", monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
  pts.forEach(([n, v]) => {
    const x = C + v.x * 10, y = C - v.y * 10;
    c.globalAlpha = 0.45 + 0.55 * THREE.MathUtils.clamp((v.z + 1) / 2, 0, 1);
    c.strokeStyle = n === 'U' ? '#e6ebf0' : '#8d99a6'; c.lineWidth = 1.5;
    seg(c, C, C, x, y);
    c.fillStyle = c.strokeStyle;
    c.fillText(n, C + v.x * 15, C - v.y * 15);
  });
  c.globalAlpha = 1;
}

// ============================================================================
// PLAYBACK CONTROLS
// ============================================================================
const ICON_PLAY = '<svg viewBox="0 0 16 16"><path d="M5 3.2v9.6a.7.7 0 0 0 1.06.6l7.4-4.8a.7.7 0 0 0 0-1.2l-7.4-4.8A.7.7 0 0 0 5 3.2z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16"><rect x="4" y="3" width="2.8" height="10" rx=".8" fill="currentColor"/><rect x="9.2" y="3" width="2.8" height="10" rx=".8" fill="currentColor"/></svg>';

const speedsEl = $('speeds');
SPEEDS.forEach((s) => {
  const b = document.createElement('button');
  b.textContent = `${s}×`; b.dataset.speed = s; b.title = `Playback speed ${s}×`;
  b.onclick = () => setSpeed(s);
  speedsEl.appendChild(b);
});
export function setSpeed(s) {
  state.speed = s;
  speedsEl.querySelectorAll('button').forEach((b) => {
    const on = +b.dataset.speed === s;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on);
  });
}
setSpeed(hashArgs.has('speed') ? +hashArgs.get('speed') : state.speed);
if (hashArgs.has('t')) state.simTime = THREE.MathUtils.clamp(+hashArgs.get('t'), tStart, tEnd);

export function setPlaying(p) {
  if (p && state.simTime >= tEnd - 1e-6) { state.simTime = tStart; state.bdaDismissed = false; }
  state.playing = p;
  $('play').innerHTML = p ? ICON_PAUSE : ICON_PLAY;
  $('play').setAttribute('aria-label', p ? 'Pause' : 'Play');
}
setPlaying(true);
$('play').onclick = () => setPlaying(!state.playing);
$('restart').onclick = () => { seek(tStart); setPlaying(true); };
$('bda-replay').onclick = () => { seek(tStart); setPlaying(true); };
$('bda-close').onclick = () => { state.bdaDismissed = true; };

const tl = $('timeline');
tl.min = tStart; tl.max = tEnd;
tl.addEventListener('pointerdown', () => { state.scrubbing = true; });
window.addEventListener('pointerup', () => { state.scrubbing = false; });
tl.addEventListener('input', () => seek(+tl.value));
tl.addEventListener('pointermove', (e) => {
  const r = tl.getBoundingClientRect();
  const f = THREE.MathUtils.clamp((e.clientX - r.left) / r.width, 0, 1);
  const tip = $('scrub-tip');
  tip.style.left = `${f * 100}%`;
  tip.textContent = `T+${(tStart + f * (tEnd - tStart)).toFixed(1)} s`;
});

export const CAMERAS = ['cinematic', 'orbit', 'threat', 'kv', 'top'];
const CAM_NAMES = { cinematic: 'Cinematic', orbit: 'Free orbit', threat: 'Chase threat', kv: 'Chase interceptor', top: 'Top-down' };
const CAM_KEYS = { cinematic: 'C', orbit: 'O', threat: 'T', kv: 'I', top: 'V' };
export function setCam(m, quiet = false, follow = null) {
  state.camMode = m;
  state.orbitFollow = m === 'orbit' ? follow : null;
  state.followFresh = true;
  document.querySelectorAll('#cam-btns button').forEach((b) => {
    const on = b.dataset.cam === m;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on);
  });
  controls.enabled = m === 'orbit';
  if (m === 'orbit') controls.target.copy(camLookTarget);
  if (!quiet) toast(`${CAM_NAMES[m]} camera${m === 'orbit' ? ' — drag to rotate, scroll to zoom, right-drag to pan' : ''} <kbd>${CAM_KEYS[m]}</kbd>`);
}

// Grabbing the scene in any directed view hands the camera to the user.
// Capture phase, so OrbitControls (listening on the same element) is already
// enabled when it sees this same pointerdown / wheel event.
const VEHICLE_NAMES = { threat: 'the threat', kv: 'the interceptor' };
function takeOver(follow) {
  setCam('orbit', true, follow);
  toast(`Free orbit${follow ? ` around ${VEHICLE_NAMES[follow]}` : ''} · <kbd>C</kbd> returns to cinematic`, 2600);
}
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (state.camMode === 'orbit' || e.button > 2) return;
  takeOver(state.camMode === 'threat' || state.camMode === 'kv' ? state.camMode : null);
}, { capture: true });
renderer.domElement.addEventListener('wheel', () => {
  if (state.camMode === 'cinematic' || state.camMode === 'top') takeOver(null);
}, { capture: true, passive: true });
document.querySelectorAll('#cam-btns button').forEach((b) => { b.onclick = () => setCam(b.dataset.cam); });

const LAYER_NAMES = {
  radar: 'Radar returns', track: 'Kalman track', cov: 'Track uncertainty (3σ)', los: 'Line of sight',
  labels: 'Labels', slowmo: 'Slow motion at intercept',
};
function syncLayerButtons() {
  document.querySelectorAll('#layer-btns button').forEach((b) => {
    const on = state.layers[b.dataset.layer];
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', on);
  });
}
export function toggleLayer(k) {
  state.layers[k] = !state.layers[k];
  syncLayerButtons();
  toast(`${LAYER_NAMES[k]} ${state.layers[k] ? 'on' : 'off'}`);
}
document.querySelectorAll('#layer-btns button').forEach((b) => { b.onclick = () => toggleLayer(b.dataset.layer); });
syncLayerButtons();

const layersBtn = $('layers-btn'), layersMenu = $('layer-btns');
function setMenu(open) {
  layersMenu.classList.toggle('open', open);
  layersBtn.setAttribute('aria-expanded', open);
  layersBtn.classList.toggle('on-toggle', open);
}
layersBtn.onclick = (e) => { e.stopPropagation(); setMenu(!layersMenu.classList.contains('open')); };
layersMenu.addEventListener('click', (e) => e.stopPropagation());
document.addEventListener('click', () => setMenu(false));

function setFocus(on) {
  document.body.classList.toggle('focus', on);
  $('focus-btn').classList.toggle('on-toggle', on);
  $('focus-btn').setAttribute('aria-pressed', on);
}
$('focus-btn').onclick = () => setFocus(!document.body.classList.contains('focus'));
if (hashArgs.has('focus')) setFocus(true);

const help = $('help');
function setHelp(open) { help.hidden = !open; if (open) $('help-close').focus(); }
$('help-btn').onclick = () => setHelp(help.hidden);
$('help-close').onclick = () => setHelp(false);

renderer.domElement.addEventListener('wheel', (e) => {
  if (state.camMode === 'threat' || state.camMode === 'kv') {
    state.chaseDist = THREE.MathUtils.clamp(state.chaseDist * Math.exp(e.deltaY * 0.0015), 0.08, 40);
    e.preventDefault();
  }
}, { passive: false });

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'escape') { setHelp(false); setMenu(false); return; }
  // Let a focused button handle its own activation keys
  if (e.target.tagName === 'BUTTON' && (k === ' ' || k === 'enter')) return;
  if (e.target.tagName === 'INPUT' && k !== ' ') return;
  if (k === '?') setHelp(help.hidden);
  else if (k === ' ') { setPlaying(!state.playing); e.preventDefault(); }
  else if (k === 'arrowright') seek(state.simTime + (e.shiftKey ? 5 : 1));
  else if (k === 'arrowleft') seek(state.simTime - (e.shiftKey ? 5 : 1));
  else if (k === 'home') seek(tStart);
  else if (k === 'end') seek(tEnd);
  else if (k === 'f') setFocus(!document.body.classList.contains('focus'));
  else if (k === 'a') setAnalysis(!state.analysisOpen);
  else if (k === 'g') setStyle(!state.sciStyle);
  else if (k === 'c') setCam('cinematic');
  else if (k === 'o') setCam('orbit');
  else if (k === 't') setCam('threat');
  else if (k === 'i') setCam('kv');
  else if (k === 'v') setCam('top');
  else if (k === 'r') toggleLayer('radar');
  else if (k === 'k') toggleLayer('track');
  else if (k === 'e') toggleLayer('cov');
  else if (k === 'l') toggleLayer('los');
  else if (k === 'h') toggleLayer('labels');
  else if (k === 'm') toggleLayer('slowmo');
  else if (k >= '1' && k <= '6') { setSpeed(SPEEDS[+k - 1]); toast(`Playback speed ${SPEEDS[+k - 1]}×`); }
});

window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h); labelRenderer.setSize(w, h);
});

// Rendering style: scientific (default) or cinematic
export function setStyle(sci, quiet = false) {
  state.sciStyle = sci;
  document.body.classList.toggle('sci', sci);
  bloom.strength = sci ? 0.35 : 0.85;
  clouds.visible = !sci;
  tubeMeshes.forEach((m) => { m.visible = !sci; });
  trailUniforms.forEach((u, i) => { u.uMin.value = sci ? Math.max(trailMinDefaults[i], 0.55) : trailMinDefaults[i]; });
  terrainMat.uniforms.uSweepOn.value = sci ? 0 : 1;
  const b = $('style-btn');
  b.textContent = sci ? 'Scientific' : 'Cinematic';
  b.setAttribute('aria-pressed', sci);
  if (!quiet) toast(`${sci ? 'Scientific' : 'Cinematic'} style <kbd>G</kbd>`);
}
$('style-btn').onclick = () => setStyle(!state.sciStyle);
setStyle(hashArgs.get('style') !== 'cinematic', true);

// Analysis drawer
const TABS = ['kin', 'gui', 'kf', 'methods'];
function setAnalysisTab(tab) {
  state.analysisTab = tab;
  document.querySelectorAll('#analysis .tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === tab));
  TABS.forEach((t) => { $(`tab-${t}`).hidden = t !== tab; });
  $('analysis-intro').hidden = tab === 'methods';
}
export function setAnalysis(open) {
  state.analysisOpen = open;
  if (open) buildAnalysis();
  document.body.classList.toggle('analysis', open);
  $('analysis-btn').classList.toggle('on-toggle', open);
  $('analysis-btn').setAttribute('aria-pressed', open);
}
document.querySelectorAll('#analysis .tabs button').forEach((b) => { b.onclick = () => setAnalysisTab(b.dataset.tab); });
$('analysis-btn').onclick = () => setAnalysis(!state.analysisOpen);
$('analysis-close').onclick = () => setAnalysis(false);
$('csv-btn').onclick = exportCsv;
if (hashArgs.has('analysis')) {
  setAnalysis(true);
  if (TABS.includes(hashArgs.get('tab'))) setAnalysisTab(hashArgs.get('tab'));
}

// Slow down around the decisive moment so it is watchable
export function slowFactor(time) {
  if (!state.layers.slowmo || endEventT == null) return 1;
  const slow = killT != null ? 0.1 : 0.3;
  const a = Math.abs(time - endEventT);
  if (a > 2.5) return 1;
  if (a < 0.8) return slow;
  return THREE.MathUtils.lerp(slow, 1, (a - 0.8) / 1.7);
}

// ============================================================================
// PER-FRAME HUD
// ============================================================================
let lastEventCount = -1;

export function updateHud(h) {
  const { time, fi, stage, tp, ip, tv, iv, g, sep, closing, ttg, launched, killed, impacted, missed, threatAlive, kvAlive } = h;
  $('clock').innerHTML = `<span class="unit">T+</span>${time.toFixed(2).padStart(6, '0')}`;
  const chip = $('status-chip');
  let st = 'Tracking', sc = '';
  if (killed) { st = 'Kill'; sc = 'ok'; }
  else if (impacted) { st = D.outcome === 'NO ENGAGE' ? 'Impact · outside' : 'Impact'; sc = D.outcome === 'NO ENGAGE' ? '' : 'bad'; }
  else if (!launched) { if (D.outcome === 'NO ENGAGE' && time >= (D.minTrackTime ?? tStart)) st = 'Weapons hold'; }
  else if (stage === 1) { st = 'Boost'; sc = 'warn'; }
  else if (missed) { st = 'Missed'; sc = 'bad'; }
  else if (terminalT != null && time >= terminalT) { st = 'Terminal'; sc = ttg < 4 ? 'kv pulse' : 'kv'; }
  else { st = 'Midcourse'; sc = 'kv'; }
  if (chip.textContent !== st) chip.textContent = st;
  chip.className = `flag ${sc}`;

  let cur = 0;
  phases.forEach((p, i) => { if (time >= p.t) cur = i; });
  phases.forEach((p, i) => { p.el.classList.toggle('done', i < cur); p.el.classList.toggle('current', i === cur); });

  $('th-alt').textContent = threatAlive ? fmtKm(tp.y) : '—';
  $('th-spd').textContent = threatAlive ? fmtSpd(tv.length()) : '—';
  $('th-rng').textContent = threatAlive ? fmtKm(Math.hypot(tp.x, tp.z)) : '—';
  {
    const k6 = 6 * fi;
    $('th-sig').textContent = threatAlive ? `±${Math.hypot(D.kfSig[k6], D.kfSig[k6 + 1], D.kfSig[k6 + 2]).toFixed(1)} m` : '—';
    $('th-err').textContent = threatAlive ? `${Math.hypot(D.kfErr[k6], D.kfErr[k6 + 1], D.kfErr[k6 + 2]).toFixed(1)} m` : '—';
  }
  $('kv-stage').textContent = !launched ? 'On launcher' : killed ? 'Hit target' : impacted || missed ? 'Missed' : stage === 1 ? 'Boost' : 'Coasting';
  $('kv-alt').textContent = launched && kvAlive ? fmtKm(ip.y) : '—';
  $('kv-spd').textContent = launched && kvAlive ? fmtSpd(iv.length()) : '—';
  $('kv-g').textContent = launched && kvAlive ? `${g.toFixed(1)} g` : '—';
  $('kv-gbar').style.width = `${launched && kvAlive ? Math.min(100, (g / (D.maxG || 40)) * 100) : 0}%`;
  $('en-sep').textContent = launched && !killed && !impacted ? fmtKm(sep) : '—';
  // Negative closing speed means the range is opening
  $('en-cls').textContent = launched && !killed && !impacted ? fmtSpd(closing) : '—';
  $('en-ttg').textContent = launched && !killed && !impacted && isFinite(ttg) ? `${ttg.toFixed(1)} s` : '—';

  let count = 0;
  events.forEach((e) => { const past = time >= e.t; e.el.classList.toggle('past', past); if (past) count++; });
  if (count !== lastEventCount) {
    events.forEach((e, i) => e.el.classList.toggle('latest', i === count - 1 && lastEventCount >= 0 && count > lastEventCount));
    lastEventCount = count;
  }

  if (!state.scrubbing) tl.value = time;
  tl.style.setProperty('--p', `${((time - tStart) / (tEnd - tStart)) * 100}%`);
  $('tlabel').innerHTML = `<b>${time.toFixed(1)}</b> / ${tEnd.toFixed(1)} s`;

  const showBda = !state.bdaDismissed && (endEventT != null ? time >= endEventT + 2 : time >= tEnd - 1e-6);
  if (showBda !== state.bdaShown) {
    state.bdaShown = showBda;
    document.body.classList.toggle('bda-on', showBda);
    const el = $('bda');
    if (showBda) { el.classList.add('mounted'); requestAnimationFrame(() => el.classList.add('show')); }
    else el.classList.remove('show', 'mounted');
  }
}
