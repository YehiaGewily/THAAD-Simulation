// Canvas figures: the side-profile chart and the analysis figure engine (a
// static base layer per figure plus a per-frame overlay with the time cursor).
import * as THREE from 'three';
import {
  D, GNC, N, P, T, burnT, cpaMidPos, cpaT, defR, endEventT, impactT, killPos, killT, launchIdx,
  launchT, locate, missCpaT, tStart, trueImpactPos,
} from './data.js';
import { $, COL } from './dom.js';
import { seek, state } from './state.js';

export const xMin = tStart, xMax = T[N - 1];

// Interactive multi-mode mini-chart (#profile): Side profile ('elev'), Plan view ('plan'), or Endgame ('zem')
const prof = $('profile');
const profTip = $('prof-tip');
const pctx = prof.getContext('2d');
let profHover = null;
let profScrubbing = false;
let profTimeAt = null;

const profData = (() => {
  const step = Math.max(1, Math.floor(N / 500));
  const th = [], kv = [], pip = [];
  let maxR = defR * 1.1, maxA = 10, maxSpan = defR * 1.15;
  for (let i = 0; i < N; i += step) {
    const tx = P.threat[3 * i], ty = P.threat[3 * i + 1], tz = P.threat[3 * i + 2];
    const r = Math.hypot(tx, tz);
    th.push([T[i], r, ty, tx, -tz]);
    maxR = Math.max(maxR, r);
    maxA = Math.max(maxA, ty);
    maxSpan = Math.max(maxSpan, Math.abs(tx), Math.abs(tz));
    if (launchIdx >= 0 && i >= launchIdx) {
      const ix = P.kv[3 * i], iy = P.kv[3 * i + 1], iz = P.kv[3 * i + 2];
      kv.push([T[i], Math.hypot(ix, iz), iy, ix, -iz]);
    }
    if (GNC.pipValid[i]) {
      const px = GNC.pipTrack[3 * i], pz = GNC.pipTrack[3 * i + 2];
      pip.push([T[i], Math.hypot(px, pz), px, -pz]);
      maxR = Math.max(maxR, Math.hypot(px, pz));
      maxSpan = Math.max(maxSpan, Math.abs(px), Math.abs(pz));
    }
  }
  return { th, kv, pip, maxR: maxR * 1.05, maxA: maxA * 1.12, maxSpan: maxSpan * 1.12 };
})();

function bindProfileEvents() {
  const pickTime = (e) => {
    if (!profTimeAt) return null;
    const r = prof.getBoundingClientRect();
    return profTimeAt(e.clientX - r.left, e.clientY - r.top);
  };
  prof.addEventListener('pointerdown', (e) => {
    profScrubbing = true;
    const t = pickTime(e);
    if (t != null) seek(t);
  });
  window.addEventListener('pointerup', () => { profScrubbing = false; });
  prof.addEventListener('pointermove', (e) => {
    const t = pickTime(e);
    profHover = t;
    if (profScrubbing && t != null) seek(t);
  });
  prof.addEventListener('pointerleave', () => {
    profHover = null;
    if (profTip) profTip.style.display = 'none';
  });
}
bindProfileEvents();

export function drawProfile(time, tp, ip) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = prof.clientWidth, h = prof.clientHeight;
  if (!w) return;
  if (prof.width !== w * dpr || prof.height !== h * dpr) { prof.width = w * dpr; prof.height = h * dpr; }
  const c = pctx;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.font = '500 10px "Geist Mono", monospace';
  const mode = state.profileMode || 'elev';
  const [fi] = locate(time);
  const dot = (x, y, col, r = 3.2) => { c.fillStyle = col; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); };

  if (mode === 'plan') {
    const L = 26, B = 16, Rp = 6, Tp = 6;
    const pw = w - L - Rp, ph = h - B - Tp;
    const S = profData.maxSpan;
    const cx = L + pw / 2, cy = Tp + ph / 2, sc = Math.min(pw, ph) / (2 * S);
    const X = (e) => cx + e * sc, Y = (n) => cy - n * sc;
    profTimeAt = (px, py) => {
      let bestT = time, bestD = Infinity;
      for (const [t, , , e, n] of profData.th) {
        const d = Math.hypot(X(e) - px, Y(n) - py);
        if (d < bestD) { bestD = d; bestT = t; }
      }
      return bestT;
    };
    c.strokeStyle = 'rgba(214,228,240,0.07)'; c.lineWidth = 1;
    seg(c, L, cy, w - Rp, cy); seg(c, cx, Tp, cx, h - B);
    c.fillStyle = COL.faint;
    c.fillText('E', w - Rp - 8, cy - 4); c.fillText('N', cx + 4, Tp + 9);
    // Defended area circle
    c.strokeStyle = COL.safe; c.globalAlpha = 0.55; c.setLineDash([3, 3]);
    c.beginPath(); c.arc(cx, cy, defR * sc, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
    c.fillStyle = COL.safe; c.globalAlpha = 0.07; c.fill(); c.globalAlpha = 1;
    // PIP migration trail
    const pathXY = (pts, col, upTo, width, alpha, ix = 3, iy = 4) => {
      c.strokeStyle = col; c.lineWidth = width; c.globalAlpha = alpha; c.beginPath();
      let st = false;
      for (const p of pts) { if (p[0] > upTo) break; const x = X(p[ix]), y = Y(p[iy]); if (!st) { c.moveTo(x, y); st = true; } else c.lineTo(x, y); }
      c.stroke(); c.globalAlpha = 1;
    };
    pathXY(profData.th, COL.threat, Infinity, 1, 0.2);
    pathXY(profData.kv, COL.kv, Infinity, 1, 0.2);
    pathXY(profData.pip, COL.accent, time, 1.3, 0.75, 2, 3);
    pathXY(profData.th, COL.threat, time, 1.75, 1);
    pathXY(profData.kv, COL.kv, time, 1.75, 1);
    dot(cx, cy, COL.safe, 2.6);
    if (GNC.pipValid[fi]) dot(X(GNC.pipTrack[3 * fi]), Y(-GNC.pipTrack[3 * fi + 2]), COL.accent, 2.8);
    dot(X(tp.x), Y(-tp.z), COL.threat, 3.2);
    if (launchT != null && time >= launchT) dot(X(ip.x), Y(-ip.z), COL.kv, 3.2);
  } else if (mode === 'zem') {
    const L = 32, B = 16, Rp = 8, Tp = 6;
    const pw = w - L - Rp, ph = h - B - Tp;
    const t0 = launchT ?? xMin, t1 = xMax;
    const X = (t) => L + ((t - t0) / Math.max(t1 - t0, 1)) * pw;
    const lMin = -1, lMax = 4.2; // 0.1 m to ~15 km
    const Y = (v) => Tp + (1 - (Math.log10(Math.max(v, 0.1)) - lMin) / (lMax - lMin)) * ph;
    profTimeAt = (px) => THREE.MathUtils.clamp(t0 + ((px - L) / pw) * (t1 - t0), xMin, xMax);
    c.strokeStyle = 'rgba(214,228,240,0.06)'; c.fillStyle = COL.faint;
    [1, 10, 100, 1000, 10000].forEach((v) => {
      const y = Y(v);
      seg(c, L, y, w - Rp, y);
      c.fillText(v >= 1000 ? `${v / 1000}k` : String(v), 2, y + 3);
    });
    // Kill radius threshold
    const rk = D.killRadius || 10;
    c.strokeStyle = COL.safe; c.setLineDash([3, 3]); c.globalAlpha = 0.65;
    seg(c, L, Y(rk), w - Rp, Y(rk)); c.setLineDash([]); c.globalAlpha = 1;
    const drawZem = (arr, col, dash, upTo, alpha) => {
      c.strokeStyle = col; c.lineWidth = dash.length ? 1.25 : 1.75; c.globalAlpha = alpha; c.setLineDash(dash);
      c.beginPath(); let st = false;
      const stride = Math.max(1, Math.floor(N / 300));
      for (let i = Math.max(0, launchIdx); i < N; i += stride) {
        if (T[i] > upTo) break;
        const v = arr[i]; if (!Number.isFinite(v) || v <= 0) continue;
        const x = X(T[i]), y = Y(v);
        if (!st) { c.moveTo(x, y); st = true; } else c.lineTo(x, y);
      }
      c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
    };
    drawZem(GNC.zemTrue, COL.accent, [], Infinity, 0.22);
    drawZem(GNC.zemEst, COL.kv, [3, 2], time, 0.8);
    drawZem(GNC.zemTrue, COL.accent, [], time, 1);
    const xc = X(Math.max(time, t0));
    c.strokeStyle = COL.accent; c.globalAlpha = 0.5; seg(c, xc, Tp, xc, h - B); c.globalAlpha = 1;
    if (Number.isFinite(GNC.zemTrue[fi])) dot(xc, Y(GNC.zemTrue[fi]), COL.accent, 3.2);
  } else {
    // Default 'elev': Altitude vs Ground range (km)
    const L = 26, B = 16, Rp = 4, Tp = 6;
    const X = (r) => L + (r / profData.maxR) * (w - L - Rp);
    const Y = (a) => h - B - (a / profData.maxA) * (h - B - Tp);
    profTimeAt = (px, py) => {
      let bestT = time, bestD = Infinity;
      for (const [t, r, a] of profData.th) {
        const d = Math.hypot(X(r) - px, Y(a) - py);
        if (d < bestD) { bestD = d; bestT = t; }
      }
      return bestT;
    };
    c.lineWidth = 1;
    c.strokeStyle = 'rgba(214,228,240,0.06)';
    c.fillStyle = COL.faint;
    for (let r = 0; r <= profData.maxR; r += 10) { c.beginPath(); c.moveTo(X(r), Tp); c.lineTo(X(r), h - B); c.stroke(); c.fillText(r.toFixed(0), X(r) - 4, h - 3); }
    for (let a = 0; a <= profData.maxA; a += 20) { c.beginPath(); c.moveTo(L, Y(a)); c.lineTo(w - Rp, Y(a)); c.stroke(); if (a) c.fillText(a.toFixed(0), 2, Y(a) + 3); }
    // Stratopause reference line at 50 km
    if (profData.maxA >= 52) {
      const y50 = Y(50);
      c.strokeStyle = 'rgba(214,228,240,0.12)'; c.setLineDash([2, 3]);
      seg(c, L, y50, w - Rp, y50); c.setLineDash([]);
    }
    // Defended area band on the ground axis
    c.fillStyle = COL.safe; c.globalAlpha = 0.08;
    c.fillRect(L, Tp, X(defR) - L, h - B - Tp);
    c.globalAlpha = 0.55; c.strokeStyle = COL.safe; c.setLineDash([3, 3]);
    c.beginPath(); c.moveTo(X(defR), Tp); c.lineTo(X(defR), h - B); c.stroke(); c.setLineDash([]);
    c.globalAlpha = 1;

    const path = (pts, col, upTo, width, alpha) => {
      c.strokeStyle = col; c.lineWidth = width; c.globalAlpha = alpha; c.lineJoin = 'round'; c.beginPath();
      let started = false;
      for (const [t, r, a] of pts) { if (t > upTo) break; if (!started) { c.moveTo(X(r), Y(a)); started = true; } else c.lineTo(X(r), Y(a)); }
      c.stroke(); c.globalAlpha = 1;
    };
    path(profData.th, COL.threat, Infinity, 1, 0.22);
    path(profData.kv, COL.kv, Infinity, 1, 0.22);
    path(profData.th, COL.threat, time, 1.75, 1);
    path(profData.kv, COL.kv, time, 1.75, 1);
    // Live vacuum PIP marker along the ground axis (z = 0)
    if (GNC.pipValid[fi] && !(killT != null && time >= killT)) {
      const pr = Math.hypot(GNC.pipTrack[3 * fi], GNC.pipTrack[3 * fi + 2]);
      dot(X(pr), Y(0), COL.accent, 2.6);
    }
    if (killT != null && time >= killT) {
      const x = X(Math.hypot(killPos.x, killPos.z)), y = Y(killPos.y);
      c.globalAlpha = 0.25; dot(x, y, COL.safe, 8); c.globalAlpha = 1; dot(x, y, COL.safe, 4);
    } else {
      if (missCpaT != null && time >= missCpaT && cpaMidPos) {
        dot(X(Math.hypot(cpaMidPos.x, cpaMidPos.z)), Y(cpaMidPos.y), COL.accent, 3.0);
      }
      if (!(impactT != null && time >= impactT)) dot(X(Math.hypot(tp.x, tp.z)), Y(tp.y), COL.threat);
      if (launchT != null && time >= launchT) dot(X(Math.hypot(ip.x, ip.z)), Y(ip.y), COL.kv);
    }
  }

  // Hover readout tooltip on the mini-chart
  if (profTip) {
    if (profHover != null) {
      const [hi] = locate(profHover);
      const tr = Math.hypot(P.threat[3 * hi], P.threat[3 * hi + 2]), ta = P.threat[3 * hi + 1];
      const zm = GNC.zemTrue[hi];
      profTip.innerHTML = `<span>T+${profHover.toFixed(1)}s · alt ${ta.toFixed(1)} km · rng ${tr.toFixed(1)} km${Number.isFinite(zm) ? ` · ZEM ${zm < 100 ? zm.toFixed(1) : Math.round(zm)} m` : ''}</span>`;
      profTip.style.display = 'block';
      profTip.style.left = '8px';
      profTip.style.top = '6px';
    } else profTip.style.display = 'none';
  }
}

// ---- figure engine (canvas 2D: static base layer + per-frame overlay) ----
const PAD = { l: 46, r: 12, t: 12, b: 22 };
const phaseMarks = [
  [launchT, 'Launch'],
  [burnT, 'Burnout'],
  [missCpaT, 'CPA'],
  [endEventT, killT != null ? 'Kill' : 'Impact'],
].filter(([t]) => t != null);
export const figures = [];
let figCount = 0;

export function niceStep(range, count) {
  const raw = range / count, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
export function linTicks(lo, hi, count) {
  const st = niceStep(hi - lo || 1, count), out = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-6; v += st) out.push(+v.toPrecision(12));
  return out;
}
export function logTicks(lo, hi) { const out = []; for (let e = Math.round(Math.log10(lo)); e <= Math.round(Math.log10(hi)); e++) out.push(10 ** e); return out; }
export function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(0).replace('e+', 'e');
  return String(+v.toPrecision(4));
}
export function seg(c, x0, y0, x1, y1) { c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke(); }
export function valueAt(arr, time) {
  const [i, fr] = locate(time);
  const a = arr[i], b = arr[i + 1];
  if (!Number.isFinite(a)) return Number.isFinite(b) && fr > 0.5 ? b : NaN;
  if (!Number.isFinite(b)) return a;
  return a + (b - a) * fr;
}

export function figure(host, tab, spec) {
  const no = ++figCount;
  const legend = [];
  if (spec.series.length > 1 || spec.band || spec.refs) {
    spec.series.forEach((sr) => legend.push(`<span><i class="${sr.dash ? 'dash' : ''}" style="--c:${sr.color}"></i>${sr.label}</span>`));
    if (spec.band) legend.push(`<span><i class="band" style="--c:${spec.band.color}"></i>${spec.band.label}</span>`);
    (spec.refs || []).forEach((r) => legend.push(`<span><i class="dash" style="--c:${COL.muted}"></i>${r.label}</span>`));
  }
  const el = document.createElement('figure');
  el.className = 'fig';
  el.innerHTML =
    `<div class="fig-head"><span class="fig-title">${spec.title}</span><span class="fig-unit">${spec.unit}${spec.log ? ' · log' : ''}</span>` +
    `<span class="legend">${legend.join('')}</span></div>` +
    `<div class="plot"><canvas class="base"></canvas><canvas class="ov"></canvas><div class="tip"></div></div>` +
    `<figcaption><b>Fig. ${no}.</b> ${spec.caption}</figcaption>`;
  host.appendChild(el);
  const f = { spec, tab, el, base: el.querySelector('.base'), ov: el.querySelector('.ov'), tip: el.querySelector('.tip'), hover: null, size: '' };
  const tAt = (e) => { const r = f.ov.getBoundingClientRect(); return f.xInv(e.clientX - r.left); };
  f.ov.addEventListener('pointermove', (e) => { if (f.xInv) f.hover = tAt(e); });
  f.ov.addEventListener('pointerleave', () => { f.hover = null; });
  f.ov.addEventListener('click', (e) => { if (f.xInv) seek(tAt(e)); });
  figures.push(f);
  return f;
}

export function layoutFig(f) {
  const w = f.ov.clientWidth, h = f.ov.clientHeight;
  if (!w || !h) return false;
  const key = `${w}x${h}`;
  if (f.size === key) return true;
  f.size = key;
  const dpr = Math.min(window.devicePixelRatio, 2);
  for (const cv of [f.base, f.ov]) { cv.width = w * dpr; cv.height = h * dpr; cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0); }
  const { spec } = f;
  let lo = Infinity, hi = -Infinity;
  const take = (v) => { if (!Number.isFinite(v) || (spec.log && v <= 0)) return; if (v < lo) lo = v; if (v > hi) hi = v; };
  spec.series.forEach((sr) => sr.data.forEach(take));
  if (spec.band) spec.band.hi.forEach((v) => { take(v); if (spec.band.symmetric) take(-v); });
  (spec.refs || []).forEach((r) => take(r.y));
  if (!Number.isFinite(lo)) { lo = spec.log ? 1 : 0; hi = spec.log ? 10 : 1; }
  if (spec.log) {
    hi = 10 ** Math.ceil(Math.log10(hi));
    lo = Math.max(10 ** Math.floor(Math.log10(lo)), hi / 1e7);
    if (hi <= lo) hi = lo * 10;
  } else {
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
    lo = spec.yMin != null ? spec.yMin : lo - pad;
    hi += pad;
  }
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
  Object.assign(f, { w, h, lo, hi, pw, ph });
  f.X = (t) => PAD.l + ((t - xMin) / (xMax - xMin)) * pw;
  f.xInv = (x) => THREE.MathUtils.clamp(xMin + ((x - PAD.l) / pw) * (xMax - xMin), xMin, xMax);
  const l0 = Math.log10(lo), l1 = Math.log10(hi);
  f.Y = spec.log ? (v) => PAD.t + (1 - (Math.log10(v) - l0) / (l1 - l0)) * ph : (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * ph;
  drawBase(f);
  return true;
}

function drawSeries(c, f, data, color, width, dash = null) {
  const stride = Math.max(1, Math.floor(N / (f.pw * 1.5)));
  c.strokeStyle = color; c.lineWidth = width; c.lineJoin = 'round'; c.lineCap = 'round';
  if (dash) c.setLineDash(dash);
  c.beginPath();
  let pen = false;
  const put = (i) => {
    const v = data[i];
    if (!Number.isFinite(v) || (f.spec.log && v <= 0)) { pen = false; return; }
    const x = f.X(T[i]), y = f.Y(v);
    if (pen) c.lineTo(x, y); else { c.moveTo(x, y); pen = true; }
  };
  for (let i = 0; i < N; i += stride) put(i);
  put(N - 1);
  c.stroke();
  if (dash) c.setLineDash([]);
}

function drawBase(f) {
  const c = f.base.getContext('2d'), { spec, w, h } = f;
  c.clearRect(0, 0, w, h);
  c.font = '500 10px "Geist Mono", ui-monospace, monospace';
  // horizontal grid + y ticks
  c.textAlign = 'right'; c.textBaseline = 'middle';
  (spec.log ? logTicks(f.lo, f.hi) : linTicks(f.lo, f.hi, 4)).forEach((v) => {
    const y = f.Y(v);
    if (y < PAD.t - 1 || y > h - PAD.b + 1) return;
    c.strokeStyle = !spec.log && v === 0 ? 'rgba(214,228,240,0.2)' : 'rgba(214,228,240,0.06)';
    c.lineWidth = 1;
    seg(c, PAD.l, y, w - PAD.r, y);
    c.fillStyle = COL.faint;
    c.fillText(fmtTick(v), PAD.l - 7, y);
  });
  // x ticks
  c.textAlign = 'center'; c.textBaseline = 'top';
  linTicks(xMin, xMax, Math.max(3, Math.floor(f.pw / 70))).forEach((t) => {
    const x = f.X(t);
    c.strokeStyle = 'rgba(214,228,240,0.04)';
    seg(c, x, PAD.t, x, h - PAD.b);
    c.fillStyle = COL.faint;
    c.fillText(String(+t.toFixed(1)), x, h - PAD.b + 6);
  });
  c.strokeStyle = 'rgba(214,228,240,0.16)';
  seg(c, PAD.l, h - PAD.b + 0.5, w - PAD.r, h - PAD.b + 0.5);

  c.save();
  c.beginPath(); c.rect(PAD.l, PAD.t - 2, f.pw, f.ph + 4); c.clip();
  // phase markers
  c.setLineDash([2, 3]); c.strokeStyle = 'rgba(214,228,240,0.25)'; c.lineWidth = 1;
  c.textAlign = 'left'; c.textBaseline = 'top'; c.fillStyle = COL.faint;
  phaseMarks.forEach(([t, label]) => { const x = f.X(t); seg(c, x, PAD.t, x, h - PAD.b); c.fillText(label, x + 3, PAD.t); });
  c.setLineDash([]);
  // uncertainty band
  if (spec.band) {
    const b = spec.band, stride = Math.max(1, Math.floor(N / (f.pw * 1.5)));
    const idx = [];
    for (let i = 0; i < N; i += stride) if (Number.isFinite(b.hi[i])) idx.push(i);
    const lo = (i) => (b.symmetric ? -b.hi[i] : 0);
    c.fillStyle = b.color; c.globalAlpha = 0.16;
    c.beginPath();
    idx.forEach((i, j) => (j ? c.lineTo(f.X(T[i]), f.Y(b.hi[i])) : c.moveTo(f.X(T[i]), f.Y(b.hi[i]))));
    for (let j = idx.length - 1; j >= 0; j--) c.lineTo(f.X(T[idx[j]]), f.Y(Math.max(lo(idx[j]), spec.log ? f.lo : -Infinity)));
    c.closePath(); c.fill();
    c.globalAlpha = 0.55; c.strokeStyle = b.color; c.lineWidth = 1;
    drawSeries(c, f, b.hi, b.color, 1);
    if (b.symmetric) drawSeries(c, f, b.hi.map((v) => -v), b.color, 1);
    c.globalAlpha = 1;
  }
  // reference lines, labelled directly
  (spec.refs || []).forEach((r) => {
    const y = f.Y(r.y);
    if (y < PAD.t || y > h - PAD.b) return;
    c.setLineDash([5, 4]); c.strokeStyle = COL.muted; c.globalAlpha = 0.8; c.lineWidth = 1;
    seg(c, PAD.l, y, w - PAD.r, y);
    c.setLineDash([]); c.globalAlpha = 1;
    c.fillStyle = COL.muted; c.textAlign = 'right'; c.textBaseline = 'bottom';
    c.fillText(r.label, w - PAD.r - 4, y - 3);
  });
  spec.series.forEach((sr) => drawSeries(c, f, sr.data, sr.color, sr.dash ? 1.5 : (spec.thin ? 1.25 : 2), sr.dash));
  // direct labels for multi-series figures, placed where the series exist
  if (spec.series.length > 1) {
    c.font = '500 11px Geist, ui-sans-serif, sans-serif'; c.textBaseline = 'bottom';
    spec.series.forEach((sr, j) => {
      const finite = [];
      for (let i = 0; i < N; i += 8) if (Number.isFinite(sr.data[i]) && !(spec.log && sr.data[i] <= 0)) finite.push(i);
      if (!finite.length) return;
      const i = finite[Math.floor(finite.length * (0.3 + 0.35 * j))] ?? finite[0];
      const x = f.X(T[i]), y = f.Y(sr.data[i]);
      c.fillStyle = COL.muted; c.textAlign = x > w - 90 ? 'right' : 'left';
      c.fillText(sr.label, x + (c.textAlign === 'left' ? 6 : -6), y - 5);
    });
  }
  c.restore();
}

export function drawOverlay(f, time) {
  if (!layoutFig(f)) return;
  const c = f.ov.getContext('2d'), { spec, w, h } = f;
  c.clearRect(0, 0, w, h);
  const mark = (t, hover) => {
    const x = f.X(t);
    c.strokeStyle = hover ? 'rgba(214,228,240,0.5)' : COL.accent; c.lineWidth = 1;
    c.setLineDash(hover ? [3, 3] : []);
    seg(c, x, PAD.t, x, h - PAD.b);
    c.setLineDash([]);
    spec.series.forEach((sr) => {
      const v = valueAt(sr.data, t);
      if (!Number.isFinite(v) || (spec.log && v <= 0)) return;
      const y = f.Y(v);
      if (y < PAD.t || y > h - PAD.b) return;
      c.fillStyle = '#0f141b'; c.beginPath(); c.arc(x, y, 5, 0, Math.PI * 2); c.fill();
      c.fillStyle = sr.color; c.beginPath(); c.arc(x, y, 3.2, 0, Math.PI * 2); c.fill();
    });
  };
  mark(THREE.MathUtils.clamp(time, xMin, xMax), false);
  if (f.hover != null) {
    mark(f.hover, true);
    const rows = [`<span>t = ${f.hover.toFixed(2)} s</span>`];
    spec.series.forEach((sr) => {
      const v = valueAt(sr.data, f.hover);
      rows.push(`<span><i style="--c:${sr.color}"></i>${sr.label}: ${Number.isFinite(v) ? spec.fmt(v) : '—'}</span>`);
    });
    if (spec.band) {
      const v = valueAt(spec.band.hi, f.hover);
      rows.push(`<span><i style="--c:${spec.band.color}"></i>${spec.band.label}: ${Number.isFinite(v) ? (spec.band.symmetric ? '±' : '') + spec.fmt(v) : '—'}</span>`);
    }
    f.tip.innerHTML = rows.join('');
    f.tip.style.display = 'block';
    const x = f.X(f.hover), tw = f.tip.offsetWidth;
    f.tip.style.left = `${x + 14 + tw > w ? x - tw - 14 : x + 14}px`;
    f.tip.style.top = `${PAD.t + 4}px`;
  } else f.tip.style.display = 'none';
}
