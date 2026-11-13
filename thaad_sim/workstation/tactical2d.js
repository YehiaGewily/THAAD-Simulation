// High-precision 2D CAD Tactical Views: Elevation Profile & East-North Plan Map
import { locateFrame } from './viewport3d.js';

export class Tactical2DViews {
  constructor(elevCanvas, planCanvas, onSeek) {
    this.elevCanvas = elevCanvas;
    this.planCanvas = planCanvas;
    this.onSeek = onSeek;
    this.data = null;
    this.compareRuns = [];
    this.simTime = 0;

    this._bindInteractions(this.elevCanvas, 'elev');
    this._bindInteractions(this.planCanvas, 'plan');

    // Canvases only resize inside update(); redraw on layout changes too, so a
    // window resize while paused does not leave a stretched bitmap.
    if ('ResizeObserver' in window) {
      const ro = new ResizeObserver(() => this.update(this.simTime));
      ro.observe(this.elevCanvas);
      ro.observe(this.planCanvas);
    }
  }

  _bindInteractions(canvas, kind) {
    let dragging = false;
    const pick = (e) => {
      if (!this.data || !this._mapFn?.[kind]) return;
      const r = canvas.getBoundingClientRect();
      const t = this._mapFn[kind](e.clientX - r.left, e.clientY - r.top);
      if (t != null && this.onSeek) this.onSeek(t);
    };
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      pick(e);
    });
    window.addEventListener('pointerup', () => {
      dragging = false;
    });
    canvas.addEventListener('pointermove', (e) => {
      if (dragging) pick(e);
    });
  }

  setData(data, compareRuns = []) {
    this.data = data;
    this.compareRuns = compareRuns;
    this._mapFn = {};

    const drv = data.derived;
    const defR = (data.defendedRadius || 30000) / 1000;
    let maxR = defR * 1.15;
    let maxA = 20;
    let maxSpan = defR * 1.2;

    const N = data.t.length;
    for (let i = 0; i < N; i++) {
      const ex = data.threat[3 * i] / 1000;
      const ny = data.threat[3 * i + 1] / 1000;
      const uz = data.threat[3 * i + 2] / 1000;
      maxR = Math.max(maxR, drv.rngT[i] || 0);
      maxA = Math.max(maxA, uz);
      maxSpan = Math.max(maxSpan, Math.abs(ex), Math.abs(ny));
      if (drv.pipValid[i]) {
        const px = drv.pipTrack[3 * i] / 1000;
        const py = drv.pipTrack[3 * i + 1] / 1000;
        maxR = Math.max(maxR, Math.hypot(px, py));
        maxSpan = Math.max(maxSpan, Math.abs(px), Math.abs(py));
      }
    }
    this.bounds = {
      maxR: maxR * 1.08,
      maxA: maxA * 1.12,
      maxSpan: maxSpan * 1.12,
      defR,
    };
  }

  update(time) {
    this.simTime = time;
    if (!this.data) return;
    this._drawElevation(time);
    this._drawPlan(time);
  }

  _prepCanvas(canvas) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w < 10 || h < 10) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  _drawElevation(time) {
    const prep = this._prepCanvas(this.elevCanvas);
    if (!prep) return;
    const { ctx: c, w, h } = prep;
    const D = this.data;
    const drv = D.derived;
    const { maxR, maxA, defR } = this.bounds;
    const pad = { l: 34, r: 12, t: 10, b: 22 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    const X = (r) => pad.l + (r / maxR) * pw;
    const Y = (a) => pad.t + ph - (a / maxA) * ph;

    // Nearest point time lookup on click
    this._mapFn.elev = (px, py) => {
      let bestT = time, bestD = Infinity;
      for (let i = 0; i < D.t.length; i += 4) {
        const d = Math.hypot(X(drv.rngT[i]) - px, Y(drv.altT[i]) - py);
        if (d < bestD) { bestD = d; bestT = D.t[i]; }
      }
      return bestT;
    };

    // Grid lines
    c.font = '10.5px "Cascadia Mono", "SF Mono", Consolas, ui-monospace, monospace';
    c.strokeStyle = 'rgba(155, 157, 162, 0.1)';
    c.fillStyle = '#6a6d73';
    c.lineWidth = 1;

    for (let r = 0; r <= maxR; r += 10) {
      const x = X(r);
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, pad.t + ph); c.stroke();
      c.fillText(`${r}`, x - 6, h - 6);
    }
    for (let a = 0; a <= maxA; a += 15) {
      const y = Y(a);
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(pad.l + pw, y); c.stroke();
      if (a > 0) c.fillText(`${a}`, 6, y + 3);
    }

    // Stratopause at 50 km
    if (maxA >= 52) {
      const y50 = Y(50);
      c.strokeStyle = 'rgba(155, 157, 162, 0.22)';
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(pad.l, y50); c.lineTo(pad.l + pw, y50); c.stroke();
      c.setLineDash([]);
      c.fillStyle = '#4a4d53';
      c.fillText('Stratopause 50km', pad.l + 6, y50 - 4);
    }

    // Defended area band
    const xDef = X(defR);
    c.fillStyle = 'rgba(107, 189, 140, 0.07)';
    c.fillRect(pad.l, pad.t, xDef - pad.l, ph);
    c.strokeStyle = 'rgba(107, 189, 140, 0.6)';
    c.setLineDash([4, 3]);
    c.beginPath(); c.moveTo(xDef, pad.t); c.lineTo(xDef, pad.t + ph); c.stroke();
    c.setLineDash([]);

    const [fi] = locateFrame(D.t, time);

    // Draw trajectory helper
    const drawCurve = (rngArr, altArr, upToIdx, color, width, alpha, dash = []) => {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.globalAlpha = alpha;
      c.setLineDash(dash);
      c.beginPath();
      let started = false;
      for (let i = 0; i <= upToIdx && i < rngArr.length; i += 2) {
        const r = rngArr[i], a = altArr[i];
        if (r == null || a == null) continue;
        const x = X(r), y = Y(a);
        if (!started) { c.moveTo(x, y); started = true; }
        else c.lineTo(x, y);
      }
      c.stroke();
      c.setLineDash([]);
      c.globalAlpha = 1;
    };

    const N = D.t.length;
    drawCurve(drv.rngT, drv.altT, N - 1, '#e0594a', 1.2, 0.22);
    drawCurve(drv.rngI, drv.altI, N - 1, '#5aa6d6', 1.2, 0.22);

    // Comparison runs
    this.compareRuns.forEach((run) => {
      const rDrv = run.derived;
      const [rFi] = locateFrame(run.t, time);
      drawCurve(rDrv.rngI, rDrv.altI, rFi, run.compareColor || '#a08ad4', 1.8, 0.85, [3, 2]);
    });

    drawCurve(drv.rngT, drv.altT, fi, '#e0594a', 2.2, 1);
    drawCurve(drv.rngI, drv.altI, fi, '#5aa6d6', 2.2, 1);

    // Current positions & PIP marker
    const curTr = drv.rngT[fi], curTa = drv.altT[fi];
    if (curTr != null && curTa != null) {
      c.fillStyle = '#e0594a';
      c.beginPath(); c.arc(X(curTr), Y(curTa), 4, 0, Math.PI * 2); c.fill();
    }
    const curIr = drv.rngI[fi], curIa = drv.altI[fi];
    if (curIr != null && curIa != null) {
      c.fillStyle = '#5aa6d6';
      c.beginPath(); c.arc(X(curIr), Y(curIa), 4, 0, Math.PI * 2); c.fill();
    }
    if (drv.pipValid[fi]) {
      const pr = Math.hypot(drv.pipTrack[3 * fi], drv.pipTrack[3 * fi + 1]) / 1000;
      c.fillStyle = '#d4a24c';
      c.beginPath(); c.arc(X(pr), Y(0), 3.8, 0, Math.PI * 2); c.fill();
    }
    if (D.cpa && time >= D.cpa.time) {
      const cr = Math.hypot(D.cpa.tPos[0], D.cpa.tPos[1]) / 1000;
      const ca = D.cpa.tPos[2] / 1000;
      c.strokeStyle = D.outcome === 'KILL' ? '#6bbd8c' : '#d4a24c';
      c.lineWidth = 2;
      c.beginPath(); c.arc(X(cr), Y(ca), 6, 0, Math.PI * 2); c.stroke();
    }
  }

  _drawPlan(time) {
    const prep = this._prepCanvas(this.planCanvas);
    if (!prep) return;
    const { ctx: c, w, h } = prep;
    const D = this.data;
    const drv = D.derived;
    const { maxSpan, defR } = this.bounds;
    const cx = w * 0.5;
    const cy = h * 0.52;
    const sc = Math.min(w - 28, h - 28) / (2 * maxSpan);

    const X = (eKm) => cx + eKm * sc;
    const Y = (nKm) => cy - nKm * sc;

    this._mapFn.plan = (px, py) => {
      let bestT = time, bestD = Infinity;
      for (let i = 0; i < D.t.length; i += 4) {
        const e = D.threat[3 * i] / 1000;
        const n = D.threat[3 * i + 1] / 1000;
        const d = Math.hypot(X(e) - px, Y(n) - py);
        if (d < bestD) { bestD = d; bestT = D.t[i]; }
      }
      return bestT;
    };

    // Axes & Radar range rings
    c.strokeStyle = 'rgba(155, 157, 162, 0.12)';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(8, cy); c.lineTo(w - 8, cy);
    c.moveTo(cx, 8); c.lineTo(cx, h - 8);
    c.stroke();

    [15, 30, 45].forEach((r) => {
      c.beginPath(); c.arc(cx, cy, r * sc, 0, Math.PI * 2); c.stroke();
    });

    // Defended perimeter
    c.strokeStyle = 'rgba(107, 189, 140, 0.7)';
    c.fillStyle = 'rgba(107, 189, 140, 0.06)';
    c.setLineDash([4, 3]);
    c.beginPath(); c.arc(cx, cy, defR * sc, 0, Math.PI * 2); c.fill(); c.stroke();
    c.setLineDash([]);

    const [fi] = locateFrame(D.t, time);
    const N = D.t.length;

    const drawXY = (flatArr, upToIdx, color, width, alpha) => {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.globalAlpha = alpha;
      c.beginPath();
      let st = false;
      for (let i = 0; i <= upToIdx && i < N; i += 2) {
        const x = X(flatArr[3 * i] / 1000);
        const y = Y(flatArr[3 * i + 1] / 1000);
        if (!st) { c.moveTo(x, y); st = true; }
        else c.lineTo(x, y);
      }
      c.stroke();
      c.globalAlpha = 1;
    };

    drawXY(D.threat, N - 1, '#e0594a', 1.1, 0.22);
    if (drv.launchIdx >= 0) {
      drawXY(D.interceptor, N - 1, '#5aa6d6', 1.1, 0.22);
    }

    // PIP migration trail
    c.strokeStyle = '#d4a24c';
    c.lineWidth = 1.4;
    c.globalAlpha = 0.8;
    c.beginPath();
    let pSt = false;
    for (let i = 0; i <= fi; i += 2) {
      if (!drv.pipValid[i]) continue;
      const px = X(drv.pipTrack[3 * i] / 1000);
      const py = Y(drv.pipTrack[3 * i + 1] / 1000);
      if (!pSt) { c.moveTo(px, py); pSt = true; }
      else c.lineTo(px, py);
    }
    c.stroke();
    c.globalAlpha = 1;

    // 3-sigma PIP footprint ellipse
    if (drv.pipValid[fi]) {
      const px = X(drv.pipTrack[3 * fi] / 1000);
      const py = Y(drv.pipTrack[3 * fi + 1] / 1000);
      const rx = Math.max(2, ((drv.pipSigE[fi] || 0) / 1000) * sc);
      const ry = Math.max(2, ((drv.pipSigN[fi] || 0) / 1000) * sc);
      c.strokeStyle = 'rgba(212, 162, 76, 0.75)';
      c.fillStyle = 'rgba(212, 162, 76, 0.14)';
      c.beginPath(); c.ellipse(px, py, rx, ry, 0, 0, Math.PI * 2); c.fill(); c.stroke();
      c.fillStyle = '#d4a24c';
      c.beginPath(); c.arc(px, py, 3, 0, Math.PI * 2); c.fill();
    }

    drawXY(D.threat, fi, '#e0594a', 2.0, 1);
    if (drv.launchIdx >= 0 && fi >= drv.launchIdx) {
      drawXY(D.interceptor, fi, '#5aa6d6', 2.0, 1);
      const ix = X(D.interceptor[3 * fi] / 1000);
      const iy = Y(D.interceptor[3 * fi + 1] / 1000);
      c.fillStyle = '#5aa6d6';
      c.beginPath(); c.arc(ix, iy, 3.8, 0, Math.PI * 2); c.fill();
    }

    const tx = X(D.threat[3 * fi] / 1000);
    const ty = Y(D.threat[3 * fi + 1] / 1000);
    c.fillStyle = '#e0594a';
    c.beginPath(); c.arc(tx, ty, 3.8, 0, Math.PI * 2); c.fill();

    // Battery origin dot
    c.fillStyle = '#6bbd8c';
    c.beginPath(); c.arc(cx, cy, 3, 0, Math.PI * 2); c.fill();
  }
}

