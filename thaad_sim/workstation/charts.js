// Synchronized Multi-Channel Engineering Telemetry Charts
import { locateFrame } from './viewport3d.js';

function niceStep(range, count = 4) {
  const raw = Math.max(range, 1e-6) / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

function linTicks(lo, hi, count = 4) {
  const step = niceStep(hi - lo || 1, count);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    out.push(+v.toPrecision(10));
  }
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  const e0 = Math.ceil(Math.log10(Math.max(lo, 1e-6)));
  const e1 = Math.floor(Math.log10(Math.max(hi, 1e-5)));
  for (let e = e0; e <= e1; e++) out.push(10 ** e);
  return out;
}

function fmtVal(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5) {
    const e = Math.floor(Math.log10(a));
    const m = v / 10 ** e;
    return `${Math.abs(m - Math.round(m)) < 0.05 ? m.toFixed(0) : m.toFixed(1)}e${e}`;
  }
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}

export class TelemetryChartsManager {
  constructor(gridEl, onSeek) {
    this.gridEl = gridEl;
    this.onSeek = onSeek;
    this.activeGroup = 'essential';
    this.data = null;
    this.compareRuns = [];
    this.simTime = 0;
    this.hoverTime = null;
    this.charts = [];

    if ('ResizeObserver' in window) {
      new ResizeObserver(() => this.redrawAll(true)).observe(this.gridEl);
    }
  }

  setFilterGroup(grp) {
    this.activeGroup = grp;
    this.charts.forEach((ch) => {
      const show = grp === 'all' || ch.spec.groups.includes(grp);
      ch.card.hidden = !show;
    });
    requestAnimationFrame(() => this.redrawAll(true));
  }

  setData(data, compareRuns = []) {
    this.data = data;
    this.compareRuns = compareRuns;
    this._buildSpecs();
    this._mountDOM();
    this.redrawAll(true);
  }

  _buildSpecs() {
    const D = this.data;
    const drv = D.derived;
    const prm = D.params || {};
    const trueDragScaled = (prm.threat_drag ?? 0.00004) * 1e5;
    const estDragSeries = D.estDrag || drv.altT.map(() => 0);
    const trueDragSeries = drv.altT.map(() => trueDragScaled);

    this.specs = [
      {
        id: 'alt',
        title: 'Altitude',
        unit: 'km',
        groups: ['essential', 'kin'],
        yMin: 0,
        series: [
          { label: 'Threat', color: '#e0594a', data: drv.altT },
          { label: 'Interceptor', color: '#5aa6d6', data: drv.altI, compareKey: 'altI' },
        ],
      },
      {
        id: 'zem',
        title: 'Zero-effort miss',
        unit: 'm (log)',
        groups: ['essential', 'gui'],
        log: true,
        series: [
          { label: 'True', color: '#d4a24c', data: drv.zemTrue, compareKey: 'zemTrue' },
          { label: 'Est', color: '#5aa6d6', data: drv.zemEst, dash: [3, 3] },
        ],
        refs: [{ y: D.killRadius || 10, label: `Kill ${D.killRadius || 10}m`, color: '#6bbd8c' }],
      },
      {
        id: 'sep',
        title: 'Separation ‖r‖',
        unit: 'km (log)',
        groups: ['essential', 'kin'],
        log: true,
        series: [
          { label: 'Separation', color: '#a08ad4', data: drv.sep, compareKey: 'sep' },
        ],
        refs: [{ y: (D.killRadius || 10) / 1000, label: 'Kill Radius', color: '#6bbd8c' }],
      },
      {
        id: 'gcmd',
        title: 'Commanded acceleration',
        unit: 'g',
        groups: ['essential', 'gui'],
        yMin: 0,
        series: [
          { label: 'a_c', color: '#5aa6d6', data: D.g, compareRootKey: 'g' },
        ],
        refs: [{ y: D.maxG || 30, label: `Max ${D.maxG || 30}g`, color: '#e0594a' }],
      },
      {
        id: 'dv',
        title: 'Divert Δv used',
        unit: 'm/s',
        groups: ['essential', 'gui'],
        yMin: 0,
        series: [
          { label: 'Used', color: '#d4a24c', data: drv.dvUsed, compareKey: 'dvUsed' },
        ],
        refs: prm.kv_divert_dv ? [{ y: prm.kv_divert_dv, label: `Budget ${prm.kv_divert_dv}m/s`, color: '#e0594a' }] : [],
      },
      {
        id: 'epos',
        title: 'Position error',
        unit: 'm',
        groups: ['essential', 'kf'],
        yMin: 0,
        series: [
          { label: 'Raw', color: '#4a4d53', data: drv.eraw, thin: true },
          { label: 'Filter', color: '#6bbd8c', data: drv.epos, compareKey: 'epos' },
          { label: '3σ', color: '#d4a24c', data: drv.spos3, dash: [3, 3] },
        ],
      },
      {
        id: 'spd',
        title: 'Speed',
        unit: 'm/s',
        groups: ['kin'],
        yMin: 0,
        series: [
          { label: 'Threat', color: '#e0594a', data: drv.spdT },
          { label: 'Interceptor', color: '#5aa6d6', data: drv.spdI, compareKey: 'spdI' },
        ],
      },
      {
        id: 'vc',
        title: 'Closing velocity',
        unit: 'm/s',
        groups: ['kin'],
        series: [
          { label: 'V_c', color: '#5aa6d6', data: drv.vc, compareKey: 'vc' },
        ],
      },
      {
        id: 'los',
        title: 'LOS rate ‖ω‖',
        unit: 'mrad/s',
        groups: ['gui'],
        yMin: 0,
        clipQuantile: 0.97,
        series: [
          { label: 'True', color: '#e0594a', data: drv.losTrue, compareKey: 'losTrue' },
          { label: 'Est', color: '#5aa6d6', data: drv.losEst, dash: [3, 3] },
        ],
      },
      {
        id: 'pipbias',
        title: 'Impact prediction error',
        unit: 'm (log)',
        groups: ['gui'],
        log: true,
        series: [
          { label: 'Active', color: '#6bbd8c', data: drv.pipBias, compareKey: 'pipBias' },
          { label: 'Vacuum', color: '#e0594a', data: drv.pipVacBias, dash: [3, 3] },
        ],
      },
      {
        id: 'evel',
        title: 'Velocity error',
        unit: 'm/s',
        groups: ['kf'],
        yMin: 0,
        series: [
          { label: 'Filter', color: '#5aa6d6', data: drv.evel, compareKey: 'evel' },
          { label: '3σ', color: '#d4a24c', data: drv.svel3, dash: [3, 3] },
        ],
      },
      {
        id: 'exyz',
        title: 'Position error by axis',
        unit: 'm',
        groups: ['kf'],
        series: [
          { label: 'e_E', color: '#e0594a', data: drv.ex },
          { label: 'e_N', color: '#5aa6d6', data: drv.ey },
          { label: 'e_U', color: '#6bbd8c', data: drv.ez },
          { label: '+3σ', color: '#6a6d73', data: drv.sx3, dash: [2, 2] },
        ],
      },
      {
        id: 'nees',
        title: 'NEES / NIS',
        unit: '6-DOF χ²',
        groups: ['kf'],
        log: true,
        series: [
          { label: 'NEES', color: '#d4a24c', data: D.nees, compareRootKey: 'nees' },
          { label: 'NIS', color: '#5aa6d6', data: D.nis },
        ],
        refs: [
          { y: 14.45, label: '95% Upper (14.4)', color: '#6a6d73' },
          { y: 1.24, label: '95% Lower (1.24)', color: '#6a6d73' },
        ],
      },
      {
        id: 'drag',
        title: 'Drag estimate k̂_T',
        unit: '10⁻⁵ m⁻¹',
        groups: ['kf'],
        yMin: 0,
        series: [
          { label: 'Est', color: '#6bbd8c', data: estDragSeries, compareRootKey: 'estDrag' },
          { label: 'True', color: '#e0594a', data: trueDragSeries, dash: [4, 3] },
        ],
      },
    ];
  }

  _mountDOM() {
    this.gridEl.innerHTML = '';
    this.charts = [];

    this.specs.forEach((spec) => {
      const card = document.createElement('div');
      card.className = 'chart-card';
      card.hidden = !(this.activeGroup === 'all' || spec.groups.includes(this.activeGroup));

      const legendHtml = spec.series
        .map((s) => `<span><i style="--c:${s.color}"></i>${s.label}</span>`)
        .join('');

      card.innerHTML = `
        <div class="chart-head">
          <div><span class="chart-title">${spec.title}</span><span class="chart-unit">[${spec.unit}]</span></div>
          <div class="chart-legend">${legendHtml}</div>
        </div>
        <div class="chart-canvas-wrap">
          <canvas class="cv-base"></canvas>
          <canvas class="cv-over"></canvas>
          <div class="chart-readout"></div>
        </div>
      `;
      this.gridEl.appendChild(card);

      const ch = {
        spec,
        card,
        base: card.querySelector('.cv-base'),
        over: card.querySelector('.cv-over'),
        readout: card.querySelector('.chart-readout'),
        sizeKey: '',
      };
      this._bindChartEvents(ch);
      this.charts.push(ch);
    });
  }

  _bindChartEvents(ch) {
    let dragging = false;
    const tFromEvent = (e) => {
      if (!ch.xInv) return null;
      const r = ch.over.getBoundingClientRect();
      return ch.xInv(e.clientX - r.left);
    };
    ch.over.addEventListener('pointerdown', (e) => {
      dragging = true;
      const t = tFromEvent(e);
      if (t != null && this.onSeek) this.onSeek(t);
    });
    window.addEventListener('pointerup', () => {
      dragging = false;
    });
    ch.over.addEventListener('pointermove', (e) => {
      const t = tFromEvent(e);
      this.hoverTime = t;
      if (dragging && t != null && this.onSeek) this.onSeek(t);
      this.updateCursor(this.simTime);
    });
    ch.over.addEventListener('pointerleave', () => {
      this.hoverTime = null;
      this.updateCursor(this.simTime);
    });
  }

  redrawAll(forceBase = false) {
    if (!this.data) return;
    this.charts.forEach((ch) => {
      if (ch.card.hidden) return;
      if (forceBase) ch.sizeKey = '';
      this._layoutAndDrawBase(ch);
      this._drawOverlay(ch, this.simTime);
    });
  }

  updateCursor(time) {
    this.simTime = time;
    if (!this.data) return;
    this.charts.forEach((ch) => {
      if (ch.card.hidden) return;
      if (!ch.sizeKey) this._layoutAndDrawBase(ch);
      this._drawOverlay(ch, time);
    });
  }

  _layoutAndDrawBase(ch) {
    const w = ch.over.clientWidth;
    const h = ch.over.clientHeight;
    if (w < 10 || h < 10) return false;
    const key = `${w}x${h}`;
    if (ch.sizeKey === key) return true;
    ch.sizeKey = key;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    [ch.base, ch.over].forEach((cv) => {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    });

    const { spec } = ch;
    const T = this.data.t;
    const xMin = T[0];
    const xMax = T[T.length - 1];

    let lo = Infinity, hi = -Infinity;
    const take = (v) => {
      if (v == null || !Number.isFinite(v) || (spec.log && v <= 0)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    spec.series.forEach((s) => (s.data || []).forEach(take));
    (spec.refs || []).forEach((r) => take(r.y));

    // Include comparison series in bounds
    this.compareRuns.forEach((run) => {
      spec.series.forEach((s) => {
        const arr = s.compareKey ? run.derived[s.compareKey] : s.compareRootKey ? run[s.compareRootKey] : null;
        if (arr) arr.forEach(take);
      });
    });

    if (spec.clipQuantile && Number.isFinite(hi)) {
      const vals = spec.series.flatMap((s) => (s.data || []).filter((v) => v != null && Number.isFinite(v)));
      vals.sort((a, b) => a - b);
      const q = vals[Math.floor((vals.length - 1) * spec.clipQuantile)];
      if (q > 0) hi = Math.min(hi, q * 1.6);
    }
    if (!Number.isFinite(lo)) {
      lo = spec.log ? 0.1 : 0;
      hi = spec.log ? 100 : 10;
    }
    if (spec.log) {
      hi = 10 ** Math.ceil(Math.log10(Math.max(hi, 1e-3)));
      lo = Math.max(10 ** Math.floor(Math.log10(Math.max(lo, 1e-4))), hi / 1e6);
      if (hi <= lo) hi = lo * 10;
    } else {
      const span = hi - lo || Math.abs(hi) || 1;
      lo = spec.yMin != null ? spec.yMin : lo - span * 0.08;
      hi = hi + span * 0.1;
    }

    const pad = { l: 42, r: 10, t: 8, b: 18 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ch.w = w; ch.h = h; ch.pad = pad; ch.pw = pw; ch.ph = ph;
    ch.lo = lo; ch.hi = hi; ch.xMin = xMin; ch.xMax = xMax;
    ch.X = (t) => pad.l + ((t - xMin) / Math.max(xMax - xMin, 1e-6)) * pw;
    ch.xInv = (px) => Math.max(xMin, Math.min(xMax, xMin + ((px - pad.l) / Math.max(pw, 1)) * (xMax - xMin)));

    const l0 = Math.log10(Math.max(lo, 1e-9));
    const l1 = Math.log10(Math.max(hi, 1e-8));
    ch.Y = spec.log
      ? (v) => pad.t + (1 - (Math.log10(Math.max(v, lo)) - l0) / Math.max(l1 - l0, 1e-6)) * ph
      : (v) => pad.t + (1 - (v - lo) / Math.max(hi - lo, 1e-6)) * ph;

    const c = ch.base.getContext('2d');
    c.clearRect(0, 0, w, h);
    c.font = '10.5px "Cascadia Mono", "SF Mono", Consolas, ui-monospace, monospace';

    // Horizontal Y grid
    const yTicks = spec.log ? logTicks(lo, hi) : linTicks(lo, hi, 3);
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    yTicks.forEach((v) => {
      const y = ch.Y(v);
      if (y < pad.t - 1 || y > pad.t + ph + 1) return;
      c.strokeStyle = 'rgba(155, 157, 162, 0.09)';
      c.lineWidth = 1;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
      c.fillStyle = '#6a6d73';
      c.fillText(fmtVal(v), pad.l - 5, y);
    });

    // Vertical X grid
    const xTicks = linTicks(xMin, xMax, 5);
    c.textAlign = 'center';
    c.textBaseline = 'top';
    xTicks.forEach((t) => {
      const x = ch.X(t);
      c.strokeStyle = 'rgba(155, 157, 162, 0.06)';
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, pad.t + ph); c.stroke();
      c.fillStyle = '#6a6d73';
      c.fillText(`${t.toFixed(0)}s`, x, pad.t + ph + 4);
    });

    // Phase vertical markers (Launch, Burnout, End)
    const drv = this.data.derived;
    [
      [drv.launchT, '#d4a24c'],
      [drv.burnT, '#5aa6d6'],
      [drv.endEventT, this.data.outcome === 'KILL' ? '#6bbd8c' : '#e0594a'],
    ].forEach(([mt, col]) => {
      if (mt == null) return;
      const x = ch.X(mt);
      c.strokeStyle = col;
      c.globalAlpha = 0.32;
      c.setLineDash([2, 3]);
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, pad.t + ph); c.stroke();
      c.setLineDash([]);
      c.globalAlpha = 1;
    });

    // Reference lines
    (spec.refs || []).forEach((r) => {
      const y = ch.Y(r.y);
      if (y < pad.t || y > pad.t + ph) return;
      c.strokeStyle = r.color || '#9b9da2';
      c.globalAlpha = 0.7;
      c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
      c.setLineDash([]);
      c.fillStyle = r.color || '#9b9da2';
      // Left edge, clear of the value readout at the top right; below the
      // line when it sits too close to the top of the plot
      c.textAlign = 'left';
      const below = y - pad.t < 14;
      c.textBaseline = below ? 'top' : 'bottom';
      c.fillText(r.label, pad.l + 4, below ? y + 3 : y - 2);
      c.globalAlpha = 1;
    });

    // Plot series helper
    const plotLine = (tArr, yArr, color, width, dash = []) => {
      if (!yArr) return;
      c.strokeStyle = color;
      c.lineWidth = width;
      c.setLineDash(dash);
      c.beginPath();
      let pen = false;
      const step = Math.max(1, Math.floor(tArr.length / Math.max(pw, 120)));
      for (let i = 0; i < tArr.length; i += step) {
        const v = yArr[i];
        if (v == null || !Number.isFinite(v) || (spec.log && v <= 0)) {
          pen = false;
          continue;
        }
        const x = ch.X(tArr[i]);
        const y = Math.max(pad.t, Math.min(pad.t + ph, ch.Y(v)));
        if (!pen) { c.moveTo(x, y); pen = true; }
        else c.lineTo(x, y);
      }
      c.stroke();
      c.setLineDash([]);
    };

    // Comparison runs first
    this.compareRuns.forEach((run) => {
      spec.series.forEach((s) => {
        const arr = s.compareKey ? run.derived[s.compareKey] : s.compareRootKey ? run[s.compareRootKey] : null;
        if (arr) plotLine(run.t, arr, run.compareColor || '#a08ad4', 1.4, [3, 2]);
      });
    });

    // Primary series
    spec.series.forEach((s) => {
      plotLine(T, s.data, s.color, s.thin ? 1 : 1.4, s.dash || []);
    });

    return true;
  }

  _drawOverlay(ch, time) {
    if (!ch.w) return;
    const c = ch.over.getContext('2d');
    const { w, h, pad, ph, spec } = ch;
    c.clearRect(0, 0, w, h);

    const T = this.data.t;
    const drawCursor = (tVal, isHover) => {
      const x = ch.X(tVal);
      if (x < pad.l || x > w - pad.r) return;
      c.strokeStyle = isHover ? 'rgba(214, 215, 217, 0.35)' : 'rgba(240, 240, 241, 0.85)';
      c.lineWidth = 1;
      if (isHover) c.setLineDash([2, 2]);
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, pad.t + ph); c.stroke();
      c.setLineDash([]);

      const [fi] = locateFrame(T, tVal);
      spec.series.forEach((s) => {
        const v = s.data?.[fi];
        if (v == null || !Number.isFinite(v) || (spec.log && v <= 0)) return;
        const y = ch.Y(v);
        if (y < pad.t || y > pad.t + ph) return;
        c.fillStyle = s.color;
        c.beginPath(); c.arc(x, y, 2.5, 0, Math.PI * 2); c.fill();
      });
    };

    drawCursor(time, false);

    const inspectT = this.hoverTime != null ? this.hoverTime : time;
    if (this.hoverTime != null) {
      drawCursor(this.hoverTime, true);
    }

    const [fi] = locateFrame(T, inspectT);
    const parts = spec.series.map((s) => {
      const v = s.data?.[fi];
      return `<span style="color:${s.color}">${s.label}: ${fmtVal(v)}</span>`;
    });
    ch.readout.style.display = 'block';
    ch.readout.innerHTML = parts.join(' · ');
  }
}

