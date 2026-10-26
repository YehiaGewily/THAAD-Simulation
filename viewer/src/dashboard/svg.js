// Minimal SVG charts for the batch dashboard: horizontal bars with interval
// whiskers, histograms and line charts (linear or log axes, an optional band,
// reference lines). Colours come from CSS custom properties, so the charts
// follow the light/dark theme without re-rendering. Every chart has a hover
// layer; the values it shows are also in the tables under each section.
const NS = 'http://www.w3.org/2000/svg';

export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

export const fmt = (v, d = 3) => {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(1).replace('e+', 'e');
  return (+v.toPrecision(d)).toLocaleString('en-US');
};
export const pct = (p, d = 1) => `${(100 * p).toFixed(d)} %`;

function niceStep(range, count) {
  const raw = range / count, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
function linTicks(lo, hi, count) {
  const st = niceStep(hi - lo || 1, count), out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-6; v += st) out.push(+v.toPrecision(12));
  return out;
}
function logTicks(lo, hi) {
  const e0 = Math.ceil(Math.log10(lo) - 1e-9), e1 = Math.floor(Math.log10(hi) + 1e-9);
  const every = Math.max(1, Math.ceil((e1 - e0 + 1) / 6));   // at most about six labelled decades
  const out = [];
  for (let e = e0; e <= e1; e++) if ((e - e0) % every === 0) out.push(10 ** e);
  return out;
}
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
// Decades as 0.01 … 1000 where short, else as powers of ten
export const fmtDecade = (v) => {
  const e = Math.round(Math.log10(v));
  return e >= -2 && e <= 3 ? fmt(v) : `10${String(e).replace(/./g, (c) => SUP[c])}`;
};

function scale(lo, hi, r0, r1, log) {
  if (log) {
    const a = Math.log10(lo), b = Math.log10(hi);
    return (v) => r0 + ((Math.log10(v) - a) / (b - a)) * (r1 - r0);
  }
  return (v) => r0 + ((v - lo) / (hi - lo || 1)) * (r1 - r0);
}

// One tooltip per chart; rows are [value, label, colour-var or null]. Text goes in via textContent.
function tooltip(host) {
  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.hidden = true;
  host.appendChild(tip);
  return {
    show(x, y, title, rows) {
      tip.replaceChildren();
      const h = document.createElement('div'); h.className = 'tip-title'; h.textContent = title; tip.appendChild(h);
      rows.forEach(([value, label, color]) => {
        const r = document.createElement('div'); r.className = 'tip-row';
        if (color) { const k = document.createElement('i'); k.style.background = `var(${color})`; r.appendChild(k); }
        const b = document.createElement('b'); b.textContent = value; r.appendChild(b);
        if (label) { const s = document.createElement('span'); s.textContent = label; r.appendChild(s); }
        tip.appendChild(r);
      });
      tip.hidden = false;
      const w = tip.offsetWidth, hw = host.clientWidth;
      // Right of the pointer, or left of it when it would overflow the chart
      tip.style.left = `${x + 14 + w > hw ? Math.max(0, x - w - 14) : x + 14}px`;
      tip.style.top = `${Math.max(0, y - 10)}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function frame(host, height, pad) {
  host.replaceChildren();
  host.classList.add('chart');
  const w = Math.max(host.clientWidth, 260), h = height;
  const svg = el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img' }, host);
  return { svg, w, h, pw: w - pad.l - pad.r, ph: h - pad.t - pad.b, tip: tooltip(host) };
}

function axisY(svg, ticks, y, pad, w, fmtTick) {
  const g = el('g', { class: 'axis' }, svg);
  ticks.forEach((v) => {
    const yy = Math.round(y(v)) + 0.5;
    el('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, class: 'grid' }, g);
    el('text', { x: pad.l - 8, y: yy, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, g).textContent = fmtTick(v);
  });
}
function axisX(svg, ticks, x, pad, h, fmtTick, label, w) {
  const g = el('g', { class: 'axis' }, svg);
  el('line', { x1: pad.l, x2: w - pad.r, y1: h - pad.b + 0.5, y2: h - pad.b + 0.5, class: 'baseline' }, g);
  ticks.forEach((v) => {
    el('text', { x: x(v), y: h - pad.b + 16, 'text-anchor': 'middle' }, g).textContent = fmtTick(v);
  });
  if (label) el('text', { x: pad.l + (w - pad.l - pad.r) / 2, y: h - 4, 'text-anchor': 'middle', class: 'axis-label' }, g).textContent = label;
}

/** Horizontal bars, 0–100 %, with Wilson whiskers. rows: {label, k, n, p, lo, hi, group}. */
export function rateBars(host, rows) {
  const narrow = host.clientWidth < 480;
  const rowH = 34, pad = { l: narrow ? 104 : 132, r: narrow ? 62 : 92, t: 8, b: 30 };
  const groups = [...new Set(rows.map((r) => r.group))];
  const gap = 18;
  const f = frame(host, pad.t + pad.b + rows.length * rowH + (groups.length - 1) * gap, pad);
  const x = scale(0, 1, pad.l, f.w - pad.r);
  const g = el('g', { class: 'axis' }, f.svg);
  (f.pw < 320 ? [0, 0.5, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1]).forEach((v) => {
    const xx = Math.round(x(v)) + 0.5;
    el('line', { x1: xx, x2: xx, y1: pad.t, y2: f.h - pad.b, class: v === 0 ? 'baseline' : 'grid' }, g);
    el('text', { x: xx, y: f.h - pad.b + 16, 'text-anchor': 'middle' }, g).textContent = `${v * 100} %`;
  });
  let y = pad.t, prev = rows[0]?.group;
  rows.forEach((r) => {
    if (r.group !== prev) { y += gap; prev = r.group; el('line', { x1: 8, x2: f.w - 8, y1: y - gap / 2, y2: y - gap / 2, class: 'divider' }, f.svg); }
    const cy = y + rowH / 2, bh = 16;
    const row = el('g', { class: 'bar-row', tabindex: 0 }, f.svg);
    el('rect', { x: 0, y, width: f.w, height: rowH, class: 'hit' }, row);
    el('text', { x: pad.l - 12, y: cy, 'text-anchor': 'end', 'dominant-baseline': 'middle', class: 'cat' }, row).textContent = r.label;
    const x0 = x(0), x1 = x(r.p);
    if (x1 - x0 > 0.5) {
      // 4px rounded data end, square at the baseline
      const rr = Math.min(4, (x1 - x0) / 2);
      el('path', { d: `M${x0},${cy - bh / 2}H${x1 - rr}a${rr},${rr} 0 0 1 ${rr},${rr}V${cy + bh / 2 - rr}a${rr},${rr} 0 0 1 ${-rr},${rr}H${x0}Z`, class: `bar ${r.emph ? 'emph' : ''}` }, row);
    }
    // 95 % interval whisker
    const wl = x(r.lo), wr = x(r.hi);
    el('line', { x1: wl, x2: wr, y1: cy, y2: cy, class: 'whisker' }, row);
    el('line', { x1: wl, x2: wl, y1: cy - 6, y2: cy + 6, class: 'whisker' }, row);
    el('line', { x1: wr, x2: wr, y1: cy - 6, y2: cy + 6, class: 'whisker' }, row);
    el('text', { x: Math.max(wr, x1) + 8, y: cy, 'dominant-baseline': 'middle', class: 'val' }, row).textContent = pct(r.p);
    const show = () => f.tip.show(Math.max(wr, x1), y, r.label, [
      [pct(r.p), `${r.k.toLocaleString('en-US')} / ${r.n.toLocaleString('en-US')}`, null],
      [`${pct(r.lo)} – ${pct(r.hi)}`, '95 % Wilson interval', null],
    ]);
    row.addEventListener('pointerenter', show);
    row.addEventListener('focus', show);
    row.addEventListener('pointerleave', () => f.tip.hide());
    row.addEventListener('blur', () => f.tip.hide());
    y += rowH;
  });
  return f;
}

/** Histogram from precomputed bins: {edges, counts, median, log, unit, reference}. */
export function histogram(host, h) {
  const pad = { l: 40, r: 12, t: 14, b: 38 };
  const f = frame(host, 196, pad);
  const lo = h.edges[0], hi = h.edges[h.edges.length - 1];
  // A reference far beyond the data (raw radar error) is noted at the edge rather than
  // stretching the axis until the histogram is a sliver
  const ref = h.reference?.value;
  const refOff = ref != null && (ref > hi * 1.25 || ref < lo);
  const xHi = ref != null && !refOff && ref > hi ? ref * (h.log ? 1.4 : 1.06) : hi;
  const x = scale(lo, xHi, pad.l, f.w - pad.r, h.log);
  const cMax = Math.max(...h.counts, 1);
  const yt = linTicks(0, cMax, 3);
  const y = scale(0, Math.max(cMax, yt[yt.length - 1]), f.h - pad.b, pad.t);
  axisY(f.svg, yt, y, pad, f.w, (v) => fmt(v));
  const xt = h.log ? logTicks(lo, xHi) : linTicks(lo, xHi, Math.max(3, Math.floor(f.pw / 70)));
  axisX(f.svg, xt.filter((v) => v >= lo && v <= xHi), x, pad, f.h, (v) => fmt(v), `${h.unit}${h.log ? ' · log scale' : ''}`, f.w);
  const bars = el('g', {}, f.svg);
  h.counts.forEach((c, i) => {
    const a = x(h.edges[i]), b = x(h.edges[i + 1]);
    const bw = Math.max(b - a - 2, 1);  // 2px surface gap between touching bars
    const top = y(c), base = y(0);
    const g = el('g', { class: 'bin', tabindex: c ? 0 : -1 }, bars);
    el('rect', { x: a, y: pad.t, width: Math.max(b - a, 1), height: f.h - pad.b - pad.t, class: 'hit' }, g);
    if (c > 0) {
      const rr = Math.min(2, bw / 2, (base - top) / 2);
      el('path', { d: `M${a + 1},${base}V${top + rr}a${rr},${rr} 0 0 1 ${rr},${-rr}H${a + 1 + bw - rr}a${rr},${rr} 0 0 1 ${rr},${rr}V${base}Z`, class: 'bar' }, g);
    }
    const show = () => f.tip.show(b, pad.t, `${fmt(h.edges[i])} – ${fmt(h.edges[i + 1])} ${h.unit}`, [[c.toLocaleString('en-US'), c === 1 ? 'run' : 'runs', null]]);
    g.addEventListener('pointerenter', show);
    g.addEventListener('focus', show);
    g.addEventListener('pointerleave', () => f.tip.hide());
    g.addEventListener('blur', () => f.tip.hide());
  });
  const vline = (v, cls, label, anchor) => {
    const xx = Math.round(x(v)) + 0.5;
    el('line', { x1: xx, x2: xx, y1: pad.t, y2: f.h - pad.b, class: cls }, f.svg);
    const right = xx > f.w - 110;
    el('text', { x: xx + (right ? -5 : 5), y: pad.t + (anchor ?? 0) + 4, 'text-anchor': right ? 'end' : 'start', 'dominant-baseline': 'hanging', class: 'note' }, f.svg).textContent = label;
  };
  vline(h.median, 'median', `median ${fmt(h.median)}`);
  if (ref != null && !refOff) vline(ref, 'ref', `${h.reference.label} ${fmt(ref)}`, 16);
  else if (refOff) {
    el('text', { x: f.w - pad.r, y: pad.t + 20, 'text-anchor': 'end', 'dominant-baseline': 'hanging', class: 'note' }, f.svg)
      .textContent = `${h.reference.label} ${fmt(ref)} ${h.unit} (off scale ${ref > hi ? '→' : '←'})`;
  }
  return f;
}

/**
 * Line chart. spec: { series: [{label, color, x, y, markers}], band: {lo, hi, label},
 * refs: [{y, label}], guides: [{x: [..], y: [..], label}], logX, logY, xLabel, yMin, height }
 */
export function lineChart(host, spec) {
  const pad = { l: 52, r: 16, t: 14, b: 40 };
  const f = frame(host, spec.height ?? 240, pad);
  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  const take = (xs, ys) => xs.forEach((v, i) => {
    const w = ys[i];
    if (!Number.isFinite(v) || !Number.isFinite(w) || (spec.logY && w <= 0)) return;
    xlo = Math.min(xlo, v); xhi = Math.max(xhi, v); ylo = Math.min(ylo, w); yhi = Math.max(yhi, w);
  });
  spec.series.forEach((s) => take(s.x, s.y));
  (spec.guides || []).forEach((g) => take(g.x, g.y));
  if (spec.band) { ylo = Math.min(ylo, spec.band.lo); yhi = Math.max(yhi, spec.band.hi); }
  (spec.refs || []).forEach((r) => { ylo = Math.min(ylo, r.y); yhi = Math.max(yhi, r.y); });
  if (spec.logX) { xlo /= 1.3; xhi *= 1.3; }
  if (spec.logY) { ylo = 10 ** Math.floor(Math.log10(ylo)); yhi = 10 ** Math.ceil(Math.log10(yhi)); }
  else { const p = (yhi - ylo) * 0.08; ylo = spec.yMin ?? ylo - p; yhi += p; }
  const x = scale(xlo, xhi, pad.l, f.w - pad.r, spec.logX);
  const y = scale(ylo, yhi, f.h - pad.b, pad.t, spec.logY);
  const yt = spec.logY ? logTicks(ylo, yhi) : linTicks(ylo, yhi, 4);
  axisY(f.svg, yt, y, pad, f.w, spec.logY ? fmtDecade : (v) => fmt(v));
  const xt = spec.xTicks ?? (spec.logX ? logTicks(xlo, xhi) : linTicks(xlo, xhi, Math.max(3, Math.floor(f.pw / 80))));
  axisX(f.svg, xt, x, pad, f.h, (v) => fmt(v), spec.xLabel, f.w);

  const clip = `c${Math.random().toString(36).slice(2)}`;
  el('rect', { x: pad.l, y: pad.t, width: f.pw, height: f.ph }, el('clipPath', { id: clip }, el('defs', {}, f.svg)));
  const plot = el('g', { 'clip-path': `url(#${clip})` }, f.svg);
  if (spec.band) {
    el('rect', { x: pad.l, y: y(spec.band.hi), width: f.pw, height: y(spec.band.lo) - y(spec.band.hi), class: 'band' }, plot);
  }
  (spec.refs || []).forEach((r) => {
    const yy = y(r.y);
    el('line', { x1: pad.l, x2: f.w - pad.r, y1: yy, y2: yy, class: 'ref' }, plot);
    if (r.label) el('text', { x: f.w - pad.r - 4, y: yy - 4, 'text-anchor': 'end', class: 'note' }, plot).textContent = r.label;
  });
  const path = (xs, ys) => {
    let d = '', pen = false;
    xs.forEach((v, i) => {
      const w = ys[i];
      if (!Number.isFinite(w) || (spec.logY && w <= 0)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(v).toFixed(1)},${y(w).toFixed(1)}`; pen = true;
    });
    return d;
  };
  (spec.guides || []).forEach((g) => {
    el('path', { d: path(g.x, g.y), class: 'guide' }, plot);
    if (g.label) {
      const i = g.x.length - 1;
      el('text', { x: x(g.x[i]) + 6, y: y(g.y[i]), 'dominant-baseline': 'middle', class: 'note' }, plot).textContent = g.label;
    }
  });
  spec.series.forEach((s) => {
    el('path', { d: path(s.x, s.y), class: 'line', style: `stroke:var(${s.color})` }, plot);
    if (s.markers) s.x.forEach((v, i) => el('circle', { cx: x(v), cy: y(s.y[i]), r: 4.5, class: 'dot', style: `fill:var(${s.color})` }, plot));
  });

  // Hover: crosshair snaps to the nearest x of the first series
  const cross = el('line', { y1: pad.t, y2: f.h - pad.b, class: 'cross', visibility: 'hidden' }, f.svg);
  const dots = spec.series.map((s) => el('circle', { r: 4.5, class: 'dot', style: `fill:var(${s.color})`, visibility: 'hidden' }, f.svg));
  const xs0 = spec.series[0].x;
  const hit = el('rect', { x: pad.l, y: pad.t, width: f.pw, height: f.ph, class: 'hit' }, f.svg);
  const nearest = (arr, v) => {
    let best = 0, bd = Infinity;
    arr.forEach((a, i) => { const d = Math.abs((spec.logX ? Math.log10(a) : a) - (spec.logX ? Math.log10(v) : v)); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  const xInv = (px) => {
    const t = (px - pad.l) / f.pw;
    return spec.logX ? 10 ** (Math.log10(xlo) + t * (Math.log10(xhi) - Math.log10(xlo))) : xlo + t * (xhi - xlo);
  };
  hit.addEventListener('pointermove', (e) => {
    const r = f.svg.getBoundingClientRect();
    const xv = xs0[nearest(xs0, xInv(e.clientX - r.left))];
    const px = x(xv);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
    const rows = [];
    spec.series.forEach((s, j) => {
      const i = nearest(s.x, xv);
      const ok = Math.abs(s.x[i] - xv) <= Math.abs(xv) * 0.02 + 1e-9 || s.x.length > 50;
      const v = s.y[i];
      if (ok && Number.isFinite(v) && !(spec.logY && v <= 0)) {
        dots[j].setAttribute('cx', x(s.x[i])); dots[j].setAttribute('cy', y(v)); dots[j].setAttribute('visibility', 'visible');
        rows.push([fmt(v, 4), s.label, s.color]);
      } else dots[j].setAttribute('visibility', 'hidden');
    });
    f.tip.show(px, pad.t, `${spec.xName ?? 'x'} = ${fmt(xv, 4)}${spec.xUnit ? ` ${spec.xUnit}` : ''}`, rows);
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden'); dots.forEach((d) => d.setAttribute('visibility', 'hidden')); f.tip.hide();
  });
  return f;
}

/** Re-render on width changes. */
export function responsive(host, render) {
  let w = 0;
  const ro = new ResizeObserver(() => { if (host.clientWidth !== w) { w = host.clientWidth; render(); } });
  ro.observe(host);
}
