// Batch results dashboard. The payload comes from python -m thaad_sim.dashboard;
// every statistic in it is already computed, this page only lays it out.
import './styles.css';
import { fmt, histogram, lineChart, pct, rateBars, responsive } from './svg.js';

const D = JSON.parse(document.getElementById('dashboard-data').textContent);
const $ = (id) => document.getElementById(id);

const MODE = { engineering: 'Engineering mode', demo: 'Max-performance demo' };
const OUTCOME_NOTES = {
  KILL: 'interceptor within the kill radius',
  GROUND: 'threat reached the ground',
  'NO ENGAGE': 'PIP outside the defended area, held fire',
  TIMEOUT: 'still airborne when the run ended',
  engaged: 'an interceptor was launched',
  'KILL | engaged': 'kills among engaged threats',
};

function text(tag, cls, content, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (content != null) e.textContent = content;
  if (parent) parent.appendChild(e);
  return e;
}

function section(id, title, intro) {
  const s = text('section', 'block', null, $('main'));
  s.id = id;
  text('h2', null, title, s);
  if (intro) text('p', 'intro', intro, s);
  return s;
}

function figure(parent, title, sub, caption) {
  const f = text('figure', 'fig', null, parent);
  const head = text('div', 'fig-head', null, f);
  text('h3', null, title, head);
  if (sub) text('span', 'sub', sub, head);
  const host = text('div', 'plot', null, f);
  if (caption) text('figcaption', null, caption, f);
  return { f, head, host };
}

function legend(head, items) {
  const l = text('div', 'legend', null, head);
  items.forEach(([label, color, kind]) => {
    const s = text('span', null, null, l);
    const k = text('i', kind ?? 'line', null, s);
    k.style.setProperty('--c', `var(${color})`);
    s.appendChild(document.createTextNode(label));
  });
}

function table(parent, summary, cols, rows) {
  const d = text('details', 'table-view', null, parent);
  text('summary', null, summary, d);
  const t = text('table', null, null, d);
  const tr = text('tr', null, null, text('thead', null, null, t));
  cols.forEach((c) => text('th', null, c, tr));
  const tb = text('tbody', null, null, t);
  rows.forEach((r) => { const row = text('tr', null, null, tb); r.forEach((c) => text('td', null, c, row)); });
}

// ---- header ----------------------------------------------------------------
document.title = `Batch results · ${MODE[D.name] ?? D.name}`;
$('title').textContent = `${MODE[D.name] ?? D.name}: ${D.runs.toLocaleString('en-US')} seeded engagements`;
$('meta').textContent = `Seeds ${D.seeds[0]}–${D.seeds[1]} · ${D.source}`;

const byLabel = Object.fromEntries(D.rates.map((r) => [r.label, r]));
{
  const hero = byLabel['KILL | engaged'];
  const tiles = $('tiles');
  if (hero) {
    const t = text('div', 'tile hero', null, tiles);
    text('div', 'label', 'Kill rate among engaged threats', t);
    text('div', 'value', pct(hero.p), t);
    text('div', 'note', `95 % CI ${pct(hero.lo)}–${pct(hero.hi)} · ${hero.k.toLocaleString('en-US')} of ${hero.n.toLocaleString('en-US')}`, t);
  }
  const tile = (label, value, note) => {
    const t = text('div', 'tile', null, tiles);
    text('div', 'label', label, t); text('div', 'value', value, t); if (note) text('div', 'note', note, t);
  };
  if (byLabel.engaged) tile('Engaged', pct(byLabel.engaged.p), `${byLabel.engaged.k} of ${byLabel.engaged.n} runs`);
  if (byLabel['NO ENGAGE']?.k) tile('Held fire', pct(byLabel['NO ENGAGE'].p), 'PIP outside the defended area');
  const h = Object.fromEntries(D.histograms.map((x) => [x.key, x]));
  if (h.miss_distance_m) tile('Median closest approach', `${fmt(h.miss_distance_m.median)} m`, `IQR ${fmt(h.miss_distance_m.q1)}–${fmt(h.miss_distance_m.q3)} m, engaged runs`);
  if (h.kf_rms_pos_m) tile('Median filter RMS error', `${fmt(h.kf_rms_pos_m.median)} m`, `raw radar ${fmt(h.kf_rms_pos_m.reference.value)} m (√3·σ)`);
}

// ---- outcomes ----------------------------------------------------------------
{
  const s = section('outcomes', 'Outcome rates',
    'Each bar is a share of the runs with its 95 % Wilson score interval, which stays inside 0–100 % and keeps its coverage near the extremes. The last row conditions on the threats that were engaged.');
  const { host } = figure(s, 'Share of runs by outcome', `n = ${D.runs.toLocaleString('en-US')}`);
  const rows = D.rates.map((r) => ({ ...r, group: r.of, emph: r.label === 'KILL | engaged' }));
  responsive(host, () => rateBars(host, rows));
  table(s, 'Table: outcome counts and intervals', ['Outcome', 'Meaning', 'k / n', 'Rate', '95 % Wilson CI'],
    D.rates.map((r) => [r.label, OUTCOME_NOTES[r.label] ?? '', `${r.k} / ${r.n}`, pct(r.p), `${pct(r.lo)} – ${pct(r.hi)}`]));
}

// ---- metrics -----------------------------------------------------------------
{
  const s = section('metrics', 'Engagement metrics',
    'Distributions over the runs each metric applies to. The solid line marks the median. For the filter errors, the note gives the error of a single raw radar return, √3·σ, for comparison.');
  const grid = text('div', 'grid', null, s);
  D.histograms.forEach((h) => {
    const { host } = figure(grid, h.title, `${h.subset}, n = ${h.n.toLocaleString('en-US')}`);
    responsive(host, () => histogram(host, h));
  });
  table(s, 'Table: medians and interquartile ranges', ['Metric', 'Runs', 'n', 'Median', 'Interquartile range'],
    D.histograms.map((h) => [`${h.title} [${h.unit}]`, h.subset, h.n.toLocaleString('en-US'), fmt(h.median, 4), `${fmt(h.q1, 4)} – ${fmt(h.q3, 4)}`]));
}

// ---- consistency -----------------------------------------------------------------
{
  const c = D.consistency;
  const s = section('consistency', 'Kalman filter consistency',
    c ? `Average normalized estimation error squared (NEES) and normalized innovation squared (NIS) over ${c.runs} seeded tracking runs. For a consistent filter both are χ² with ${c.state_dim} degrees of freedom, so the average must stay inside the shaded 95 % bounds about 95 % of the time. Matched model: drag off and the truth driven by the filter's own Q. As configured: the real threat, whose drag the filter does not model (q = ${c.q} m²/s³).`
      : null);
  if (!c) {
    text('p', 'missing', `No consistency data. Run ${D.commands.consistency} and rebuild the dashboard.`, s);
  } else {
    const grid = text('div', 'grid two', null, s);
    [['nees', 'Average NEES', c.nees_bounds], ['nis', 'Average NIS', c.nis_bounds]].forEach(([k, title, b]) => {
      const m = c.matched[k], g = c.configured[k];
      const { head, host } = figure(grid, title, `dof ${k === 'nees' ? c.state_dim : c.meas_dim}`,
        `Matched model: mean ${m.mean.toFixed(2)}, ${pct(m.inside)} of ${m.steps.toLocaleString('en-US')} steps inside [${b[0].toFixed(2)}, ${b[1].toFixed(2)}]. ` +
        `As configured: mean ${g.mean.toFixed(2)}, ${pct(g.inside)} of ${g.steps.toLocaleString('en-US')} steps inside.`);
      legend(head, [['Matched model', '--s1'], ['As configured', '--s2'], ['95 % bounds', '--band', 'swatch']]);
      responsive(host, () => lineChart(host, {
        series: [{ label: 'Matched model', color: '--s1', x: m.t, y: m.value }, { label: 'As configured', color: '--s2', x: g.t, y: g.value }],
        band: { lo: b[0], hi: b[1] }, refs: [{ y: k === 'nees' ? c.state_dim : c.meas_dim }],
        xLabel: 'time since first detection [s]', xName: 't', xUnit: 's', yMin: 0,
      }));
    });
    table(s, 'Table: consistency summary', ['Statistic', 'Setting', 'Mean', 'Steps inside 95 % bounds', 'Bounds'],
      ['nees', 'nis'].flatMap((k) => [['matched', 'Matched model'], ['configured', 'As configured']].map(([key, name]) => {
        const v = c[key][k], b = c[`${k}_bounds`];
        return [k.toUpperCase(), name, v.mean.toFixed(2), pct(v.inside), `${b[0].toFixed(2)} – ${b[1].toFixed(2)}`];
      })));
  }
}

// ---- convergence -----------------------------------------------------------------
{
  const c = D.convergence;
  const s = section('convergence', 'Integrator convergence',
    c ? 'Final-position error against an RK4 run at a 16× finer step, at dt, dt/2, dt/4 and dt/8. On log–log axes the slope is the observed order: about 1 for forward Euler and 4 for RK4, also through the interceptor\'s burnout, where the step that contains it is split.'
      : null);
  if (!c) {
    text('p', 'missing', `No convergence data. Run ${D.commands.convergence} and rebuild the dashboard.`, s);
  } else {
    const grid = text('div', 'grid two', null, s);
    c.convergence.forEach((p) => {
      const last = (m) => p[m].orders[p[m].orders.length - 1];
      const { head, host } = figure(grid, p.title, 'final-position error [m], log–log',
        `Observed order at the finest step: Euler ${last('euler').toFixed(2)}, RK4 ${last('rk4').toFixed(2)}. Dashed: slopes 1 and 4.`);
      legend(head, [['Forward Euler', '--s1'], ['RK4', '--s2']]);
      const ref = (m, k) => p.dts.map((dt) => 0.2 * p[m].errors[0] * (dt / p.dts[0]) ** k);
      responsive(host, () => lineChart(host, {
        series: [
          { label: 'Forward Euler', color: '--s1', x: p.dts, y: p.euler.errors, markers: true },
          { label: 'RK4', color: '--s2', x: p.dts, y: p.rk4.errors, markers: true },
        ],
        guides: [{ x: p.dts, y: ref('euler', 1), label: 'slope 1' }, { x: p.dts, y: ref('rk4', 4), label: 'slope 4' }],
        logX: true, logY: true, xTicks: p.dts, xLabel: 'time step dt [s]', xName: 'dt', xUnit: 's', height: 260,
      }));
    });
    table(s, 'Table: errors and observed orders', ['Problem', 'dt [s]', 'Euler error [m]', 'RK4 error [m]', 'Euler order', 'RK4 order'],
      c.convergence.flatMap((p) => p.dts.map((dt, i) => [
        p.problem, String(dt), fmt(p.euler.errors[i], 4), fmt(p.rk4.errors[i], 4),
        i ? p.euler.orders[i - 1].toFixed(2) : '', i ? p.rk4.orders[i - 1].toFixed(2) : '',
      ])));
    const a = c.analytic;
    if (a) {
      text('p', 'intro', `Analytic check (drag off, dt = ${a.dt} s): forward Euler's altitude error matches its predicted truncation error g·t·dt/2 to ${fmt(a.euler_max_dev, 2)} m over ${a.t[a.t.length - 1]} s, and RK4 matches the closed-form solution to ${fmt(a.rk4_max_abs, 2)} m.`, s);
    }
  }
}

// ---- reproduce -----------------------------------------------------------------
{
  const s = section('reproduce', 'Reproduce', 'Every number on this page comes from these commands, run from the project folder.');
  const pre = text('pre', null, null, s);
  pre.textContent = Object.values(D.commands).join('\n');
}
