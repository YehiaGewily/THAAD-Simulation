// Monte Carlo results, verification tables and the methods page

// Three significant figures without exponent notation (1540, not 1.54e+3)
const sig3 = (v) => String(Number(Number(v).toPrecision(3)));

function pct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function drawHistogramCanvas(canvas, h) {
  const w = canvas.clientWidth || 280;
  const height = 140;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = w * dpr;
  canvas.height = height * dpr;
  const c = canvas.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, height);

  const pad = { l: 32, r: 10, t: 10, b: 22 };
  const pw = w - pad.l - pad.r;
  const ph = height - pad.t - pad.b;

  const counts = h.counts || [];
  const edges = h.edges || [];
  const maxC = Math.max(...counts, 1);
  const nBins = counts.length;
  if (!nBins) return;

  c.fillStyle = 'rgba(155, 157, 162, 0.55)';
  c.strokeStyle = '#9b9da2';
  c.lineWidth = 1;

  const bw = pw / nBins;
  for (let i = 0; i < nBins; i++) {
    const bh = (counts[i] / maxC) * ph;
    const x = pad.l + i * bw;
    const y = pad.t + ph - bh;
    c.fillRect(x + 0.5, y, Math.max(1, bw - 1), bh);
  }

  // Median marker
  const eMin = edges[0], eMax = edges[edges.length - 1];
  const xPos = (val) => {
    if (h.log && eMin > 0 && val > 0) {
      return pad.l + ((Math.log10(val) - Math.log10(eMin)) / Math.max(Math.log10(eMax) - Math.log10(eMin), 1e-6)) * pw;
    }
    return pad.l + ((val - eMin) / Math.max(eMax - eMin, 1e-6)) * pw;
  };

  if (h.median != null) {
    const xm = xPos(h.median);
    c.strokeStyle = '#f0f0f1';
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(xm, pad.t); c.lineTo(xm, pad.t + ph); c.stroke();
  }

  c.font = '10.5px "Cascadia Mono", "SF Mono", Consolas, ui-monospace, monospace';
  c.fillStyle = '#6a6d73';
  c.textAlign = 'left';
  c.fillText(sig3(eMin), pad.l, height - 6);
  c.textAlign = 'right';
  c.fillText(`${sig3(eMax)} ${h.unit || ''}`, w - pad.r, height - 6);
}

export function renderBatchView(container, payload, onSelectSeed) {
  if (!payload) {
    container.innerHTML = `<div class="batch-section"><h3>No batch data</h3><p class="sub">Use “Run batch” above to simulate one.</p></div>`;
    return;
  }

  const byLabel = Object.fromEntries((payload.rates || []).map((r) => [r.label, r]));
  const hero = byLabel['KILL | engaged'];
  const engaged = byLabel.engaged;
  const hMap = Object.fromEntries((payload.histograms || []).map((x) => [x.key, x]));

  let html = `
    <div class="batch-tiles">
      ${
        hero
          ? `<div class="b-tile hero">
              <div class="lbl">Kill rate, engaged</div>
              <div class="val">${pct(hero.p)}</div>
              <div class="note">95% CI ${pct(hero.lo)}–${pct(hero.hi)} · ${hero.k}/${hero.n}</div>
            </div>`
          : ''
      }
      ${
        engaged
          ? `<div class="b-tile">
              <div class="lbl">Engaged</div>
              <div class="val">${pct(engaged.p)}</div>
              <div class="note">${engaged.k}/${engaged.n} runs</div>
            </div>`
          : ''
      }
      ${
        hMap.miss_distance_m
          ? `<div class="b-tile">
              <div class="lbl">Median miss</div>
              <div class="val">${hMap.miss_distance_m.median.toFixed(2)} m</div>
              <div class="note">IQR ${hMap.miss_distance_m.q1.toFixed(2)} – ${hMap.miss_distance_m.q3.toFixed(2)} m</div>
            </div>`
          : ''
      }
      ${
        hMap.divert_dv_used_mps
          ? `<div class="b-tile">
              <div class="lbl">Median divert Δv</div>
              <div class="val">${hMap.divert_dv_used_mps.median.toFixed(0)} m/s</div>
              <div class="note">IQR ${hMap.divert_dv_used_mps.q1.toFixed(0)} – ${hMap.divert_dv_used_mps.q3.toFixed(0)} m/s</div>
            </div>`
          : ''
      }
      ${
        hMap.kf_rms_pos_m
          ? `<div class="b-tile">
              <div class="lbl">Median filter RMS error</div>
              <div class="val">${hMap.kf_rms_pos_m.median.toFixed(1)} m</div>
              <div class="note">IQR ${hMap.kf_rms_pos_m.q1.toFixed(1)} – ${hMap.kf_rms_pos_m.q3.toFixed(1)} m</div>
            </div>`
          : ''
      }
    </div>

    <div class="batch-section">
      <h3>Outcome rates</h3>
      <p class="sub">${payload.name} · n = ${payload.runs} · seeds ${payload.seeds[0]}–${payload.seeds[1]} · ${String(payload.source || '').split(/[\\/]/).pop()} · 95% Wilson score intervals</p>
      <table class="data-table">
        <thead>
          <tr><th>Outcome</th><th>k / n</th><th>Rate</th><th>95% CI</th></tr>
        </thead>
        <tbody>
          ${(payload.rates || [])
            .map(
              (r) => `<tr>
                <td><strong>${r.label}</strong></td>
                <td>${r.k} / ${r.n}</td>
                <td>${pct(r.p)}</td>
                <td>${pct(r.lo)} – ${pct(r.hi)}</td>
              </tr>`
            )
            .join('')}
        </tbody>
      </table>
    </div>

    <div class="batch-section">
      <h3>Distributions</h3>
      <p class="sub">Over the relevant subset of runs (kills, engaged, or all). The white line marks the median.</p>
      <div class="hist-grid" id="batch-hist-grid"></div>
    </div>
  `;

  if (payload.seedSummaries && payload.seedSummaries.length) {
    html += `
      <div class="batch-section">
        <h3>Runs</h3>
        <p class="sub">Open any seed in the engagement view.</p>
        <table class="data-table">
          <thead>
            <tr><th>Seed</th><th>Outcome</th><th>Miss</th><th>Kill alt</th><th>Divert Δv</th><th>Peak g</th><th>Filter RMS</th><th></th></tr>
          </thead>
          <tbody>
            ${payload.seedSummaries
              .map(
                (s) => `<tr>
                  <td>${s.seed}</td>
                  <td><span class="badge ${s.outcome === 'KILL' ? 'ok' : s.outcome === 'NO ENGAGE' ? 'info' : 'bad'}">${s.outcome}</span></td>
                  <td>${s.miss_m != null ? s.miss_m + ' m' : '—'}</td>
                  <td>${s.kill_alt_km != null ? s.kill_alt_km + ' km' : '—'}</td>
                  <td>${s.dv_used != null ? s.dv_used + ' m/s' : '—'}</td>
                  <td>${s.max_g != null ? s.max_g + ' g' : '—'}</td>
                  <td>${s.kf_rms_pos_m != null ? s.kf_rms_pos_m + ' m' : '—'}</td>
                  <td><button type="button" class="btn-mini btn-replay-seed" data-seed="${s.seed}">Open</button></td>
                </tr>`
              )
              .join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  if (payload.consistency && payload.convergence) {
    const cs = payload.consistency;
    const cv = payload.convergence;
    html += `
      <div class="batch-section">
        <h3>Verification</h3>
        <p class="sub">Filter consistency over ${cs.runs} seeded tracking runs; integrator order from step-halving convergence.</p>
        <table class="data-table">
          <thead>
            <tr><th>Check</th><th>Case</th><th>Observed</th><th>Reference</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>NEES (6-DOF)</strong></td><td>Matched Linear Model</td><td>Mean ${cs.matched.nees.mean.toFixed(2)} (${pct(cs.matched.nees.inside)} inside)</td><td>95% Bounds [${cs.nees_bounds[0].toFixed(2)}, ${cs.nees_bounds[1].toFixed(2)}]</td></tr>
            <tr><td><strong>NEES (6-DOF)</strong></td><td>With Unmodeled Drag (KF6)</td><td>Mean ${cs.configured.nees.mean.toFixed(2)} (${pct(cs.configured.nees.inside)} inside)</td><td>95% Bounds [${cs.nees_bounds[0].toFixed(2)}, ${cs.nees_bounds[1].toFixed(2)}]</td></tr>
            <tr><td><strong>NIS (6-DOF)</strong></td><td>Matched Linear Model</td><td>Mean ${cs.matched.nis.mean.toFixed(2)} (${pct(cs.matched.nis.inside)} inside)</td><td>95% Bounds [${cs.nis_bounds[0].toFixed(2)}, ${cs.nis_bounds[1].toFixed(2)}]</td></tr>
            ${cv.convergence
              .map(
                (p) => `<tr>
                  <td><strong>Integrator order</strong></td>
                  <td>${p.title}</td>
                  <td>Euler: ${p.euler.orders[p.euler.orders.length - 1].toFixed(2)} · RK4: ${p.rk4.orders[p.rk4.orders.length - 1].toFixed(2)}</td>
                  <td>Euler 1, RK4 4</td>
                </tr>`
              )
              .join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  container.innerHTML = html;

  const histGrid = container.querySelector('#batch-hist-grid');
  if (histGrid && payload.histograms) {
    payload.histograms.forEach((h) => {
      const box = document.createElement('div');
      box.className = 'hist-box';
      box.innerHTML = `
        <h4><span>${h.title}</span><span class="mono dim">median ${sig3(h.median)} ${h.unit}</span></h4>
        <canvas></canvas>
      `;
      histGrid.appendChild(box);
      requestAnimationFrame(() => drawHistogramCanvas(box.querySelector('canvas'), h));
    });
  }

  container.querySelectorAll('.btn-replay-seed').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (onSelectSeed) onSelectSeed(+btn.dataset.seed);
    });
  });
}

const TeX = String.raw;

const escAttr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

// Inline maths with a readable plain-text fallback for offline use
const m = (tex, fallback) => `<span class="tex" data-tex="${escAttr(tex)}">${fallback}</span>`;

const eq = (label, tex, fallback, note = '') => `
  <div class="eq-label">${label}</div>
  <div class="eq"><span class="tex" data-tex="${escAttr('\\displaystyle ' + tex)}"><span class="tex-fallback">${fallback}</span></span>${note ? `<span class="note">${note}</span>` : ''}</div>`;

function typeset(container) {
  if (!window.katex) return false;
  container.querySelectorAll('[data-tex]').forEach((el) => {
    try {
      window.katex.render(el.dataset.tex, el, { throwOnError: false, output: 'html' });
    } catch { /* keep the fallback text */ }
  });
  return true;
}

// Round long floats (89.97495996864073 -> 89.975) without touching integers or text
function fmtParam(v) {
  if (typeof v === 'string' && v.trim() !== '' && !Number.isNaN(Number(v))) v = Number(v);
  if (typeof v === 'number' && Number.isFinite(v) && !Number.isInteger(v)) return String(+v.toPrecision(6));
  return v == null ? '—' : String(v);
}

export function renderMethodsView(container, data) {
  if (!data) return;
  const prm = data.params || {};
  const kfStats = data.derived?.kfStats;
  const vT = TeX`\mathbf{v}_T`;

  container.innerHTML = `
    <div class="methods-grid">
      <div class="method-card">
        <h3>Equations of motion</h3>
        <p>Both vehicles are point masses integrated in East-North-Up coordinates with fourth-order Runge-Kutta, splitting the step exactly at burnout and interpolating ground and closest-approach crossings.</p>
        <div class="eq-list">
          ${eq('Threat', TeX`\ddot{\mathbf{r}}_T = -k_T\,\lVert ${vT} \rVert\, ${vT} - g\,\hat{\mathbf{z}}`, 'r̈_T = −k_T ‖v_T‖ v_T − g ẑ')}
          ${eq('Interceptor', TeX`\ddot{\mathbf{r}}_I = \frac{T\,\hat{\mathbf{u}} - k_I\,\lVert \mathbf{v}_I \rVert\,\mathbf{v}_I}{m(t)} + \mathbf{a}_c - g\,\hat{\mathbf{z}}`, 'r̈_I = (T û − k_I ‖v_I‖ v_I) / m(t) + a_c − g ẑ')}
          ${eq('Mass', TeX`m(t) = \begin{cases} m_\text{wet} - \dot{m}\,(t - t_\text{launch}) & t < t_\text{burn} \\ m_\text{KV} & \text{otherwise} \end{cases}`, 'm(t) = m_wet − ṁ (t − t_launch) for t &lt; t_burn, else m_KV')}
        </div>
      </div>

      <div class="method-card">
        <h3>State estimation: KF6 and EKF7</h3>
        <p>The baseline <code>kf6</code> tracks position and velocity assuming vacuum free fall. The <code>ekf7</code> adds the threat's ballistic drag coefficient ${m(TeX`k_T`, 'k_T')} to the state and linearises the drag term each step.</p>
        <div class="eq-list">
          ${eq('State', TeX`\mathbf{x} = \begin{bmatrix} \mathbf{r}_T & ${vT} & k_T \end{bmatrix}^\top`, 'x = [r_T, v_T, k_T]ᵀ')}
          ${eq('Dynamics', TeX`\dot{\mathbf{x}} = \begin{bmatrix} ${vT} & -k_T \lVert ${vT} \rVert ${vT} - g\hat{\mathbf{z}} & 0 \end{bmatrix}^\top`, 'ẋ = [v_T, −k_T ‖v_T‖ v_T − g ẑ, 0]ᵀ')}
          ${eq('Jacobian', TeX`\frac{\partial \mathbf{a}_d}{\partial \mathbf{v}} = -k_T\!\left(\lVert\mathbf{v}\rVert I_3 + \frac{\mathbf{v}\mathbf{v}^\top}{\lVert\mathbf{v}\rVert}\right), \quad \frac{\partial \mathbf{a}_d}{\partial k_T} = -\lVert\mathbf{v}\rVert\,\mathbf{v}`, '∂a_d/∂v = −k_T (‖v‖ I₃ + v vᵀ/‖v‖),  ∂a_d/∂k_T = −‖v‖ v')}
        </div>
      </div>

      <div class="method-card">
        <h3>Radar model and impact prediction</h3>
        <p>In <code>rae</code> mode the radar measures range, azimuth, elevation and Doppler range rate. Their noise is rotated into ENU through the beam basis ${m(TeX`(\mathbf{u}_r, \mathbf{u}_{az}, \mathbf{u}_{el})`, '(u_r, u_az, u_el)')}. The predicted impact point (PIP) decides whether to launch.</p>
        <div class="eq-list">
          ${eq('Noise', TeX`R_p = \sigma_r^2\,\mathbf{u}_r\mathbf{u}_r^\top + (r\cos\epsilon\,\sigma_\theta)^2\,\mathbf{u}_{az}\mathbf{u}_{az}^\top + (r\,\sigma_\theta)^2\,\mathbf{u}_{el}\mathbf{u}_{el}^\top`, 'R_p = σ_r² u_r u_rᵀ + (r cos ε σ_θ)² u_az u_azᵀ + (r σ_θ)² u_el u_elᵀ')}
          ${eq('Vacuum PIP', TeX`t_\text{imp} = \frac{v_z + \sqrt{v_z^2 + 2gz}}{g}`, 't_imp = (v_z + √(v_z² + 2gz)) / g')}
          ${eq('Drag PIP', TeX`[\hat{\mathbf{r}}_T,\ \hat{\mathbf{v}}_T] \xrightarrow{\ \text{RK4},\ \hat{k}_T\ } z = 0`, 'RK4 of [r̂_T, v̂_T] with k̂_T until z = 0')}
        </div>
      </div>

      <div class="method-card">
        <h3>Guidance laws</h3>
        <p>During the kill vehicle's coast, uniform gravity cancels in the relative motion. APN and ZEM feed forward the estimated target drag deceleration ${m(TeX`\hat{\mathbf{a}}_T = -\hat{k}_T \lVert \hat{\mathbf{v}}_T \rVert\, \hat{\mathbf{v}}_T`, 'â_T = −k̂_T ‖v̂_T‖ v̂_T')}.</p>
        <div class="eq-list">
          ${eq('PN', TeX`\mathbf{a}_c = N\,(\mathbf{v}_\text{rel} \times \boldsymbol{\omega}) + g\,\hat{\mathbf{z}}`, 'a_c = N (v_rel × ω) + g ẑ')}
          ${eq('APN', TeX`\mathbf{a}_c = N\,(\mathbf{v}_\text{rel} \times \boldsymbol{\omega}) + \tfrac{1}{2} N\,\hat{\mathbf{a}}_{T,\perp}`, 'a_c = N (v_rel × ω) + ½ N â_T,⊥', '+ g ẑ during boost')}
          ${eq('ZEM', TeX`\mathbf{a}_c = \frac{N}{t_\text{go}^2}\,\mathrm{ZEM}_\perp, \quad \mathrm{ZEM} = \mathbf{r} + \mathbf{v}_\text{rel}\,t_\text{go} + \tfrac{1}{2}\hat{\mathbf{a}}_T\,t_\text{go}^2`, 'a_c = (N / t_go²) ZEM_⊥,  ZEM = r + v_rel t_go + ½ â_T t_go²')}
        </div>
      </div>
    </div>

    <div class="batch-section">
      <h3>Run parameters · seed ${data.seed ?? '—'} · ${data.config ?? ''}</h3>
      <p class="sub">
        Filter <strong>${(data.filterType || 'kf6').toUpperCase()}</strong> ·
        PIP <strong>${(data.pipMode || 'vacuum').toUpperCase()}</strong> ·
        Radar <strong>${(data.sensorModel || 'cartesian').toUpperCase()}</strong> ·
        Guidance <strong>${(data.guidanceLaw || 'pn').toUpperCase()}</strong>
        ${kfStats ? ` · post-transient RMS position error <strong>${kfStats.rmsPos} m</strong>, 3σ coverage ${kfStats.inside3SigmaPct}%` : ''}
      </p>
      <table class="data-table param-table">
        <thead>
          <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
        </thead>
        <tbody>
          ${(() => {
            const entries = Object.entries(prm);
            const half = Math.ceil(entries.length / 2);
            const rows = [];
            for (let i = 0; i < half; i++) {
              const [k1, v1] = entries[i];
              const [k2, v2] = entries[i + half] || ['', ''];
              rows.push(`<tr><td>${k1}</td><td>${fmtParam(v1)}</td><td>${k2}</td><td>${k2 ? fmtParam(v2) : ''}</td></tr>`);
            }
            return rows.join('');
          })()}
        </tbody>
      </table>
    </div>
  `;

  if (!typeset(container)) {
    window.addEventListener('load', () => typeset(container), { once: true });
  }
}
