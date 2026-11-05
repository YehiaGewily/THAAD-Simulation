// Replay data: the payload written by thaad_sim/replay_viewer.py, converted to
// scene units, plus interpolation helpers and the key event times.
import * as THREE from 'three';

export const D = JSON.parse(document.getElementById('replay-data').textContent);
export const KM = 0.001;
export const G_KM = 9.81 * KM;
export const N = D.t.length;
export const T = Float64Array.from(D.t);

export const MODE_NAMES = { 'ENGINEERING MODE': 'Engineering', 'MAX-PERFORMANCE MODE': 'Max performance' };
export const RULE_NAMES = { 'PIP-GATED LAUNCH': 'PIP-gated launch', 'FIRE ON TRACK': 'Fires on track' };

// Sim frame is x = East, y = North, z = Up (metres). Three.js is y-up, so we
// map (x, y, z) -> (x, z, -y) and work in kilometres.
function toThree(flat) {
  const out = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 3) {
    out[i] = flat[i] * KM;
    out[i + 1] = flat[i + 2] * KM;
    out[i + 2] = -flat[i + 1] * KM;
  }
  return out;
}
export const vec = (p) => new THREE.Vector3(p[0] * KM, p[2] * KM, -p[1] * KM);

export const P = {
  threat: toThree(D.threat),
  kv: toThree(D.interceptor),
  track: toThree(D.track),
  radar: toThree(D.radar),
};

export function locate(time) {
  if (time <= T[0]) return [0, 0];
  if (time >= T[N - 1]) return [N - 2, 1];
  let lo = 0, hi = N - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (T[m] <= time) lo = m; else hi = m;
  }
  return [lo, (time - T[lo]) / Math.max(T[lo + 1] - T[lo], 1e-9)];
}
export function sample(arr, time, out = new THREE.Vector3()) {
  const [i, f] = locate(time);
  const a = 3 * i, b = 3 * (i + 1);
  return out.set(
    arr[a] + (arr[b] - arr[a]) * f,
    arr[a + 1] + (arr[b + 1] - arr[a + 1]) * f,
    arr[a + 2] + (arr[b + 2] - arr[a + 2]) * f,
  );
}
export function velocity(arr, time, out = new THREE.Vector3()) {
  const h = 0.05;
  const tc = Math.min(Math.max(time, T[0]), T[N - 1]);
  const t0 = Math.max(T[0], tc - h), t1 = Math.min(T[N - 1], tc + h);
  if (t1 - t0 < 1e-6) return out.set(0, 0, 0);
  const a = sample(arr, t0, new THREE.Vector3());
  return sample(arr, t1, out).sub(a).divideScalar(t1 - t0);
}

// Key events
export const launchIdx = D.status.findIndex((s) => s > 0);
export const launchT = launchIdx >= 0 ? T[launchIdx] : null;
export const burnIdx = D.stage.findIndex((s) => s === 2);
export const burnT = burnIdx >= 0 ? T[burnIdx] : null;
export const killT = D.kill ? D.kill.time : null;
export const impactT = (D.outcome === 'GROUND' || D.outcome === 'NO ENGAGE') ? T[N - 1] : null;
export const endEventT = killT ?? impactT;
export const tStart = T[0];
export const tEnd = T[N - 1] + (endEventT != null ? 7 : 1);

export let terminalT = null;
if (launchIdx >= 0) {
  for (let i = Math.max(launchIdx, burnIdx); i < N && i >= 0; i++) {
    const dx = P.threat[3 * i] - P.kv[3 * i], dy = P.threat[3 * i + 1] - P.kv[3 * i + 1], dz = P.threat[3 * i + 2] - P.kv[3 * i + 2];
    if (Math.hypot(dx, dy, dz) < 15) { terminalT = T[i]; break; }
  }
}

export const killPos = D.kill ? vec(D.kill.pos) : null;
export const impactPos = impactT != null ? sample(P.threat, impactT).setY(0) : null;
export const pipPos = D.launchPip ? vec(D.launchPip).setY(0) : null;
export const threatStart = sample(P.threat, tStart);
export const defR = D.defendedRadius * KM;

const PRM = D.params || {};
export const simG = PRM.g ?? 9.81;
export const simDt = PRM.dt ?? 0.02;
export const threatDrag = PRM.threat_drag ?? 0.00004;
export const pnGain = PRM.pn_gain ?? 4.0;
export const maxG = D.maxG ?? PRM.max_g ?? 30.0;
export const divertBudget = PRM.kv_divert_dv ?? null;
export const massWet = PRM.mass_wet ?? 900.0;
export const massDry = PRM.mass_dry ?? 450.0;
export const kvMass = PRM.kv_mass ?? 200.0;
export const massFlow = PRM.mass_flow ?? ((massWet - massDry) / Math.max(PRM.burn_time ?? 5.0, 1e-6));

// Closest Point of Approach (CPA): exact sub-step chord minimum over all launched frames.
// Present for both KILL and failed intercepts (GROUND / TIMEOUT).
export const cpaInfo = (() => {
  if (D.cpa) {
    return {
      time: D.cpa.time,
      alt: D.cpa.alt,
      dist: D.cpa.dist,
      closing: D.cpa.closing,
      tPos: vec(D.cpa.tPos),
      iPos: vec(D.cpa.iPos),
    };
  }
  if (D.kill) {
    const p = vec(D.kill.pos);
    return { time: D.kill.time, alt: D.kill.alt, dist: D.kill.dist, closing: D.kill.closing, tPos: p.clone(), iPos: sample(P.kv, D.kill.time) };
  }
  if (launchIdx < 0) return null;
  let bestD = Infinity, bestT = T[launchIdx], bestTp = new THREE.Vector3(), bestIp = new THREE.Vector3(), bestCls = 0;
  for (let i = Math.max(0, launchIdx - 1); i < N - 1; i++) {
    const dt = T[i + 1] - T[i];
    if (dt <= 0) continue;
    const a = 3 * i, b = 3 * (i + 1);
    const tx0 = D.threat[a], ty0 = D.threat[a + 1], tz0 = D.threat[a + 2];
    const ix0 = D.interceptor[a], iy0 = D.interceptor[a + 1], iz0 = D.interceptor[a + 2];
    const tvx = (D.threat[b] - tx0) / dt, tvy = (D.threat[b + 1] - ty0) / dt, tvz = (D.threat[b + 2] - tz0) / dt;
    const ivx = (D.interceptor[b] - ix0) / dt, ivy = (D.interceptor[b + 1] - iy0) / dt, ivz = (D.interceptor[b + 2] - iz0) / dt;
    const dpx = tx0 - ix0, dpy = ty0 - iy0, dpz = tz0 - iz0;
    const dvx = tvx - ivx, dvy = tvy - ivy, dvz = tvz - ivz;
    const dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
    let tau = dv2 > 1e-6 ? -(dpx * dvx + dpy * dvy + dpz * dvz) / dv2 : 0;
    tau = Math.max(0, Math.min(dt, tau));
    const dx = dpx + dvx * tau, dy = dpy + dvy * tau, dz = dpz + dvz * tau;
    const d = Math.hypot(dx, dy, dz);
    if (d < bestD) {
      bestD = d;
      bestT = T[i] + tau;
      bestTp.set((tx0 + tvx * tau) * KM, (tz0 + tvz * tau) * KM, -(ty0 + tvy * tau) * KM);
      bestIp.set((ix0 + ivx * tau) * KM, (iz0 + ivz * tau) * KM, -(iy0 + ivy * tau) * KM);
      bestCls = Math.sqrt(dv2);
    }
  }
  return Number.isFinite(bestD) ? { time: bestT, alt: bestTp.y / KM, dist: bestD, closing: bestCls, tPos: bestTp, iPos: bestIp } : null;
})();

export const cpaT = cpaInfo ? cpaInfo.time : null;
export const missCpaT = (!killT && cpaInfo) ? cpaInfo.time : null;
export const cpaMidPos = cpaInfo ? cpaInfo.tPos.clone().add(cpaInfo.iPos).multiplyScalar(0.5) : null;

// True unintercepted ground impact point (sim metres [x, y, 0] and scene vec).
// If the threat was intercepted in mid-air, integrate its final state to z = 0 with RK4 under drag.
export const trueImpactSim = (() => {
  const last = 3 * (N - 1);
  let x = D.threat[last], y = D.threat[last + 1], z = D.threat[last + 2];
  let vx = D.threatVel[last], vy = D.threatVel[last + 1], vz = D.threatVel[last + 2];
  if (z <= 0) return [x, y, 0];
  const h = 0.05, k = threatDrag, g = simG;
  const deriv = (s) => {
    const sp = Math.hypot(s[3], s[4], s[5]);
    return [s[3], s[4], s[5], -k * sp * s[3], -k * sp * s[4], -k * sp * s[5] - g];
  };
  let st = [x, y, z, vx, vy, vz];
  for (let step = 0; step < 4000 && st[2] > 0; step++) {
    const prev = st.slice();
    const k1 = deriv(st);
    const k2 = deriv(st.map((v, j) => v + 0.5 * h * k1[j]));
    const k3 = deriv(st.map((v, j) => v + 0.5 * h * k2[j]));
    const k4 = deriv(st.map((v, j) => v + h * k3[j]));
    st = st.map((v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    if (st[2] <= 0) {
      const f = prev[2] / Math.max(prev[2] - st[2], 1e-9);
      return [prev[0] + f * (st[0] - prev[0]), prev[1] + f * (st[1] - prev[1]), 0];
    }
  }
  return [st[0], st[1], 0];
})();
export const trueImpactPos = vec(trueImpactSim);

// Precompute instantaneous vacuum PIP (from Kalman estimate), 3-sigma ground footprint,
// divert delta-v, interceptor mass, and perceived vs. true guidance series for every frame.
export const GNC = (() => {
  const pipTrack = new Float32Array(3 * N);      // scene coords (x, 0.02, z) of instantaneous PIP
  const pipValid = new Uint8Array(N);
  const pipSigE = new Float32Array(N);           // 3-sigma ground footprint semi-axis East (km)
  const pipSigN = new Float32Array(N);           // 3-sigma ground footprint semi-axis North (km)
  const pipBias = new Float32Array(N).fill(NaN); // |PIP(t) - trueImpact| in metres
  const dvUsed = new Float32Array(N);            // cumulative KV divert delta-v used (m/s)
  const mass = new Float32Array(N);              // interceptor mass (kg)
  const acScene = new Float32Array(3 * N);       // 3D commanded acceleration in scene frame (m/s^2)
  const zemScene = new Float32Array(3 * N);      // 3D true ZEM vector in scene frame (km)
  const zemTrue = new Float32Array(N).fill(NaN); // true ZEM magnitude (m)
  const zemEst = new Float32Array(N).fill(NaN);  // filter-perceived ZEM magnitude (m)
  const losTrue = new Float32Array(N).fill(NaN); // true LOS rate (mrad/s)
  const losEst = new Float32Array(N).fill(NaN);  // filter-perceived LOS rate (mrad/s)
  const nees = new Float32Array(N).fill(NaN);    // 6-DOF NEES
  const nis = new Float32Array(N).fill(NaN);     // 6-DOF NIS

  const hasPayloadNees = Array.isArray(D.nees) && D.nees.length === N;
  const hasPayloadNis = Array.isArray(D.nis) && D.nis.length === N;

  let dv = 0;
  for (let i = 0; i < N; i++) {
    const a = 3 * i, k = 6 * i;
    const tx = D.threat[a], ty = D.threat[a + 1], tz = D.threat[a + 2];
    const tvx = D.threatVel[a], tvy = D.threatVel[a + 1], tvz = D.threatVel[a + 2];
    const ex = D.kfErr[k], ey = D.kfErr[k + 1], ez = D.kfErr[k + 2];
    const evx = D.kfErr[k + 3], evy = D.kfErr[k + 4], evz = D.kfErr[k + 5];
    const sx = D.kfSig[k], sy = D.kfSig[k + 1], sz = D.kfSig[k + 2];
    const svx = D.kfSig[k + 3], svy = D.kfSig[k + 4], svz = D.kfSig[k + 5];

    // Filter state estimate in sim frame
    const epx = tx + ex, epy = ty + ey, epz = tz + ez;
    const epvx = tvx + evx, epvy = tvy + evy, epvz = tvz + evz;

    // NEES & NIS (from payload or diagonal fallback for legacy replays)
    if (hasPayloadNees && Number.isFinite(D.nees[i])) nees[i] = D.nees[i];
    else {
      nees[i] = (ex / Math.max(sx, 1e-6)) ** 2 + (ey / Math.max(sy, 1e-6)) ** 2 + (ez / Math.max(sz, 1e-6)) ** 2
              + (evx / Math.max(svx, 1e-6)) ** 2 + (evy / Math.max(svy, 1e-6)) ** 2 + (evz / Math.max(svz, 1e-6)) ** 2;
    }
    if (hasPayloadNis && D.nis[i] != null && Number.isFinite(D.nis[i])) nis[i] = D.nis[i];

    // Instantaneous vacuum PIP from Kalman estimate
    if (epvz < 0 && epz > 0) {
      const disc = epvz * epvz + 2 * simG * epz;
      const root = Math.sqrt(Math.max(0, disc));
      const tImp = (epvz + root) / simG;
      if (tImp > 0) {
        const px = epx + epvx * tImp;
        const py = epy + epvy * tImp;
        pipTrack[a] = px * KM;
        pipTrack[a + 1] = 0.025;
        pipTrack[a + 2] = -py * KM;
        pipValid[i] = 1;
        pipBias[i] = Math.hypot(px - trueImpactSim[0], py - trueImpactSim[1]);
        // Linearized 3-sigma ground footprint from state covariance
        const dtdz = 1 / Math.max(root, 1);
        const dtdvz = tImp / Math.max(root, 1);
        const sigX = Math.hypot(sx, tImp * svx, epvx * dtdz * sz, epvx * dtdvz * svz);
        const sigY = Math.hypot(sy, tImp * svy, epvy * dtdz * sz, epvy * dtdvz * svz);
        pipSigE[i] = 3 * sigX * KM;
        pipSigN[i] = 3 * sigY * KM;
      }
    }
    if (!pipValid[i] && i > 0) {
      pipTrack[a] = pipTrack[a - 3];
      pipTrack[a + 1] = pipTrack[a - 2];
      pipTrack[a + 2] = pipTrack[a - 1];
      pipSigE[i] = pipSigE[i - 1];
      pipSigN[i] = pipSigN[i - 1];
    }

    // Interceptor mass & guidance quantities
    const stg = D.stage[i];
    if (D.status[i] === 0 || launchT == null) {
      mass[i] = massWet;
    } else if (stg === 1) {
      mass[i] = Math.max(massDry, massWet - massFlow * Math.max(0, T[i] - launchT));
    } else {
      mass[i] = kvMass;
    }

    if (D.status[i] > 0) {
      const dt = i > 0 ? T[i] - T[i - 1] : simDt;
      if (stg === 2) dv += (D.g[i] || 0) * 9.81 * dt;
      dvUsed[i] = dv;

      const ix = D.interceptor[a], iy = D.interceptor[a + 1], iz = D.interceptor[a + 2];
      const ivx = D.interceptorVel[a], ivy = D.interceptorVel[a + 1], ivz = D.interceptorVel[a + 2];

      // True relative state
      const rx = tx - ix, ry = ty - iy, rz = tz - iz;
      const vx = tvx - ivx, vy = tvy - ivy, vz = tvz - ivz;
      const r = Math.max(Math.hypot(rx, ry, rz), 1e-6);
      const rv = rx * vx + ry * vy + rz * vz, vv = vx * vx + vy * vy + vz * vz;
      const wx = (ry * vz - rz * vy) / (r * r), wy = (rz * vx - rx * vz) / (r * r), wz = (rx * vy - ry * vx) / (r * r);
      losTrue[i] = 1000 * Math.hypot(wx, wy, wz);
      const tgo = vv > 1e-9 ? Math.max(0, -rv / vv) : 0;
      const zx = rx + vx * tgo, zy = ry + vy * tgo, zz = rz + vz * tgo;
      zemTrue[i] = Math.hypot(zx, zy, zz);
      zemScene[a] = zx * KM; zemScene[a + 1] = zz * KM; zemScene[a + 2] = -zy * KM;

      // Filter-perceived relative state (what PN guidance actually sees)
      const erx = epx - ix, ery = epy - iy, erz = epz - iz;
      const ervx = epvx - ivx, ervy = epvy - ivy, ervz = epvz - ivz;
      const er2 = erx * erx + ery * ery + erz * erz;
      const er = Math.max(Math.sqrt(er2), 1e-6);
      const erv = erx * ervx + ery * ervy + erz * ervz;
      const evv = ervx * ervx + ervy * ervy + ervz * ervz;
      const ewx = (ery * ervz - erz * ervy) / (er2 + 1e-6);
      const ewy = (erz * ervx - erx * ervz) / (er2 + 1e-6);
      const ewz = (erx * ervy - ery * ervx) / (er2 + 1e-6);
      losEst[i] = 1000 * Math.hypot(ewx, ewy, ewz);
      const etgo = evv > 1e-9 ? Math.max(0, -erv / evv) : 0;
      zemEst[i] = Math.hypot(erx + ervx * etgo, ery + ervy * etgo, erz + ervz * etgo);

      // Commanded acceleration vector a_c = N (v_rel x omega) + g z_hat
      if (er >= 20.0) {
        let acx = pnGain * (ervy * ewz - ervz * ewy);
        let acy = pnGain * (ervz * ewx - ervx * ewz);
        let acz = pnGain * (ervx * ewy - ervy * ewx) + 9.81;
        const mag = Math.hypot(acx, acy, acz);
        const actualMag = (D.g[i] || 0) * 9.81;
        const scale = mag > 1e-6 ? actualMag / mag : 0;
        acx *= scale; acy *= scale; acz *= scale;
        acScene[a] = acx; acScene[a + 1] = acz; acScene[a + 2] = -acy;
      }
    }
  }
  return { pipTrack, pipValid, pipSigE, pipSigN, pipBias, dvUsed, mass, acScene, zemScene, zemTrue, zemEst, losTrue, losEst, nees, nis };
})();

/** Compute a 3D ballistic trajectory arc (scene coords) from frame fi to ground (z = 0). */
export function computeBallisticArc(fi, useDrag, steps = 48) {
  const a = 3 * fi, k = 6 * fi;
  // Drag-free arc starts from the Kalman estimate (showing the filter's vacuum PIP prediction);
  // dragged arc starts from the true threat state.
  let x = D.threat[a] + (useDrag ? 0 : D.kfErr[k]);
  let y = D.threat[a + 1] + (useDrag ? 0 : D.kfErr[k + 1]);
  let z = D.threat[a + 2] + (useDrag ? 0 : D.kfErr[k + 2]);
  let vx = D.threatVel[a] + (useDrag ? 0 : D.kfErr[k + 3]);
  let vy = D.threatVel[a + 1] + (useDrag ? 0 : D.kfErr[k + 4]);
  let vz = D.threatVel[a + 2] + (useDrag ? 0 : D.kfErr[k + 5]);
  if (z <= 0 || vz >= 0) return null;
  const pts = [new THREE.Vector3(x * KM, z * KM, -y * KM)];
  if (!useDrag) {
    const tImp = (vz + Math.sqrt(Math.max(0, vz * vz + 2 * simG * z))) / simG;
    if (tImp <= 0) return null;
    for (let s = 1; s <= steps; s++) {
      const t = (s / steps) * tImp;
      pts.push(new THREE.Vector3((x + vx * t) * KM, Math.max(0, z + vz * t - 0.5 * simG * t * t) * KM, -(y + vy * t) * KM));
    }
    return pts;
  }
  // RK4 integration with quadratic drag
  const h = 0.25, cd = threatDrag;
  const deriv = (st) => {
    const sp = Math.hypot(st[3], st[4], st[5]);
    return [st[3], st[4], st[5], -cd * sp * st[3], -cd * sp * st[4], -cd * sp * st[5] - simG];
  };
  let st = [x, y, z, vx, vy, vz];
  const raw = [[x, y, z]];
  for (let s = 0; s < 800 && st[2] > 0; s++) {
    const prev = st.slice();
    const k1 = deriv(st);
    const k2 = deriv(st.map((v, j) => v + 0.5 * h * k1[j]));
    const k3 = deriv(st.map((v, j) => v + 0.5 * h * k2[j]));
    const k4 = deriv(st.map((v, j) => v + h * k3[j]));
    st = st.map((v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    if (st[2] <= 0) {
      const f = prev[2] / Math.max(prev[2] - st[2], 1e-9);
      raw.push([prev[0] + f * (st[0] - prev[0]), prev[1] + f * (st[1] - prev[1]), 0]);
      break;
    }
    raw.push([st[0], st[1], st[2]]);
  }
  const n = raw.length - 1;
  if (n < 1) return null;
  for (let s = 1; s <= steps; s++) {
    const u = (s / steps) * n, i0 = Math.min(Math.floor(u), n - 1), fr = u - i0;
    const p0 = raw[i0], p1 = raw[i0 + 1];
    pts.push(new THREE.Vector3(
      (p0[0] + (p1[0] - p0[0]) * fr) * KM,
      Math.max(0, p0[2] + (p1[2] - p0[2]) * fr) * KM,
      -(p0[1] + (p1[1] - p0[1]) * fr) * KM,
    ));
  }
  return pts;
}
