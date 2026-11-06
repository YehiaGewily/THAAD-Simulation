// In-browser GNC engagement simulator: runs a complete seeded engagement
// (RK4 dynamics, 6-state KF or 7-state drag-estimating EKF, Cartesian or
// Polar RAE radar, Vacuum or Drag-integrated PIP, PN / APN / ZEM guidance,
// and actuator lag) in ~3 ms client-side and returns a replay payload.

const G0 = 9.81;
const STD_G0 = 9.80665;

function makeRng(seed) {
  let s = (Number(seed) || 0) >>> 0;
  function nextU32() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }
  function uniform(lo = 0, hi = 1) {
    return lo + ((nextU32() + 0.5) / 4294967296) * (hi - lo);
  }
  let spare = null;
  function normal(mean = 0, std = 1) {
    if (spare !== null) {
      const z = spare;
      spare = null;
      return mean + std * z;
    }
    let u = 0, v = 0;
    while (u <= 1e-12) u = uniform(0, 1);
    v = uniform(0, 1);
    const mag = Math.sqrt(-2.0 * Math.log(u));
    spare = mag * Math.sin(2.0 * Math.PI * v);
    return mean + std * (mag * Math.cos(2.0 * Math.PI * v));
  }
  return { uniform, normal };
}

function rk4Step(state, deriv, dt) {
  const n = state.length;
  const k1 = deriv(state);
  const s2 = new Float64Array(n);
  for (let i = 0; i < n; i++) s2[i] = state[i] + 0.5 * dt * k1[i];
  const k2 = deriv(s2);
  const s3 = new Float64Array(n);
  for (let i = 0; i < n; i++) s3[i] = state[i] + 0.5 * dt * k2[i];
  const k3 = deriv(s3);
  const s4 = new Float64Array(n);
  for (let i = 0; i < n; i++) s4[i] = state[i] + dt * k3[i];
  const k4 = deriv(s4);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return out;
}

// Invert a symmetric positive-definite 6x6 matrix via Gauss-Jordan
function solveSym6(A, B) {
  // Solves A * X = B where A is 6x6 (flattened 36) and B is 6xM (flattened 6*M)
  const M = B.length / 6;
  const aug = new Float64Array(6 * (6 + M));
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) aug[r * (6 + M) + c] = A[r * 6 + c];
    for (let c = 0; c < M; c++) aug[r * (6 + M) + 6 + c] = B[r * M + c];
  }
  const W = 6 + M;
  for (let i = 0; i < 6; i++) {
    let piv = aug[i * W + i];
    if (Math.abs(piv) < 1e-18) piv = 1e-18;
    const invP = 1.0 / piv;
    for (let c = i; c < W; c++) aug[i * W + c] *= invP;
    for (let r = 0; r < 6; r++) {
      if (r === i) continue;
      const f = aug[r * W + i];
      if (f === 0) continue;
      for (let c = i; c < W; c++) aug[r * W + c] -= f * aug[i * W + c];
    }
  }
  const X = new Float64Array(6 * M);
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < M; c++) X[r * M + c] = aug[r * W + 6 + c];
  }
  return X;
}

function matMul(A, B, R, K, C) {
  const out = new Float64Array(R * C);
  for (let r = 0; r < R; r++) {
    for (let k = 0; k < K; k++) {
      const a = A[r * K + k];
      if (a === 0) continue;
      for (let c = 0; c < C; c++) out[r * C + c] += a * B[k * C + c];
    }
  }
  return out;
}

function matTrans(A, R, C) {
  const out = new Float64Array(C * R);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) out[c * R + r] = A[r * C + c];
  }
  return out;
}

function predictImpactDrag(px, py, pz, vx, vy, vz, drag, g) {
  if (vz >= 0 || pz <= 0) return null;
  const kd = Math.max(0, drag);
  if (kd < 1e-11) {
    const d = vz * vz + 2 * g * pz;
    if (d < 0) return null;
    const t = (vz + Math.sqrt(d)) / g;
    return t > 0 ? [px + vx * t, py + vy * t, 0] : null;
  }
  let st = new Float64Array([px, py, pz, vx, vy, vz]);
  const h = 0.25;
  const deriv = (s) => {
    const sp = Math.hypot(s[3], s[4], s[5]);
    return [s[3], s[4], s[5], -kd * sp * s[3], -kd * sp * s[4], -kd * sp * s[5] - g];
  };
  for (let k = 0; k < 1200; k++) {
    const prev = st;
    st = rk4Step(prev, deriv, h);
    if (st[2] <= 0) {
      const f = prev[2] / Math.max(prev[2] - st[2], 1e-12);
      return [prev[0] + f * (st[0] - prev[0]), prev[1] + f * (st[1] - prev[1]), 0];
    }
    if (st[5] >= 0) return null;
  }
  return null;
}

export function runBrowserSim(opts = {}) {
  const seed = Number.isFinite(+opts.seed) ? Math.max(0, Math.floor(+opts.seed)) : 13;
  const filterType = opts.filterType || 'ekf7';   // 'kf6' | 'ekf7'
  const pipMode = opts.pipMode || (filterType === 'ekf7' ? 'drag' : 'vacuum');
  const sensorModel = opts.sensorModel || 'rae';  // 'cartesian' | 'rae'
  const guidanceLaw = opts.guidanceLaw || 'apn';  // 'pn' | 'apn' | 'zem'
  const divertBudget = opts.divertBudget === null || opts.divertBudget === 'unlimited'
    ? null : (Number.isFinite(+opts.divertBudget) ? +opts.divertBudget : 435.0);
  const tau = Number.isFinite(+opts.tau) ? Math.max(0, +opts.tau) : 0.0;
  const maxG = Number.isFinite(+opts.maxG) ? +opts.maxG : 30.0;
  const pnGain = Number.isFinite(+opts.pnGain) ? +opts.pnGain : 4.0;
  const posNoise = Number.isFinite(+opts.posNoise) ? +opts.posNoise : 25.0;
  const velNoise = 8.0;
  const rangeNoise = 10.0;
  const angleNoise = 0.0012;
  const rangeRateNoise = 3.0;

  const dt = 0.02, g = 9.81;
  const threatDrag = 0.00004;
  const boostThrust = 150000.0, isp = 170.0, massWet = 900.0, massDry = 450.0, kvMass = 200.0;
  const kvDrag = 0.00008, boosterDrag = 0.0001;
  const massFlow = boostThrust / (isp * STD_G0);
  const burnTime = (massWet - massDry) / massFlow;
  const defendedRadius = 30000.0, killRadius = 10.0, minTrackTime = 5.0;
  const qSpec = filterType === 'ekf7' ? 5.0 : 30.0;

  const rng = makeRng(seed);

  // Spawn threat scenario
  const tImp = rng.uniform(70.0, 100.0);
  const z0 = rng.uniform(55000.0, 75000.0);
  const r0 = rng.uniform(35000.0, 55000.0);
  const az = rng.uniform(0, 2 * Math.PI);
  const x0 = r0 * Math.cos(az), y0 = r0 * Math.sin(az);
  const aimR = rng.uniform(0.0, 35000.0);
  const aimAz = rng.uniform(0, 2 * Math.PI);
  const aimX = aimR * Math.cos(aimAz), aimY = aimR * Math.sin(aimAz);
  const dragComp = 1.03;
  const vx0 = ((aimX - x0) / tImp) * dragComp;
  const vy0 = ((aimY - y0) / tImp) * dragComp;
  const vz0 = ((0.5 * g * tImp * tImp - z0) / tImp) * dragComp;

  let tState = new Float64Array([x0, y0, z0, vx0, vy0, vz0]);
  let iState = new Float64Array([0, 0, 0, 0, 0, 0, massWet]);
  let actAcc = [0, 0, 0];
  let launched = false, launchTime = null, launchPip = null, stage = 'IDLE';
  let divertUsed = 0.0;

  // Filter state (D = 6 or 7)
  const D_DIM = filterType === 'ekf7' ? 7 : 6;
  const xHat = new Float64Array(D_DIM);
  xHat[6 - (D_DIM === 6 ? 1 : 0)] = 0;
  if (D_DIM === 7) xHat[6] = 0.000035;
  let P = new Float64Array(D_DIM * D_DIM);
  let kfInit = false;

  const Q = new Float64Array(D_DIM * D_DIM);
  for (let a = 0; a < 3; a++) {
    Q[a * D_DIM + a] = qSpec * (dt ** 3) / 3;
    Q[a * D_DIM + (a + 3)] = qSpec * (dt ** 2) / 2;
    Q[(a + 3) * D_DIM + a] = qSpec * (dt ** 2) / 2;
    Q[(a + 3) * D_DIM + (a + 3)] = qSpec * dt;
  }
  if (D_DIM === 7) Q[48] = 1e-12 * dt;

  const tArr = [], threatArr = [], intArr = [], trackArr = [], radarArr = [];
  const tVelArr = [], iVelArr = [], kfErrArr = [], kfSigArr = [], kfCovArr = [];
  const neesArr = [], nisArr = [], estDragArr = [], statusArr = [], stageArr = [], gArr = [];

  const tMax = Math.max(150.0, tImp * 2.0 + 30.0);
  let t = 0.0, outcome = 'UNKNOWN', kill = null, cpa = null, minSep = Infinity;

  function measure(pos, vel) {
    if (sensorModel === 'cartesian') {
      const mp = [pos[0] + rng.normal(0, posNoise), pos[1] + rng.normal(0, posNoise), pos[2] + rng.normal(0, posNoise)];
      const mv = [vel[0] + rng.normal(0, velNoise), vel[1] + rng.normal(0, velNoise), vel[2] + rng.normal(0, velNoise)];
      const R = new Float64Array(36);
      for (let i = 0; i < 3; i++) { R[i * 6 + i] = posNoise * posNoise; R[(i + 3) * 6 + (i + 3)] = velNoise * velNoise; }
      return { mp, mv, R };
    }
    const rho = Math.hypot(pos[0], pos[1]);
    const r = Math.max(Math.hypot(rho, pos[2]), 1.0);
    const az0 = Math.atan2(pos[1], pos[0]);
    const el0 = Math.atan2(pos[2], Math.max(rho, 1e-9));
    const rm = Math.max(r + rng.normal(0, rangeNoise), 1.0);
    const azm = az0 + rng.normal(0, angleNoise);
    const elm = el0 + rng.normal(0, angleNoise);
    const ca = Math.cos(azm), sa = Math.sin(azm), ce = Math.cos(elm), se = Math.sin(elm);
    const ur = [ce * ca, ce * sa, se];
    const uaz = [-sa, ca, 0];
    const uel = [-se * ca, -se * sa, ce];
    const mp = [rm * ur[0], rm * ur[1], rm * ur[2]];
    const stdAz = Math.max(rm * Math.abs(ce) * angleNoise, rangeNoise);
    const stdEl = Math.max(rm * angleNoise, rangeNoise);
    const R = new Float64Array(36);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        R[i * 6 + j] = (rangeNoise ** 2) * ur[i] * ur[j] + (stdAz ** 2) * uaz[i] * uaz[j] + (stdEl ** 2) * uel[i] * uel[j];
        R[(i + 3) * 6 + (j + 3)] = (rangeRateNoise ** 2) * ur[i] * ur[j] + (velNoise ** 2) * (uaz[i] * uaz[j] + uel[i] * uel[j]);
      }
    }
    const nrr = rng.normal(0, rangeRateNoise), nva = rng.normal(0, velNoise), nve = rng.normal(0, velNoise);
    const mv = [
      vel[0] + nrr * ur[0] + nva * uaz[0] + nve * uel[0],
      vel[1] + nrr * ur[1] + nva * uaz[1] + nve * uel[1],
      vel[2] + nrr * ur[2] + nva * uaz[2] + nve * uel[2],
    ];
    return { mp, mv, R };
  }

  while (t < tMax) {
    const { mp, mv, R } = measure(tState, tState.subarray(3, 6));
    let nisVal = null;
    if (!kfInit) {
      for (let i = 0; i < 3; i++) { xHat[i] = mp[i]; xHat[i + 3] = mv[i]; }
      if (D_DIM === 7) xHat[6] = 0.000035;
      P.fill(0);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) P[r * D_DIM + c] = R[r * 6 + c];
      if (D_DIM === 7) P[48] = (0.000025) ** 2;
      kfInit = true;
    } else {
      // Predict
      const Phi = new Float64Array(D_DIM * D_DIM);
      for (let i = 0; i < D_DIM; i++) Phi[i * D_DIM + i] = 1.0;
      if (D_DIM === 6) {
        for (let i = 0; i < 3; i++) {
          xHat[i] += xHat[i + 3] * dt + (i === 2 ? -0.5 * g * dt * dt : 0);
          Phi[i * 6 + (i + 3)] = dt;
        }
        xHat[5] -= g * dt;
      } else {
        const vx = xHat[3], vy = xHat[4], vz = xHat[5], kt = Math.max(0, xHat[6]);
        const sp = Math.hypot(vx, vy, vz);
        const F = new Float64Array(49);
        F[0 * 7 + 3] = 1; F[1 * 7 + 4] = 1; F[2 * 7 + 5] = 1;
        if (sp > 1e-9) {
          const vs = [vx, vy, vz];
          for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
              F[(r + 3) * 7 + (c + 3)] = -kt * ((r === c ? sp : 0) + (vs[r] * vs[c]) / sp);
            }
            F[(r + 3) * 7 + 6] = -sp * vs[r];
          }
        }
        const Fdt = F.map((v) => v * dt);
        const Fdt2 = matMul(Fdt, Fdt, 7, 7, 7);
        for (let i = 0; i < 49; i++) Phi[i] += Fdt[i] + 0.5 * Fdt2[i];
        const nextX = rk4Step(xHat, (s) => {
          const sp0 = Math.hypot(s[3], s[4], s[5]), k0 = Math.max(0, s[6]);
          return [s[3], s[4], s[5], -k0 * sp0 * s[3], -k0 * sp0 * s[4], -k0 * sp0 * s[5] - g, 0];
        }, dt);
        nextX[6] = Math.max(0, Math.min(5e-4, nextX[6]));
        xHat.set(nextX);
      }
      const PhiP = matMul(Phi, P, D_DIM, D_DIM, D_DIM);
      const PhiPPhiT = matMul(PhiP, matTrans(Phi, D_DIM, D_DIM), D_DIM, D_DIM, D_DIM);
      for (let i = 0; i < D_DIM * D_DIM; i++) P[i] = PhiPPhiT[i] + Q[i];

      // Update
      const y = new Float64Array([mp[0] - xHat[0], mp[1] - xHat[1], mp[2] - xHat[2], mv[0] - xHat[3], mv[1] - xHat[4], mv[2] - xHat[5]]);
      const S = new Float64Array(36);
      const HP = new Float64Array(6 * D_DIM);
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) S[r * 6 + c] = P[r * D_DIM + c] + R[r * 6 + c];
        for (let c = 0; c < D_DIM; c++) HP[r * D_DIM + c] = P[r * D_DIM + c];
      }
      const Kt = solveSym6(S, HP); // 6 x D_DIM
      const K = matTrans(Kt, 6, D_DIM); // D_DIM x 6
      const Sy = solveSym6(S, y);
      nisVal = 0;
      for (let i = 0; i < 6; i++) nisVal += y[i] * Sy[i];
      for (let r = 0; r < D_DIM; r++) {
        let d = 0;
        for (let c = 0; c < 6; c++) d += K[r * 6 + c] * y[c];
        xHat[r] += d;
      }
      if (D_DIM === 7) xHat[6] = Math.max(0, Math.min(5e-4, xHat[6]));
      // Joseph form P = (I - K H) P (I - K H)^T + K R K^T
      const IKH = new Float64Array(D_DIM * D_DIM);
      for (let r = 0; r < D_DIM; r++) {
        IKH[r * D_DIM + r] = 1.0;
        for (let c = 0; c < 6; c++) IKH[r * D_DIM + c] -= K[r * 6 + c];
      }
      const term1 = matMul(matMul(IKH, P, D_DIM, D_DIM, D_DIM), matTrans(IKH, D_DIM, D_DIM), D_DIM, D_DIM, D_DIM);
      const term2 = matMul(matMul(K, R, D_DIM, 6, 6), Kt, D_DIM, 6, D_DIM);
      for (let i = 0; i < D_DIM * D_DIM; i++) P[i] = term1[i] + term2[i];
    }

    const ep = [xHat[0], xHat[1], xHat[2]], ev = [xHat[3], xHat[4], xHat[5]];
    const estDrag = D_DIM === 7 ? xHat[6] : 0.0;
    const kfErr = [ep[0] - tState[0], ep[1] - tState[1], ep[2] - tState[2], ev[0] - tState[3], ev[1] - tState[4], ev[2] - tState[5]];
    const P6 = new Float64Array(36);
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) P6[r * 6 + c] = P[r * D_DIM + c];
    const PinvErr = solveSym6(P6, kfErr);
    let neesVal = 0;
    for (let i = 0; i < 6; i++) neesVal += kfErr[i] * PinvErr[i];

    // Launch decision
    if (t >= minTrackTime && !launched && ev[2] < 0) {
      const pip = predictImpactDrag(ep[0], ep[1], ep[2], ev[0], ev[1], ev[2], pipMode === 'drag' ? estDrag : 0.0, g);
      if (pip && Math.hypot(pip[0], pip[1]) <= defendedRadius) {
        launched = true;
        launchTime = t;
        launchPip = pip;
        stage = 'BOOST';
        const d0 = Math.max(Math.hypot(ep[0], ep[1], ep[2]), 1e-6);
        iState[3] = (ep[0] / d0) * 80.0;
        iState[4] = (ep[1] / d0) * 80.0;
        iState[5] = (ep[2] / d0) * 80.0;
      }
    }

    // Step interceptor & threat
    const tNext = t + dt;
    const ip0 = [iState[0], iState[1], iState[2]];
    const tp0 = [tState[0], tState[1], tState[2]];
    let gLoad = 0;

    if (launched) {
      const burnLeft = burnTime - (t - launchTime);
      const tBoost = Math.min(Math.max(burnLeft, 0.0), dt);
      const tCoast = dt - tBoost;
      stage = burnLeft > dt ? 'BOOST' : 'KV_COAST';

      const rx = ep[0] - iState[0], ry = ep[1] - iState[1], rz = ep[2] - iState[2];
      const vx = ev[0] - iState[3], vy = ev[1] - iState[4], vz = ev[2] - iState[5];
      const rMag = Math.hypot(rx, ry, rz);
      let ac = [0, 0, 0];
      if (rMag >= 20.0) {
        const r2 = rMag * rMag + 1e-6;
        const wx = (ry * vz - rz * vy) / r2, wy = (rz * vx - rx * vz) / r2, wz = (rx * vy - ry * vx) / r2;
        const esp = Math.hypot(ev[0], ev[1], ev[2]);
        const at = [-estDrag * esp * ev[0], -estDrag * esp * ev[1], -estDrag * esp * ev[2]];
        if (guidanceLaw === 'zem') {
          const vsq = Math.max(vx * vx + vy * vy + vz * vz, 1e-6);
          const rv = rx * vx + ry * vy + rz * vz;
          const tgo = rv < 0 ? Math.max(-rv / vsq, 0.05) : Math.max(rMag / Math.sqrt(vsq), 0.05);
          const zx = rx + vx * tgo + 0.5 * at[0] * tgo * tgo;
          const zy = ry + vy * tgo + 0.5 * at[1] * tgo * tgo;
          const zz = rz + vz * tgo + 0.5 * at[2] * tgo * tgo;
          const dot = (zx * rx + zy * ry + zz * rz) / (rMag * rMag);
          ac = [
            (pnGain / (tgo * tgo)) * (zx - dot * rx),
            (pnGain / (tgo * tgo)) * (zy - dot * ry),
            (pnGain / (tgo * tgo)) * (zz - dot * rz) + (tBoost > 0 ? g : 0),
          ];
        } else {
          ac = [
            pnGain * (vy * wz - vz * wy),
            pnGain * (vz * wx - vx * wz),
            pnGain * (vx * wy - vy * wx) + (guidanceLaw === 'pn' || tBoost > 0 ? g : 0),
          ];
          if (guidanceLaw === 'apn') {
            const dot = (at[0] * rx + at[1] * ry + at[2] * rz) / (rMag * rMag);
            ac[0] += 0.5 * pnGain * (at[0] - dot * rx);
            ac[1] += 0.5 * pnGain * (at[1] - dot * ry);
            ac[2] += 0.5 * pnGain * (at[2] - dot * rz);
          }
        }
      }
      let acMag = Math.hypot(ac[0], ac[1], ac[2]);
      const lim = maxG * G0;
      if (acMag > lim) { ac = ac.map((c) => (c / acMag) * lim); acMag = lim; }
      if (tau > 0) {
        const alpha = 1.0 - Math.exp(-dt / tau);
        actAcc = actAcc.map((a, j) => a + alpha * (ac[j] - a));
        ac = actAcc.slice();
        acMag = Math.hypot(ac[0], ac[1], ac[2]);
        if (acMag > lim) { ac = ac.map((c) => (c / acMag) * lim); acMag = lim; }
      }
      gLoad = acMag / G0;

      if (tBoost > 0) {
        const u = acMag > 1e-3 ? ac.map((c) => c / acMag) : [0, 0, 1];
        iState = rk4Step(iState, (s) => {
          const sp = Math.hypot(s[3], s[4], s[5]), m = s[6];
          return [
            s[3], s[4], s[5],
            (u[0] * boostThrust - boosterDrag * sp * s[3]) / m + 0.1 * ac[0],
            (u[1] * boostThrust - boosterDrag * sp * s[4]) / m + 0.1 * ac[1],
            (u[2] * boostThrust - boosterDrag * sp * s[5]) / m + 0.1 * ac[2] - g,
            -massFlow,
          ];
        }, tBoost);
        if (burnLeft <= dt) iState[6] = kvMass;
      }
      if (tCoast > 0) {
        let kvCmd = ac.slice(), kvMag = acMag;
        if (divertBudget !== null) {
          const rem = Math.max(divertBudget - divertUsed, 0);
          if (kvMag * tCoast > rem) {
            const sc = rem / tCoast / Math.max(kvMag, 1e-9);
            kvCmd = kvCmd.map((c) => c * sc);
            kvMag = rem / tCoast;
          }
        }
        divertUsed += kvMag * tCoast;
        if (tBoost === 0) gLoad = kvMag / G0;
        iState = rk4Step(iState, (s) => {
          const sp = Math.hypot(s[3], s[4], s[5]), m = s[6];
          return [
            s[3], s[4], s[5],
            (-kvDrag * sp * s[3]) / m + kvCmd[0],
            (-kvDrag * sp * s[4]) / m + kvCmd[1],
            (-kvDrag * sp * s[5]) / m + kvCmd[2] - g,
            0,
          ];
        }, tCoast);
      }
    }

    // Propagate threat
    const tPrev = tState;
    tState = rk4Step(tPrev, (s) => {
      const sp = Math.hypot(s[3], s[4], s[5]);
      return [s[3], s[4], s[5], -threatDrag * sp * s[3], -threatDrag * sp * s[4], -threatDrag * sp * s[5] - g];
    }, dt);
    let tGround = null;
    if (tState[2] <= 0) {
      const f = tPrev[2] / Math.max(tPrev[2] - tState[2], 1e-12);
      for (let j = 0; j < 6; j++) tState[j] = tPrev[j] + f * (tState[j] - tPrev[j]);
      tState[2] = 0;
      tGround = f * dt;
    }
    const win = tGround === null ? dt : tGround;
    const iv = [(iState[0] - ip0[0]) / dt, (iState[1] - ip0[1]) / dt, (iState[2] - ip0[2]) / dt];
    const tv = [(tState[0] - tp0[0]) / win, (tState[1] - tp0[1]) / win, (tState[2] - tp0[2]) / win];

    if (launched) {
      const dpx = tp0[0] - ip0[0], dpy = tp0[1] - ip0[1], dpz = tp0[2] - ip0[2];
      const dvx = tv[0] - iv[0], dvy = tv[1] - iv[1], dvz = tv[2] - iv[2];
      const dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
      const tOff = dv2 > 1e-6 ? Math.max(0, Math.min(win, -(dpx * dvx + dpy * dvy + dpz * dvz) / dv2)) : 0;
      const dMin = Math.hypot(dpx + dvx * tOff, dpy + dvy * tOff, dpz + dvz * tOff);
      if (dMin < minSep) {
        minSep = dMin;
        cpa = {
          time: +(t + tOff).toFixed(3),
          alt: +(tp0[2] + tv[2] * tOff).toFixed(1),
          dist: +dMin.toFixed(2),
          closing: +Math.sqrt(dv2).toFixed(1),
          tPos: [+(tp0[0] + tv[0] * tOff).toFixed(1), +(tp0[1] + tv[1] * tOff).toFixed(1), +(tp0[2] + tv[2] * tOff).toFixed(1)],
          iPos: [+(ip0[0] + iv[0] * tOff).toFixed(1), +(ip0[1] + iv[1] * tOff).toFixed(1), +(ip0[2] + iv[2] * tOff).toFixed(1)],
        };
      }
      if (dMin < killRadius) {
        const kTime = t + tOff;
        const ktPos = [tp0[0] + tv[0] * tOff, tp0[1] + tv[1] * tOff, tp0[2] + tv[2] * tOff];
        const kiPos = [ip0[0] + iv[0] * tOff, ip0[1] + iv[1] * tOff, ip0[2] + iv[2] * tOff];
        tArr.push(+kTime.toFixed(3));
        threatArr.push(+ktPos[0].toFixed(1), +ktPos[1].toFixed(1), +ktPos[2].toFixed(1));
        intArr.push(+kiPos[0].toFixed(1), +kiPos[1].toFixed(1), +kiPos[2].toFixed(1));
        trackArr.push(+ep[0].toFixed(1), +ep[1].toFixed(1), +ep[2].toFixed(1));
        radarArr.push(+mp[0].toFixed(1), +mp[1].toFixed(1), +mp[2].toFixed(1));
        tVelArr.push(+tv[0].toFixed(1), +tv[1].toFixed(1), +tv[2].toFixed(1));
        iVelArr.push(+iv[0].toFixed(1), +iv[1].toFixed(1), +iv[2].toFixed(1));
        kfErr.forEach((v) => kfErrArr.push(+v.toFixed(2)));
        for (let i = 0; i < 6; i++) kfSigArr.push(+Math.sqrt(Math.max(0, P6[i * 6 + i])).toFixed(2));
        kfCovArr.push(+P6[0].toFixed(2), +P6[7].toFixed(2), +P6[14].toFixed(2), +P6[1].toFixed(2), +P6[2].toFixed(2), +P6[8].toFixed(2));
        neesArr.push(+neesVal.toFixed(2));
        nisArr.push(nisVal === null ? null : +nisVal.toFixed(2));
        estDragArr.push(+(estDrag * 1e5).toFixed(3));
        statusArr.push(2);
        stageArr.push(stage === 'BOOST' ? 1 : 2);
        gArr.push(+gLoad.toFixed(2));
        outcome = 'KILL';
        kill = { time: +kTime.toFixed(3), alt: +ktPos[2].toFixed(1), dist: +dMin.toFixed(2), closing: +Math.sqrt(dv2).toFixed(1), pos: ktPos.map((c) => +c.toFixed(1)) };
        break;
      }
    }

    const recTime = tGround !== null ? t + tGround : tNext;
    tArr.push(+recTime.toFixed(3));
    threatArr.push(+tState[0].toFixed(1), +tState[1].toFixed(1), +tState[2].toFixed(1));
    intArr.push(+iState[0].toFixed(1), +iState[1].toFixed(1), +iState[2].toFixed(1));
    trackArr.push(+ep[0].toFixed(1), +ep[1].toFixed(1), +ep[2].toFixed(1));
    radarArr.push(+mp[0].toFixed(1), +mp[1].toFixed(1), +mp[2].toFixed(1));
    tVelArr.push(+tState[3].toFixed(1), +tState[4].toFixed(1), +tState[5].toFixed(1));
    iVelArr.push(+iState[3].toFixed(1), +iState[4].toFixed(1), +iState[5].toFixed(1));
    kfErr.forEach((v) => kfErrArr.push(+v.toFixed(2)));
    for (let i = 0; i < 6; i++) kfSigArr.push(+Math.sqrt(Math.max(0, P6[i * 6 + i])).toFixed(2));
    kfCovArr.push(+P6[0].toFixed(2), +P6[7].toFixed(2), +P6[14].toFixed(2), +P6[1].toFixed(2), +P6[2].toFixed(2), +P6[8].toFixed(2));
    neesArr.push(+neesVal.toFixed(2));
    nisArr.push(nisVal === null ? null : +nisVal.toFixed(2));
    estDragArr.push(+(estDrag * 1e5).toFixed(3));
    statusArr.push(launched ? 1 : 0);
    stageArr.push(!launched ? 0 : stage === 'BOOST' ? 1 : 2);
    gArr.push(launched ? +gLoad.toFixed(2) : 0);

    if (tGround !== null) {
      const impRange = Math.hypot(tState[0], tState[1]);
      outcome = (!launched && impRange > defendedRadius) ? 'NO ENGAGE' : 'GROUND';
      break;
    }
    t = tNext;
  }

  if (outcome === 'UNKNOWN') outcome = 'TIMEOUT';

  const params = {
    dt, t_max: 150.0, g, pos_noise: posNoise, vel_noise: velNoise,
    boost_thrust: boostThrust, isp, mass_flow: massFlow, burn_time: burnTime,
    mass_wet: massWet, mass_dry: massDry, kv_mass: kvMass, kv_drag: kvDrag,
    pn_gain: pnGain, max_g: maxG, kill_radius: killRadius,
    booster_drag: boosterDrag, threat_drag: threatDrag, kf_q: qSpec,
    defended_radius: defendedRadius, min_track_time: minTrackTime,
  };
  if (divertBudget !== null) params.kv_divert_dv = divertBudget;
  if (tau > 0) params.tau = tau;
  if (sensorModel === 'rae') {
    params.range_noise = rangeNoise;
    params.angle_noise = angleNoise;
    params.range_rate_noise = rangeRateNoise;
  }

  return {
    title: `LIVE BROWSER LAB | ${filterType.toUpperCase()} + ${guidanceLaw.toUpperCase()}`,
    mode: filterType === 'ekf7' ? 'ADVANCED GNC MODE' : 'ENGINEERING MODE',
    engageRule: 'PIP-GATED LAUNCH',
    outcome,
    integrator: 'rk4',
    t: tArr,
    threat: threatArr,
    interceptor: intArr,
    track: trackArr,
    radar: radarArr,
    threatVel: tVelArr,
    interceptorVel: iVelArr,
    kfErr: kfErrArr,
    kfSig: kfSigArr,
    kfCov: kfCovArr,
    nees: neesArr,
    nis: nisArr,
    estDrag: filterType === 'ekf7' ? estDragArr : null,
    filterType,
    pipMode,
    sensorModel,
    guidanceLaw,
    status: statusArr,
    stage: stageArr,
    g: gArr,
    kill,
    cpa,
    launchPip: launchPip ? launchPip.map((c) => +c.toFixed(1)) : null,
    impact: (outcome === 'GROUND' || outcome === 'NO ENGAGE') ? [+tState[0].toFixed(1), +tState[1].toFixed(1), 0] : null,
    defendedRadius,
    killRadius,
    maxG,
    minTrackTime,
    params,
    seed,
    config: filterType === 'ekf7' ? 'configs/advanced.yaml' : 'configs/engineering.yaml',
  };
}

