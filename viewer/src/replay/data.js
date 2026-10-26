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
