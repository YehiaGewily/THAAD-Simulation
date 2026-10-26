// The camera director: where each directed camera (cinematic, chase threat,
// chase interceptor, top-down) wants to be at a given moment.
import * as THREE from 'three';
import { impactPos, killPos, threatStart } from './data.js';
import { realClock, state } from './state.js';

// ============================================================================
// CAMERA DIRECTOR
// ============================================================================
const UP = new THREE.Vector3(0, 1, 0);
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Vector3();
export const desiredPos = new THREE.Vector3(), desiredTarget = new THREE.Vector3();

// Establishing side direction: perpendicular to the threat's ground track
const trackDir = new THREE.Vector3(threatStart.x, 0, threatStart.z).normalize();
const sideDir = new THREE.Vector3(-trackDir.z, 0, trackDir.x);

export function shot(st) {
  const { tp, ip, tv, launched, stage, sep, killed, impacted } = st;
  if (state.camMode === 'top') {
    tmpA.copy(tp).multiplyScalar(0.5).setY(0);
    desiredTarget.copy(tmpA);
    desiredPos.set(tmpA.x, Math.max(90, threatStart.length() * 1.6), tmpA.z + 0.01);
    return;
  }
  const focus = killed ? killPos : impacted ? impactPos : null;
  if (focus) {
    // after the engagement every view (except top-down) orbits the aftermath
    const ang = realClock.t * 0.07;
    const r = killed ? 9 : 7;
    desiredPos.set(focus.x + Math.cos(ang) * r, focus.y + r * 0.35 + 0.5, focus.z + Math.sin(ang) * r);
    desiredTarget.copy(focus);
    return;
  }
  if (state.camMode === 'threat') {
    const vdir = tmpA.copy(tv).normalize();
    if (vdir.lengthSq() < 0.5) vdir.set(0, -1, 0);
    desiredPos.copy(tp).addScaledVector(vdir, -state.chaseDist).addScaledVector(UP, state.chaseDist * 0.35).addScaledVector(sideDir, state.chaseDist * 0.2);
    desiredTarget.copy(tp).addScaledVector(vdir, state.chaseDist * 0.6);
    return;
  }
  if (state.camMode === 'kv') {
    if (!launched) {
      desiredPos.copy(sideDir).multiplyScalar(0.3).setY(0.08);
      desiredTarget.copy(tp);
      return;
    }
    const los = tmpA.copy(tp).sub(ip).normalize();
    desiredPos.copy(ip).addScaledVector(los, -state.chaseDist).addScaledVector(UP, state.chaseDist * 0.25);
    desiredTarget.copy(ip).addScaledVector(los, state.chaseDist * 4);
    return;
  }
  // cinematic
  if (!launched) {
    const mid = tmpA.copy(tp).multiplyScalar(0.5);
    const span = Math.max(tp.length(), 20);
    desiredPos.copy(mid).addScaledVector(sideDir, span * 1.25).addScaledVector(UP, span * 0.1).addScaledVector(trackDir, span * 0.25);
    desiredPos.y = Math.max(desiredPos.y, 3);
    desiredTarget.copy(mid);
    return;
  }
  if (stage === 1) {
    desiredPos.copy(sideDir).multiplyScalar(5).addScaledVector(trackDir, -2).setY(1.2);
    desiredPos.y += ip.y * 0.4;
    desiredTarget.copy(ip);
    return;
  }
  const ttg = st.ttg;
  if (sep > 6 && !(ttg < 2.5)) {
    const mid = tmpA.copy(tp).add(ip).multiplyScalar(0.5);
    const los = tmpB.copy(tp).sub(ip).normalize();
    const side = tmpC.crossVectors(los, UP);
    if (side.lengthSq() < 1e-4) side.copy(sideDir); else side.normalize();
    const dist = sep * 0.95 + 6;
    desiredPos.copy(mid).addScaledVector(side, dist).addScaledVector(UP, dist * 0.18);
    desiredTarget.copy(mid);
    return;
  }
  // terminal: over the kill vehicle's shoulder, looking at the threat
  const los = tmpA.copy(tp).sub(ip).normalize();
  const side = tmpB.crossVectors(los, UP);
  if (side.lengthSq() < 1e-4) side.copy(sideDir); else side.normalize();
  desiredPos.copy(ip).addScaledVector(los, -0.8).addScaledVector(side, 0.7).addScaledVector(UP, 0.25);
  desiredTarget.copy(tp);
}
