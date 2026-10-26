// Kalman track uncertainty: the filter's 3-sigma position ellipsoid.
//
// The payload carries the position block of the covariance P for every frame
// (kfCov: xx, yy, zz, xy, xz, yz in m², sim frame). Each frame the 3×3 block
// is rotated into the scene frame and diagonalized; the ellipsoid's axes are
// the eigenvectors and its semi-axes 3·sqrt(eigenvalue).
//
// A few metres would vanish at kilometre scale, so the geometry is drawn
// EXAGGERATION times larger about the true threat position: both the
// ellipsoid's size and the estimate's offset from the truth are scaled. The
// threat therefore sits inside the shell exactly when the estimate error is
// inside the 3-sigma ellipsoid, as it would at true scale.
import * as THREE from 'three';
import { D, KM, N } from './data.js';
import { fresnelMaterial } from './materials.js';
import { PALETTE, hex } from './palette.js';
import { scene } from './scene.js';

export const EXAGGERATION = 10;
export const N_SIGMA = 3;
export const hasCov = Array.isArray(D.kfCov) && D.kfCov.length === 6 * N;

// Cyclic Jacobi eigen-decomposition of a symmetric 3×3 matrix (row-major
// array of 9). Returns { values: [3], vectors: [[3] x3] } with unit vectors.
export function eigSym3(m) {
  const a = m.slice();
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off < 1e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8] + 1e-300)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[3 * p + q];
      if (Math.abs(apq) < 1e-300) continue;
      const app = a[3 * p + p], aqq = a[3 * q + q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {           // A <- A J  (columns p, q)
        const akp = a[3 * k + p], akq = a[3 * k + q];
        a[3 * k + p] = c * akp - s * akq;
        a[3 * k + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {           // A <- J^T A  (rows p, q)
        const apk = a[3 * p + k], aqk = a[3 * q + k];
        a[3 * p + k] = c * apk - s * aqk;
        a[3 * q + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {           // V <- V J
        const vkp = v[3 * k + p], vkq = v[3 * k + q];
        v[3 * k + p] = c * vkp - s * vkq;
        v[3 * k + q] = s * vkp + c * vkq;
      }
    }
  }
  return {
    values: [a[0], a[4], a[8]],
    vectors: [0, 1, 2].map((j) => [v[j], v[3 + j], v[6 + j]]),
  };
}

// Position covariance of frame i in the scene frame (km²), row-major 3×3.
// The scene maps sim (x, y, z) -> (x, z, -y).
export function sceneCov(i) {
  const c = D.kfCov, k = 6 * i, s = KM * KM;
  const xx = c[k], yy = c[k + 1], zz = c[k + 2], xy = c[k + 3], xz = c[k + 4], yz = c[k + 5];
  return [
    xx * s, xz * s, -xy * s,
    xz * s, zz * s, -yz * s,
    -xy * s, -yz * s, yy * s,
  ];
}

const group = new THREE.Group();
group.visible = false;
const shell = new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 24),
  fresnelMaterial(hex(PALETTE.cov), { power: 2.4, opacity: 0.22 }),
);
shell.renderOrder = 3;
group.add(shell);
{
  // Principal-axis rings make the orientation readable when the shell is not a sphere
  const pts = [];
  for (let j = 0; j <= 96; j++) { const a = (j / 96) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0)); }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color: hex(PALETTE.cov), transparent: true, opacity: 0.3, depthWrite: false });
  [[0, 0, 0], [Math.PI / 2, 0, 0], [0, Math.PI / 2, 0]].forEach(([x, y, z]) => {
    const ring = new THREE.LineLoop(geo, mat);
    ring.rotation.set(x, y, z);
    group.add(ring);
  });
}
scene.add(group);

const basis = new THREE.Matrix4();
const e = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
const err = new THREE.Vector3();
let lastFrame = -1;
let sigma3 = 0;  // 3-sigma radius along the largest axis, metres

/** Place the ellipsoid for frame ``fi`` around the true threat position ``truth`` (scene units). */
export function updateCovariance(visible, fi, truth) {
  group.visible = visible && hasCov;
  if (!group.visible) return;
  if (fi !== lastFrame) {
    lastFrame = fi;
    const { values, vectors } = eigSym3(sceneCov(fi));
    vectors.forEach((v, j) => e[j].set(v[0], v[1], v[2]));
    e[2].crossVectors(e[0], e[1]);            // right-handed basis
    basis.makeBasis(e[0], e[1], e[2]);
    group.quaternion.setFromRotationMatrix(basis);
    const r = values.map((l) => N_SIGMA * Math.sqrt(Math.max(l, 0)) * EXAGGERATION);
    group.scale.set(Math.max(r[0], 1e-6), Math.max(r[1], 1e-6), Math.max(r[2], 1e-6));
    sigma3 = (N_SIGMA * Math.sqrt(Math.max(...values, 0))) / KM;
  }
  // Estimate minus truth, sim frame (m) -> scene frame (km), exaggerated like the shell
  const k = 6 * fi;
  err.set(D.kfErr[k] * KM, D.kfErr[k + 2] * KM, -D.kfErr[k + 1] * KM).multiplyScalar(EXAGGERATION);
  group.position.copy(truth).add(err);
}

export const covSigma3 = () => sigma3;
