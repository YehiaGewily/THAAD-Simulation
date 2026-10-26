// Vehicle models: the threat re-entry vehicle, the two-stage interceptor and
// the spent booster.
import * as THREE from 'three';
import { P, burnT, sample, velocity } from './data.js';
import { SOFT, glowSprite } from './materials.js';
import { PALETTE, hex } from './palette.js';
import { darkMetal, metal, scene } from './scene.js';

// ---------------------------------------------------------------------------
// Vehicles
// ---------------------------------------------------------------------------
const MODEL_SCALE = 8; // models are exaggerated so they read in close-up cameras

function makeThreat() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.ConeGeometry(0.0009, 0.0035, 20).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x3a2e2a, metalness: 0.3, roughness: 0.5, emissive: 0x401006 }));
  g.add(body);
  const model = new THREE.Group(); model.add(g); model.scale.setScalar(MODEL_SCALE);
  const root = new THREE.Group();
  root.add(model);
  const heat = glowSprite(0xff7a2a, 0.045, { attenuate: true, tex: SOFT });
  heat.position.z = 0.012;
  model.add(heat);
  const far = glowSprite(hex(PALETTE.threat), 0.028);
  root.add(far);
  root.userData = { model, heat, far };
  scene.add(root);
  return root;
}

function makeInterceptor() {
  const root = new THREE.Group();
  const model = new THREE.Group(); model.scale.setScalar(MODEL_SCALE);
  const booster = new THREE.Mesh(new THREE.CylinderGeometry(0.00035, 0.00035, 0.0045, 16).rotateX(Math.PI / 2), metal);
  booster.position.z = -0.0012;
  const kvBody = new THREE.Mesh(new THREE.CylinderGeometry(0.00028, 0.00035, 0.0012, 16).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xe8edf2, metalness: 0.4, roughness: 0.35 }));
  kvBody.position.z = 0.0017;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.00028, 0.0009, 16).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xdfe6ec, metalness: 0.5, roughness: 0.3 }));
  nose.position.z = 0.0027;
  model.add(booster, kvBody, nose);

  const flame = glowSprite(0xffb050, 0.03, { attenuate: true, tex: SOFT });
  flame.position.z = -0.0055;
  model.add(flame);
  const plume = glowSprite(0xff8a3a, 0.08, { attenuate: true, tex: SOFT, opacity: 0.5 });
  plume.position.z = -0.012;
  model.add(plume);
  const divert = glowSprite(0xcfeaf7, 0.012, { attenuate: true });
  model.add(divert);

  root.add(model);
  const far = glowSprite(hex(PALETTE.kv), 0.03);
  root.add(far);
  root.userData = { model, booster, flame, plume, divert, far, farBoost: new THREE.Color(hex(PALETTE.accent)), farCoast: new THREE.Color(hex(PALETTE.kv)) };
  scene.add(root);
  return root;
}

const threatObj = makeThreat();
const kvObj = makeInterceptor();

// Spent booster, separated at burnout and falling ballistically
const spentBooster = (() => {
  const root = new THREE.Group();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(0.00035, 0.00035, 0.0045, 16).rotateX(Math.PI / 2), metal);
  const model = new THREE.Group(); model.add(m); model.scale.setScalar(MODEL_SCALE);
  root.add(model);
  const ember = glowSprite(0xff7a30, 0.012, { opacity: 0.8 });
  root.add(ember);
  root.visible = false;
  root.userData = { model, ember };
  scene.add(root);
  if (burnT != null) {
    root.userData.p0 = sample(P.kv, burnT);
    root.userData.v0 = velocity(P.kv, burnT);
  }
  return root;
})();

export { MODEL_SCALE, threatObj, kvObj, spentBooster };
