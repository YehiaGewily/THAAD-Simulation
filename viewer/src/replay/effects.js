// Things that change with replay time: trails, radar returns, explosions and
// the line-of-sight and radar-beam lines.
import * as THREE from 'three';
import { G_KM, N, P, T, burnT, killPos, impactPos, killT, launchIdx, sample, velocity } from './data.js';
import { fresnelMaterial, glowSprite } from './materials.js';
import { PALETTE, hex } from './palette.js';
import { renderer, scene } from './scene.js';

// ---------------------------------------------------------------------------
// Trails: a thin always-visible line plus a tube for close-up views.
// Both fade with age and only draw up to the current replay time.
// ---------------------------------------------------------------------------
const trailUniforms = [];
const trailMinDefaults = [];
const tubeMeshes = [];
const trailVS = `attribute float aTime; varying float vTime; varying vec3 vW;
  void main(){ vTime = aTime; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const trailFS = `uniform float uNow; uniform float uFade; uniform float uMin; uniform float uSplit;
  uniform vec3 uColA; uniform vec3 uColB; uniform float uGain; varying float vTime; varying vec3 vW;
  void main(){
    if (vTime > uNow) discard;
    float age = uNow - vTime;
    float a = mix(uMin, 1.0, exp(-age / uFade));
    a *= smoothstep(0.6, 3.0, length(cameraPosition - vW));
    vec3 c = vTime < uSplit ? uColA : uColB;
    gl_FragColor = vec4(c * a * uGain, 1.0);
  }`;
function trailMaterial(opts) {
  const u = {
    uNow: { value: -1e9 }, uFade: { value: opts.fade }, uMin: { value: opts.min },
    uSplit: { value: opts.split ?? 1e9 }, uColA: { value: new THREE.Color(opts.colorA) },
    uColB: { value: new THREE.Color(opts.colorB ?? opts.colorA) }, uGain: { value: opts.gain },
  };
  trailUniforms.push(u);
  trailMinDefaults.push(opts.min);
  return new THREE.ShaderMaterial({ uniforms: u, vertexShader: trailVS, fragmentShader: trailFS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
}

class SampledCurve extends THREE.Curve {
  constructor(arr, t0, t1) { super(); this.arr = arr; this.t0 = t0; this.t1 = t1; }
  getPoint(u, out = new THREE.Vector3()) { return sample(this.arr, this.t0 + u * (this.t1 - this.t0), out); }
  getUtoTmapping(u) { return u; }
}

function buildTrail(arr, i0, i1, opts) {
  const group = new THREE.Group();
  if (i1 - i0 < 2) return group;
  const t0 = T[i0], t1 = T[i1];

  const pts = arr.slice(3 * i0, 3 * (i1 + 1));
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  lineGeo.setAttribute('aTime', new THREE.BufferAttribute(Float32Array.from(T.slice(i0, i1 + 1)), 1));
  const line = new THREE.Line(lineGeo, trailMaterial({ ...opts, gain: opts.lineGain ?? 1.0 }));
  line.frustumCulled = false;
  group.add(line);

  if (opts.radius) {
    const segs = Math.min(4000, Math.max(64, Math.round((t1 - t0) / 0.04)));
    const radial = 8;
    const tube = new THREE.TubeGeometry(new SampledCurve(arr, t0, t1), segs, opts.radius, radial, false);
    const at = new Float32Array((segs + 1) * (radial + 1));
    for (let i = 0; i <= segs; i++) for (let j = 0; j <= radial; j++) at[i * (radial + 1) + j] = t0 + (i / segs) * (t1 - t0);
    tube.setAttribute('aTime', new THREE.BufferAttribute(at, 1));
    const mesh = new THREE.Mesh(tube, trailMaterial({ ...opts, gain: opts.tubeGain ?? 0.5 }));
    mesh.frustumCulled = false;
    group.add(mesh);
    tubeMeshes.push(mesh);
  }
  scene.add(group);
  return group;
}

const lastIdx = N - 1;
buildTrail(P.threat, 0, lastIdx, { colorA: hex(PALETTE.threat), fade: 14, min: 0.22, radius: 0.018, lineGain: 1.1, tubeGain: 0.55 });
if (launchIdx >= 0) {
  buildTrail(P.kv, launchIdx, lastIdx, {
    colorA: hex(PALETTE.accent), colorB: hex(PALETTE.kv), split: burnT ?? 1e9,
    fade: 10, min: 0.25, radius: 0.03, lineGain: 1.2, tubeGain: 0.4,
  });
}
const trackTrail = buildTrail(P.track, 0, lastIdx, { colorA: hex(PALETTE.track), fade: 6, min: 0.08, lineGain: 0.55 });

// Radar returns: raw noisy measurements shown for a few seconds after each scan
const radarUniforms = { uNow: { value: -1e9 }, uWin: { value: 3.5 }, uPx: { value: 3.0 * renderer.getPixelRatio() }, uCol: { value: new THREE.Color(hex(PALETTE.radar)) } };
const radarPts = (() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(P.radar, 3));
  g.setAttribute('aTime', new THREE.BufferAttribute(Float32Array.from(T), 1));
  const pts = new THREE.Points(g, new THREE.ShaderMaterial({
    uniforms: radarUniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float aTime; uniform float uNow; uniform float uWin; uniform float uPx; varying float vA;
      void main(){ float age = uNow - aTime; vA = (age < 0.0 || age > uWin) ? 0.0 : 1.0 - age / uWin;
        gl_PointSize = vA > 0.0 ? uPx : 0.0; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform vec3 uCol; varying float vA;
      void main(){ if (vA <= 0.0) discard; float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
        gl_FragColor = vec4(uCol * vA * (1.0 - d * 1.6), 1.0); }`,
  }));
  pts.frustumCulled = false;
  scene.add(pts);
  return pts;
})();
// ---------------------------------------------------------------------------
// Explosions (kill in the air, or threat ground impact)
// ---------------------------------------------------------------------------
function createExplosion(origin, { ground = false, inherit = new THREE.Vector3(), scale = 1 } = {}) {
  const group = new THREE.Group();
  group.position.copy(origin);
  group.visible = false;
  scene.add(group);

  const flash = glowSprite(0xfff1d0, 1, { attenuate: true });
  const fireball = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ color: ground ? 0xff7a2a : 0xff9a3a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  const shockMat = fresnelMaterial(ground ? 0xffb070 : 0xd6e6f2, { power: 2.2, opacity: 0.8 });
  const shock = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), shockMat);
  if (ground) shock.scale.y = 0.35;
  group.add(fireball, shock, flash);

  const n = ground ? 900 : 1200;
  const vel = new Float32Array(n * 3), seed = new Float32Array(n);
  const dir = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    dir.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
    if (dir.lengthSq() > 1 || dir.lengthSq() < 1e-4) { i--; continue; }
    dir.normalize();
    if (ground) dir.y = Math.abs(dir.y) * 0.9 + 0.1;
    const speed = (0.08 + Math.pow(Math.random(), 2.2) * 1.3) * scale;
    vel[3 * i] = dir.x * speed + inherit.x;
    vel[3 * i + 1] = dir.y * speed + inherit.y;
    vel[3 * i + 2] = dir.z * speed + inherit.z;
    seed[i] = Math.random();
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  pg.setAttribute('aVel', new THREE.BufferAttribute(vel, 3));
  pg.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
  const pu = { uAge: { value: 0 }, uG: { value: G_KM }, uPx: { value: renderer.getPixelRatio() } };
  const debris = new THREE.Points(pg, new THREE.ShaderMaterial({
    uniforms: pu, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute vec3 aVel; attribute float aSeed; uniform float uAge; uniform float uG; uniform float uPx;
      varying float vLife; varying float vSeed;
      void main(){
        float k = 0.55 + aSeed * 0.6;
        vec3 p = aVel * (1.0 - exp(-k * uAge)) / k - vec3(0.0, 0.5 * uG * uAge * uAge, 0.0);
        vLife = clamp(uAge / (2.5 + aSeed * 6.0), 0.0, 1.0); vSeed = aSeed;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = clamp((1.0 - vLife) * (1.0 + 2.0 * aSeed) * uPx * 35.0 / -mv.z, 0.0, 10.0 * uPx);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `varying float vLife; varying float vSeed;
      void main(){
        float d = length(gl_PointCoord - 0.5); if (d > 0.5 || vLife >= 1.0) discard;
        vec3 hot = vec3(1.0, 0.95, 0.75), mid = vec3(1.0, 0.45, 0.12), cool = vec3(0.45, 0.08, 0.04);
        vec3 c = vLife < 0.25 ? mix(hot, mid, vLife / 0.25) : mix(mid, cool, (vLife - 0.25) / 0.75);
        float a = (1.0 - vLife) * (1.0 - d * 2.0);
        gl_FragColor = vec4(c * a * 0.9, 1.0);
      }`,
  }));
  debris.frustumCulled = false;
  group.add(debris);

  const light = new THREE.PointLight(0xffb070, 0, 60 * scale, 1.5);
  group.add(light);

  return {
    group,
    update(age) {
      group.visible = age >= 0 && age < 14;
      if (!group.visible) { light.intensity = 0; return; }
      const fs = (age < 0.06 ? age / 0.06 : 1) * 2.2 * scale;
      flash.scale.setScalar(Math.max(fs, 1e-3));
      flash.material.opacity = Math.exp(-age * 5);
      const fr = 0.45 * scale * (1 - Math.exp(-age * 4));
      fireball.scale.setScalar(Math.max(fr, 1e-3));
      if (ground) fireball.scale.y *= 0.6;
      fireball.material.opacity = 0.35 * Math.exp(-age * 2.2);
      const sr = 8 * scale * (1 - Math.exp(-age * 0.9));
      shock.scale.set(sr, ground ? sr * 0.35 : sr, sr);
      shockMat.uniforms.uOpacity.value = 0.35 * Math.exp(-age * 1.1);
      pu.uAge.value = age;
      light.intensity = 200 * scale * Math.exp(-age * 4);
    },
  };
}

let killFx = null, impactFx = null;
if (killPos) {
  const inh = velocity(P.threat, killT - 0.1).add(velocity(P.kv, killT - 0.1)).multiplyScalar(0.18);
  killFx = createExplosion(killPos, { inherit: inh, scale: 1 });
}
if (impactPos) impactFx = createExplosion(impactPos, { ground: true, scale: 0.8 });

// ---------------------------------------------------------------------------
// LOS + radar beam lines
// ---------------------------------------------------------------------------
function dynLine(color, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  l.frustumCulled = false;
  scene.add(l);
  return l;
}
const losLine = dynLine(hex(PALETTE.accent), 0.32);
const beamLine = dynLine(hex(PALETTE.safe), 0.14);
function setLine(l, a, b) { const p = l.geometry.attributes.position; p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z); p.needsUpdate = true; }

export { trailUniforms, trailMinDefaults, tubeMeshes, trackTrail, radarUniforms, radarPts, killFx, impactFx, losLine, beamLine, setLine };
