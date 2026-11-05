// Things that change with replay time: trails, radar returns, explosions and
// the line-of-sight and radar-beam lines.
import * as THREE from 'three';
import {
  D, GNC, G_KM, KM, N, P, T, burnT, computeBallisticArc, cpaInfo, cpaT, impactPos, killPos, killT,
  launchIdx, maxG, sample, velocity,
} from './data.js';
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

// ---------------------------------------------------------------------------
// Academic GNC Layer: 3D velocity & acceleration vectors, ZEM bar,
// dynamic PIP ground migration trace, 3-sigma ground footprint & ballistic arcs
// ---------------------------------------------------------------------------
const gncGroup = new THREE.Group();
scene.add(gncGroup);

// 1. Dynamic PIP migration ground trail (shows drag-mismatch drift over time)
const pipValidFirst = GNC.pipValid.findIndex((v) => v > 0);
const pipTrailGroup = pipValidFirst >= 0
  ? buildTrail(GNC.pipTrack, pipValidFirst, lastIdx, { colorA: hex(PALETTE.accent), fade: 90, min: 0.5, lineGain: 1.1 })
  : new THREE.Group();
gncGroup.add(pipTrailGroup);

// 2. Live PIP 3-sigma ground impact footprint ellipse + center crosshair
const pipFootprint = (() => {
  const grp = new THREE.Group();
  const pts = [];
  for (let i = 0; i <= 64; i++) {
    const a = (i / 64) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
  }
  const ring = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: hex(PALETTE.accent), transparent: true, opacity: 0.55, depthWrite: false }),
  );
  const crossGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.35, 0, 0), new THREE.Vector3(0.35, 0, 0),
    new THREE.Vector3(0, 0, -0.35), new THREE.Vector3(0, 0, 0.35),
  ]);
  const cross = new THREE.LineSegments(
    crossGeo,
    new THREE.LineBasicMaterial({ color: hex(PALETTE.accent), transparent: true, opacity: 0.8, depthWrite: false }),
  );
  grp.add(ring, cross);
  grp.userData = { ring, cross };
  grp.visible = false;
  gncGroup.add(grp);
  return grp;
})();

// 3. Vacuum vs. Dragged 3D ballistic projection arcs (from threat to ground)
const ARC_STEPS = 48;
function makeArcLine(color, dashSize, gapSize, opacity) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((ARC_STEPS + 1) * 3), 3));
  const line = new THREE.Line(
    geo,
    new THREE.LineDashedMaterial({ color, dashSize, gapSize, transparent: true, opacity, depthWrite: false }),
  );
  line.frustumCulled = false;
  line.visible = false;
  gncGroup.add(line);
  return line;
}
const vacArcLine = makeArcLine(hex(PALETTE.accent), 0.7, 0.5, 0.45);
const dragArcLine = makeArcLine(hex(PALETTE.threat), 0.35, 0.35, 0.38);

// 4. 3D Vehicle Vectors (v_T, v_I, commanded lateral acceleration a_c, and ZEM bar)
const velThreatArrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), 1, hex(PALETTE.threat), 0.22, 0.11);
const velKvArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, hex(PALETTE.kv), 0.22, 0.11);
const acKvArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, hex(PALETTE.accent), 0.26, 0.13);
[velThreatArrow, velKvArrow, acKvArrow].forEach((a) => {
  a.line.material.transparent = true; a.line.material.opacity = 0.85; a.line.material.depthWrite = false;
  a.cone.material.transparent = true; a.cone.material.opacity = 0.9; a.cone.material.depthWrite = false;
  a.visible = false;
  gncGroup.add(a);
});

const zemLine = dynLine(hex(PALETTE.cov), 0.75);
gncGroup.add(zemLine);

// 5. CPA Endgame Kill-Radius Sphere & Miss Chord (shown near/after CPA)
const cpaGroup = (() => {
  const grp = new THREE.Group();
  const rKm = Math.max((D.killRadius || 10) * KM, 0.01);
  // Drawn with a minimum visual radius when camera is far, or exact scale when close
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshBasicMaterial({ color: hex(PALETTE.safe), wireframe: true, transparent: true, opacity: 0.32, depthWrite: false }),
  );
  const chordGeo = new THREE.BufferGeometry();
  chordGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const chord = new THREE.Line(
    chordGeo,
    new THREE.LineBasicMaterial({ color: hex(PALETTE.accent), transparent: true, opacity: 0.9, depthTest: false }),
  );
  chord.frustumCulled = false;
  grp.add(sphere, chord);
  grp.userData = { sphere, chord, rKm };
  grp.visible = false;
  scene.add(grp);
  return grp;
})();

const tmpDir = new THREE.Vector3(), tmpAc = new THREE.Vector3(), tmpZemEnd = new THREE.Vector3();
const colAccent = new THREE.Color(hex(PALETTE.accent)), colThreat = new THREE.Color(hex(PALETTE.threat));
let lastArcFrame = -1;

export function updateGncEffects({ visible, time, fi, tp, ip, tv, iv, launched, stage, g, threatAlive, kvAlive, camPos }) {
  gncGroup.visible = visible;
  if (visible) {
    // Live PIP footprint & ballistic projection arcs while threat is airborne
    const hasPip = threatAlive && GNC.pipValid[fi] > 0;
    pipFootprint.visible = hasPip;
    if (hasPip) {
      const a = 3 * fi;
      pipFootprint.position.set(GNC.pipTrack[a], 0.03, GNC.pipTrack[a + 2]);
      const se = THREE.MathUtils.clamp(GNC.pipSigE[fi], 0.18, 22);
      const sn = THREE.MathUtils.clamp(GNC.pipSigN[fi], 0.18, 22);
      pipFootprint.userData.ring.scale.set(se, 1, sn);
    }

    // Recompute ballistic projection arcs every 4 frames to keep frame time tiny
    vacArcLine.visible = hasPip;
    dragArcLine.visible = hasPip;
    if (hasPip && Math.abs(fi - lastArcFrame) >= 4) {
      lastArcFrame = fi;
      const fillArc = (line, pts) => {
        if (!pts || pts.length <= ARC_STEPS) { line.visible = false; return; }
        const pos = line.geometry.attributes.position;
        for (let i = 0; i <= ARC_STEPS; i++) pos.setXYZ(i, pts[i].x, pts[i].y, pts[i].z);
        pos.needsUpdate = true;
        line.computeLineDistances();
      };
      fillArc(vacArcLine, computeBallisticArc(fi, false, ARC_STEPS));
      fillArc(dragArcLine, computeBallisticArc(fi, true, ARC_STEPS));
    }

    // 3D Velocity & Commanded Acceleration Arrows (scaled by camera distance so they read at any zoom)
    if (threatAlive && tv.lengthSq() > 1e-6) {
      const dT = THREE.MathUtils.clamp(camPos.distanceTo(tp) * 0.085, 0.06, 4.2);
      velThreatArrow.visible = true;
      velThreatArrow.position.copy(tp);
      velThreatArrow.setDirection(tmpDir.copy(tv).normalize());
      velThreatArrow.setLength(dT, dT * 0.22, dT * 0.1);
    } else velThreatArrow.visible = false;

    if (launched && kvAlive && iv.lengthSq() > 1e-6) {
      const dI = THREE.MathUtils.clamp(camPos.distanceTo(ip) * 0.085, 0.06, 4.2);
      velKvArrow.visible = true;
      velKvArrow.position.copy(ip);
      velKvArrow.setDirection(tmpDir.copy(iv).normalize());
      velKvArrow.setLength(dI, dI * 0.22, dI * 0.1);

      const a = 3 * fi;
      tmpAc.set(GNC.acScene[a], GNC.acScene[a + 1], GNC.acScene[a + 2]);
      const dvRem = GNC.dvUsed[fi];
      const exhausted = stage === 2 && D.params?.kv_divert_dv != null && dvRem >= D.params.kv_divert_dv - 0.5;
      if (tmpAc.lengthSq() > 0.25 && !exhausted) {
        acKvArrow.visible = true;
        acKvArrow.position.copy(ip);
        acKvArrow.setDirection(tmpDir.copy(tmpAc).normalize());
        const gFrac = THREE.MathUtils.clamp(g / (maxG || 30), 0.18, 1.15);
        const len = dI * 0.95 * gFrac;
        acKvArrow.setLength(len, len * 0.26, len * 0.12);
        acKvArrow.setColor(g >= (maxG || 30) * 0.98 ? colThreat : colAccent);
      } else acKvArrow.visible = false;

      // ZEM bar at the current threat position showing the miss offset vector
      const zemM = GNC.zemTrue[fi];
      if (Number.isFinite(zemM) && zemM > 0.05 && zemM < 15000) {
        zemLine.visible = true;
        // Exaggerate small ZEM slightly at macro distances so the miss direction is visible
        const zemScale = camPos.distanceTo(tp) > 4 ? Math.max(1, Math.min(10, 0.8 / Math.max(zemM * KM, 0.02))) : 1;
        tmpZemEnd.set(tp.x - GNC.zemScene[a] * zemScale, tp.y - GNC.zemScene[a + 1] * zemScale, tp.z - GNC.zemScene[a + 2] * zemScale);
        setLine(zemLine, tp, tmpZemEnd);
      } else zemLine.visible = false;
    } else {
      velKvArrow.visible = false;
      acKvArrow.visible = false;
      zemLine.visible = false;
    }
  }

  // CPA Endgame Kill-Radius Sphere & Miss Chord: visible near CPA (within 1.5 s) or after a missed CPA
  const showCpa = visible && cpaInfo != null && (Math.abs(time - cpaT) < 1.5 || (!killT && time >= cpaT));
  cpaGroup.visible = showCpa;
  if (showCpa) {
    const { sphere, chord, rKm } = cpaGroup.userData;
    const atCpa = time >= cpaT;
    const pT = atCpa ? cpaInfo.tPos : tp;
    const pI = atCpa ? cpaInfo.iPos : ip;
    sphere.position.copy(pT);
    // Keep the kill-radius sphere visible even when the camera is several hundred metres away
    const camD = camPos.distanceTo(pT);
    const visR = Math.max(rKm, Math.min(camD * 0.022, 0.18));
    sphere.scale.setScalar(visR);
    setLine(chord, pI, pT);
  }
}

export {
  trailUniforms, trailMinDefaults, tubeMeshes, trackTrail, radarUniforms, radarPts,
  killFx, impactFx, losLine, beamLine, setLine, gncGroup, cpaGroup,
};
