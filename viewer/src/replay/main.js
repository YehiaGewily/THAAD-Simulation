// Entry point of the replay viewer: builds the scene, wires the HUD and runs
// the frame loop. Module evaluation order follows the imports below.
import './styles.css';
import * as THREE from 'three';
import {
  D, GNC, G_KM, N, P, burnT, cpaMidPos, defR, divertBudget, impactPos, impactT, killPos, killT,
  launchT, locate, missCpaT, pipPos, sample, tEnd, terminalT, threatStart, tStart, velocity,
} from './data.js';
import { camLookTarget, hashArgs, realClock, state } from './state.js';
import { $, fmtSpd, toast } from './dom.js';
import {
  DOME_OPACITY, battery, camera, cloudMat, composer, controls, cpaMarker, domeMat, dropI, dropT, gridLabels,
  impactMarker, killMarker, labelRenderer, lblBattery, lblCpa, lblImpact, lblKV, lblKill, lblPip, lblThreat,
  refGeo, renderer, scene, seekerCam, setDrop, sky, smooth, terrainMat,
} from './scene.js';
import {
  beamLine, impactFx, killFx, losLine, radarPts, radarUniforms, setLine, trackTrail, trailUniforms,
  updateGncEffects,
} from './effects.js';
import { kvObj, spentBooster, threatObj } from './models.js';
import { covSigma3, updateCovariance } from './covariance.js';
import { desiredPos, desiredTarget, shot } from './director.js';
import { drawOverlay, drawProfile, figures } from './charts.js';
import {
  CAMERAS, drawRefs, setAnalysis, setCam, setPlaying, setSpeed, setStyle, slowFactor, toggleLayer, updateHud,
} from './ui.js';

// ============================================================================
// FRAME UPDATE
// ============================================================================
const tp = new THREE.Vector3(), ip = new THREE.Vector3(), tv = new THREE.Vector3(), iv = new THREE.Vector3();
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Vector3();
const tmpProj = new THREE.Vector3();
const followPos = new THREE.Vector3();

// Keep vehicle labels on screen: flip them to the other side near the edges
function placeLabel(lbl, pos, preferLeft) {
  tmpProj.copy(pos).project(camera);
  const x = (tmpProj.x * 0.5 + 0.5) * window.innerWidth;
  let left = preferLeft;
  if (x < 240) left = false;
  else if (x > window.innerWidth - 240) left = true;
  lbl.element.classList.toggle('left', left);
}

// Fade 3D labels while they sit under a HUD panel instead of drawing over or behind it
const SIDE_PANELS = ['brand', 'bda', 'pip-box', 'events', 'profilebox'].map((id) => $(id)).concat([...document.querySelectorAll('#telemetry .inst')]);
const FIXED_PANELS = ['mission', 'transport', 'analysis'].map((id) => $(id));
let sceneLabels = null;
function fadeLabelsUnderPanels() {
  sceneLabels ??= [lblThreat, lblKV, lblBattery, lblPip, lblKill, lblImpact, lblCpa, ...gridLabels].filter(Boolean);
  const panels = document.body.classList.contains('focus') ? FIXED_PANELS : FIXED_PANELS.concat(SIDE_PANELS);
  const rects = panels.filter(Boolean).map((el) => el.getBoundingClientRect()).filter((r) => r.width && r.height);
  sceneLabels.forEach((o) => {
    if (!o.visible) return;
    if (o.userData.kind === 'alt' && !keepOnScreen(o.userData.inner)) { o.element.style.opacity = '0'; return; }
    const r = o.userData.inner.getBoundingClientRect();
    const covered = rects.some((p) => r.left < p.right && r.right > p.left && r.top < p.bottom && r.bottom > p.top);
    o.element.style.opacity = covered ? '0' : '';
  });
}
// The altitude scale can sit at the very left edge in narrow views; slide its labels
// right just enough to be read in full, instead of losing the "60 km" part off-screen.
// When the scale is well off-screen the label is hidden rather than dragged into the scene.
const LABEL_MARGIN = 8, LABEL_MAX_SHIFT = 72;
function keepOnScreen(el) {
  const prev = el._shift || 0;
  const left = el.getBoundingClientRect().left - prev;
  const need = Math.max(0, LABEL_MARGIN - left);
  const shift = need <= LABEL_MAX_SHIFT ? need : 0;
  if (shift !== prev) { el._shift = shift; el.style.translate = shift ? `${shift.toFixed(1)}px 0` : ''; }
  return need <= LABEL_MAX_SHIFT;
}
const tmpLook = new THREE.Vector3();

function update(time, dtReal) {
  realClock.t += dtReal;
  const [fi] = locate(time);
  const stage = D.stage[Math.min(fi + 1, N - 1)];
  sample(P.threat, time, tp);
  sample(P.kv, time, ip);
  velocity(P.threat, time, tv);
  velocity(P.kv, time, iv);
  const launched = launchT != null && time >= launchT;
  const killed = killT != null && time >= killT;
  const impacted = impactT != null && time >= impactT;
  const sep = launched ? tp.distanceTo(ip) : tp.length();
  const closing = launched ? -tmpA.copy(tp).sub(ip).normalize().dot(tmpB.copy(tv).sub(iv)) : 0;
  const ttg = launched && closing > 0 ? sep / closing : Infinity;
  // After terminal homing, an opening range means the kill vehicle has passed the threat
  const missed = launched && !killed && terminalT != null && time >= terminalT && closing < 0;
  const { layers, sciStyle } = state;

  // Trails & layers
  trailUniforms.forEach((u) => { u.uNow.value = time; });
  trackTrail.visible = layers.track;
  radarPts.visible = layers.radar;
  radarUniforms.uNow.value = Math.min(time, killT ?? Infinity);

  // Threat
  const threatAlive = !killed && !impacted;
  threatObj.visible = threatAlive;
  threatObj.position.copy(tp);
  if (tv.lengthSq() > 1e-8) threatObj.userData.model.lookAt(tmpLook.copy(tp).add(tv));
  {
    const heat = threatObj.userData.heat;
    const spd = tv.length();
    const atmDensity = Math.exp(-Math.max(0, tp.y) / 18);
    heat.material.opacity = THREE.MathUtils.clamp((spd - 1.2) * 0.9, 0.15, 1) * (0.85 + Math.random() * 0.15);
    if (threatObj.userData.shockCone) {
      threatObj.userData.shockCone.material.opacity = THREE.MathUtils.clamp((spd - 1.0) * 0.16 * (0.35 + 0.65 * atmDensity), 0.04, 0.28);
    }
  }
  updateCovariance(layers.cov && threatAlive, fi, tp);

  // Interceptor
  const kvAlive = !killed;
  kvObj.visible = kvAlive;
  kvObj.position.copy(ip);
  const ud = kvObj.userData;
  if (launched) {
    if (iv.lengthSq() > 1e-8) ud.model.lookAt(tmpLook.copy(ip).add(iv));
  } else {
    ud.model.lookAt(tmpLook.set(0.2, 1, 0).add(ip));
  }
  const boosting = launched && stage === 1;
  ud.booster.visible = !launched || stage !== 2;
  ud.flame.visible = boosting;
  ud.plume.visible = boosting;
  if (boosting) {
    const f = 0.85 + Math.random() * 0.3;
    ud.flame.scale.setScalar(0.03 * f);
    ud.plume.scale.setScalar(0.08 * (0.9 + Math.random() * 0.2));
  }
  const g = launched ? (D.g[Math.min(fi + 1, N - 1)] || 0) : 0;
  const dvExhausted = divertBudget != null && (GNC.dvUsed[fi] || 0) >= divertBudget - 0.5;
  const dacsFiring = launched && stage === 2 && g > 0.8 && !dvExhausted;
  ud.divert.visible = dacsFiring && Math.random() > 0.25;
  if (ud.dacsJet) {
    ud.dacsJet.visible = dacsFiring;
    if (dacsFiring) {
      const k3 = 3 * fi;
      tmpA.set(GNC.acScene[k3], GNC.acScene[k3 + 1], GNC.acScene[k3 + 2]);
      // Project perpendicular to velocity iv so the DACS jet fires purely laterally
      if (iv.lengthSq() > 1e-8) {
        tmpB.copy(iv).normalize();
        tmpA.addScaledVector(tmpB, -tmpA.dot(tmpB));
      }
      if (tmpA.lengthSq() > 1e-6) {
        // Thruster plume exhausts opposite to the commanded acceleration vector
        tmpA.normalize().negate();
        ud.dacsJet.lookAt(tmpLook.copy(ip).add(tmpA));
        const jetScale = THREE.MathUtils.clamp(0.45 + (g / 15) * 0.85, 0.45, 1.5) * (0.85 + Math.random() * 0.3);
        ud.dacsJet.scale.set(1, 1, jetScale);
      } else {
        ud.dacsJet.visible = false;
      }
    }
  }
  ud.far.material.color.copy(boosting ? ud.farBoost : ud.farCoast);
  ud.far.scale.setScalar(launched ? (boosting ? 0.04 : 0.028) : 0.012);

  // 3D GNC vectors, dynamic PIP footprint, ballistic arcs & CPA endgame geometry
  updateGncEffects({
    visible: layers.vectors,
    time, fi, tp, ip, tv, iv, launched, stage, g, threatAlive, kvAlive,
    camPos: camera.position,
  });

  // Spent booster
  if (burnT != null && time >= burnT && time < burnT + 30) {
    const a = time - burnT, k = 0.06;
    const sb = spentBooster;
    sb.visible = true;
    sb.position.copy(sb.userData.p0).addScaledVector(sb.userData.v0, (1 - Math.exp(-k * a)) / k);
    sb.position.y -= 0.5 * G_KM * a * a;
    if (sb.position.y < 0) sb.visible = false;
    sb.userData.model.rotation.set(a * 1.3, a * 0.7, 0);
    sb.userData.ember.material.opacity = Math.max(0, 0.8 - a / 12);
  } else spentBooster.visible = false;

  // Explosions
  if (killFx) killFx.update(time - killT);
  if (impactFx) impactFx.update(time - impactT);

  // Scientific style swaps the effects for exact markers and reference geometry
  if (sciStyle) {
    if (killFx) killFx.group.visible = false;
    if (impactFx) impactFx.group.visible = false;
  }
  killMarker.visible = sciStyle && killed;
  impactMarker.visible = sciStyle && impacted;
  if (cpaMarker) cpaMarker.visible = sciStyle && missCpaT != null && time >= missCpaT;
  dropT.visible = sciStyle && threatAlive;
  if (dropT.visible) setDrop(dropT, tp);
  dropI.visible = sciStyle && launched && kvAlive;
  if (dropI.visible) setDrop(dropI, ip);
  refGeo.visible = sciStyle;

  // LOS + radar beam
  losLine.visible = layers.los && launched && !killed;
  if (losLine.visible) setLine(losLine, ip, tp);
  beamLine.visible = layers.los && threatAlive;
  if (beamLine.visible) setLine(beamLine, tmpC.set(0, 0.05, 0), tp);

  // Terrain uniforms
  terrainMat.uniforms.uSweep.value = (realClock.t * 1.4) % (Math.PI * 2);
  terrainMat.uniforms.uReal.value = realClock.t;
  cloudMat.uniforms.uTime.value = time;
  terrainMat.uniforms.uPipOn.value = pipPos && launched ? 1 : 0;
  domeMat.uniforms.uScan.value = (realClock.t * 0.12) % 1;
  // Seen from inside, the dome would wash out the whole frame
  domeMat.uniforms.uOpacity.value = DOME_OPACITY * smooth(defR * 0.95, defR * 1.3, camera.position.length());
  battery.userData.beacon.material.opacity = 0.55 + 0.45 * Math.sin(realClock.t * 4);

  // Labels
  const showLabels = layers.labels;
  lblThreat.visible = showLabels && threatAlive;
  lblThreat.position.copy(tp);
  const cov = layers.cov && covSigma3() > 0 ? ` · 3σ ${covSigma3().toFixed(0)} m` : '';
  lblThreat.userData.inner.innerHTML = `Threat<small>${tp.y.toFixed(1)} km · ${fmtSpd(tv.length())}${cov}</small>`;
  lblKV.visible = showLabels && launched && kvAlive;
  lblKV.position.copy(ip);
  lblKV.userData.inner.innerHTML = `${stage === 1 ? 'Interceptor' : 'Kill vehicle'}<small>${fmtSpd(iv.length())} · ${g.toFixed(1)} g</small>`;
  lblBattery.visible = showLabels;
  lblPip.visible = showLabels && !!pipPos && launched && !killed;
  lblKill.visible = showLabels && killed;
  lblImpact.visible = showLabels && impacted;
  if (lblCpa) lblCpa.visible = showLabels && missCpaT != null && time >= missCpaT;
  gridLabels.forEach((o) => { o.visible = sciStyle && showLabels && (o.userData.kind === 'map') === (state.camMode === 'top'); });

  // Update Seeker Picture-in-Picture camera pose
  if (layers.pip) {
    if (!launched) {
      seekerCam.position.set(0, 0.12, 0.08);
      seekerCam.fov = 1.8;
      seekerCam.lookAt(threatAlive ? tp : (impactPos || tp));
    } else if (!killed && !impacted) {
      // Mount slightly ahead of KV nose along its look direction so KV mesh doesn't occlude
      tmpA.copy(tp).sub(ip);
      const dist = tmpA.length();
      if (dist > 1e-4) tmpA.divideScalar(dist);
      else if (iv.lengthSq() > 1e-8) tmpA.copy(iv).normalize();
      else tmpA.set(0, 1, 0);
      seekerCam.position.copy(ip).addScaledVector(tmpA, 0.018);
      seekerCam.fov = THREE.MathUtils.clamp(dist > 0.05 ? Math.min(6.0, Math.max(0.8, 12 / dist)) : 6.0, 0.8, 6.0);
      seekerCam.lookAt(tp);
    } else {
      const focusPt = killed ? killPos : (cpaMidPos || impactPos || tp);
      seekerCam.position.copy(focusPt).add(tmpA.set(0.35, 0.18, 0.45));
      seekerCam.fov = 3.5;
      seekerCam.lookAt(focusPt);
    }
    seekerCam.updateProjectionMatrix();
  }

  // Camera
  if (state.camMode === 'orbit') {
    if (state.orbitFollow) {
      const fp = state.orbitFollow === 'threat' ? (killed ? killPos : impacted ? impactPos : tp) : (killed ? killPos : ip);
      if (fp) {
        // Carry the camera along with the vehicle; the user's rotation and zoom are kept
        if (state.followFresh) controls.target.copy(fp);
        else { tmpA.copy(fp).sub(followPos); camera.position.add(tmpA); controls.target.add(tmpA); }
        followPos.copy(fp);
        state.followFresh = false;
      }
    }
    controls.update();
    camLookTarget.copy(controls.target);
  } else {
    shot({ time, tp, ip, tv, launched, stage, sep, killed, impacted, ttg });
    const k = state.snapCamera ? 1 : 1 - Math.exp(-dtReal * (state.camMode === 'cinematic' ? 1.8 : 6));
    state.snapCamera = false;
    camera.position.lerp(desiredPos, k);
    camLookTarget.lerp(desiredTarget, k);
    if (camera.position.y < 0.02) camera.position.y = 0.02;
    camera.lookAt(camLookTarget);
  }
  sky.position.copy(camera.position);
  camera.updateMatrixWorld();
  placeLabel(lblThreat, tp, false);
  placeLabel(lblKV, ip, true);
  if (lblPip.visible) {
    // Point the PIP and battery labels away from each other so they never overlap
    const px = tmpProj.copy(pipPos).project(camera).x;
    const bx = tmpProj.set(0, 0.05, 0).project(camera).x;
    lblPip.element.classList.toggle('left', px < bx);
    lblBattery.element.classList.toggle('left', px >= bx);
  } else lblBattery.element.classList.remove('left');
  fadeLabelsUnderPanels();
  if (sciStyle) drawRefs();

  updateHud({ time, fi, stage, tp, ip, tv, iv, g, sep, closing, ttg, launched, killed, impacted, missed, threatAlive, kvAlive });
  drawProfile(time, tp, ip);
  if (state.analysisOpen) figures.forEach((f) => { if (f.tab === state.analysisTab) drawOverlay(f, time); });
}

// Initial framing
{
  const st = { time: tStart, tp: threatStart.clone(), ip: new THREE.Vector3(), tv: new THREE.Vector3(), launched: false, stage: 0, sep: 0, killed: false, impacted: false, ttg: Infinity };
  shot(st);
  camera.position.copy(desiredPos);
  camLookTarget.copy(desiredTarget);
  camera.lookAt(camLookTarget);
  controls.target.copy(camLookTarget);
}
setCam(CAMERAS.includes(hashArgs.get('cam')) ? hashArgs.get('cam') : 'cinematic', true);
if (hashArgs.has('paused')) setPlaying(false);
else {
  // Keyboard hints mean nothing on a touch screen; tell those visitors how to move the camera instead
  const touch = window.matchMedia?.('(pointer: coarse)').matches;
  const hint = touch ? 'Drag to orbit · pinch to zoom' : 'Press <kbd>?</kbd> for shortcuts · <kbd>F</kbd> hides the panels';
  setTimeout(() => toast(hint, 4500), 1200);
}

// A small handle for scripted use (recording the README animation, smoke tests)
window.thaadReplay = {
  get time() { return state.simTime; },
  get duration() { return [tStart, tEnd]; },
  seek(t) { state.simTime = THREE.MathUtils.clamp(t, tStart, tEnd); },
  setPlaying, setSpeed, setCam, setStyle, setAnalysis, toggleLayer,
  state,
};

// ============================================================================
// LOOP
// ============================================================================
const pipViewEl = $('pip-view');
let last = performance.now();
function frame(now) {
  const dtReal = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (state.playing && !state.scrubbing) {
    state.simTime += dtReal * state.speed * slowFactor(state.simTime);
    if (state.simTime >= tEnd) { state.simTime = tEnd; setPlaying(false); }
  }
  update(state.simTime, dtReal);
  composer.render();

  if (state.layers.pip && pipViewEl && !document.body.classList.contains('focus') && !state.analysisOpen) {
    const r = pipViewEl.getBoundingClientRect();
    if (r.width > 10 && r.height > 10) {
      const h = window.innerHeight;
      const vx = Math.round(r.left);
      const vy = Math.round(h - r.bottom);
      const vw = Math.round(r.width);
      const vh = Math.round(r.height);
      seekerCam.aspect = vw / vh;
      seekerCam.updateProjectionMatrix();
      const prevAutoClear = renderer.autoClear;
      renderer.autoClear = false;
      renderer.setViewport(vx, vy, vw, vh);
      renderer.setScissor(vx, vy, vw, vh);
      renderer.setScissorTest(true);
      renderer.clearDepth();
      sky.position.copy(seekerCam.position);
      renderer.render(scene, seekerCam);
      sky.position.copy(camera.position);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, window.innerWidth, h);
      renderer.autoClear = prevAutoClear;
    }
  }

  labelRenderer.render(scene, camera);
  requestAnimationFrame(frame);
}
window.__viewerReady = true;
requestAnimationFrame((now) => {
  last = now;
  frame(now);
  document.getElementById('loading').classList.add('gone');
});
