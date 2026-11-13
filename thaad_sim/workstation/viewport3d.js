// Clean, modular 3D Tactical Arena & Optical Seeker Renderer (Three.js)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const KM = 0.001;

export function enuToScene(x, y, z, out = new THREE.Vector3()) {
  return out.set(x * KM, z * KM, -y * KM);
}

export function locateFrame(T, time) {
  const n = T.length;
  if (!n || time <= T[0]) return [0, 0];
  if (time >= T[n - 1]) return [Math.max(0, n - 2), 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (T[m] <= time) lo = m;
    else hi = m;
  }
  const dt = Math.max(T[lo + 1] - T[lo], 1e-9);
  return [lo, (time - T[lo]) / dt];
}

export function sampleVec3(flatArr, i, f, out = new THREE.Vector3()) {
  const a = 3 * i, b = 3 * (i + 1);
  const x = flatArr[a] + (flatArr[b] - flatArr[a]) * f;
  const y = flatArr[a + 1] + (flatArr[b + 1] - flatArr[a + 1]) * f;
  const z = flatArr[a + 2] + (flatArr[b + 2] - flatArr[a + 2]) * f;
  return enuToScene(x, y, z, out);
}

// Compute 3x3 symmetric positive-definite matrix square root via Jacobi eigendecomposition
function sym3Sqrt(xx, yy, zz, xy, xz, yz, outMat4) {
  const A = [
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iter = 0; iter < 10; iter++) {
    let p = 0, q = 1, maxOff = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = Math.abs(A[0][2]); }
    if (Math.abs(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = Math.abs(A[1][2]); }
    if (maxOff < 1e-10) break;
    const app = A[p][p], aqq = A[q][q], apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let k = 0; k < 3; k++) {
      const arp = A[k][p], arq = A[k][q];
      A[k][p] = c * arp - s * arq;
      A[k][q] = s * arp + c * arq;
    }
    for (let k = 0; k < 3; k++) {
      const apk = A[p][k], aqk = A[q][k];
      A[p][k] = c * apk - s * aqk;
      A[q][k] = s * apk + c * aqk;
    }
    A[p][q] = A[q][p] = 0;
    for (let k = 0; k < 3; k++) {
      const vkp = V[k][p], vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }
  const s0 = Math.sqrt(Math.max(A[0][0], 1e-6));
  const s1 = Math.sqrt(Math.max(A[1][1], 1e-6));
  const s2 = Math.sqrt(Math.max(A[2][2], 1e-6));
  // M = V * diag(s) * V^T in ENU, then map ENU (x, y, z) -> Scene (x, z, -y)
  const M = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const S = [s0, s1, s2];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      M[r][c] = V[r][0] * S[0] * V[c][0] + V[r][1] * S[1] * V[c][1] + V[r][2] * S[2] * V[c][2];
    }
  }
  // Scene basis permutation P = [[1,0,0],[0,0,1],[0,-1,0]]
  const P = [
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
  ];
  const MS = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          sum += P[r][i] * M[i][j] * P[c][j];
        }
      }
      MS[r][c] = sum;
    }
  }
  outMat4.set(
    MS[0][0], MS[0][1], MS[0][2], 0,
    MS[1][0], MS[1][1], MS[1][2], 0,
    MS[2][0], MS[2][1], MS[2][2], 0,
    0, 0, 0, 1
  );
}

export class TacticalViewport3D {
  constructor(containerEl, labelsContainerEl, seekerContainerEl) {
    this.container = containerEl;
    this.labelsEl = labelsContainerEl;
    this.seekerContainer = seekerContainerEl;

    this.visualScale = 12;
    this.camMode = 'arena';
    this.layers = {
      radar: true,
      track: true,
      cov: true,
      vectors: true,
      pip: true,
      los: true,
      dome: true,
      grid: true,
    };

    this.data = null;
    this.compareRuns = [];

    this._initScene();
    this._initArenaGeometry();
    this._initVehicles();
    this._initTrajectoriesAndVectors();
    this._initLabels();
    this._bindResize();
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0c0e);
    this.scene.fog = new THREE.FogExp2(0x0b0c0e, 0.0028);

    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 500;
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.02, 600);
    this.camera.position.set(35, 28, 48);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.container.insertBefore(this.renderer.domElement, this.container.firstChild);

    // Seeker dedicated WebGL canvas so it works in any tab/layout cleanly
    this.seekerCanvas = document.createElement('canvas');
    this.seekerCanvas.id = 'canvas-seeker';
    this.seekerContainer.insertBefore(this.seekerCanvas, this.seekerContainer.firstChild);
    this.seekerRenderer = new THREE.WebGLRenderer({ canvas: this.seekerCanvas, antialias: true });
    this.seekerRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.seekerCam = new THREE.PerspectiveCamera(2.5, 16 / 9, 0.005, 300);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxDistance = 260;
    this.controls.minDistance = 0.2;
    this.controls.target.set(8, 14, -8);

    // Lighting
    this.scene.add(new THREE.HemisphereLight(0x9b9da2, 0x111214, 1.35));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(40, 80, 50);
    this.scene.add(dir);
  }

  _initArenaGeometry() {
    this.gridGroup = new THREE.Group();
    this.scene.add(this.gridGroup);

    // Ground grid (120 km x 120 km, 10 km major divisions)
    const grid = new THREE.GridHelper(120, 24, 0x303237, 0x1a1b1e);
    grid.position.y = 0.01;
    this.gridGroup.add(grid);

    // Concentric ground range rings (10, 20, 30, 40, 50 km)
    const ringMat = new THREE.LineBasicMaterial({ color: 0x303237, transparent: true, opacity: 0.55 });
    [10, 20, 30, 40, 50].forEach((r) => {
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * r, 0.02, Math.sin(a) * r));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      this.gridGroup.add(new THREE.Line(geo, ringMat));
    });

    // Vertical altitude reference pole at (-30, 0, -30) up to 70 km
    const polePts = [new THREE.Vector3(-30, 0, -30), new THREE.Vector3(-30, 70, -30)];
    for (let a = 10; a <= 70; a += 10) {
      polePts.push(new THREE.Vector3(-30, a, -30), new THREE.Vector3(-28.5, a, -30));
      polePts.push(new THREE.Vector3(-30, a, -30));
    }
    const poleGeo = new THREE.BufferGeometry().setFromPoints(polePts);
    this.gridGroup.add(new THREE.Line(poleGeo, new THREE.LineBasicMaterial({ color: 0x4a4d53 })));

    // Defended perimeter ring + translucent dome
    this.domeGroup = new THREE.Group();
    this.scene.add(this.domeGroup);

    const defRingPts = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      defRingPts.push(new THREE.Vector3(Math.cos(a), 0.04, Math.sin(a)));
    }
    this.defRing = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(defRingPts),
      new THREE.LineBasicMaterial({ color: 0x6bbd8c, transparent: true, opacity: 0.85 })
    );
    this.domeGroup.add(this.defRing);

    const domeGeo = new THREE.SphereGeometry(1, 40, 20, 0, Math.PI * 2, 0, Math.PI * 0.5);
    this.defDome = new THREE.Mesh(
      domeGeo,
      new THREE.MeshBasicMaterial({ color: 0x6bbd8c, transparent: true, opacity: 0.045, wireframe: true, depthWrite: false })
    );
    this.domeGroup.add(this.defDome);

    // Battery & AN/TPY-2 Radar at origin
    const batGroup = new THREE.Group();
    const pad = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.55, 0.12, 24),
      new THREE.MeshStandardMaterial({ color: 0x1f2125, roughness: 0.6 })
    );
    pad.position.y = 0.06;
    batGroup.add(pad);
    const radarDish = new THREE.Mesh(
      new THREE.BoxGeometry(0.35, 0.22, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x6bbd8c, emissive: 0x1d3f2c })
    );
    radarDish.position.set(0, 0.22, 0);
    radarDish.rotation.x = -0.35;
    batGroup.add(radarDish);
    this.scene.add(batGroup);
  }

  _initVehicles() {
    // Threat Reentry Vehicle
    this.threatGroup = new THREE.Group();
    this.threatModel = new THREE.Group();
    const rvCone = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.36, 16),
      new THREE.MeshStandardMaterial({ color: 0xe0594a, emissive: 0x4a1a15, roughness: 0.3 })
    );
    rvCone.rotation.x = Math.PI * 0.5;
    this.threatModel.add(rvCone);

    const heatGlow = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xec8a7f, transparent: true, opacity: 0.55 })
    );
    heatGlow.position.z = 0.12;
    this.threatModel.add(heatGlow);
    this.threatGroup.add(this.threatModel);
    this.scene.add(this.threatGroup);

    // 3-sigma Kalman Covariance Ellipsoid
    this.covMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 20, 14),
      new THREE.MeshBasicMaterial({ color: 0xd4a24c, wireframe: true, transparent: true, opacity: 0.42 })
    );
    this.covMesh.matrixAutoUpdate = false;
    this.scene.add(this.covMesh);

    // Interceptor / Kill Vehicle
    this.kvGroup = new THREE.Group();
    this.kvModel = new THREE.Group();

    this.kvHead = new THREE.Mesh(
      new THREE.ConeGeometry(0.065, 0.22, 16),
      new THREE.MeshStandardMaterial({ color: 0x5aa6d6, emissive: 0x1b3a50, roughness: 0.25 })
    );
    this.kvHead.rotation.x = Math.PI * 0.5;
    this.kvHead.position.z = 0.08;
    this.kvModel.add(this.kvHead);

    this.boosterBody = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.065, 0.28, 16),
      new THREE.MeshStandardMaterial({ color: 0xd6d7d9, roughness: 0.4 })
    );
    this.boosterBody.rotation.x = Math.PI * 0.5;
    this.boosterBody.position.z = -0.16;
    this.kvModel.add(this.boosterBody);

    this.boostFlame = new THREE.Mesh(
      new THREE.ConeGeometry(0.08, 0.42, 12),
      new THREE.MeshBasicMaterial({ color: 0xd4a24c, transparent: true, opacity: 0.85 })
    );
    this.boostFlame.rotation.x = -Math.PI * 0.5;
    this.boostFlame.position.z = -0.48;
    this.kvModel.add(this.boostFlame);

    // Lateral DACS divert jet
    this.dacsJet = new THREE.Mesh(
      new THREE.ConeGeometry(0.045, 0.28, 10),
      new THREE.MeshBasicMaterial({ color: 0x5aa6d6, transparent: true, opacity: 0.85 })
    );
    this.dacsJet.rotation.x = Math.PI * 0.5;
    this.kvGroup.add(this.dacsJet);

    this.kvGroup.add(this.kvModel);
    this.scene.add(this.kvGroup);

    // Kill / CPA marker sphere
    this.cpaMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.35, 20, 20),
      new THREE.MeshBasicMaterial({ color: 0x6bbd8c, wireframe: true, transparent: true, opacity: 0.85 })
    );
    this.cpaMarker.visible = false;
    this.scene.add(this.cpaMarker);
  }

  _initTrajectoriesAndVectors() {
    this.trajGroup = new THREE.Group();
    this.scene.add(this.trajGroup);

    // Dynamic lines helper
    const makeLine = (color, opacity = 1, dashed = false) => {
      const geo = new THREE.BufferGeometry();
      const mat = dashed
        ? new THREE.LineDashedMaterial({ color, transparent: opacity < 1, opacity, dashSize: 0.4, gapSize: 0.25 })
        : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
      return new THREE.Line(geo, mat);
    };

    this.lineThreatFull = makeLine(0xe0594a, 0.22);
    this.lineThreatActive = makeLine(0xe0594a, 0.95);
    this.lineKvFull = makeLine(0x5aa6d6, 0.22);
    this.lineKvBoost = makeLine(0xd4a24c, 0.95);
    this.lineKvCoast = makeLine(0x5aa6d6, 0.95);
    this.lineKfTrack = makeLine(0xe0b866, 0.7);
    this.linePipTrace = makeLine(0xd4a24c, 0.75);
    this.lineBallisticTrue = makeLine(0xe0594a, 0.65, true);
    this.lineBallisticPip = makeLine(0xd4a24c, 0.75, true);

    this.lineLos = makeLine(0x9b9da2, 0.5, true);
    this.lineRadarBeam = makeLine(0x6bbd8c, 0.35);
    this.dropThreat = makeLine(0xe0594a, 0.35);
    this.dropKv = makeLine(0x5aa6d6, 0.35);

    // Ground PIP 3-sigma ellipse
    this.pipEllipse = makeLine(0xd4a24c, 0.85);
    this.pipMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.18, 0.32, 24),
      new THREE.MeshBasicMaterial({ color: 0xd4a24c, side: THREE.DoubleSide })
    );
    this.pipMarker.rotation.x = -Math.PI * 0.5;
    this.pipMarker.position.y = 0.04;

    this.trueImpactMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.18, 0.32, 24),
      new THREE.MeshBasicMaterial({ color: 0xe0594a, side: THREE.DoubleSide })
    );
    this.trueImpactMarker.rotation.x = -Math.PI * 0.5;
    this.trueImpactMarker.position.y = 0.04;

    // Radar raw point cloud
    this.radarPointsGeo = new THREE.BufferGeometry();
    this.radarPoints = new THREE.Points(
      this.radarPointsGeo,
      new THREE.PointsMaterial({ color: 0x8ccfa6, size: 0.22, transparent: true, opacity: 0.55 })
    );

    // 3D GNC Arrows (v_T, v_I, a_c, ZEM)
    this.arrowVt = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, 0xe0594a, 0.35, 0.2);
    this.arrowVi = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, 0x5aa6d6, 0.35, 0.2);
    this.arrowAc = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, 0xd4a24c, 0.4, 0.22);
    this.arrowZem = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, 0xa08ad4, 0.35, 0.2);

    this.trajGroup.add(
      this.lineThreatFull, this.lineThreatActive,
      this.lineKvFull, this.lineKvBoost, this.lineKvCoast,
      this.lineKfTrack, this.linePipTrace,
      this.lineBallisticTrue, this.lineBallisticPip,
      this.lineLos, this.lineRadarBeam,
      this.dropThreat, this.dropKv,
      this.pipEllipse, this.pipMarker, this.trueImpactMarker,
      this.radarPoints,
      this.arrowVt, this.arrowVi, this.arrowAc, this.arrowZem
    );

    // Comparison trajectories group
    this.compareGroup = new THREE.Group();
    this.scene.add(this.compareGroup);
  }

  _initLabels() {
    this.lblThreat = this._makeLabel('threat');
    this.lblKv = this._makeLabel('kv');
    this.lblPip = this._makeLabel('pip');
    this.lblCpa = this._makeLabel('cpa');
  }

  _makeLabel(cls) {
    const el = document.createElement('div');
    el.className = `lbl-3d ${cls}`;
    el.style.display = 'none';
    this.labelsEl.appendChild(el);
    return el;
  }

  _placeLabel(el, worldPos, html) {
    if (!html) {
      el.style.display = 'none';
      return;
    }
    const v = worldPos.clone().project(this.camera);
    if (v.z > 1 || v.z < -1) {
      el.style.display = 'none';
      return;
    }
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    const x = (v.x * 0.5 + 0.5) * w;
    const y = (-v.y * 0.5 + 0.5) * h;
    if (x < 10 || x > w - 10 || y < 10 || y > h - 10) {
      el.style.display = 'none';
      return;
    }
    el.style.display = 'block';
    el.style.left = `${x.toFixed(1)}px`;
    el.style.top = `${y.toFixed(1)}px`;
    el.innerHTML = html;
  }

  _bindResize() {
    const onResize = () => {
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      if (w > 10 && h > 10) {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
      }
      const sw = this.seekerContainer.clientWidth;
      const sh = this.seekerContainer.clientHeight;
      if (sw > 10 && sh > 10) {
        this.seekerCam.aspect = sw / sh;
        this.seekerCam.updateProjectionMatrix();
        this.seekerRenderer.setSize(sw, sh);
      }
    };
    if ('ResizeObserver' in window) {
      const ro = new ResizeObserver(onResize);
      ro.observe(this.container);
      ro.observe(this.seekerContainer);
    }
    window.addEventListener('resize', onResize);
  }

  setData(data, compareRuns = []) {
    this.data = data;
    this.compareRuns = compareRuns;
    const N = data.t.length;
    const defR = (data.defendedRadius || 30000) * KM;
    this.defRing.scale.set(defR, 1, defR);
    this.defDome.scale.set(defR, defR, defR);

    // Prebuild full Vector3 arrays
    this.ptsThreat = [];
    this.ptsKv = [];
    this.ptsTrack = [];
    this.ptsRadar = [];
    this.ptsPip = [];
    const drv = data.derived;

    for (let i = 0; i < N; i++) {
      const a = 3 * i;
      this.ptsThreat.push(enuToScene(data.threat[a], data.threat[a + 1], data.threat[a + 2]));
      this.ptsKv.push(enuToScene(data.interceptor[a], data.interceptor[a + 1], data.interceptor[a + 2]));
      this.ptsTrack.push(enuToScene(data.track[a], data.track[a + 1], data.track[a + 2]));
      this.ptsRadar.push(enuToScene(data.radar[a], data.radar[a + 1], data.radar[a + 2]));
      if (drv.pipValid[i]) {
        this.ptsPip.push(enuToScene(drv.pipTrack[a], drv.pipTrack[a + 1], 40));
      }
    }

    this.lineThreatFull.geometry.setFromPoints(this.ptsThreat);
    const lIdx = Math.max(0, drv.launchIdx);
    this.lineKvFull.geometry.setFromPoints(drv.launchIdx >= 0 ? this.ptsKv.slice(lIdx) : []);
    this.radarPointsGeo.setFromPoints(this.ptsRadar);

    if (drv.trueImpact) {
      this.trueImpactMarker.position.copy(enuToScene(drv.trueImpact[0], drv.trueImpact[1], 40));
    }

    // Build comparison trajectories if active
    while (this.compareGroup.children.length) {
      this.compareGroup.remove(this.compareGroup.children[0]);
    }
    compareRuns.forEach((run) => {
      const col = new THREE.Color(run.compareColor || '#a08ad4');
      const rN = run.t.length;
      const lStart = Math.max(0, run.derived.launchIdx);
      if (run.derived.launchIdx >= 0) {
        const kvPts = [];
        for (let i = lStart; i < rN; i++) {
          const a = 3 * i;
          kvPts.push(enuToScene(run.interceptor[a], run.interceptor[a + 1], run.interceptor[a + 2]));
        }
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(kvPts),
          new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.85 })
        );
        this.compareGroup.add(line);
      }
    });

    this.setCameraMode(this.camMode, true);
  }

  setCameraMode(mode, snap = false) {
    this.camMode = mode;
    if (!this.data) return;
    const drv = this.data.derived;
    if (mode === 'arena') {
      const startPt = this.ptsThreat[0] || new THREE.Vector3(25, 55, -35);
      const mid = new THREE.Vector3(startPt.x * 0.45, Math.max(18, startPt.y * 0.42), startPt.z * 0.45);
      this.controls.target.copy(mid);
      if (snap) {
        const span = Math.max(55, startPt.length() * 1.05);
        this.camera.position.set(mid.x + span * 0.72, mid.y + span * 0.32, mid.z + span * 0.78);
      }
    } else if (mode === 'top') {
      this.controls.target.set(0, 0, 0);
      this.camera.position.set(0, 85, 0.01);
    } else if (mode === 'side') {
      const startPt = this.ptsThreat[0] || new THREE.Vector3(25, 45, -25);
      this.controls.target.set(startPt.x * 0.4, startPt.y * 0.4, startPt.z * 0.4);
      this.camera.position.set(startPt.x * 0.4, startPt.y * 0.42, startPt.z * 0.4 + 65);
    } else if (mode === 'cpa' && this.data.cpa) {
      const cpaPt = enuToScene(this.data.cpa.tPos[0], this.data.cpa.tPos[1], this.data.cpa.tPos[2]);
      this.controls.target.copy(cpaPt);
      this.camera.position.copy(cpaPt).add(new THREE.Vector3(2.2, 1.2, 2.8));
    }
    this.controls.update();
  }

  update(time) {
    if (!this.data) return;
    const D = this.data;
    const drv = D.derived;
    const T = D.t;
    const N = T.length;
    const [fi, frac] = locateFrame(T, time);
    const idx = Math.min(fi + 1, N - 1);

    const tp = sampleVec3(D.threat, fi, frac);
    const ip = sampleVec3(D.interceptor, fi, frac);
    const tv = sampleVec3(D.threatVel, fi, frac);
    const iv = sampleVec3(D.interceptorVel, fi, frac);

    const launched = drv.launchT != null && time >= drv.launchT;
    const killed = drv.killT != null && time >= drv.killT;
    const impacted = drv.impactT != null && time >= drv.impactT;
    const threatAlive = !killed && !impacted;
    const kvAlive = launched && !killed;
    const stg = D.stage[idx];
    const vScale = this.visualScale * 0.08;

    // Layer visibilities
    this.gridGroup.visible = this.layers.grid;
    this.domeGroup.visible = this.layers.dome;
    this.radarPoints.visible = this.layers.radar;
    this.lineKfTrack.visible = this.layers.track;

    // Active trajectory slices
    this.lineThreatActive.geometry.setFromPoints(this.ptsThreat.slice(0, idx + 1));
    if (launched && drv.launchIdx >= 0) {
      const bEnd = drv.burnIdx >= 0 ? Math.min(idx + 1, drv.burnIdx + 1) : idx + 1;
      this.lineKvBoost.geometry.setFromPoints(this.ptsKv.slice(drv.launchIdx, bEnd));
      this.lineKvCoast.geometry.setFromPoints(
        drv.burnIdx >= 0 && idx >= drv.burnIdx ? this.ptsKv.slice(drv.burnIdx, idx + 1) : []
      );
      this.lineKvBoost.visible = true;
      this.lineKvCoast.visible = true;
    } else {
      this.lineKvBoost.visible = false;
      this.lineKvCoast.visible = false;
    }

    if (this.layers.track) {
      this.lineKfTrack.geometry.setFromPoints(this.ptsTrack.slice(0, idx + 1));
    }
    if (this.layers.radar) {
      this.radarPointsGeo.setDrawRange(0, idx + 1);
    }

    // Threat pose & scale
    this.threatGroup.visible = threatAlive;
    this.threatGroup.position.copy(tp);
    this.threatGroup.scale.setScalar(vScale);
    if (tv.lengthSq() > 1e-9) {
      this.threatModel.lookAt(tp.clone().add(tv));
    }

    // 3-sigma covariance ellipsoid (scaled by visualScale so it's readable)
    this.covMesh.visible = this.layers.cov && threatAlive;
    if (this.covMesh.visible) {
      const c6 = 6 * fi;
      const covScale = 3.0 * KM * Math.max(2, this.visualScale * 0.65);
      const s2 = covScale * covScale;
      const mat = new THREE.Matrix4();
      sym3Sqrt(
        D.kfCov[c6] * s2, D.kfCov[c6 + 1] * s2, D.kfCov[c6 + 2] * s2,
        D.kfCov[c6 + 3] * s2, D.kfCov[c6 + 4] * s2, D.kfCov[c6 + 5] * s2,
        mat
      );
      mat.setPosition(tp);
      this.covMesh.matrix.copy(mat);
    }

    // Interceptor / KV pose & scale
    this.kvGroup.visible = !killed;
    this.kvGroup.position.copy(ip);
    this.kvGroup.scale.setScalar(vScale);
    if (launched && iv.lengthSq() > 1e-9) {
      this.kvModel.lookAt(ip.clone().add(iv));
    } else {
      this.kvModel.lookAt(ip.clone().add(new THREE.Vector3(0.2, 1, -0.2)));
    }
    this.boosterBody.visible = !launched || stg === 1;
    this.boostFlame.visible = launched && stg === 1;

    // DACS lateral thruster jet during Stage 2
    const gLoad = D.g[idx] || 0;
    const k3 = 3 * fi;
    const acScene = enuToScene(drv.acVec[k3], drv.acVec[k3 + 1], drv.acVec[k3 + 2]);
    this.dacsJet.visible = launched && stg === 2 && gLoad > 0.5 && !killed;
    if (this.dacsJet.visible && acScene.lengthSq() > 1e-6) {
      const jetDir = acScene.clone().normalize().negate();
      this.dacsJet.lookAt(ip.clone().add(jetDir));
    }

    // Drop lines
    this.dropThreat.visible = threatAlive && this.layers.grid;
    if (this.dropThreat.visible) {
      this.dropThreat.geometry.setFromPoints([tp, new THREE.Vector3(tp.x, 0, tp.z)]);
    }
    this.dropKv.visible = kvAlive && this.layers.grid;
    if (this.dropKv.visible) {
      this.dropKv.geometry.setFromPoints([ip, new THREE.Vector3(ip.x, 0, ip.z)]);
    }

    // LOS & Radar Beam
    this.lineLos.visible = this.layers.los && kvAlive && threatAlive;
    if (this.lineLos.visible) {
      this.lineLos.geometry.setFromPoints([ip, tp]);
      this.lineLos.computeLineDistances();
    }
    this.lineRadarBeam.visible = this.layers.los && threatAlive;
    if (this.lineRadarBeam.visible) {
      this.lineRadarBeam.geometry.setFromPoints([new THREE.Vector3(0, 0.1, 0), tp]);
    }

    // GNC Vectors (v_T, v_I, a_c, ZEM)
    const showVec = this.layers.vectors;
    this.arrowVt.visible = showVec && threatAlive && tv.lengthSq() > 1e-6;
    if (this.arrowVt.visible) {
      this.arrowVt.position.copy(tp);
      this.arrowVt.setDirection(tv.clone().normalize());
      this.arrowVt.setLength(Math.min(6, Math.max(1.2, tv.length() * 1.4)), 0.4, 0.22);
    }
    this.arrowVi.visible = showVec && kvAlive && iv.lengthSq() > 1e-6;
    if (this.arrowVi.visible) {
      this.arrowVi.position.copy(ip);
      this.arrowVi.setDirection(iv.clone().normalize());
      this.arrowVi.setLength(Math.min(6, Math.max(1.2, iv.length() * 1.4)), 0.4, 0.22);
    }
    this.arrowAc.visible = showVec && kvAlive && acScene.lengthSq() > 1e-6;
    if (this.arrowAc.visible) {
      this.arrowAc.position.copy(ip);
      this.arrowAc.setDirection(acScene.clone().normalize());
      this.arrowAc.setLength(Math.min(5, Math.max(0.9, gLoad * 0.12)), 0.38, 0.2);
    }
    const zemScene = enuToScene(drv.zemVec[k3], drv.zemVec[k3 + 1], drv.zemVec[k3 + 2]);
    this.arrowZem.visible = showVec && kvAlive && zemScene.lengthSq() > 1e-8;
    if (this.arrowZem.visible) {
      this.arrowZem.position.copy(tp);
      this.arrowZem.setDirection(zemScene.clone().normalize());
      this.arrowZem.setLength(Math.min(5, Math.max(0.8, zemScene.length() * 2)), 0.35, 0.18);
    }

    // PIP Footprint & Ground trace
    const showPip = this.layers.pip;
    this.trueImpactMarker.visible = showPip;
    this.pipMarker.visible = showPip && !!drv.pipValid[fi] && threatAlive;
    this.pipEllipse.visible = this.pipMarker.visible;
    this.linePipTrace.visible = showPip;
    this.lineBallisticTrue.visible = showPip && threatAlive;

    if (this.pipMarker.visible) {
      const pipPt = enuToScene(drv.pipTrack[k3], drv.pipTrack[k3 + 1], 40);
      this.pipMarker.position.copy(pipPt);
      const sigE = (drv.pipSigE[fi] || 100) * KM;
      const sigN = (drv.pipSigN[fi] || 100) * KM;
      const ellPts = [];
      for (let s = 0; s <= 48; s++) {
        const a = (s / 48) * Math.PI * 2;
        ellPts.push(new THREE.Vector3(pipPt.x + Math.cos(a) * sigE, 0.05, pipPt.z + Math.sin(a) * sigN));
      }
      this.pipEllipse.geometry.setFromPoints(ellPts);
      this.lineBallisticPip.visible = true;
      this.lineBallisticPip.geometry.setFromPoints([tp, pipPt]);
      this.lineBallisticPip.computeLineDistances();
    } else {
      this.lineBallisticPip.visible = false;
    }

    if (this.lineBallisticTrue.visible && drv.trueImpact) {
      const impPt = enuToScene(drv.trueImpact[0], drv.trueImpact[1], 0);
      this.lineBallisticTrue.geometry.setFromPoints([tp, impPt]);
      this.lineBallisticTrue.computeLineDistances();
    }

    // CPA / Kill marker
    if (D.cpa && (killed || (drv.endEventT != null && time >= D.cpa.time))) {
      this.cpaMarker.visible = true;
      this.cpaMarker.position.copy(enuToScene(D.cpa.tPos[0], D.cpa.tPos[1], D.cpa.tPos[2]));
    } else {
      this.cpaMarker.visible = false;
    }

    // Camera follow modes (smoothly shifts OrbitControls target while preserving user orbit rotation/zoom)
    if (this.camMode === 'threat') {
      const delta = tp.clone().sub(this.controls.target);
      this.camera.position.add(delta);
      this.controls.target.copy(tp);
    } else if (this.camMode === 'kv' && launched) {
      const delta = ip.clone().sub(this.controls.target);
      this.camera.position.add(delta);
      this.controls.target.copy(ip);
    } else if (this.camMode === 'both') {
      const mid = launched ? tp.clone().add(ip).multiplyScalar(0.5) : tp;
      const delta = mid.clone().sub(this.controls.target);
      this.camera.position.add(delta);
      this.controls.target.copy(mid);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    // 3D Floating labels
    this._placeLabel(
      this.lblThreat,
      tp,
      threatAlive ? `<b>THREAT</b> ${tp.y.toFixed(1)}km · ${(tv.length() * 1000).toFixed(0)}m/s` : null
    );
    this._placeLabel(
      this.lblKv,
      ip,
      kvAlive ? `<b>${stg === 1 ? 'BOOSTER' : 'KILL VEHICLE'}</b> ${ip.y.toFixed(1)}km · ${gLoad.toFixed(1)}g` : null
    );
    if (this.pipMarker.visible) {
      this._placeLabel(this.lblPip, this.pipMarker.position, `<b>PIP</b> ±${((drv.pipSigE[fi] || 0) / 1000).toFixed(1)}km`);
    } else {
      this.lblPip.style.display = 'none';
    }
    if (this.cpaMarker.visible && D.cpa) {
      this._placeLabel(
        this.lblCpa,
        this.cpaMarker.position,
        `<b>${D.outcome === 'KILL' ? 'HIT-TO-KILL' : 'CPA MISS'}</b> ${D.cpa.dist.toFixed(2)} m`
      );
    } else {
      this.lblCpa.style.display = 'none';
    }

    // Render Optical Seeker View
    this._renderSeeker(tp, ip, tv, iv, launched, threatAlive, kvAlive, stg, fi);
  }

  _renderSeeker(tp, ip, tv, iv, launched, threatAlive, kvAlive, stg, fi) {
    const sw = this.seekerContainer.clientWidth;
    const sh = this.seekerContainer.clientHeight;
    if (sw < 10 || sh < 10) return;

    const drv = this.data.derived;
    const sepKm = tp.distanceTo(ip);
    let fov = 2.0;

    if (!launched) {
      this.seekerCam.position.set(0, 0.15, 0);
      fov = 2.0;
      this.seekerCam.lookAt(tp);
    } else if (threatAlive && kvAlive) {
      const lookDir = tp.clone().sub(ip);
      const d = lookDir.length();
      if (d > 1e-4) lookDir.divideScalar(d);
      this.seekerCam.position.copy(ip).addScaledVector(lookDir, 0.05);
      fov = THREE.MathUtils.clamp(d > 0.05 ? 10 / d : 6.0, 0.8, 6.0);
      this.seekerCam.lookAt(tp);
    } else if (this.data.cpa) {
      const cpaPt = enuToScene(this.data.cpa.tPos[0], this.data.cpa.tPos[1], this.data.cpa.tPos[2]);
      this.seekerCam.position.copy(cpaPt).add(new THREE.Vector3(1.2, 0.6, 1.5));
      fov = 3.5;
      this.seekerCam.lookAt(cpaPt);
    }

    this.seekerCam.fov = fov;
    this.seekerCam.aspect = sw / sh;
    this.seekerCam.updateProjectionMatrix();

    // Hide host KV & local KV vectors while rendering through the nosecone seeker
    const prevKvVis = this.kvGroup.visible;
    const prevViVis = this.arrowVi.visible;
    const prevAcVis = this.arrowAc.visible;
    this.kvGroup.visible = false;
    this.arrowVi.visible = false;
    this.arrowAc.visible = false;
    this.seekerRenderer.render(this.scene, this.seekerCam);
    this.kvGroup.visible = prevKvVis;
    this.arrowVi.visible = prevViVis;
    this.arrowAc.visible = prevAcVis;

    // Update Seeker HUD overlays
    const titleEl = document.getElementById('seeker-title');
    const fovEl = document.getElementById('seeker-fov');
    const hudL = document.getElementById('seeker-hud-l');
    const hudR = document.getElementById('seeker-hud-r');
    const dot = document.getElementById('ret-los-dot');
    const acLine = document.getElementById('ret-ac-vec');
    const zemRing = document.getElementById('ret-zem-ring');

    if (titleEl) {
      titleEl.textContent = !launched
        ? 'Radar track (pre-launch)'
        : stg === 1
        ? 'Boost camera'
        : 'KV seeker'
    }
    if (fovEl) fovEl.textContent = `FOV ${fov.toFixed(1)}°`;
    const zem = drv.zemTrue[fi];
    if (hudL) hudL.textContent = launched ? `RNG ${sepKm.toFixed(2)} km` : `ALT ${tp.y.toFixed(1)} km`;
    if (hudR) hudR.textContent = zem != null ? `ZEM ${zem.toFixed(1)} m` : this.data.outcome;

    const k3 = 3 * fi;
    const ax = drv.acVec[k3] || 0;
    const az = drv.acVec[k3 + 2] || 0;
    const zx = drv.zemVec[k3] || 0;
    const zz = drv.zemVec[k3 + 2] || 0;
    if (dot) {
      const ox = THREE.MathUtils.clamp(zx * 0.08, -32, 32);
      const oy = THREE.MathUtils.clamp(-zz * 0.08, -22, 22);
      dot.setAttribute('cx', (120 + ox).toFixed(1));
      dot.setAttribute('cy', (67.5 + oy).toFixed(1));
    }
    if (acLine) {
      const vx = THREE.MathUtils.clamp(ax * 0.16, -38, 38);
      const vy = THREE.MathUtils.clamp(-az * 0.16, -28, 28);
      acLine.setAttribute('x2', (120 + vx).toFixed(1));
      acLine.setAttribute('y2', (67.5 + vy).toFixed(1));
    }
    if (zemRing) {
      const r = zem != null ? THREE.MathUtils.clamp(6 + Math.log10(Math.max(1, zem)) * 7, 6, 30) : 16;
      zemRing.setAttribute('r', r.toFixed(1));
    }
  }
}

