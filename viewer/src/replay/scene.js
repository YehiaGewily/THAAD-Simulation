// The static world: renderer, camera, sky, terrain, clouds, defended dome,
// battery, scientific reference geometry and the 3D labels.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { D, defR, impactPos, killPos, pipPos, threatStart } from './data.js';
import { glowSprite, glowTexture, fresnelMaterial, ringTexture } from './materials.js';
import { PALETTE, hex } from './palette.js';

// ============================================================================
// RENDERER / SCENE
// ============================================================================
const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
container.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
Object.assign(labelRenderer.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
container.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
const FOG_COLOR = new THREE.Color().setRGB(0.028, 0.05, 0.09);
scene.fog = new THREE.Fog(FOG_COLOR, 160, 700);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxDistance = 600;
controls.minDistance = 0.05;
controls.maxPolarAngle = Math.PI * 0.495;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.85, 0.5, 0.16);
composer.addPass(bloom);
composer.addPass(new OutputPass());

const SUN_DIR = new THREE.Vector3(-0.55, 0.22, -0.8).normalize();
scene.add(new THREE.HemisphereLight(0x5a78b0, 0x0a0806, 0.7));
const sun = new THREE.DirectionalLight(0xffd7b0, 2.2);
sun.position.copy(SUN_DIR).multiplyScalar(100);
scene.add(sun);

// ---------------------------------------------------------------------------
// Sky dome + stars (follows the camera)
// ---------------------------------------------------------------------------
const sky = new THREE.Group();
sky.add(new THREE.Mesh(
  new THREE.SphereGeometry(1800, 48, 24),
  new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { uSun: { value: SUN_DIR } },
    vertexShader: `varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
      uniform vec3 uSun; varying vec3 vDir;
      void main(){
        vec3 d = normalize(vDir); float h = d.y;
        vec3 zen = vec3(0.002, 0.004, 0.012);
        vec3 hor = vec3(0.028, 0.05, 0.09);
        vec3 col = mix(hor, zen, pow(clamp(h, 0.0, 1.0), 0.4));
        float s = max(dot(d, uSun), 0.0);
        col += vec3(1.0, 0.42, 0.16) * pow(s, 6.0) * 0.22 * (1.0 - clamp(h * 2.5, 0.0, 1.0));
        col += vec3(1.0, 0.7, 0.4) * pow(s, 400.0) * 3.0;
        if (h < 0.0) col = mix(hor, vec3(0.006, 0.008, 0.012), clamp(-h * 6.0, 0.0, 1.0));
        gl_FragColor = vec4(col, 1.0);
      }`,
  }),
));
{
  const n = 3000, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = Math.random(), v = Math.random() * 0.95 + 0.05;
    const th = 2 * Math.PI * u, y = v, r = Math.sqrt(1 - y * y);
    pos.set([Math.cos(th) * r * 1700, y * 1700, Math.sin(th) * r * 1700], 3 * i);
    const b = 0.06 + Math.pow(Math.random(), 5) * 0.9;
    col.set([b * 0.85, b * 0.92, b], 3 * i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const starTex = glowTexture([[0, 'rgba(255,255,255,1)'], [0.35, 'rgba(255,255,255,0.4)'], [1, 'rgba(255,255,255,0)']]);
  sky.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 3, map: starTex, sizeAttenuation: false, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })));
}
sky.children.forEach((c) => { c.renderOrder = -10; c.frustumCulled = false; });
scene.add(sky);

// ---------------------------------------------------------------------------
// Terrain: flat defended plain (sim ground is z = 0) rising into hills
// ---------------------------------------------------------------------------
function hash2(x, y) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y) { let s = 0, a = 0.5, f = 1; for (let o = 0; o < 5; o++) { s += a * vnoise(x * f, y * f); f *= 2.03; a *= 0.5; } return s; }
const smooth = (e0, e1, x) => { const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); };

const terrainMat = (() => {
  const geo = new THREE.PlaneGeometry(520, 520, 300, 300);
  geo.rotateX(-Math.PI / 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i), r = Math.hypot(x, z);
    const ridge = fbm(x / 38 + 7.3, z / 38 - 3.1);
    p.setY(i, Math.pow(ridge, 2.2) * 10 * smooth(60, 170, r));
  }
  geo.computeVertexNormals();
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uSun: { value: SUN_DIR }, uFog: { value: FOG_COLOR }, uDefR: { value: defR },
      uSweep: { value: 0 }, uSweepOn: { value: 1 }, uReal: { value: 0 }, uPip: { value: new THREE.Vector2() }, uPipOn: { value: 0 },
    },
    vertexShader: `
      varying vec3 vW; varying vec3 vN;
      void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `
      uniform vec3 uSun; uniform vec3 uFog; uniform float uDefR; uniform float uSweep; uniform float uReal;
      uniform vec2 uPip; uniform float uPipOn; uniform float uSweepOn;
      varying vec3 vW; varying vec3 vN;
      float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
        return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
      float fb(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a*vn(p); p *= 2.07; a *= 0.5; } return s; }
      float line(float c, float w){ float d = abs(fract(c - 0.5) - 0.5) / fwidth(c); return 1.0 - min(d / w, 1.0); }
      void main(){
        vec2 p = vW.xz; float r = length(p);
        float m = fb(p * 0.09) * 0.7 + fb(p * 0.6) * 0.3;
        vec3 base = mix(vec3(0.028, 0.034, 0.036), vec3(0.075, 0.072, 0.062), m);
        base = mix(base, vec3(0.11, 0.095, 0.075), clamp(vW.y / 4.0, 0.0, 1.0));
        float diff = max(dot(normalize(vN), uSun), 0.0);
        vec3 col = base * (0.45 + 1.4 * diff);

        float camD = length(cameraPosition - vW);
        float g10 = max(line(p.x / 10.0, 1.0), line(p.y / 10.0, 1.0));
        float g2 = max(line(p.x / 2.0, 1.0), line(p.y / 2.0, 1.0)) * (1.0 - smoothstep(15.0, 45.0, camD));
        float gridFade = 1.0 - smoothstep(90.0, 160.0, r);
        col += vec3(0.28, 0.38, 0.48) * (g10 * 0.16 + g2 * 0.06) * gridFade;

        float rings = line(r / 10.0, 1.2) * (1.0 - smoothstep(50.0, 62.0, r));
        col += vec3(0.3, 0.42, 0.52) * rings * 0.16;

        float dr = 1.0 - smoothstep(0.0, fwidth(r) * 2.0 + 0.04, abs(r - uDefR));
        dr *= clamp(0.12 / fwidth(r), 0.08, 1.0);  // dim the ring where it is seen edge-on
        col += vec3(0.42, 0.8, 0.62) * dr * 0.75;
        col += vec3(0.03, 0.08, 0.06) * (1.0 - smoothstep(uDefR - 0.5, uDefR, r)) * 0.5;

        float ang = atan(p.y, p.x);
        float sd = mod(uSweep - ang, 6.2831853);
        float sweep = exp(-sd * 2.5) * (1.0 - smoothstep(uDefR * 1.5, uDefR * 1.75, r)) * smoothstep(0.2, 1.5, r);
        col += vec3(0.42, 0.8, 0.62) * sweep * 0.14 * uSweepOn;

        if (uPipOn > 0.0) {
          float d = length(p - uPip);
          float ph = fract(uReal * 0.55);
          float ring = 1.0 - smoothstep(0.0, 0.18 + fwidth(d), abs(d - ph * 4.5));
          col += vec3(1.0, 0.55, 0.12) * ring * (1.0 - ph) * 2.0 * uPipOn;
          col += vec3(1.0, 0.5, 0.1) * (1.0 - smoothstep(0.0, 0.4, d)) * 1.5 * uPipOn;
          float cross = (1.0 - smoothstep(0.0, 0.05 + fwidth(d), min(abs(p.x - uPip.x), abs(p.y - uPip.y)))) * (1.0 - smoothstep(0.6, 1.4, d));
          col += vec3(1.0, 0.6, 0.2) * cross * uPipOn;
        }

        float fog = 1.0 - exp(-pow(camD * 0.0024, 1.5));
        col = mix(col, uFog, clamp(fog, 0.0, 1.0));
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(geo, mat));
  return mat;
})();
if (pipPos) terrainMat.uniforms.uPip.value.set(pipPos.x, pipPos.z);

// ---------------------------------------------------------------------------
// Cloud deck well below the engagement, for depth and scale
// ---------------------------------------------------------------------------
const cloudMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  uniforms: { uTime: { value: 0 }, uSun: { value: SUN_DIR }, uFog: { value: FOG_COLOR } },
  vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `
    uniform float uTime; uniform vec3 uSun; uniform vec3 uFog; varying vec3 vW;
    float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
      return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
    float fb(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ s += a*vn(p); p = p*2.03 + 7.1; a *= 0.5; } return s; }
    void main(){
      vec2 p = vW.xz * 0.045 + vec2(uTime * 0.004, uTime * 0.0015);
      float d = fb(p);
      float r = length(vW.xz);
      // keep the defended area clear and fade out toward the horizon
      float cover = smoothstep(0.5, 0.78, d) * smoothstep(20.0, 48.0, r) * (1.0 - smoothstep(220.0, 300.0, r));
      if (cover < 0.01) discard;
      float lit = smoothstep(0.5, 0.9, d);
      vec3 col = mix(vec3(0.07, 0.085, 0.11), vec3(0.14, 0.138, 0.133), lit);
      float sunSide = max(dot(normalize(vec3(uSun.x, 0.0, uSun.z)), normalize(vec3(vW.x, 0.0, vW.z))), 0.0);
      col += vec3(0.22, 0.12, 0.05) * pow(sunSide, 4.0) * 0.5;
      float camD = length(cameraPosition - vW);
      col = mix(col, uFog, clamp(1.0 - exp(-pow(camD * 0.0028, 1.4)), 0.0, 1.0));
      gl_FragColor = vec4(col, cover * 0.38);
    }`,
});
const clouds = new THREE.Mesh(new THREE.PlaneGeometry(640, 640).rotateX(-Math.PI / 2), cloudMat);
clouds.position.y = 9;
clouds.renderOrder = 1;
scene.add(clouds);
// ---------------------------------------------------------------------------
// Defended volume dome, battery, PIP marker
// ---------------------------------------------------------------------------
const DOME_OPACITY = 0.09;
const domeMat = fresnelMaterial(hex(PALETTE.safe), { power: 3.0, opacity: DOME_OPACITY, scan: 1, radius: defR });
const dome = new THREE.Mesh(new THREE.SphereGeometry(defR, 96, 48, 0, Math.PI * 2, 0, Math.PI / 2), domeMat);
scene.add(dome);

const metal = new THREE.MeshStandardMaterial({ color: 0x8c949c, metalness: 0.6, roughness: 0.45 });
const darkMetal = new THREE.MeshStandardMaterial({ color: 0x2a2f35, metalness: 0.5, roughness: 0.6 });

const battery = new THREE.Group();
{
  const truck = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.018, 0.022), darkMetal);
  truck.position.y = 0.009;
  const pod = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.016, 0.02), metal);
  pod.position.set(0.004, 0.03, 0); pod.rotation.z = 0.9;
  battery.add(truck, pod);
  const beacon = glowSprite(hex(PALETTE.safe), 0.02);
  beacon.position.y = 0.05;
  battery.add(beacon);
  battery.userData.beacon = beacon;
}
scene.add(battery);
// ---------------------------------------------------------------------------
// Reference geometry for the scientific style: drop lines, axis labels,
// an altitude scale at the battery and exact intercept/impact markers
// ---------------------------------------------------------------------------
function dropLine(color) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color, dashSize: 0.6, gapSize: 0.45, transparent: true, opacity: 0.5, depthWrite: false }));
  l.frustumCulled = false;
  scene.add(l);
  return l;
}
function setDrop(l, p) {
  const a = l.geometry.attributes.position;
  a.setXYZ(0, p.x, p.y, p.z); a.setXYZ(1, p.x, 0, p.z); a.needsUpdate = true;
  l.computeLineDistances();
}
const dropT = dropLine(hex(PALETTE.threat));
const dropI = dropLine(hex(PALETTE.kv));

const gridLabels = [];
// kind 'alt' labels the altitude scale (perspective views); 'map' labels the ground axes (top-down view only)
function gridLabel(text, pos, kind) {
  const outer = document.createElement('div'); outer.className = 'label3d gridlbl';
  const inner = document.createElement('div'); inner.textContent = text; outer.appendChild(inner);
  const o = new CSS2DObject(outer); o.position.copy(pos); scene.add(o);
  o.userData.kind = kind;
  o.userData.inner = inner;
  gridLabels.push(o);
  return o;
}
const refGeo = new THREE.Group();
{
  const maxAlt = Math.ceil(Math.max(threatStart.y, 10) / 10) * 10;
  const pts = [0, 0, 0, 0, maxAlt, 0];
  for (let a = 10; a <= maxAlt; a += 10) {
    pts.push(-0.4, a, 0, 0.4, a, 0);
    if (a % 20 === 0) gridLabel(`${a} km`, new THREE.Vector3(0.6, a, 0), 'alt');
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  refGeo.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xd6e4f0, transparent: true, opacity: 0.28, depthWrite: false })));
  for (let k = 10; k <= 60; k += 10) {
    gridLabel(`${k} km E`, new THREE.Vector3(k, 0, 0), 'map');
    gridLabel(`${k} km N`, new THREE.Vector3(0, 0, -k), 'map');
  }
  scene.add(refGeo);
}

const RING = ringTexture();
function marker(color, pos) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: RING, color, sizeAttenuation: false, depthTest: false, transparent: true }));
  sp.scale.setScalar(0.045);
  sp.renderOrder = 5;
  sp.visible = false;
  if (pos) sp.position.copy(pos);
  scene.add(sp);
  return sp;
}
const killMarker = marker(hex(PALETTE.safe), killPos);
const impactMarker = marker(hex(PALETTE.threat), impactPos);

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------
function label(color, side = 'right') {
  const outer = document.createElement('div'); outer.className = side === 'left' ? 'label3d left' : 'label3d';
  const inner = document.createElement('div'); inner.style.setProperty('--c', color); outer.appendChild(inner);
  const obj = new CSS2DObject(outer); scene.add(obj);
  obj.userData.inner = inner;
  return obj;
}
const lblThreat = label(PALETTE.threat);
const lblKV = label(PALETTE.kv, 'left');
const lblBattery = label(PALETTE.safe);
lblBattery.position.set(0, 0.05, 0);
lblBattery.userData.inner.innerHTML = 'THAAD battery';
const lblPip = label(PALETTE.accent);
if (pipPos) { lblPip.position.copy(pipPos); lblPip.userData.inner.innerHTML = 'PIP<small>predicted impact point</small>'; }
const lblKill = label(PALETTE.safe);
if (killPos) { lblKill.position.copy(killPos); lblKill.userData.inner.innerHTML = `Intercept<small>${(D.kill.alt / 1000).toFixed(1)} km · miss ${D.kill.dist.toFixed(2)} m</small>`; }
const lblImpact = label(PALETTE.threat);
if (impactPos) {
  lblImpact.position.copy(impactPos);
  const r = impactPos.length();
  lblImpact.userData.inner.innerHTML = `Impact<small>${r.toFixed(1)} km from battery${r > defR ? ' · outside DA' : ''}</small>`;
}

export {
  renderer, labelRenderer, scene, camera, controls, composer, bloom, sky, terrainMat, cloudMat, clouds,
  DOME_OPACITY, domeMat, battery, metal, darkMetal, smooth, dropT, dropI, setDrop, gridLabels, refGeo,
  killMarker, impactMarker, lblThreat, lblKV, lblBattery, lblPip, lblKill, lblImpact,
};
