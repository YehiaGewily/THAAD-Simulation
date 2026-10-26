// Textures and materials shared by the scene, the vehicles and the effects.
import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Shared materials / textures
// ---------------------------------------------------------------------------
function glowTexture(stops) {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  stops.forEach(([o, col]) => grd.addColorStop(o, col));
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const GLOW = glowTexture([[0, 'rgba(255,255,255,1)'], [0.12, 'rgba(255,255,255,0.9)'], [0.35, 'rgba(255,255,255,0.25)'], [1, 'rgba(255,255,255,0)']]);
const SOFT = glowTexture([[0, 'rgba(255,255,255,0.9)'], [0.5, 'rgba(255,255,255,0.25)'], [1, 'rgba(255,255,255,0)']]);

function glowSprite(color, scale, { attenuate = false, tex = GLOW, opacity = 1 } = {}) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    sizeAttenuation: attenuate, opacity, fog: false,
  }));
  s.scale.setScalar(scale);
  return s;
}

function fresnelMaterial(color, { power = 2.5, opacity = 1, scan = 0, radius = 1 } = {}) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uCol: { value: new THREE.Color(color) }, uPow: { value: power }, uOpacity: { value: opacity }, uScanOn: { value: scan }, uScan: { value: 0 }, uR: { value: radius } },
    vertexShader: `varying vec3 vN; varying vec3 vW;
      void main(){ vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform vec3 uCol; uniform float uPow; uniform float uOpacity; uniform float uScanOn; uniform float uScan; uniform float uR;
      varying vec3 vN; varying vec3 vW;
      void main(){
        vec3 v = normalize(cameraPosition - vW);
        float fr = pow(1.0 - abs(dot(v, normalize(vN))), uPow);
        float band = uScanOn * exp(-pow((vW.y / uR - uScan) * 28.0, 2.0)) * 0.4;
        gl_FragColor = vec4(uCol * (fr + band) * uOpacity, 1.0);
      }`,
  });
}
function ringTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  g.strokeStyle = '#fff'; g.lineWidth = 7;
  g.beginPath(); g.arc(64, 64, 38, 0, Math.PI * 2); g.stroke();
  g.lineWidth = 5;
  [[64, 4, 64, 34], [64, 94, 64, 124], [4, 64, 34, 64], [94, 64, 124, 64]].forEach(([a, b, c2, d]) => { g.beginPath(); g.moveTo(a, b); g.lineTo(c2, d); g.stroke(); });
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

export { glowTexture, GLOW, SOFT, glowSprite, fresnelMaterial, ringTexture };
