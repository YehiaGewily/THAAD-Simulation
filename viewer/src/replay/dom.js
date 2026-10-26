// DOM helpers, number formats, the toast and the colour tokens read by the canvases.
import { PALETTE, applyPalette } from './palette.js';

applyPalette();

export const $ = (id) => document.getElementById(id);
export const fmtKm = (km) => `${km.toFixed(2)} km`;
export const fmtSpd = (kms) => `${Math.round(kms * 1000).toLocaleString('en-US')} m/s`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const COL = {
  threat: PALETTE.threat, kv: PALETTE.kv, safe: PALETTE.safe, accent: PALETTE.accent,
  muted: cssVar('--muted'), faint: cssVar('--faint'),
};

let toastTimer = 0;
export function toast(html, ms = 1600) {
  const el = $('toast');
  el.innerHTML = html;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}
