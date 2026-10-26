// The scene and UI palette in one place. styles.css derives its colour tokens
// from these values at startup (see applyPalette), so the 3D scene, the HUD
// and the figures cannot drift apart. docs/methods.md reports the colour-vision
// and contrast checks run on it (scripts/palette_check.py).
export const PALETTE = {
  threat: '#f0705a',   // threat, its trail and labels; failure states
  kv: '#9ccfe8',       // interceptor / kill vehicle after burnout
  accent: '#f2b45c',   // UI accent, boost phase, line of sight, PIP
  safe: '#7cc4a4',     // defended area, battery, kill marker; success states
  track: '#a9fcba',    // Kalman track
  radar: '#8e99ab',    // raw radar returns: neutral, the filtered track carries the colour
  cov: '#ab79f5',      // Kalman 3-sigma position ellipsoid
};

// Series hues for the analysis figures, on the dark chart surface
export const SERIES = { threat: '#e8664f', kv: '#4598cc', eng: '#d9a03a' };

export const hex = (css) => parseInt(css.slice(1), 16);

export function applyPalette(root = document.documentElement) {
  for (const k of ['threat', 'kv', 'accent', 'safe']) root.style.setProperty(`--${k}`, PALETTE[k]);
}
