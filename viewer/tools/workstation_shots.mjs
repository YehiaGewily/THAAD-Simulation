// Screenshots of the live workstation for the README and docs.
//
//   python -m thaad_sim --serve --no-browser        (in another terminal)
//   node tools/workstation_shots.mjs [base-url] [out-dir]
//
// Defaults: http://127.0.0.1:8765 and docs/figures/. Loads the default preset
// (advanced config, seed 13), pauses in terminal homing, and captures each tab.
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, frames, launch } from './browser.mjs';

const [base = 'http://127.0.0.1:8765', outDir = path.join(ROOT, 'docs', 'figures')] = process.argv.slice(2);
const AT = 60; // s, KV in terminal homing a few seconds before the kill
fs.mkdirSync(outDir, { recursive: true });

const browser = await launch({ width: 1600, height: 900 });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

const idle = () => page.waitForFunction(
  () => document.getElementById('sim-meta-tag').textContent && !document.body.classList.contains('is-busy'),
  { timeout: 60000 },
);
const tab = async (ws) => {
  await page.click(`#workspace-tabs button[data-ws="${ws}"]`);
  await frames(page, 20);
};
const shot = async (name) => {
  const file = path.join(outDir, `workstation_${name}.png`);
  await page.screenshot({ path: file });
  console.log('wrote', path.relative(ROOT, file));
};

await page.goto(base, { waitUntil: 'load' });
await idle();

// Pause, then seek to the moment worth showing
await page.evaluate((t) => {
  const play = document.getElementById('btn-play');
  if (play.classList.contains('is-playing')) play.click();
  const slider = document.getElementById('timeline-slider');
  slider.value = t;
  slider.dispatchEvent(new Event('input'));
}, AT);
// Camera items live in a closed dropdown: open it, pick Midpoint, close it
await page.evaluate(() => { document.getElementById('cam-mode-btns').closest('details').open = true; });
await page.click('#cam-mode-btns button[data-cam="both"]');
await page.evaluate(() => document.querySelectorAll('details.dropdown[open]').forEach((d) => { d.open = false; }));
await frames(page, 30);
await shot('engagement');

await tab('charts');
await page.click('#chart-filter-tabs button[data-cgrp="all"]');
await frames(page, 20);
await shot('analysis');

await tab('batch');
await page.waitForSelector('#batch-content .b-tile', { timeout: 30000 });
await frames(page, 10);
await shot('montecarlo');

await tab('methods');
await page.waitForFunction(() => document.querySelectorAll('#methods-content .katex').length > 0, { timeout: 15000 }).catch(() => {});
await frames(page, 10);
await shot('methods');

await browser.close();
if (errors.length) {
  console.error('Console errors:\n  ' + errors.join('\n  '));
  process.exit(1);
}
