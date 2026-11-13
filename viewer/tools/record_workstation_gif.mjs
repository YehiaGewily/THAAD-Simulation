// Records the README animation from the live workstation: loads the default
// preset (advanced config, seed 13), steps through the engagement in sim time
// (slowing down around the intercept), screenshots each step in headless
// Chrome, and encodes a GIF with ffmpeg.
//
//   python -m thaad_sim --serve --no-browser        (in another terminal)
//   node tools/record_workstation_gif.mjs [base-url] [out.gif]
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT, frames, launch } from './browser.mjs';

const [base = 'http://127.0.0.1:8765', out = path.join(ROOT, 'docs', 'figures', 'workstation.gif')] = process.argv.slice(2);
const WIDTH = 1280, HEIGHT = 720, GIF_WIDTH = 960, FPS = 12;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'thaad-ws-gif-'));

const browser = await launch({ width: WIDTH, height: HEIGHT });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

await page.goto(base, { waitUntil: 'load' });
await page.waitForFunction(
  () => document.getElementById('sim-meta-tag').textContent && !document.body.classList.contains('is-busy'),
  { timeout: 60000 },
);

// Pause, pick the Midpoint camera (it tracks between the two vehicles)
await page.evaluate(() => {
  const play = document.getElementById('btn-play');
  if (play.classList.contains('is-playing')) play.click();
  document.getElementById('cam-mode-btns').closest('details').open = true;
});
await page.click('#cam-mode-btns button[data-cam="both"]');
await page.evaluate(() => document.querySelectorAll('details.dropdown[open]').forEach((d) => { d.open = false; }));

// Dolly in so the vehicles fill more of the frame (the Midpoint camera keeps
// this distance while it tracks)
const box = await (await page.$('#viewport-3d canvas')).boundingBox();
await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.45);
for (let k = 0; k < 6; k++) {
  await page.mouse.wheel({ deltaY: -250 });
  await frames(page, 4);
}
await page.mouse.move(box.x + 5, box.y + 5);
await frames(page, 60);

// End of the engagement: the kill (or closest approach) from the event log
const { endT, tMax } = await page.evaluate(() => {
  const slider = document.getElementById('timeline-slider');
  const items = [...document.querySelectorAll('#event-log-list li')];
  const hit = items.find((li) => /Hit-to-kill|Closest approach|Ground impact/.test(li.textContent));
  return { endT: hit ? +hit.dataset.t : +slider.max - 5, tMax: +slider.max };
});

// Sim times: brisk through the flight, slow around the intercept, then hold
const times = [];
for (let t = 0; t < endT - 3; t += 0.9) times.push(t);
for (let t = endT - 3; t < endT + 1; t += 0.1) times.push(t);
for (let t = endT + 1; t <= Math.min(tMax, endT + 3); t += 0.25) times.push(t);

const seek = (t) => page.evaluate((v) => {
  const slider = document.getElementById('timeline-slider');
  slider.value = v;
  slider.dispatchEvent(new Event('input'));
}, t);

await seek(times[0]);
await frames(page, 40);
for (let i = 0; i < times.length; i++) {
  await seek(times[i]);
  await frames(page, 3);
  await page.screenshot({ path: path.join(tmp, `f${String(i).padStart(4, '0')}.png`) });
}
await browser.close();
if (errors.length) throw new Error('Console errors:\n' + errors.join('\n'));

// Hold the last frame for a second, then encode with a palette made for these frames
const last = path.join(tmp, `f${String(times.length - 1).padStart(4, '0')}.png`);
for (let k = 0; k < FPS; k++) fs.copyFileSync(last, path.join(tmp, `f${String(times.length + k).padStart(4, '0')}.png`));
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(tmp, 'f%04d.png'),
  '-vf', `fps=${FPS},scale=${GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle`,
  '-loop', '0', path.resolve(out)]);
fs.rmSync(tmp, { recursive: true, force: true });
console.log(`Wrote ${path.relative(ROOT, path.resolve(out))}: ${times.length + FPS} frames, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB`);
