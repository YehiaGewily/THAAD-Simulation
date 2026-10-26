// Records the README animation from a replay: steps the viewer through the
// engagement in sim time (slowing down around the intercept), screenshots each
// step in headless Chrome, and encodes a GIF with ffmpeg.
//
//   python -m thaad_sim --config configs/engineering.yaml --seed 913 --no-browser --out site/replay.html
//   node tools/record_gif.mjs ../site/replay.html ../docs/figures/demo.gif
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT, frames, launch, openReplay, serve } from './browser.mjs';

const [file = '../site/replay.html', out = '../docs/figures/demo.gif'] = process.argv.slice(2);
const WIDTH = 960, HEIGHT = 540, GIF_WIDTH = 640, FPS = 12;
const rel = path.relative(ROOT, path.resolve(file)).replaceAll('\\', '/');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'thaad-gif-'));

const server = await serve();
const browser = await launch({ width: WIDTH, height: HEIGHT });
const page = await browser.newPage();
const errors = await openReplay(page, `${server.url}/${rel}?gif#t=0&cam=cinematic&paused&focus&style=cinematic`);
if (errors.length) throw new Error(errors.join('\n'));
const { killT, tEnd } = await page.evaluate(() => {
  const [, end] = window.thaadReplay.duration;
  const kill = [...document.querySelectorAll('#log li')].map((li) => li.textContent).find((t) => t.includes('destroyed'));
  return { killT: kill ? parseFloat(kill.slice(2)) : end - 7, tEnd: end };
});

// Sim times of the frames: brisk through the flight, slow around the kill
const times = [];
for (let t = 2; t < killT - 2.5; t += 0.75) times.push(t);
for (let t = killT - 2.5; t < killT + 1.5; t += 0.1) times.push(t);
for (let t = killT + 1.5; t <= Math.min(tEnd, killT + 5); t += 0.25) times.push(t);

// Settle the camera at the first frame, then step
await page.evaluate((t) => window.thaadReplay.seek(t), times[0]);
await frames(page, 60);
for (let i = 0; i < times.length; i++) {
  await page.evaluate((t) => window.thaadReplay.seek(t), times[i]);
  await frames(page, 4);
  await page.screenshot({ path: path.join(tmp, `f${String(i).padStart(4, '0')}.png`) });
}
await browser.close();
server.close();

// Hold the last frame for a moment, then encode with a palette made for these frames
const last = path.join(tmp, `f${String(times.length - 1).padStart(4, '0')}.png`);
for (let k = 0; k < FPS; k++) fs.copyFileSync(last, path.join(tmp, `f${String(times.length + k).padStart(4, '0')}.png`));
const scale = `fps=${FPS},scale=${GIF_WIDTH}:-1:flags=lanczos`;
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(tmp, 'f%04d.png'),
  '-vf', `${scale},split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`,
  '-loop', '0', path.resolve(out)]);
fs.rmSync(tmp, { recursive: true, force: true });
console.log(`Wrote ${path.resolve(out)}: ${times.length + FPS} frames, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB`);
