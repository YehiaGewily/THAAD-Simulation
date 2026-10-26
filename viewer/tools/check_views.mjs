// Smoke test of a replay in a real browser: screenshots of every camera in
// both rendering styles, every keyboard shortcut checked against the viewer's
// state, and any console errors.
//
//   node tools/check_views.mjs ../site/replay.html [out-dir] [time]
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, frames, launch, openReplay, serve } from './browser.mjs';

const [file = 'site/replay.html', outDir = 'view-check', at = '40'] = process.argv.slice(2);
const rel = path.relative(ROOT, path.resolve(file)).replaceAll('\\', '/');
fs.mkdirSync(outDir, { recursive: true });

const server = await serve();
const browser = await launch();
const page = await browser.newPage();
const failures = [];
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) failures.push(what); };

const CAMS = ['cinematic', 'orbit', 'threat', 'kv', 'top'];
for (const style of ['scientific', 'cinematic']) {
  for (const cam of CAMS) {
    const errors = await openReplay(page, `${server.url}/${rel}?view=${style}-${cam}#t=${at}&cam=${cam}&paused&style=${style}`);
    const st = await page.evaluate(() => ({ cam: window.thaadReplay.state.camMode, sci: window.thaadReplay.state.sciStyle }));
    check(st.cam === cam && st.sci === (style === 'scientific') && !errors.length,
      `${style} / ${cam}${errors.length ? `: ${errors.join('; ')}` : ''}`);
    await page.screenshot({ path: path.join(outDir, `${style}_${cam}.png`) });
  }
}

// Keyboard shortcuts, each checked against the state it should change
await openReplay(page, `${server.url}/${rel}#t=${at}&paused`);
const state = () => page.evaluate(() => JSON.parse(JSON.stringify({ ...window.thaadReplay.state, focus: document.body.classList.contains('focus'), help: !document.getElementById('help').hidden })));
const press = async (key, opts) => { await page.keyboard.press(key, opts); await frames(page, 3); return state(); };
let s = await state();
for (const [key, cam] of [['o', 'orbit'], ['t', 'threat'], ['i', 'kv'], ['v', 'top'], ['c', 'cinematic']]) {
  s = await press(key);
  check(s.camMode === cam, `key ${key.toUpperCase()} -> ${cam} camera`);
}
for (const [key, layer] of [['r', 'radar'], ['k', 'track'], ['e', 'cov'], ['l', 'los'], ['h', 'labels'], ['m', 'slowmo']]) {
  const before = s.layers[layer];
  s = await press(key);
  check(s.layers[layer] === !before, `key ${key.toUpperCase()} toggles ${layer}`);
  s = await press(key);
}
s = await press('g'); check(!s.sciStyle, 'key G -> cinematic style');
s = await press('g'); check(s.sciStyle, 'key G -> scientific style');
s = await press('a'); check(s.analysisOpen, 'key A opens Analysis');
await page.screenshot({ path: path.join(outDir, 'analysis.png') });
s = await press('a'); check(!s.analysisOpen, 'key A closes Analysis');
s = await press('f'); check(s.focus, 'key F -> focus mode');
s = await press('f'); check(!s.focus, 'key F leaves focus mode');
s = await press('?'); check(s.help, 'key ? opens help');
s = await press('Escape'); check(!s.help, 'Esc closes help');
const t0 = s.simTime;
s = await press('ArrowRight'); check(Math.abs(s.simTime - t0 - 1) < 1e-6, 'Right arrow steps +1 s');
await page.keyboard.down('Shift');
s = await press('ArrowLeft');
await page.keyboard.up('Shift');
check(Math.abs(s.simTime - (t0 + 1 - 5)) < 1e-6, 'Shift+Left steps -5 s');
const [tStart, tEnd] = await page.evaluate(() => window.thaadReplay.duration);
s = await press('End'); check(s.simTime === tEnd, 'End jumps to the end');
s = await press('Home'); check(s.simTime === tStart, 'Home jumps to the start');
for (let k = 1; k <= 6; k++) { s = await press(String(k)); check(s.speed === [0.25, 1, 2, 5, 10, 20][k - 1], `key ${k} sets speed`); }
const playing = s.playing;
s = await press(' '); check(s.playing === !playing, 'Space toggles play');

await browser.close();
server.close();
console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nAll checks passed');
process.exit(failures.length ? 1 : 0);
