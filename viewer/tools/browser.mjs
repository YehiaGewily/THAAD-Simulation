// Shared helpers for the scripted browser tools: start a local static server
// for the repository and launch an installed Chrome or Edge (puppeteer-core
// does not download a browser). Set CHROME_PATH to use another browser.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

export const ROOT = path.resolve(import.meta.dirname, '..', '..');

const CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean);

export function findBrowser() {
  const found = CANDIDATES.find((p) => fs.existsSync(p));
  if (!found) throw new Error('No Chrome/Chromium/Edge found; set CHROME_PATH');
  return found;
}

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };

/** Serve the repository root on a free localhost port; returns { url, close }. */
export function serve() {
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = path.join(ROOT, rel);
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({
    url: `http://127.0.0.1:${server.address().port}`,
    close: () => server.close(),
  })));
}

export async function launch({ width = 1280, height = 800 } = {}) {
  return puppeteer.launch({
    executablePath: findBrowser(),
    headless: true,
    defaultViewport: { width, height, deviceScaleFactor: 1 },
    args: ['--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--hide-scrollbars'],
  });
}

/** Open a replay URL and wait until the viewer has drawn its first frames. */
export async function openReplay(page, url) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__viewerReady && document.getElementById('loading').classList.contains('gone'), { timeout: 60000 });
  await frames(page, 10);
  return errors;
}

/** Let the page render n animation frames. */
export function frames(page, n) {
  return page.evaluate((k) => new Promise((done) => {
    const step = () => (k-- <= 0 ? done() : requestAnimationFrame(step));
    requestAnimationFrame(step);
  }), n);
}
