// Builds the replay viewer and the batch dashboard into single, self-contained
// HTML templates under thaad_sim/viewer/, which the Python package fills with
// data (replay_viewer.py, dashboard.py). The built files are committed, so
// running the simulation never needs Node.
//
//   npm run build          both templates
//   npm run dev            dev server; fills the templates with the newest
//                          replay in ../replays/ and ../site/dashboard.html
import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

const ROOT = path.resolve(import.meta.dirname, '..');
const PAGES = {
  replay: { html: 'replay_template.html', token: '__REPLAY_DATA__', id: 'replay-data', dir: 'replays' },
  dashboard: { html: 'dashboard_template.html', token: '__DASHBOARD_DATA__', id: 'dashboard-data', dir: 'site' },
};

// The JSON payload of the newest generated page in ../<dir>/, for the dev server
function newestPayload({ id, dir }) {
  const folder = path.join(ROOT, dir);
  if (!fs.existsSync(folder)) return null;
  const files = fs.readdirSync(folder).filter((f) => f.endsWith('.html'))
    .map((f) => path.join(folder, f)).sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  const re = new RegExp(`<script id="${id}" type="application/json">([\s\S]*?)</script>`);
  for (const f of files) {
    const m = fs.readFileSync(f, 'utf-8').match(re);
    if (m && !m[1].startsWith('__')) return m[1];
  }
  return null;
}

function devData() {
  return {
    name: 'thaad-dev-data',
    apply: 'serve',
    transformIndexHtml(html) {
      for (const page of Object.values(PAGES)) {
        if (!html.includes(page.token)) continue;
        const data = newestPayload(page);
        if (!data) throw new Error(`No generated page with data in ${page.dir}/; generate one first`);
        return html.replace(page.token, () => data);
      }
      return html;
    },
  };
}

export default defineConfig(({ mode }) => {
  const page = PAGES[mode] ?? PAGES.replay;
  return {
    plugins: [viteSingleFile({ removeViteModuleLoader: true }), devData()],
    build: {
      outDir: path.join(ROOT, 'thaad_sim', 'viewer'),
      emptyOutDir: false,
      target: 'es2020',
      chunkSizeWarningLimit: 2000,
      rollupOptions: { input: path.resolve(import.meta.dirname, page.html) },
    },
  };
});
