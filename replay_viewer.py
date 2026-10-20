"""Browser replay for a finished engagement.

Packs the history returned by ``run()`` into the Three.js viewer template
(viewer/replay_template.html) and opens the result in the default browser.
"""
import json
import time
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "viewer" / "replay_template.html"
OUTPUT_DIR = ROOT / "replays"

STATUS_CODES = {"SEARCH": 0, "FLYING": 1, "HIT": 2}
STAGE_CODES = {"IDLE": 0, "BOOST": 1, "KV_COAST": 2}


def _flat(points):
    """List of 3-vectors (metres) -> flat list rounded to 0.1 m."""
    return np.round(np.asarray(points, dtype=float), 1).ravel().tolist()


def _flat_fine(rows):
    """Small-valued rows (errors, sigmas) -> flat list rounded to 0.01."""
    return np.round(np.asarray(rows, dtype=float), 2).ravel().tolist()


def _xyz(v):
    return None if v is None else [round(float(c), 1) for c in v]


def build_payload(hist):
    meta = hist['meta']
    status = [STATUS_CODES[s] for s in hist['status']]

    # g_hist has one entry per interceptor update, i.e. one per frame after launch
    g_hist = iter(meta['g_hist'])
    g = [round(float(next(g_hist, 0.0)), 2) if s else 0.0 for s in status]

    kill = None
    if meta['kill']:
        k = meta['kill']
        kill = {
            'time': float(k['time']),
            'alt': float(k['alt']),
            'dist': float(k['dist']),
            'closing': float(k['closing']),
            'pos': _xyz(hist['t_pos'][-1]),
        }

    return {
        'title': meta['title'],
        'mode': meta['mode'],
        'engageRule': meta['engage_rule'],
        'outcome': meta['outcome'],
        't': [round(float(t), 3) for t in hist['t']],
        'threat': _flat(hist['t_pos']),
        'interceptor': _flat(hist['i_pos']),
        'track': _flat(hist['k_pos']),
        'radar': _flat(hist['raw']),
        'threatVel': _flat(hist['t_vel']),
        'interceptorVel': _flat(hist['i_vel']),
        'kfErr': _flat_fine(hist['kf_err']),
        'kfSig': _flat_fine(hist['kf_sig']),
        'status': status,
        'stage': [STAGE_CODES[s] for s in hist['stage']],
        'g': g,
        'kill': kill,
        'launchPip': _xyz(meta['launch_pip']),
        'impact': _xyz(meta['impact']),
        'defendedRadius': meta['defended_radius'],
        'killRadius': meta['kill_radius'],
        'maxG': meta['max_g'],
        'minTrackTime': meta['min_track_time'],
        'params': {k: float(v) for k, v in meta['params'].items()},
        'seed': meta.get('seed'),
    }


def show_replay(hist, open_browser=True):
    """Write a self-contained HTML replay of ``hist`` and open it; returns the file path."""
    data = json.dumps(build_payload(hist), separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", data)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.html"
    path.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(path.as_uri())
    return path
