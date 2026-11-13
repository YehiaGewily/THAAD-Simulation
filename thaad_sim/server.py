"""FastAPI server for the live GNC workstation (thaad_sim/workstation/).

Usage:
    python -m thaad_sim --serve [--port 8765] [--no-browser]
    python -m thaad_sim.server [--port 8765] [--no-browser]

Needs the workstation extra: pip install -e ".[workstation]"
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.console import Console

from .batch import _fmt, run_jobs
from .config import SimConfig, load_config
from .dashboard import build_data, metric_histograms, rate_rows
from .engagement import simulate
from .summary import Batch
from .telemetry import build_workstation_telemetry

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSTATION_DIR = Path(__file__).resolve().parent / "workstation"
CONFIGS_DIR = ROOT_DIR / "configs"
RESULTS_DIR = ROOT_DIR / "results"

QUIET_CONSOLE = Console(quiet=True)

app = FastAPI(
    title="THAAD GNC Telemetry & Engagement Workstation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _revalidate_static(request, call_next):
    """Make browsers revalidate workstation files so edits show on reload."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


class SimOverrides(BaseModel):
    filter_type: Optional[str] = None      # "kf6" | "ekf7"
    pip_mode: Optional[str] = None         # "vacuum" | "drag"
    sensor_model: Optional[str] = None     # "cartesian" | "rae"
    guidance_law: Optional[str] = None     # "pn" | "apn" | "zem"
    launch_rule: Optional[str] = None      # "pip_gated" | "fire_on_track"
    integrator: Optional[str] = None       # "rk4" | "euler"
    dt: Optional[float] = None
    kv_divert_dv: Optional[float] = None   # <= 0 or None treated as unlimited if specified via unlimited_dv
    unlimited_dv: Optional[bool] = None
    pn_gain: Optional[float] = None
    max_g: Optional[float] = None
    tau: Optional[float] = None
    kill_radius: Optional[float] = None
    defended_radius: Optional[float] = None
    threat_drag: Optional[float] = None
    pos_noise: Optional[float] = None
    vel_noise: Optional[float] = None
    kf_q: Optional[float] = None


class SimRequest(BaseModel):
    config: str = "configs/advanced.yaml"
    seed: int = 13
    legacy_rng: bool = False
    overrides: Optional[SimOverrides] = None


class CompareItem(BaseModel):
    label: str
    color: Optional[str] = None
    config: str = "configs/engineering.yaml"
    seed: int = 13
    legacy_rng: bool = False
    overrides: Optional[SimOverrides] = None


class CompareRequest(BaseModel):
    runs: List[CompareItem]


class LiveBatchRequest(BaseModel):
    config: str = "configs/advanced.yaml"
    seed_start: int = 0
    n: int = 50
    overrides: Optional[SimOverrides] = None


def _resolve_config_path(cfg_str: str) -> Path:
    p = Path(cfg_str)
    if p.exists():
        return p.resolve()
    p2 = ROOT_DIR / cfg_str
    if p2.exists():
        return p2.resolve()
    p3 = CONFIGS_DIR / f"{cfg_str}.yaml"
    if p3.exists():
        return p3.resolve()
    raise HTTPException(status_code=404, detail=f"Config file not found: {cfg_str}")


def apply_overrides(cfg: SimConfig, ov: Optional[SimOverrides]) -> SimConfig:
    if ov is None:
        return cfg

    sim_s = cfg.sim
    sensor_s = cfg.sensor
    kalman_s = cfg.kalman
    threat_s = cfg.threat
    inter_s = cfg.interceptor
    eng_s = cfg.engagement

    if ov.integrator is not None:
        sim_s = replace(sim_s, integrator=ov.integrator)
    if ov.dt is not None and ov.dt > 0:
        sim_s = replace(sim_s, dt=float(ov.dt))

    if ov.sensor_model is not None:
        sensor_s = replace(sensor_s, model=ov.sensor_model)
    if ov.pos_noise is not None and ov.pos_noise > 0:
        sensor_s = replace(sensor_s, pos_noise=float(ov.pos_noise))
    if ov.vel_noise is not None and ov.vel_noise > 0:
        sensor_s = replace(sensor_s, vel_noise=float(ov.vel_noise))

    if ov.filter_type is not None:
        # When switching to ekf7 from a config with q=30, adjust default q unless explicitly set
        new_q = float(ov.kf_q) if ov.kf_q is not None else (5.0 if ov.filter_type == "ekf7" and kalman_s.q >= 25.0 else kalman_s.q)
        kalman_s = replace(kalman_s, filter_type=ov.filter_type, q=new_q)
    elif ov.kf_q is not None and ov.kf_q >= 0:
        kalman_s = replace(kalman_s, q=float(ov.kf_q))

    if ov.pip_mode is not None:
        kalman_s = replace(kalman_s, pip_mode=ov.pip_mode)

    if ov.threat_drag is not None and ov.threat_drag >= 0:
        threat_s = replace(threat_s, drag=float(ov.threat_drag))

    if ov.guidance_law is not None:
        inter_s = replace(inter_s, guidance_law=ov.guidance_law)
    if ov.pn_gain is not None and ov.pn_gain > 0:
        inter_s = replace(inter_s, pn_gain=float(ov.pn_gain))
    if ov.max_g is not None and ov.max_g > 0:
        inter_s = replace(inter_s, max_g=float(ov.max_g))
    if ov.tau is not None and ov.tau >= 0:
        inter_s = replace(inter_s, tau=float(ov.tau))
    if ov.unlimited_dv is True:
        inter_s = replace(inter_s, kv_divert_dv=None)
    elif ov.kv_divert_dv is not None and ov.kv_divert_dv > 0:
        inter_s = replace(inter_s, kv_divert_dv=float(ov.kv_divert_dv))

    if ov.launch_rule is not None:
        eng_s = replace(eng_s, launch_rule=ov.launch_rule)
    if ov.kill_radius is not None and ov.kill_radius > 0:
        eng_s = replace(eng_s, kill_radius=float(ov.kill_radius))
    if ov.defended_radius is not None and ov.defended_radius > 0:
        eng_s = replace(eng_s, defended_radius=float(ov.defended_radius))

    return replace(
        cfg,
        sim=sim_s,
        sensor=sensor_s,
        kalman=kalman_s,
        threat=threat_s,
        interceptor=inter_s,
        engagement=eng_s,
    )


PRESETS = [
    {
        "id": "adv_seed13",
        "name": "Seed 13 · EKF7 + APN Rescue (KILL)",
        "badge": "ADVANCED GNC",
        "description": "7-state drag-estimating EKF + RAE radar + Drag PIP + Augmented PN saves divert Δv and achieves a 1.1 m hit.",
        "config": "configs/advanced.yaml",
        "seed": 13,
        "overrides": {"filter_type": "ekf7", "pip_mode": "drag", "sensor_model": "rae", "guidance_law": "apn", "kv_divert_dv": 435.0, "pn_gain": 4.0, "tau": 0.0},
    },
    {
        "id": "eng_seed13",
        "name": "Seed 13 · KF6 + Classical PN (MISS)",
        "badge": "BASELINE FAIL",
        "description": "Same Seed 13 threat with 6-state linear KF + Cartesian radar + Classical PN exhausts 435 m/s divert budget and misses.",
        "config": "configs/engineering.yaml",
        "seed": 13,
        "overrides": {"filter_type": "kf6", "pip_mode": "vacuum", "sensor_model": "cartesian", "guidance_law": "pn", "kv_divert_dv": 435.0, "pn_gain": 4.0, "tau": 0.0},
    },
    {
        "id": "zem_seed42",
        "name": "Seed 42 · EKF7 + Optimal ZEM (Lag τ=20ms)",
        "badge": "OPTIMAL ZEM",
        "description": "Optimal Zero-Effort Miss guidance with target drag feed-forward and 20 ms DACS actuator lag.",
        "config": "configs/advanced.yaml",
        "seed": 42,
        "overrides": {"filter_type": "ekf7", "pip_mode": "drag", "sensor_model": "rae", "guidance_law": "zem", "kv_divert_dv": 435.0, "pn_gain": 4.0, "tau": 0.02},
    },
    {
        "id": "eng_seed913",
        "name": "Seed 913 · High-Altitude Engineering Hit",
        "badge": "ENGINEERING",
        "description": "Baseline engineering mode intercept at ~26 km altitude.",
        "config": "configs/engineering.yaml",
        "seed": 913,
        "overrides": {"filter_type": "kf6", "pip_mode": "vacuum", "sensor_model": "cartesian", "guidance_law": "pn", "kv_divert_dv": 435.0, "pn_gain": 4.0, "tau": 0.0},
    },
    {
        "id": "hold_seed68",
        "name": "Seed 68 · PIP Outside Perimeter (WEAPONS HOLD)",
        "badge": "PIP GATED",
        "description": "Threat predicted impact falls outside the 30 km defended perimeter; battery holds fire.",
        "config": "configs/engineering.yaml",
        "seed": 68,
        "overrides": {"filter_type": "kf6", "pip_mode": "vacuum", "sensor_model": "cartesian", "guidance_law": "pn", "kv_divert_dv": 435.0, "pn_gain": 4.0, "tau": 0.0},
    },
    {
        "id": "demo_seed3",
        "name": "Seed 3 · Max-Performance Demo",
        "badge": "MAX-PERF",
        "description": "High-thrust 8 s booster with 50 g limit and unlimited divert budget.",
        "config": "configs/demo.yaml",
        "seed": 3,
        "overrides": {"filter_type": "kf6", "pip_mode": "vacuum", "sensor_model": "cartesian", "guidance_law": "pn", "unlimited_dv": True, "pn_gain": 4.0, "tau": 0.0},
    },
]


@app.get("/api/configs")
def get_configs():
    configs_list = []
    if CONFIGS_DIR.exists():
        for f in sorted(CONFIGS_DIR.glob("*.yaml")):
            cfg = load_config(f)
            rel = f"configs/{f.name}"
            configs_list.append({
                "path": rel,
                "name": f.stem,
                "title": cfg.title,
                "mode": cfg.mode,
                "settings": asdict(cfg),
            })
    return {"configs": configs_list, "presets": PRESETS}


@app.post("/api/simulate")
def api_simulate(req: SimRequest):
    cfg_path = _resolve_config_path(req.config)
    base_cfg = load_config(cfg_path)
    try:
        cfg = apply_overrides(base_cfg, req.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    hist = simulate(cfg, req.seed, legacy_rng=req.legacy_rng, console=QUIET_CONSOLE)
    rel_cfg = f"configs/{cfg_path.name}"
    hist["meta"].update(seed=req.seed, config=rel_cfg, legacy_rng=req.legacy_rng)
    payload = build_workstation_telemetry(hist)
    return JSONResponse(payload)


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    results = []
    for item in req.runs:
        cfg_path = _resolve_config_path(item.config)
        base_cfg = load_config(cfg_path)
        try:
            cfg = apply_overrides(base_cfg, item.overrides)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{item.label}: {exc}")
        hist = simulate(cfg, item.seed, legacy_rng=item.legacy_rng, console=QUIET_CONSOLE)
        hist["meta"].update(seed=item.seed, config=f"configs/{cfg_path.name}", legacy_rng=item.legacy_rng)
        payload = build_workstation_telemetry(hist)
        payload["compareLabel"] = item.label
        payload["compareColor"] = item.color
        results.append(payload)
    return JSONResponse({"runs": results})


@app.get("/api/batch")
def api_batch(dataset: str = "engineering"):
    csv_map = {}
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.csv")):
            # Only single-config batch CSVs (skip sweep CSVs)
            if "sweep" not in f.name:
                csv_map[f.stem] = f

    if dataset not in csv_map:
        dataset = next(iter(csv_map.keys()), None)

    if dataset is None:
        return {"available": [], "data": None}

    consistency_path = RESULTS_DIR / "consistency.json"
    convergence_path = RESULTS_DIR / "convergence.json"
    data = build_data(csv_map[dataset], consistency_path, convergence_path)
    return {
        "available": list(csv_map.keys()),
        "active": dataset,
        "data": data,
    }


@app.post("/api/batch/run")
def api_batch_run(req: LiveBatchRequest):
    n = max(5, min(int(req.n), 500))
    cfg_path = _resolve_config_path(req.config)
    base_cfg = load_config(cfg_path)
    cfg = apply_overrides(base_cfg, req.overrides)

    seeds = range(req.seed_start, req.seed_start + n)
    # One process per core: a serial loop took ~1.8 s per seed.
    raw_rows = run_jobs([(cfg, s, {}) for s in seeds], workers=min(os.cpu_count() or 1, 12))

    rows = []
    seed_summaries = []
    for s, raw_row in zip(seeds, raw_rows):
        str_row = {k: _fmt(v) for k, v in raw_row.items()}
        rows.append(str_row)
        seed_summaries.append({
            "seed": s,
            "outcome": raw_row["outcome"],
            "engaged": bool(raw_row["launched"]),
            "miss_m": round(float(raw_row["miss_distance_m"]), 2) if raw_row["miss_distance_m"] is not None else None,
            "kill_alt_km": round(float(raw_row["kill_alt_m"]) / 1000.0, 2) if raw_row["kill_alt_m"] is not None else None,
            "dv_used": round(float(raw_row["divert_dv_used_mps"]), 1) if raw_row["divert_dv_used_mps"] is not None else None,
            "max_g": round(float(raw_row["max_g"]), 1) if raw_row["max_g"] is not None else None,
            "kf_rms_pos_m": round(float(raw_row["kf_rms_pos_m"]), 2) if raw_row["kf_rms_pos_m"] is not None else None,
        })

    batch = Batch(rows)
    return {
        "name": f"Live Monte Carlo ({cfg.kalman.filter_type.upper()} + {cfg.interceptor.guidance_law.upper()})",
        "source": f"Live {n} runs · {cfg_path.name}",
        "runs": batch.n,
        "seeds": [req.seed_start, req.seed_start + n - 1],
        "rates": rate_rows(batch),
        "histograms": metric_histograms(batch),
        "seedSummaries": seed_summaries,
    }


@app.get("/")
def serve_index():
    return FileResponse(
        WORKSTATION_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


# Three.js comes from viewer/node_modules (npm ci in viewer/) so the
# workstation works offline. Without it, fall back to the same pinned version
# on a CDN so a plain pip install still gets a working 3D view.
THREE_VERSION = "0.160.0"
THREE_DIR = ROOT_DIR / "viewer" / "node_modules" / "three"
if THREE_DIR.exists():
    app.mount("/vendor/three", StaticFiles(directory=str(THREE_DIR)), name="three")
else:
    @app.get("/vendor/three/{path:path}")
    def three_from_cdn(path: str):
        return RedirectResponse(f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/{path}")

if WORKSTATION_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WORKSTATION_DIR)), name="workstation")


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m thaad_sim.server",
        description="Launch the THAAD GNC Telemetry & Engagement Workstation server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    print(f"\n  THAAD GNC Telemetry Workstation -> {url}\n")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
