"""Command line: python -m thaad_sim --config configs/engineering.yaml --seed 123"""
import argparse
from pathlib import Path

import numpy as np
from rich.console import Console

from .config import load_config
from .engagement import make_rng, run
from .replay_viewer import show_replay

DEFAULT_CONFIG = "configs/engineering.yaml"


def _display_path(path):
    """Config path as shown in the reproduce command: relative to the cwd when possible."""
    p = Path(path).resolve()
    try:
        p = p.relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return p.as_posix()


def reproduce_command(config, seed, legacy_rng=False):
    config_arg = f'"{config}"' if any(c.isspace() for c in config) else config
    cmd = f"python -m thaad_sim --config {config_arg} --seed {seed}"
    return cmd + " --legacy-rng" if legacy_rng else cmd


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m thaad_sim",
        description="Run one THAAD-style intercept engagement and open its 3D replay.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"YAML config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; a run is fully determined by config and seed (default: random)")
    parser.add_argument("--legacy-rng", action="store_true",
                        help="use the np.random.seed stream of the old prototype scripts, "
                             "so seeds they printed replay the same scenario")
    parser.add_argument("--no-browser", action="store_true",
                        help="write the replay file without opening it")
    parser.add_argument("--out", type=Path, default=None, metavar="PATH",
                        help="replay file to write (default: replays/replay_<timestamp>.html)")
    args = parser.parse_args(argv)

    console = Console()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    config = _display_path(args.config)
    console.print(f"[dim]Seed {seed} – rerun this scenario with: "
                  f"{reproduce_command(config, seed, args.legacy_rng)}[/dim]")

    hist = run(cfg, make_rng(seed, args.legacy_rng), console)
    hist['meta'].update(seed=seed, config=config, legacy_rng=args.legacy_rng)
    path = show_replay(hist, open_browser=not args.no_browser, path=args.out)
    if args.no_browser:
        console.print(f"[dim]Replay written to: {path}[/dim]")
    else:
        console.print(f"[dim]Replay opened in your browser: {path}[/dim]")


if __name__ == "__main__":
    main()
