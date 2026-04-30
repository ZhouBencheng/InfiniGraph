#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and run analyzer-load-demo.

The C++/CUDA demo creates real GPU pressure while the analyzer samples live
runtime resources and compares different OpTrace task windows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_PRESETS = {
    "none": [],
    "iluvatar": ["--mutual-awareness=y", "--iluvatar-gpu=y", "--ccl=y"],
    "nvidia": ["--mutual-awareness=y", "--nv-gpu=y", "--ccl=y"],
}


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live GPU-load x task-trace mutual-awareness demo."
    )
    parser.add_argument(
        "--configure",
        choices=sorted(CONFIG_PRESETS),
        default="none",
        help="optional xmake platform configuration preset",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        help="extra xmake config argument",
    )
    parser.add_argument(
        "--xmake",
        default="xmake",
        help="xmake executable path",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=1500,
        help="per-load warmup before analyzer sampling",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build analyzer-load-demo but do not run it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = project_dir()

    preset = CONFIG_PRESETS[args.configure]
    if preset:
        run([args.xmake, "f", "-c", "-y", *preset, *args.extra_config], cwd)

    run([args.xmake, "build", "analyzer-load-demo"], cwd)
    if not args.build_only:
        run([args.xmake, "run", "analyzer-load-demo", str(args.warmup_ms)], cwd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[analyzer-load-demo] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
