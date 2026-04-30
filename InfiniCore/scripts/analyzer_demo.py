#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务与资源感知分析模块 - 真数据 Demo 入口

本脚本不生成模拟数据。它只负责构建并运行 C++ analyzer-demo，
由 C++ demo 通过 OpTrace + infinirt 读取当前运行时的真实资源快照。

示例:
  python3 scripts/analyzer_demo.py
  python3 scripts/analyzer_demo.py --configure iluvatar
  python3 scripts/analyzer_demo.py --configure metax --extra-config --use-mc=y
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_PRESETS = {
    "none": [],
    "cpu": ["--mutual-awareness=y", "--cpu=y", "--omp=n"],
    "iluvatar": ["--mutual-awareness=y", "--iluvatar-gpu=y", "--ccl=y"],
    "metax": ["--mutual-awareness=y", "--metax-gpu=y", "--ccl=y"],
    "nvidia": ["--mutual-awareness=y", "--nv-gpu=y", "--ccl=y"],
}


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the live mutual-awareness analyzer demo."
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
        help="extra xmake config argument, e.g. --extra-config --use-mc=y",
    )
    parser.add_argument(
        "--xmake",
        default="xmake",
        help="xmake executable path",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build analyzer-demo but do not run it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = project_dir()

    preset = CONFIG_PRESETS[args.configure]
    if preset:
        run([args.xmake, "f", "-c", "-y", *preset, *args.extra_config], cwd)

    run([args.xmake, "build", "analyzer-demo"], cwd)
    if not args.build_only:
        run([args.xmake, "run", "analyzer-demo"], cwd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[analyzer-demo] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
