#!/usr/bin/env python3
"""
一键对比 chunked prefill 开/关性能

用法:
  python3 scripts/test_prefill_compare.py --rounds 5 --delay 0.1 --dev cpu

该脚本会依次启动 launch_server.py (chunk-size=512/0)，运行 test_perf_cp.py 取结果，最后输出对比。
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_SERVER = os.path.join(SCRIPT_DIR, "launch_server.py")
TEST_SCRIPT = os.path.join(SCRIPT_DIR, "test_perf_cp.py")


def run_command(cmd, wait=True, capture_output=True, env=None):
    return subprocess.run(cmd, shell=True, text=True, capture_output=capture_output, env=env)


def launch_server(chunk_size, dev, ndev, max_batch, max_tokens, model_path, awq, gptq):
    args = [sys.executable, LAUNCH_SERVER,
            f"--chunk-size {chunk_size}",
            f"--dev {dev}",
            f"--ndev {ndev}",
            f"--max-batch {max_batch}"]
    if max_tokens is not None:
        args.append(f"--max-tokens {max_tokens}")
    if model_path:
        args.append(f"--model-path {model_path}")
    if awq:
        args.append("--awq")
    if gptq:
        args.append("--gptq")

    # 以子进程方式运行服务并返回 process 对象
    popen = subprocess.Popen(" ".join(args), shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return popen


def stop_server(popen):
    if popen and popen.poll() is None:
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        popen.wait(timeout=10)


def run_test_perf(rounds, delay):
    cmd = f"{sys.executable} {TEST_SCRIPT} --rounds {rounds} --delay {delay}"
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.returncode, p.stdout + p.stderr


def parse_short_stats(output):
    # 查找末尾用例中 SHORT Avg TTFT / E2E
    m_ttft = re.search(r"SHORT Avg TTFT = ([0-9.]+) ms", output)
    m_e2e = re.search(r"SHORT Avg E2E\s*= ([0-9.]+) ms", output)
    return {
        "short_avg_ttft_ms": float(m_ttft.group(1)) if m_ttft else None,
        "short_avg_e2e_ms": float(m_e2e.group(1)) if m_e2e else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较 chunked prefill 开/关的 TTFT/E2E")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--dev", type=str, default="cpu")
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--warmup", type=float, default=2.0, help="服务启动后的等待时间(秒)")
    args = parser.parse_args()

    results = []

    for chunk_size in (256, 0):
        mode = "ON" if chunk_size > 0 else "OFF"
        print("\n" + "="*78)
        print(f"启动服务，chunked prefill {mode} (chunk-size={chunk_size})...")

        server = launch_server(chunk_size=chunk_size, dev=args.dev, ndev=args.ndev,
                                max_batch=args.max_batch, max_tokens=args.max_tokens,
                                model_path=args.model_path, awq=args.awq, gptq=args.gptq)
        try:
            time.sleep(args.warmup)
            print("服务启动完成，开始跑 test_perf_cp.py")
            retcode, out = run_test_perf(args.rounds, args.delay)
            if retcode != 0:
                print("test_perf_cp.py 执行失败，退出码", retcode)
                print(out)
                raise SystemExit(1)
            stats = parse_short_stats(out)
            stats.update({"chunk_size": chunk_size, "mode": mode, "full_output": out})
            results.append(stats)
            print("完成一轮，对比指标：", stats)

        finally:
            stop_server(server)
            print("服务已停止")

    print("\n" + "#"*78)
    print("最终结果：")
    for r in results:
        print(f"chunk_size={r['chunk_size']} mode={r['mode']} SHORT Avg TTFT={r['short_avg_ttft_ms']}ms SHORT Avg E2E={r['short_avg_e2e_ms']}ms")

    if len(results) == 2:
        delta_ttft = results[1]['short_avg_ttft_ms'] - results[0]['short_avg_ttft_ms']
        delta_e2e = results[1]['short_avg_e2e_ms'] - results[0]['short_avg_e2e_ms']
        print("\n对比（chunked ON - OFF）：")
        print(f"  TTFT 变化：{delta_ttft:+.2f} ms")
        print(f"  E2E  变化：{delta_e2e:+.2f} ms")
        print(f"  TTFT 加速比例：{(1 - results[0]['short_avg_ttft_ms'] / results[1]['short_avg_ttft_ms']) * 100:.2f}%")
    print("#"*78)
