"""
End-to-end benchmark: Fused FFN vs Non-Fused FFN in Jiuge model inference.

Design — scientifically controlled variables
--------------------------------------------
* The ONLY metric is end-to-end inference wall clock latency measured with
  ``time.perf_counter`` immediately around the ``inferBatchJiuge`` C call.
  There is NO per-layer FFN profiling: on the C side, profiling events
  inject per-layer host↔device synchronisation that serialises the
  pipeline and biases the launch-overhead component of the comparison.
  Removing that instrumentation is what makes this measurement trustworthy.

* Fused and non-fused modes run on byte-identical inputs: same
  deterministic token lists, same KV cache capacity, same batch layout,
  same sampling parameters. The only variable that changes between the
  two modes is ``use_fused_ffn``.

* Measurement rounds are INTERLEAVED: round r measures non-fused then
  fused back-to-back, then round r+1 repeats. Any slow drift — GPU
  clock stepping, thermal throttling, unrelated system jitter — is
  therefore shared equally between the two modes and cancels in the
  ratio. A blocked ordering (all NF, then all F) can be re-enabled via
  ``--order blocked`` for debugging.

* The timing window wraps ONLY the C call. Task construction,
  ``KVCache`` allocation, and ``JiugeBatchedTask`` marshalling happen
  outside the window so Python/ctypes overhead cannot inflate the
  measurement — especially important for decode-shaped workloads where
  the GPU work is sub-millisecond.

* Warmup rounds are executed per-mode BEFORE the measured interleaved
  loop, priming the memory pool, kernel caches, and GPU clock state.

Usage
-----
    python bench_fused_ffn.py --nvidia /path/to/model \\
        [n_device] [--warmup 5] [--rounds 20] [--order interleaved] \\
        [--output report.md] [--plot plots/] [--plot-format png] \\
        [--skip-verify]
"""

import argparse
import os
import sys
import time
from ctypes import c_uint
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from jiuge import JiugeBatchedTask, JiugeForCauslLM  # noqa: E402
from libinfinicore_infer import DeviceType           # noqa: E402
from infer_task import InferTask, KVCache            # noqa: E402


# ── Test scenarios ─────────────────────────────────────────────────
#
# KV caches are sized to input_tokens + output_tokens per scenario
# (NOT to full max_position_embeddings, which would OOM at large bs).

TEST_CASES: List[dict] = [
    # ── Single-batch decode (baseline) ──
    {"label": "Decode (bs=1, seq=1)",            "batch_size": 1,  "input_tokens": 1,    "output_tokens": 32},
    # ── Batched decode ──
    {"label": "Batched Decode (bs=4, seq=1)",    "batch_size": 4,  "input_tokens": 1,    "output_tokens": 32},
    {"label": "Batched Decode (bs=8, seq=1)",    "batch_size": 8,  "input_tokens": 1,    "output_tokens": 32},
    {"label": "Batched Decode (bs=16, seq=1)",   "batch_size": 16, "input_tokens": 1,    "output_tokens": 32},
    {"label": "Batched Decode (bs=32, seq=1)",   "batch_size": 32, "input_tokens": 1,    "output_tokens": 32},
    # ── Batched prefill ──
    {"label": "Batched Prefill (bs=2, seq=256)", "batch_size": 2,  "input_tokens": 256,  "output_tokens": 32},
    {"label": "Batched Prefill (bs=4, seq=256)", "batch_size": 4,  "input_tokens": 256,  "output_tokens": 32},
    {"label": "Batched Prefill (bs=4, seq=512)", "batch_size": 4,  "input_tokens": 512,  "output_tokens": 1},
    {"label": "Batched Prefill (bs=8, seq=512)", "batch_size": 8,  "input_tokens": 512,  "output_tokens": 1},
    # ── Single-batch prefill: small → large ──
    {"label": "Prefill (bs=1, seq=32)",          "batch_size": 1,  "input_tokens": 32,   "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=64)",          "batch_size": 1,  "input_tokens": 64,   "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=128)",         "batch_size": 1,  "input_tokens": 128,  "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=256)",         "batch_size": 1,  "input_tokens": 256,  "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=512)",         "batch_size": 1,  "input_tokens": 512,  "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=1024)",        "batch_size": 1,  "input_tokens": 1024, "output_tokens": 32},
    {"label": "Prefill (bs=1, seq=2048)",        "batch_size": 1,  "input_tokens": 2048, "output_tokens": 1},
    {"label": "Prefill (bs=1, seq=4096)",        "batch_size": 1,  "input_tokens": 4096, "output_tokens": 1},
]


# ── Helpers ─────────────────────────────────────────────────────────


def make_dummy_tokens(n_tokens: int, vocab_size: int) -> List[int]:
    """Deterministic token list — same call → same list, every time."""
    return [i % vocab_size for i in range(n_tokens)]


def summarize(samples: List[float]) -> Dict[str, float]:
    """Robust summary statistics over measured rounds."""
    if not samples:
        return {
            "mean": 0.0, "trimmed_mean": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "stdev": 0.0, "p99": 0.0, "n": 0,
        }
    arr = sorted(samples)
    n = len(arr)
    trimmed = arr[1:-1] if n >= 4 else arr
    p99_idx = max(0, min(n - 1, int(round((n - 1) * 0.99))))
    return {
        "mean":         mean(arr),
        "trimmed_mean": mean(trimmed),
        "median":       median(arr),
        "min":          arr[0],
        "max":          arr[-1],
        "stdev":        stdev(arr) if n > 1 else 0.0,
        "p99":          arr[p99_idx],
        "n":            n,
    }


def speedup_pct(nf: float, f: float) -> float:
    return (nf - f) / nf * 100.0 if nf > 0 else 0.0


def speedup_ratio(nf: float, f: float) -> float:
    return nf / f if f > 0 else 0.0


# ── Core measurement primitive ──────────────────────────────────────


def time_infer_batch(model: JiugeForCauslLM, batch: JiugeBatchedTask,
                     output_buf) -> float:
    """
    Time ONE inferBatchJiuge call. Returns latency in milliseconds.

    inferBatchJiuge synchronously drains the stream and copies sampled
    tokens back to host before returning (see ``infinirtStreamSynchronize``
    + D2H memcpy at the end of ``inferDeviceBatch``), so the
    perf_counter delta reflects the full wall-clock cost of GPU forward
    + sampling for this one step. No extra host-side sync is required.
    """
    start = time.perf_counter()
    model.jiuge_model.infer_batch(
        model.model_instance,
        *batch.input_args(),
        output_buf,
    )
    end = time.perf_counter()
    return (end - start) * 1000.0


# ── Correctness verification ────────────────────────────────────────


def run_forward_once(model: JiugeForCauslLM, tokens_list: List[List[int]],
                     use_fused: bool, kv_max_len: int) -> torch.Tensor:
    """One forward pass that returns logits — used only by verification."""
    model.jiuge_model.set_fused_ffn(model.model_instance, use_fused)

    tasks: List[InferTask] = []
    kvs: List[KVCache] = []
    for tokens in tokens_list:
        task = InferTask(
            0, tokens, model.max_context_len(),
            1.0, 1, 1.0, model.eos_token_id,
        )
        kv = KVCache(model, max_len=kv_max_len)
        task.bind_kvcache(kv)
        tasks.append(task)
        kvs.append(kv)

    batch = JiugeBatchedTask(tasks)
    ntok = batch.ntok
    dvoc = model.meta.dvoc
    logits = torch.zeros((ntok, dvoc), dtype=model.meta.torch_dtype_logits)

    model.jiuge_model.forward_batch(
        model.model_instance,
        batch.tokens, batch.ntok,
        batch.req_lens, batch.nreq,
        batch.req_pos, batch.kv_caches,
        logits.data_ptr(),
    )
    for kv in kvs:
        kv.drop(model)
    return logits


def verify_correctness(model: JiugeForCauslLM, vocab_size: int) -> dict:
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION  (fused vs non-fused)")
    print("=" * 60)

    tokens = make_dummy_tokens(16, vocab_size)
    kv_len = len(tokens) + 1

    print("  running non-fused forward pass...")
    logits_nf = run_forward_once(model, [tokens], False, kv_len).float().numpy()
    print("  running fused forward pass...")
    logits_f = run_forward_once(model, [tokens], True,  kv_len).float().numpy()

    max_diff = float(np.max(np.abs(logits_nf - logits_f)))
    mean_diff = float(np.mean(np.abs(logits_nf - logits_f)))
    denom = np.linalg.norm(logits_nf) * np.linalg.norm(logits_f) + 1e-10
    cos_sim = float(np.sum(logits_nf * logits_f) / denom)
    passed = cos_sim > 0.999

    print(f"  max |diff|  : {max_diff:.6e}")
    print(f"  mean |diff| : {mean_diff:.6e}")
    print(f"  cos_sim     : {cos_sim:.8f}   "
          f"{'PASS' if passed else 'FAIL'} (threshold 0.999)")
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "passed": passed,
    }


# ── Scenario benchmark ──────────────────────────────────────────────


def benchmark_scenario(model: JiugeForCauslLM, case: dict,
                       vocab_size: int, warmup: int, rounds: int,
                       order: str = "interleaved"
                       ) -> Tuple[List[float], List[float]]:
    """
    Measure NF vs F end-to-end latency for one scenario.

    Parameters
    ----------
    order : {"interleaved", "blocked"}
        ``interleaved`` → NF, F, NF, F, ... (default; cancels drift).
        ``blocked``     → all NF rounds, then all F rounds (debug only).

    Returns
    -------
    (nf_samples_ms, f_samples_ms)
        Samples are paired by round index under interleaved ordering so
        that index r of each list corresponds to back-to-back
        measurements taken under the same thermal/clock state.
    """
    label = case["label"]
    bs = case["batch_size"]
    n_input = case["input_tokens"]
    n_output = case["output_tokens"]
    kv_max_len = n_input + n_output

    print(f"\n[{label}]")
    print(f"  bs={bs}, in={n_input}, out={n_output}, "
          f"warmup={warmup}, rounds={rounds}, order={order}")
    print("  " + "-" * 58)

    # Fixed deterministic inputs, shared across all calls in this
    # scenario. Only ``use_fused_ffn`` changes between measurements.
    tokens_list = [make_dummy_tokens(n_input, vocab_size) for _ in range(bs)]
    output_buf = (c_uint * bs)()

    def build_and_run(use_fused: bool) -> float:
        """
        Build fresh tasks + KV caches (excluded from timing), then time
        one ``inferBatchJiuge`` call and tear the KV caches down.
        Rebuilding per round guarantees each measurement starts from a
        clean KV state, so there is zero cross-round pollution.
        """
        model.jiuge_model.set_fused_ffn(model.model_instance, use_fused)

        tasks: List[InferTask] = []
        kvs: List[KVCache] = []
        for tokens in tokens_list:
            task = InferTask(
                0, tokens, model.max_context_len(),
                1.0, 1, 1.0, model.eos_token_id,
            )
            kv = KVCache(model, max_len=kv_max_len)
            task.bind_kvcache(kv)
            tasks.append(task)
            kvs.append(kv)
        batch = JiugeBatchedTask(tasks)

        t_ms = time_infer_batch(model, batch, output_buf)

        for kv in kvs:
            kv.drop(model)
        return t_ms

    # ── Warmup (blocked per mode: prime each path independently) ──
    for _ in range(warmup):
        build_and_run(False)
    for _ in range(warmup):
        build_and_run(True)

    # ── Measurement ──
    nf_samples: List[float] = []
    f_samples: List[float] = []
    if order == "interleaved":
        for _ in range(rounds):
            nf_samples.append(build_and_run(False))
            f_samples.append(build_and_run(True))
    elif order == "blocked":
        for _ in range(rounds):
            nf_samples.append(build_and_run(False))
        for _ in range(rounds):
            f_samples.append(build_and_run(True))
    else:
        raise ValueError(f"unknown order: {order!r}")

    nf_stat = summarize(nf_samples)
    f_stat = summarize(f_samples)

    total_tokens = bs * n_input
    nf_thr = total_tokens / (nf_stat["mean"] / 1000.0) if nf_stat["mean"] > 0 else 0.0
    f_thr = total_tokens / (f_stat["mean"] / 1000.0) if f_stat["mean"] > 0 else 0.0
    sp = speedup_pct(nf_stat["mean"], f_stat["mean"])
    ratio = speedup_ratio(nf_stat["mean"], f_stat["mean"])

    def row(name: str, nf: float, f: float) -> None:
        print(f"  {name:<22} {nf:>12.3f}  {f:>12.3f}")

    print(f"  {'metric':<22} {'non-fused':>12}  {'fused':>12}")
    print(f"  {'':->50}")
    row("mean    (ms)",    nf_stat["mean"],         f_stat["mean"])
    row("trimmed (ms)",    nf_stat["trimmed_mean"], f_stat["trimmed_mean"])
    row("median  (ms)",    nf_stat["median"],       f_stat["median"])
    row("stdev   (ms)",    nf_stat["stdev"],        f_stat["stdev"])
    row("min     (ms)",    nf_stat["min"],          f_stat["min"])
    row("p99     (ms)",    nf_stat["p99"],          f_stat["p99"])
    row("throughput tok/s", nf_thr,                 f_thr)
    print(f"  → e2e speedup: {sp:+.2f}%   ({ratio:.3f}×)")

    return nf_samples, f_samples


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fused vs Non-Fused FFN End-to-End Benchmark "
                    "(scientifically controlled)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    for dev in ("cpu", "nvidia", "qy", "cambricon", "ascend", "metax",
                "moore", "iluvatar", "kunlun", "hygon", "ali"):
        group.add_argument(f"--{dev}", action="store_const", const=dev, dest="device")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("n_device", nargs="?", type=int, default=1,
                        help="Number of devices")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup rounds per mode (default: 5)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Measured rounds per mode (default: 20)")
    parser.add_argument("--order", choices=["interleaved", "blocked"],
                        default="interleaved",
                        help="Round ordering. interleaved (default) "
                             "cancels thermal drift by pairing NF/F "
                             "back-to-back; blocked is for debugging.")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save markdown report to file (e.g. report.md)")
    parser.add_argument("--plot", type=str, default=None,
                        help="Directory to save charts (needs matplotlib)")
    parser.add_argument("--plot-format", choices=["png", "svg", "pdf"],
                        default="png",
                        help="Chart file format (default: png)")
    args = parser.parse_args()

    device_map = {
        "cpu":       DeviceType.DEVICE_TYPE_CPU,
        "nvidia":    DeviceType.DEVICE_TYPE_NVIDIA,
        "qy":        DeviceType.DEVICE_TYPE_QY,
        "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
        "ascend":    DeviceType.DEVICE_TYPE_ASCEND,
        "metax":     DeviceType.DEVICE_TYPE_METAX,
        "moore":     DeviceType.DEVICE_TYPE_MOORE,
        "iluvatar":  DeviceType.DEVICE_TYPE_ILUVATAR,
        "kunlun":    DeviceType.DEVICE_TYPE_KUNLUN,
        "hygon":     DeviceType.DEVICE_TYPE_HYGON,
        "ali":       DeviceType.DEVICE_TYPE_ALI,
    }
    device_type = device_map[args.device]

    print("=" * 60)
    print("FUSED FFN END-TO-END BENCHMARK")
    print("=" * 60)
    print(f"Model   : {args.model_path}")
    print(f"Device  : {args.device} (x{args.n_device})")
    print(f"Warmup  : {args.warmup} rounds per mode")
    print(f"Rounds  : {args.rounds} pairs, ordering={args.order}")
    print()

    model = JiugeForCauslLM(args.model_path, device_type, args.n_device)
    vocab_size = model.meta.dvoc

    verify_data = None
    if not args.skip_verify:
        verify_data = verify_correctness(model, vocab_size)
        if not verify_data["passed"]:
            print("\nCorrectness check FAILED. Aborting benchmark.")
            model.destroy_model_instance()
            return

    all_samples: Dict[str, Dict[str, List[float]]] = {}
    for case in TEST_CASES:
        nf_s, f_s = benchmark_scenario(
            model, case, vocab_size,
            warmup=args.warmup, rounds=args.rounds, order=args.order,
        )
        all_samples[case["label"]] = {"nf": nf_s, "f": f_s}

    # ── Global summary ──
    print("\n" + "=" * 76)
    print("SUMMARY — end-to-end latency")
    print("=" * 76)
    print(f"{'Scenario':<35} {'NF mean':>11} {'F mean':>11} {'Speedup':>10} {'Ratio':>7}")
    print("-" * 76)
    for case in TEST_CASES:
        label = case["label"]
        nf = mean(all_samples[label]["nf"])
        f = mean(all_samples[label]["f"])
        print(f"{label:<35} "
              f"{nf:>10.3f}  {f:>10.3f}  "
              f"{speedup_pct(nf, f):>+8.2f}%  "
              f"{speedup_ratio(nf, f):>6.3f}×")

    model.destroy_model_instance()
    print("\nBenchmark complete.")

    # ── Markdown report ──
    if args.output:
        save_markdown_report(args, verify_data, all_samples)
        print(f"Markdown report: {args.output}")

    # ── Plots ──
    if args.plot:
        try:
            from bench_plot import save_e2e_plots
            out_dir = save_e2e_plots(
                args, verify_data, all_samples, TEST_CASES,
                out_dir=args.plot, fmt=args.plot_format,
            )
            print(f"Plots directory: {out_dir}")
        except Exception as exc:
            print(f"Failed to render plots: {exc}")


# ── Markdown report ────────────────────────────────────────────────


def save_markdown_report(args, verify_data, all_samples):
    lines: List[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    # ── Header ──
    L("# Fused FFN End-to-End Benchmark Report")
    L()
    L(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"- **Model:** `{args.model_path}`")
    L(f"- **Device:** {args.device} (x{args.n_device})")
    L(f"- **Warmup:** {args.warmup} rounds per mode")
    L(f"- **Measured rounds:** {args.rounds} per mode "
      f"(ordering: {args.order})")
    L()
    L("> Only metric is end-to-end wall clock latency, measured with "
      "`time.perf_counter` around the `inferBatchJiuge` C call. Fused "
      "and non-fused runs share byte-identical inputs; rounds are "
      "interleaved so that thermal drift and system jitter affect both "
      "modes equally and cancel in the comparison.")
    L()

    # ── Correctness ──
    if verify_data:
        status = "PASS" if verify_data["passed"] else "FAIL"
        L("## Correctness Verification")
        L()
        L("| Metric | Value |")
        L("|--------|-------|")
        L(f"| Max abs diff | `{verify_data['max_diff']:.6e}` |")
        L(f"| Mean abs diff | `{verify_data['mean_diff']:.6e}` |")
        L(f"| Cosine similarity | `{verify_data['cos_sim']:.8f}` |")
        L(f"| Status | **{status}** (threshold 0.999) |")
        L()

    # ── Per-scenario results ──
    L("## Per-Scenario Results")
    L()
    for case in TEST_CASES:
        label = case["label"]
        bs = case["batch_size"]
        n_in = case["input_tokens"]
        n_out = case["output_tokens"]
        nf = summarize(all_samples[label]["nf"])
        f = summarize(all_samples[label]["f"])
        sp = speedup_pct(nf["mean"], f["mean"])
        ratio = speedup_ratio(nf["mean"], f["mean"])
        total_tokens = bs * n_in
        nf_thr = total_tokens / (nf["mean"] / 1000.0) if nf["mean"] > 0 else 0.0
        f_thr = total_tokens / (f["mean"] / 1000.0) if f["mean"] > 0 else 0.0

        L(f"### {label}")
        L()
        L(f"`batch_size={bs}, input_tokens={n_in}, output_tokens={n_out}`")
        L()
        L("| Metric | Non-Fused | Fused | Δ |")
        L("|--------|-----------|-------|---|")
        L(f"| mean latency (ms) | {nf['mean']:.3f} | {f['mean']:.3f} | {sp:+.2f}% |")
        L(f"| trimmed mean (ms) | {nf['trimmed_mean']:.3f} | {f['trimmed_mean']:.3f} | |")
        L(f"| median (ms) | {nf['median']:.3f} | {f['median']:.3f} | |")
        L(f"| stdev (ms) | {nf['stdev']:.3f} | {f['stdev']:.3f} | |")
        L(f"| min (ms) | {nf['min']:.3f} | {f['min']:.3f} | |")
        L(f"| p99 (ms) | {nf['p99']:.3f} | {f['p99']:.3f} | |")
        L(f"| throughput (tok/s) | {nf_thr:.2f} | {f_thr:.2f} | |")
        L(f"| **speedup ratio** | | | **{ratio:.3f}×** |")
        L()

    # ── Summary table ──
    L("## Summary")
    L()
    L("| Scenario | NF mean (ms) | F mean (ms) | Speedup | Ratio |")
    L("|----------|-------------|------------|---------|-------|")
    for case in TEST_CASES:
        label = case["label"]
        nf = mean(all_samples[label]["nf"])
        f = mean(all_samples[label]["f"])
        L(f"| {label} | {nf:.3f} | {f:.3f} | "
          f"{speedup_pct(nf, f):+.2f}% | {speedup_ratio(nf, f):.3f}× |")
    L()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
