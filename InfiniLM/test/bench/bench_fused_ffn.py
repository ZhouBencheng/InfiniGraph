"""
End-to-end benchmark: Fused FFN vs Non-Fused FFN in Jiuge model inference.

Usage:
    python bench_fused_ffn.py --nvidia <path/to/model> [n_device] [--warmup 5] [--rounds 10]

This script:
1. Loads the Jiuge model (e.g. 9g_8b_thinking)
2. Verifies fused FFN produces correct output (vs non-fused)
3. Runs multiple scenarios with different batch sizes and sequence lengths
4. Reports per-layer FFN GPU timing (infinirtEvent) and end-to-end latency
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import List

import numpy as np
import safetensors
import torch

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from jiuge import (
    JiugeBatchedTask,
    JiugeForCauslLM,
)
from libinfinicore_infer import DeviceType
from infer_task import InferTask, KVCache


# ── Test Scenarios ──────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "Decode (bs=1, seq=1)",
        "batch_size": 1,
        "input_tokens": 1,
        "output_tokens": 32,
    },
    {
        "label": "Short Prefill (bs=1, seq=64)",
        "batch_size": 1,
        "input_tokens": 64,
        "output_tokens": 32,
    },
    {
        "label": "Medium Prefill (bs=1, seq=512)",
        "batch_size": 1,
        "input_tokens": 512,
        "output_tokens": 32,
    },
    {
        "label": "Long Prefill (bs=1, seq=1024)",
        "batch_size": 1,
        "input_tokens": 1024,
        "output_tokens": 1,
    },
]

WARMUP_ROUNDS = 5
MEASURED_ROUNDS = 10


# ── Helpers ─────────────────────────────────────────────────────────

def make_dummy_tokens(n_tokens: int, vocab_size: int) -> List[int]:
    """Generate deterministic token sequence."""
    return [i % vocab_size for i in range(n_tokens)]


def run_inference(model: JiugeForCauslLM, tokens_list: List[List[int]],
                  use_fused: bool) -> dict:
    """
    Run a single inference round with the given tokens.
    Returns timing info and FFN profile data.
    """
    model.jiuge_model.set_fused_ffn(model.model_instance, use_fused)

    tasks = []
    kv_caches = []
    for i, tokens in enumerate(tokens_list):
        task = InferTask(0, tokens, model.max_context_len(),
                         1.0, 1, 1.0, model.eos_token_id)
        kv = KVCache(model)
        task.bind_kvcache(kv)
        tasks.append(task)
        kv_caches.append(kv)

    start = time.perf_counter()
    batch = JiugeBatchedTask(tasks)
    output = model.batch_infer_one_round(tasks)
    end = time.perf_counter()

    e2e_ms = (end - start) * 1000
    profile = model.jiuge_model.get_ffn_profile(model.model_instance)

    # Cleanup
    for kv in kv_caches:
        kv.drop(model)

    return {
        "e2e_ms": e2e_ms,
        "ffn_total_ms": profile["total_ms"],
        "ffn_per_layer_ms": profile["per_layer_ms"],
        "output_tokens": list(output),
    }


def run_forward(model: JiugeForCauslLM, tokens_list: List[List[int]],
                use_fused: bool) -> torch.Tensor:
    """Run forward pass and return logits for correctness check."""
    model.jiuge_model.set_fused_ffn(model.model_instance, use_fused)

    tasks = []
    kv_caches = []
    for tokens in tokens_list:
        task = InferTask(0, tokens, model.max_context_len(),
                         1.0, 1, 1.0, model.eos_token_id)
        kv = KVCache(model)
        task.bind_kvcache(kv)
        tasks.append(task)
        kv_caches.append(kv)

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

    for kv in kv_caches:
        kv.drop(model)

    return logits


# ── Correctness Check ───────────────────────────────────────────────

def verify_correctness(model: JiugeForCauslLM, vocab_size: int):
    """Verify fused FFN produces same output as non-fused."""
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)

    tokens = make_dummy_tokens(16, vocab_size)

    print("Running non-fused forward pass...")
    logits_non_fused = run_forward(model, [tokens], use_fused=False)

    print("Running fused forward pass...")
    logits_fused = run_forward(model, [tokens], use_fused=True)

    logits_nf = logits_non_fused.float().numpy()
    logits_f = logits_fused.float().numpy()

    max_diff = np.max(np.abs(logits_nf - logits_f))
    mean_diff = np.mean(np.abs(logits_nf - logits_f))
    cos_sim = np.sum(logits_nf * logits_f) / (
        np.linalg.norm(logits_nf) * np.linalg.norm(logits_f) + 1e-10
    )

    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Cosine similarity:  {cos_sim:.8f}")

    passed = cos_sim > 0.999
    if passed:
        print(f"  PASS: Cosine similarity {cos_sim:.8f} > 0.999")
    else:
        print(f"  FAIL: Cosine similarity {cos_sim:.8f} <= 0.999")
        print("  The fused FFN may have numerical differences. Proceed with caution.")

    print()
    return passed


# ── Benchmark Runner ────────────────────────────────────────────────

def benchmark_scenario(model: JiugeForCauslLM, case: dict, vocab_size: int):
    """Run benchmark for a single scenario, both fused and non-fused."""
    label = case["label"]
    bs = case["batch_size"]
    n_input = case["input_tokens"]
    n_output = case["output_tokens"]

    print(f"\n[{label}]  batch_size={bs}, input_tokens={n_input}, output_tokens={n_output}")
    print("-" * 60)

    results = {"non_fused": {"e2e": [], "ffn_total": [], "ffn_per_layer": []},
               "fused": {"e2e": [], "ffn_total": [], "ffn_per_layer": []}}

    for mode_name, use_fused in [("non_fused", False), ("fused", True)]:
        # Warmup
        for _ in range(WARMUP_ROUNDS):
            tokens_list = [make_dummy_tokens(n_input, vocab_size) for _ in range(bs)]
            run_inference(model, tokens_list, use_fused)

        # Measured rounds
        for _ in range(MEASURED_ROUNDS):
            tokens_list = [make_dummy_tokens(n_input, vocab_size) for _ in range(bs)]
            result = run_inference(model, tokens_list, use_fused)
            results[mode_name]["e2e"].append(result["e2e_ms"])
            results[mode_name]["ffn_total"].append(result["ffn_total_ms"])
            results[mode_name]["ffn_per_layer"].append(result["ffn_per_layer_ms"])

    # Print comparison table
    def stats(arr):
        return mean(arr), median(arr), min(arr), (stdev(arr) if len(arr) > 1 else 0)

    nf_e2e = stats(results["non_fused"]["e2e"])
    f_e2e = stats(results["fused"]["e2e"])
    nf_ffn = stats(results["non_fused"]["ffn_total"])
    f_ffn = stats(results["fused"]["ffn_total"])

    # Average per-layer from last round
    nf_per_layer = results["non_fused"]["ffn_per_layer"][-1]
    f_per_layer = results["fused"]["ffn_per_layer"][-1]
    avg_nf_layer = mean(nf_per_layer) if nf_per_layer else 0
    avg_f_layer = mean(f_per_layer) if f_per_layer else 0

    total_tokens = n_input * bs
    nf_throughput = total_tokens / (nf_e2e[0] / 1000) if nf_e2e[0] > 0 else 0
    f_throughput = total_tokens / (f_e2e[0] / 1000) if f_e2e[0] > 0 else 0

    def speedup(a, b):
        return ((a - b) / a * 100) if a > 0 else 0

    def fmt(val):
        return f"{val:8.2f}"

    print(f"{'Metric':<25} {'Non-Fused':>12} {'Fused':>12} {'Speedup':>10}")
    print(f"{'':_>60}")
    print(f"{'E2E Latency (ms)':<25} {fmt(nf_e2e[0]):>12} {fmt(f_e2e[0]):>12} {speedup(nf_e2e[0], f_e2e[0]):>+9.1f}%")
    print(f"{'E2E Median (ms)':<25} {fmt(nf_e2e[1]):>12} {fmt(f_e2e[1]):>12}")
    print(f"{'FFN Total (ms)':<25} {fmt(nf_ffn[0]):>12} {fmt(f_ffn[0]):>12} {speedup(nf_ffn[0], f_ffn[0]):>+9.1f}%")
    print(f"{'FFN Avg/Layer (ms)':<25} {fmt(avg_nf_layer):>12} {fmt(avg_f_layer):>12} {speedup(avg_nf_layer, avg_f_layer):>+9.1f}%")
    print(f"{'Throughput (tok/s)':<25} {fmt(nf_throughput):>12} {fmt(f_throughput):>12} {speedup(nf_throughput, f_throughput):>+9.1f}%")

    # Per-layer breakdown (first 5 and last 5 layers)
    n_layers = len(nf_per_layer)
    if n_layers > 0:
        print(f"\n  Per-layer FFN time (ms) - first 5 & last 5 layers:")
        print(f"  {'Layer':<10} {'Non-Fused':>12} {'Fused':>12} {'Speedup':>10}")
        indices = list(range(min(5, n_layers)))
        if n_layers > 10:
            indices += list(range(n_layers - 5, n_layers))
        elif n_layers > 5:
            indices += list(range(5, n_layers))

        for idx in indices:
            nf_l = nf_per_layer[idx]
            f_l = f_per_layer[idx]
            sp = speedup(nf_l, f_l)
            print(f"  {idx:<10} {nf_l:>12.3f} {f_l:>12.3f} {sp:>+9.1f}%")

    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fused FFN End-to-End Benchmark")
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--cpu", action="store_const", const="cpu", dest="device")
    device_group.add_argument("--nvidia", action="store_const", const="nvidia", dest="device")
    device_group.add_argument("--qy", action="store_const", const="qy", dest="device")
    device_group.add_argument("--cambricon", action="store_const", const="cambricon", dest="device")
    device_group.add_argument("--ascend", action="store_const", const="ascend", dest="device")
    device_group.add_argument("--metax", action="store_const", const="metax", dest="device")
    device_group.add_argument("--moore", action="store_const", const="moore", dest="device")
    device_group.add_argument("--iluvatar", action="store_const", const="iluvatar", dest="device")
    device_group.add_argument("--kunlun", action="store_const", const="kunlun", dest="device")
    device_group.add_argument("--hygon", action="store_const", const="hygon", dest="device")
    device_group.add_argument("--ali", action="store_const", const="ali", dest="device")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("n_device", nargs="?", type=int, default=1, help="Number of devices")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup rounds")
    parser.add_argument("--rounds", type=int, default=10, help="Measured rounds")
    parser.add_argument("--skip-verify", action="store_true", help="Skip correctness check")
    args = parser.parse_args()

    global WARMUP_ROUNDS, MEASURED_ROUNDS
    WARMUP_ROUNDS = args.warmup
    MEASURED_ROUNDS = args.rounds

    device_map = {
        "cpu": DeviceType.DEVICE_TYPE_CPU,
        "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
        "qy": DeviceType.DEVICE_TYPE_QY,
        "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
        "ascend": DeviceType.DEVICE_TYPE_ASCEND,
        "metax": DeviceType.DEVICE_TYPE_METAX,
        "moore": DeviceType.DEVICE_TYPE_MOORE,
        "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
        "kunlun": DeviceType.DEVICE_TYPE_KUNLUN,
        "hygon": DeviceType.DEVICE_TYPE_HYGON,
        "ali": DeviceType.DEVICE_TYPE_ALI,
    }
    device_type = device_map[args.device]

    # Load model
    print("=" * 60)
    print("FUSED FFN END-TO-END BENCHMARK")
    print("=" * 60)
    print(f"Model:  {args.model_path}")
    print(f"Device: {args.device} (x{args.n_device})")
    print(f"Warmup: {WARMUP_ROUNDS} rounds, Measured: {MEASURED_ROUNDS} rounds")
    print()

    model = JiugeForCauslLM(args.model_path, device_type, args.n_device)
    vocab_size = model.meta.dvoc

    # Correctness verification
    if not args.skip_verify:
        passed = verify_correctness(model, vocab_size)
        if not passed:
            print("Correctness check FAILED. Aborting benchmark.")
            model.destroy_model_instance()
            return

    # Run benchmarks
    all_results = {}
    for case in TEST_CASES:
        results = benchmark_scenario(model, case, vocab_size)
        all_results[case["label"]] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<35} {'NF FFN(ms)':>10} {'F FFN(ms)':>10} {'Speedup':>10}")
    print("-" * 70)
    for case in TEST_CASES:
        label = case["label"]
        nf = mean(all_results[label]["non_fused"]["ffn_total"])
        f = mean(all_results[label]["fused"]["ffn_total"])
        sp = ((nf - f) / nf * 100) if nf > 0 else 0
        print(f"{label:<35} {nf:>10.2f} {f:>10.2f} {sp:>+9.1f}%")

    # Cleanup
    model.destroy_model_instance()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
