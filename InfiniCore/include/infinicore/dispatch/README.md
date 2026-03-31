# KernelDispatcher -- Device-Aware Kernel Dispatch Module

## Background & Motivation

InfiniCore supports multiple GPU backends (NVIDIA, Iluvatar, MetaX, ALI, etc.), and the paged attention prefill operator has **8 kernel variants** (warp, warpcta, warpcta8, warpcta8pipe, warpcta8mma, warpcta8n128, warpcta16, ref) with significantly different performance characteristics across hardware.

**Problem**: Kernel variant selection was hardcoded in each `.cu` / `.maca` file's `default_prefill_kernel()` function. This couples dispatch logic with operator implementation, meaning:
- Adding a new GPU requires modifying `.cu` files
- Cannot dynamically adapt kernel choice based on runtime workload characteristics
- No integration with the existing `MutualAwarenessAnalyzer` that already detects prefill/decode phases and outputs `OptimizationGoal`

**Benchmark evidence** (2026-03-31):
- **NVIDIA A100**: `warpcta8pipe` wins 297/300 configurations for fp16/bf16 + page_block_size=256 + head_size=128
- **Iluvatar BI-V150**: ~50/50 split between `warp` and `warpcta8pipe` depending on head_size and batch size:
  - head_size <= 64: `warpcta8pipe` always wins
  - head_size = 128, batch >= 4 or q_per_seq >= 64: `warp` wins
  - head_size = 128, low batch: `warpcta8pipe` wins

These hardware-specific patterns motivate an independent dispatch module that can be extended per-device without touching operator code.

## Architecture

```
MutualAwarenessAnalyzer                KernelDispatcher (this module)
├── PhaseDetector                      ├── 3D lookup table:
├── ResourceSensor                     │   (OpType, DeviceType, OptimizationGoal)
├── IntentGenerator                    │       -> KernelSelectFn
│   -> OptimizationGoal                │
└──────────────┬───────────────────────┘
               |
        .cu / .maca files (consumers)
        default_prefill_kernel(info) {
            KernelDispatcher::instance()
                .selectKernel(op, device, &info)
                -> queries analyzer for current goal
                -> looks up registered rule
                -> returns kernel name (or nullptr for fallback)
        }
```

### Call Flow

```
DISPATCH_KERNEL macro
  -> default_prefill_kernel(info)
       -> KernelDispatcher::selectKernel(op, device, &info)
            1. Query MutualAwarenessAnalyzer for current OptimizationGoal
               (LATENCY_FIRST / THROUGHPUT_FIRST / MEMORY_SAFE / STABILITY_FIRST)
            2. Look up (OpType, DeviceType, Goal) in table
            3. Call registered KernelSelectFn(info) -> kernel name
       -> If nullptr: use original hardcoded heuristic as fallback
```

### Conditional Compilation

- `ENABLE_MUTUAL_AWARENESS`: When OFF, the entire dispatch module is bypassed; `.cu` files use their original hardcoded heuristics unchanged.
- `ENABLE_*_API` (NVIDIA/ILUVATAR/METAX/ALI): Determines which device's rules are registered at static init time.

## File Structure

```
include/infinicore/dispatch/
├── kernel_dispatcher.hpp        # KernelDispatcher class (singleton, thread-safe)
├── prefill_dispatch_rules.h     # Benchmark-driven rules + static registration
└── README.md                    # This file

src/infinicore/dispatch/
├── kernel_dispatcher.cc         # selectKernel() implementation
└── prefill_dispatch_rules.cc    # Stub (rules register from .cu compilation units)
```

### Why rules live in a header, not a .cc file

`PagedAttentionPrefillInfo` is defined in `src/infiniop/ops/paged_attention_prefill/info.h`, which is NOT on the include path for `src/infinicore/`. The `.cu` / `.maca` files already include `info.h` through their own headers, so `prefill_dispatch_rules.h` is included there and can access the struct.

## Adding Support for a New GPU

1. Add an `#elif defined(ENABLE_<VENDOR>_API)` block in `prefill_dispatch_rules.h`
2. Implement `KernelSelectFn` functions using benchmark data from the new hardware
3. Register them in the `registerPrefillRules()` function
4. No changes needed to `.cu` files or `kernel_dispatcher.cc`

## TODO

### Must-Have (Before Merge to Production)

- [ ] **Compilation verification**: Build with `ENABLE_MUTUAL_AWARENESS` ON/OFF crossed with `ENABLE_NVIDIA_API` / `ENABLE_ILUVATAR_API` / `ENABLE_METAX_API` on their respective platforms
- [ ] **Fallback verification**: Confirm that with `ENABLE_MUTUAL_AWARENESS=OFF`, behavior is identical to before (no regression)
- [ ] **Remote benchmark on BI-V150**: Validate that the Iluvatar dispatch rules produce correct kernel choices and match or improve latency vs the previous hardcoded `"warp"` default
- [ ] **NVIDIA A100 benchmark**: Verify `warpcta8pipe` selection under LATENCY_FIRST goal matches prior benchmark results

### Should-Have

- [ ] **BI-200 benchmark & rules**: Iluvatar BI-200 may have different performance characteristics; need new benchmark data and corresponding rules
- [ ] **Debug logging**: Add `INFINIOP_DEBUG_PREFILL_DISPATCH=1` env var support to log which kernel was selected and why (goal, device, rule match)
- [ ] **Thread safety audit**: `selectKernel()` reads `table_[]` without lock (reads are concurrent-safe for `std::array` if writes are done before reads, which static init guarantees). Verify no late registration can race.

### Nice-to-Have

- [ ] **Python bindings**: `infinicore.dispatch.get_dispatcher()` / `override_kernel()` for debugging and manual override
- [ ] **JSON-based rule loading**: Load dispatch rules from a config file instead of compiled-in functions (enables benchmark -> auto-generate rules pipeline)
- [ ] **Per-model-config override**: Allow model deployment configs to pin specific kernels
- [ ] **Extend to other ops**: Apply the same KernelDispatcher pattern to other operators beyond paged_attention_prefill (e.g., decode attention, GEMM)
