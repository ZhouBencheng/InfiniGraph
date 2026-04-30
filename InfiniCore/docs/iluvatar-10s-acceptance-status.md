# 天数 BI-V150 任务/资源负载感知 10s 目标对齐状态

本文记录天数 BI-V150 上 Mutual Awareness Analyzer 针对“任务模块与资源模块应在 10s 内完成需求分析”这一目标的实现和实测状态。

结论：当前天数侧实现已经对齐这个 10s 目标。2026-04-30 在 BI-V150 / CoreX 4.3.0 实机上，`analyzer-demo` 的 P99 为 22.48 ms，`analyzer-load-demo` 的最慢单次分析为 38.96 ms，均远低于 10000 ms 阈值。

## 目标拆解

| 目标项 | 当前天数侧实现 | 对齐状态 |
| --- | --- | --- |
| 任务模块能识别当前任务阶段 | `OpTraceRing` + `PhaseDetector`，基于 op 类型、shape、dtype、device 识别 prefill / decode / dense / KV cache / communication | 已实现并通过测试 |
| 资源模块能读取真实设备负载 | `infinirtGetDeviceResourceSnapshot()` 通过 CoreX CUDA ABI + IXML 采集显存、GPU utilization、memory bandwidth utilization，通信使用 AllReduce event sample | 已实现并通过 BI-V150 实机验证 |
| 能输出融合后的需求分析结果 | `MutualAwarenessAnalyzer::analyze()` 输出 `OptimizationIntent`，包含 phase、primary bottleneck、optimization goal、per-device resource view | 已实现并通过 demo 验证 |
| 任务+资源需求分析在 10s 内完成 | `analyze()` 单次端到端分析耗时按 demo 输出统计 | 已通过，实测最大 38.96 ms |

## 实现范围

天数侧复用 CUDA-like runtime 路径，构建时启用 `--iluvatar-gpu=y --mutual-awareness=y`，并定义 `ENABLE_ILUVATAR_API`。

资源模块当前接入的真实信号：

- 设备枚举：`cudaGetDeviceCount`。
- 显存容量和空闲量：`cudaMemGetInfo`。
- GPU compute utilization：动态加载 `libixml.so` 后调用 NVML-compatible `nvmlDeviceGetUtilizationRates`。
- memory bandwidth utilization：同一 IXML utilization 结构中的 memory 字段。
- kernel time ratio：当前由 compute utilization 估计，并在快照中标记为估计字段。
- communication ratio / bytes：infiniccl AllReduce 前后记录 CoreX event，完成后进入最近 1s 窗口统计。

任务模块当前接入的信号：

- 生产路径 op 可通过 `traceOp()` 写入 `OpTraceRing`。
- 图/op 注册路径已经补到 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 和 op type registry。
- `PhaseDetector` 基于最近 op 窗口识别任务阶段。

融合输出：

- 输出类型为 `OptimizationIntent`。
- 全局字段包括 `current_phase`、`primary_bottleneck`、`goal`、`compute_intensity`、`confidence`。
- 设备字段包括 memory usage、compute utilization、bandwidth utilization、communication ratio、local bottleneck、resource confidence。

## 实机环境

验证机器：

- 平台：天数 BI-V150。
- CoreX：4.3.0。
- `ixsmi` 可见 1 张 BI-V150。
- 运行环境需要：

```bash
export PATH=/root/.local/bin:/usr/local/corex-4.3.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.3.0/lib64:${LD_LIBRARY_PATH:-}
export XMAKE_ROOT=y
```

`LD_LIBRARY_PATH` 必须包含 CoreX `lib64`，否则 `ixsmi` 和 IXML 动态库加载会失败。

## 验证结果

### analyzer-test

命令：

```bash
xmake build analyzer-test
xmake run analyzer-test
```

结果：

```text
Results: 42 passed, 0 failed
```

覆盖范围：

- `OpTraceRing`
- `PhaseDetector`
- `ResourceSensor`
- `IntentGenerator`
- `MutualAwarenessAnalyzer`
- attention dispatch 对 analyzer goal 的消费路径
- 性能微基准

### infinirt-test-analyzer-hw

命令：

```bash
xmake build infinirt-test-analyzer-hw
xmake run infinirt-test-analyzer-hw
```

结果：

```text
Device: Iluvatar x 1
getMemInfo                 PASSED
snapshot_memory            PASSED
snapshot_utilization       PASSED (compute=0.0%, mem_bw=1.0%)
snapshot_communication     PASSED
event_timing               PASSED
malloc_memcpy              PASSED
multi_device_snapshot      PASSED (skip: need 2+ GPUs, have 1)
Results: 7 passed, 0 failed
```

这说明天数侧的基础 runtime、显存快照、IXML utilization、event timing、基础 memcpy 都已在实机路径上通过。

### analyzer-demo

命令：

```bash
python3 scripts/analyzer_demo.py --configure iluvatar
```

结果摘要：

```text
任务模块 (Phase 识别)   : 6/6 正确
资源模块 (设备感知)     : 1 个加速器成功采集
延迟需求                : <=10 s PASSED
模块状态                : READY FOR INTEGRATION
```

性能段输出：

```text
采样次数              : 100
min                   : 21.19 ms
avg                   : 21.43 ms
P50                   : 21.24 ms
P95                   : 21.35 ms
P99                   : 22.48 ms
max                   : 37.44 ms
需求阈值              : <= 10000 ms
裕量倍率              : x445
验收结论              : PASSED
```

本文按性能段中的保守 P99 值 22.48 ms 判断 10s 目标。

`analyzer-demo` 的重点是展示常规需求分析输出：任务阶段、瓶颈、优化目标、资源快照、策略建议和 10s 延迟验收。

### analyzer-load-demo

命令：

```bash
python3 scripts/analyzer_load_demo.py --configure iluvatar --warmup-ms 1500
```

这个 demo 用真实 GPU 负载压测资源模块，再用不同任务 trace 触发不同分析输出。GPU 负载是真实的 CoreX/CUDA 工作负载；任务 trace 是 demo 注入的合成 OpTrace 窗口，用于稳定复现不同任务形态。

实测摘要：

```text
GPU Load: idle
Prefill   -> prefill,       bandwidth_bound,     throughput_first, mem=0.1%,  gpu=0.0%,   bw=1.0%
Decode    -> decode,        compute_bound,       latency_first,    mem=0.1%,  gpu=0.0%,   bw=1.0%
KV Cache  -> kv_cache,      memory_bound,        memory_safe,      mem=0.1%,  gpu=0.0%,   bw=1.0%
AllReduce -> communication, communication_bound, stability_first,  mem=0.1%,  gpu=0.0%,   bw=1.0%

GPU Load: memory_pressure
held memory: 28.09 GiB
所有任务 -> memory_bound + memory_safe, mem=88.0%, gpu=0.0%, bw=89.0%

GPU Load: bandwidth_copy
copy buffer: 512 MiB x2
任务 phase 仍按任务 trace 区分，gpu=100.0%, mem=3.2%, bw=4.0%

GPU Load: compute_kernel
compute buffer: 128 MiB
任务 phase 仍按任务 trace 区分，gpu=100.0%, mem=0.5%, bw=2.0%

GPU Load: mixed
held memory: 20.75 GiB, copy buffer: 512 MiB x2, compute buffer: 128 MiB
任务 phase 仍按任务 trace 区分，mem=68.5%, gpu=100.0%, bw=69.0%
```

最终结果：

```text
Summary: phase=25/25 correct, worst analyze latency=38.96 ms, requirement=PASSED
```

这个结果说明：

- 在空闲、copy、compute、mixed 等负载下，任务模块仍能保持 phase 识别正确。
- 在高显存压力下，资源模块会把全局决策推向 `memory_bound + memory_safe`，说明真实资源压力会参与最终需求分析。
- 单次需求分析延迟最坏 38.96 ms，距离 10s 阈值有约 256 倍裕量。

## 不同任务下输出有什么不同

在同一资源状态下，任务 trace 会改变 phase、bottleneck 和 goal：

| 任务形态 | 典型 phase | 典型 bottleneck | 典型 goal |
| --- | --- | --- | --- |
| Prefill / long-seq attention + GEMM | `prefill` | `bandwidth_bound` | `throughput_first` |
| Decode / single-token attention + MLP | `decode` | `compute_bound` | `latency_first` |
| GEMM/MLP dense segment | `gemm_mlp_dense` | `compute_bound` | `throughput_first` |
| KV cache manipulation | `kv_cache` | `memory_bound` | `memory_safe` |
| AllReduce-heavy window | `communication` | `communication_bound` | `stability_first` |

## 不同 GPU 负载下输出有什么不同

在同一任务 trace 下，资源状态会改变资源列，必要时覆盖全局瓶颈和优化目标：

| GPU 负载 | 资源读数表现 | 对输出的影响 |
| --- | --- | --- |
| idle | mem/gpu/bw 均低 | 主要由任务 trace 决定 phase、bottleneck、goal |
| memory_pressure | mem 88.0%，bw 89.0% | 全部任务都切到 `memory_bound + memory_safe` |
| bandwidth_copy | gpu 100.0%，但 IXML memory bandwidth 字段仅 4.0% | phase 按任务区分；当前 IXML 对 D2D copy 更明显反映在 GPU utilization 上 |
| compute_kernel | gpu 100.0%，bw 2.0% | phase 按任务区分；compute 压力进入资源视图 |
| mixed | mem 68.5%，gpu 100.0%，bw 69.0% | 同时呈现显存和 GPU 压力；未超过 memory-safe 阈值时仍保留任务主导输出 |

## 和 10s 目标的对齐判断

对齐。

理由：

1. 任务模块已能从 OpTrace 窗口输出阶段和任务侧瓶颈，`analyzer-test` 与 `analyzer-demo` 均通过。
2. 资源模块已能在 BI-V150 上采集真实显存、compute utilization、memory bandwidth utilization 和通信窗口信息，`infinirt-test-analyzer-hw` 通过。
3. 融合模块已能把任务 trace 与资源快照合成 `OptimizationIntent`，并在高显存压力下改变全局瓶颈和优化目标。
4. 端到端分析延迟远低于 10s：常规 demo P99 22.48 ms，负载 demo 最坏 38.96 ms。

当前边界：

- `analyzer-load-demo` 的 GPU 负载是真实负载，资源读数是真实读数；任务 trace 是合成窗口，不等价于真实 LLM 生产推理 trace。
- InfiniLM 侧调用点仍需接入 analyzer API，当前验证到 analyzer 模块自身和 demo 输出。
- `kernel_time_ratio` 仍是由 compute utilization 估计，不是独立 profiler 字段。
- 单卡 BI-V150 环境下多卡通信无法完全验证；通信结构和 event sample 路径已存在，单卡硬件测试中多设备项按预期 skip。

## 快速复现

```bash
cd /root/InfiniGraph/InfiniCore
export PATH=/root/.local/bin:/usr/local/corex-4.3.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.3.0/lib64:${LD_LIBRARY_PATH:-}
export XMAKE_ROOT=y

xmake f -c -y --mutual-awareness=y --iluvatar-gpu=y --ccl=y
xmake build analyzer-test
xmake build infinirt-test-analyzer-hw
xmake build analyzer-demo
xmake build analyzer-load-demo

xmake run analyzer-test
xmake run infinirt-test-analyzer-hw
xmake run analyzer-demo
xmake run analyzer-load-demo 1500
```
