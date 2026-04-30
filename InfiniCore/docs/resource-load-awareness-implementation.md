# BI-V150 / MetaX 资源负载感知实现说明

本文记录 MutualAwarenessAnalyzer 的资源负载感知方案、平台 API 依据、当前实现状态和实机验证入口。

天数 BI-V150 针对“任务模块与资源模块应在 10s 内完成需求分析”的实机验收状态见 [天数 BI-V150 任务/资源负载感知 10s 目标对齐状态](./iluvatar-10s-acceptance-status.md)。

## 目标

目标是在天数 BI-V150 和沐曦 MetaX 上输出真实的需求分析结果，而不是脚本模拟值。分析结果由两类信号融合：

1. 任务负载信号：`OpTraceRing` 记录生产路径中的 op 类型、shape、dtype、device，`PhaseDetector` 识别 prefill / decode / dense / communication 等阶段。
2. 资源负载信号：`infinirtGetDeviceResourceSnapshot()` 返回每张加速卡的 memory、compute utilization、memory bandwidth utilization、kernel time ratio、communication ratio / bytes。

最终输出仍是 `OptimizationIntent`，包含全局阶段、主导瓶颈、优化目标和 per-device local intent。

## 平台接口依据

### 天数 BI-V150 / CoreX

运行时侧复用 CUDA ABI 分支：

- 设备/显存：`cudaGetDeviceCount`、`cudaSetDevice`、`cudaMemGetInfo`。
- 事件计时：`cudaEventCreate`、`cudaEventRecord`、`cudaEventQuery`、`cudaEventElapsedTime`。
- 通信采样：`infiniccl` AllReduce 前后插入事件，完成后进入最近 1s 窗口统计。

管理库侧当前实现使用 CoreX IXML。公开 Go binding `deep-spark/go-ixml` 的 `api.h` 显示 IXML 提供 NVML-compatible API：

- `nvmlInit_v2`
- `nvmlDeviceGetHandleByIndex_v2`
- `nvmlDeviceGetName`
- `nvmlDeviceGetUtilizationRates`
- `nvmlDeviceGetMemoryInfo`
- `nvmlDeviceGetPcieThroughput`
- `nvmlDeviceGetPowerUsage`
- `nvmlDeviceGetTemperature`

第二轮 API 搜索还确认了天数侧存在更完整的 IXDCGM 管理栈。公开 Go binding `deep-spark/go-ixdcgm` 说明 IXDCGM 需要 `libixdcgm.so` 和 IXDCGM SDK，并支持三种 hostengine 模式：Embedded、Standalone、StartHostengine。它不是第三个平台，而是 BI-V150 / CoreX 的另一条资源感知接口线，适合后续增强更细的诊断字段：

- `ixdcgm.Init(ixdcgm.Embedded)` / `dcgmInit` / hostengine 连接。
- `ixdcgm.GetDeviceStatus(gpuId)`：power、temperature、GPU utilization、memory utilization、PCIe Tx/Rx throughput、PCIe replay counter、显存 total/used/free。
- `ixdcgm.GetDeviceProfStatus(gpuId)`：`SmActive`、`SmOccupancy`、`DramActive`。
- `ixdcgm.GetDeviceRunningProcesses(gpuId)`：进程级 GPU 显存占用。
- DCGM-like field ID 包括 `DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_MEM_COPY_UTIL`、`DCGM_FI_DEV_PCIE_TX_THROUGHPUT`、`DCGM_FI_DEV_PCIE_RX_THROUGHPUT`、`DCGM_FI_PROF_SM_ACTIVE`、`DCGM_FI_PROF_DRAM_ACTIVE`、`DCGM_FI_PROF_PCIE_TX_BYTES`、`DCGM_FI_PROF_PCIE_RX_BYTES`。

当前主线优先选择 IXML，是因为它不要求 hostengine，并且已经覆盖 memory / compute utilization / PCIe throughput 这些 analyzer 的核心字段；IXDCGM 应作为目标机验证后的增强路径，用于补全 SM/Dram active、进程和诊断类指标。

运行命令侧主要使用 `ixsmi` 验证设备、驱动和实时利用率。

### 沐曦 MetaX

运行时侧使用 MACA/HCR 或 MCR：

- 设备/显存：`hcGetDeviceCount`、`hcSetDevice`、`hcMemGetInfo`。
- 事件计时：`hcEventCreate`、`hcEventRecord`、`hcEventQuery`、`hcEventElapsedTime`。
- 通信采样：`hcclAllReduce` 前后插入 `hcEvent_t`，完成后进入最近 1s 窗口统计。

管理库侧当前实现使用 MXSML extension。公开 Go binding `MetaX-MACA/go-mxsml` 的 `MxSmlExtension.h` 显示扩展 API：

- `mxSmlExInit`
- `mxSmlExGetDeviceHandleByIndex`
- `mxSmlExDeviceGetName`
- `mxSmlExDeviceGetUtilization`
- `mxSmlExDeviceGetMemoryInfo`
- `mxSmlExGetPcieThroughput`
- `mxSmlExGetPowerUsage`
- `mxSmlExDeviceGetTemperature`
- `mxSmlExDeviceGetComputeRunningProcesses`
- `mxSmlExDeviceGetFieldValues`

同一个公开 binding 还提供 base MXSML 接口 `pkg/mxsml`。这组接口比 extension 更贴近 `mx-smi`/设备管理原始结构，适合作为 MetaX 后续增强的资源带宽和互联指标来源：

- `MxSmlGetMemoryInfo`
- `MxSmlGetHbmBandWidth`
- `MxSmlGetDmaBandwidth`
- `MxSmlGetPcieThroughput`
- `MxSmlGetMetaXLinkBandwidth`
- `MxSmlGetMetaXLinkTrafficStat`
- `MxSmlGetNumberOfProcess`
- `MxSmlGetProcessInfo`
- `MxSmlGetDeviceTopology`
- `MxSmlGetDeviceIpUsage`
- `MxSmlGetXcoreApUsage`

当前主线优先选择 MXSML extension，是因为它提供接近 NVML 风格的 device handle、utilization、memory、PCIe throughput，映射到现有 `DeviceResourceSnapshot` 最直接；base MXSML 应作为目标机验证后的增强路径，用于补全 HBM/DMA/MetaXLink、进程和更细粒度 IP 使用率。

实现中按以下顺序 `dlopen`：

1. `libmxsml.so`
2. `/opt/mxdriver/lib/libmxsml.so`
3. `/opt/maca/lib/libmxsml.so`
4. `/opt/mxn100/lib/libmxsml.so`

运行命令侧主要使用 `mx-smi` 验证设备、驱动和实时利用率。

## 当前实现状态

| 层次 | 天数 BI-V150 | 沐曦 MetaX |
| --- | --- | --- |
| infinirt 后端注册 | 复用 CUDA-like 后端，`ENABLE_ILUVATAR_API` | 独立 `infinirt-metax` 后端 |
| 设备/内存基础 API | CoreX CUDA ABI | MACA `hc*` / `mc*` |
| 资源快照 - 真实设备型号 | IXML `nvmlDeviceGetName`，用于区分 BI-V150 / BI-200 等 SKU | MXSML `mxSmlExDeviceGetName`，用于区分 C500 / C550 等 SKU |
| 资源快照 - 内存容量 | `cudaMemGetInfo` | `hcMemGetInfo`，MXSML 可覆盖 |
| 资源快照 - compute_util | `dlopen libixml.so` + NVML-compatible API | `dlopen libmxsml.so` + `mxSmlExDeviceGetUtilization` |
| 资源快照 - mem_bw_util | IXML `nvmlDeviceGetUtilizationRates.memory` | MXSML `mxSmlExUtilization.memory` |
| 资源快照 - kernel_time | 由 compute utilization 估计 | 由 compute utilization 估计 |
| 资源快照 - communication | CUDA-like AllReduce event sampling | HCCL AllReduce event sampling |
| 硬件层测试 | `infinirt-test-analyzer-hw` + `scripts/test_iluvatar_analyzer.sh` | `infinirt-test-analyzer-hw` + `scripts/test_metax_analyzer.sh` |
| OpTrace 自动埋点 | `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 已在生产 op 路径记录 | 同左；已补 `AllReduce -> ALLREDUCE` 注册 |
| `collectRuntimeResourceSnapshots()` | 枚举所有非 CPU runtime 设备并拉取快照 | 同左 |
| 真实分析输出 demo | `src/analyzer-demo/main.cc` + `scripts/analyzer_demo.py --configure iluvatar`；`analyzer-load-demo` 支持真实 GPU 负载矩阵 | `src/analyzer-demo/main.cc` + `scripts/analyzer_demo.py --configure metax`；`analyzer-load-demo` 支持真实 GPU 负载矩阵 |
| InfiniLM 集成 | Analyzer C++ API 已可调用；InfiniLM 侧仍需接入调用点 | 同左 |

## 代码入口

- `src/infinirt/cuda/infinirt_cuda.cu`
  - NVIDIA / Iluvatar / CUDA-like 后端的 IXML/NVML utilization 和通信采样。
- `src/infinirt/metax/infinirt_metax.cc`
  - MetaX 内存快照、MXSML 动态加载、utilization 快照、通信窗口统计。
- `src/infiniccl/metax/infiniccl_metax.cc`
  - MetaX AllReduce 事件采样与通信字节估计。
- `src/infinicore/analyzer/mutual_awareness_analyzer.cc`
  - `collectRuntimeResourceSnapshots()` 汇总所有 runtime 设备快照。
- `include/infinicore/analyzer/op_type_registry.hpp`
  - 生产 OpTrace 的 class name 到 `OpType` 映射。
- `src/analyzer-demo/main.cc`
  - 真实需求分析输出 demo。
- `src/analyzer-load-demo/`
  - 真实 GPU 负载与任务 trace 矩阵 demo；CUDA/CoreX 后端用 `.cu` 计算压测，MetaX 后端用 `.maca` 计算压测，其余显存和 D2D copy 负载走 `infinirt`。
  - 同时包含 `Resource Replay` 段，通过显式 `DeviceResourceSnapshot` 覆盖 high-memory / high-compute / high-bandwidth / high-communication 场景，用于验证需求分析规则会随资源条件切换；该段不声称是真实 GPU 负载。
- `scripts/analyzer_demo.py`
  - 构建/运行真实 C++ demo 的薄入口，不再输出模拟数据。
- `scripts/analyzer_load_demo.py`
  - 构建/运行不同 GPU 负载 x 不同任务 trace 的需求分析矩阵。

## 验证命令

本机无目标硬件时，至少验证通用 runtime 硬件测试目标：

```bash
cd InfiniCore
xmake f -c --mutual-awareness=y --cpu=y --omp=n -y
xmake build infinirt-test-analyzer-hw
xmake run infinirt-test-analyzer-hw
```

BI-V150 目标机：

```bash
cd InfiniCore
which ixsmi
ixsmi -L
ixsmi
find /usr/local/corex* -name 'libixml.so*' -o -name 'libixdcgm.so*'
bash scripts/test_iluvatar_analyzer.sh
python3 scripts/analyzer_demo.py --configure iluvatar
python3 scripts/analyzer_load_demo.py --configure iluvatar
```

MetaX 目标机：

```bash
cd InfiniCore
which mx-smi
mx-smi
find /opt/mxdriver /opt/maca /opt/mxn100 -name 'libmxsml.so*'
bash scripts/test_metax_analyzer.sh
python3 scripts/analyzer_demo.py --configure metax
python3 scripts/analyzer_load_demo.py --configure metax
```

如 MetaX 现场使用 MC 运行时：

```bash
cd InfiniCore
METAX_USE_MC=1 bash scripts/test_metax_analyzer.sh
python3 scripts/analyzer_demo.py --configure metax --extra-config --use-mc=y
python3 scripts/analyzer_load_demo.py --configure metax --extra-config --use-mc=y
```

## 降级行为

- 找不到 `libixml.so` 或 `libmxsml.so` 时，基础内存快照仍通过 runtime API 返回，但 compute / bandwidth utilization 不置 valid bit。
- 真实设备型号只来自管理库：天数走 `nvmlDeviceGetName`，沐曦走 `mxSmlExDeviceGetName`。读取失败时 `INFINIRT_RESOURCE_FIELD_DEVICE_NAME` 不置位，demo 只显示平台族和 `model unavailable`，不会把 BI-V150 / BI-200 / C500 写死到输出里。
- `kernel_time_ratio` 当前由 compute utilization 估计，因此会在 `estimated_fields` 中标记。
- 通信占比来自最近 1s 内已完成的 AllReduce event sample；没有通信或事件未完成时，通信字节为 0。
- analyzer 输出会根据 valid bits 计算 `resource_confidence`，不会把缺失的管理库字段当成真实 0 利用率。

## 目标机到手后的快速判定

### 天数 BI-V150

1. `ixsmi -L` 确认卡可见；`ixsmi` 看实时利用率是否有非零字段。
2. `scripts/test_iluvatar_analyzer.sh` 会检查核心 IXML 符号：
   - 必需：`nvmlInit_v2`、`nvmlDeviceGetHandleByIndex_v2`、`nvmlDeviceGetName`、`nvmlDeviceGetUtilizationRates`、`nvmlDeviceGetMemoryInfo`、`nvmlShutdown`。
   - 增强：`nvmlDeviceGetPcieThroughput`、`nvmlDeviceGetPowerUsage`、`nvmlDeviceGetTemperature`。
3. 如果存在 `libixdcgm.so`，脚本会额外探测 IXDCGM / DCGM-like 符号：
   - 初始化/连接：`dcgmInit`、`dcgmShutdown`、`dcgmStartEmbedded`、`dcgmConnect`。
   - 指标：`dcgmGetLatestValuesForFields`、`dcgmWatchFields`、`dcgmGetDeviceAttributes`。
4. 当前实现验证看 `infinirt-test-analyzer-hw` 与 `analyzer-demo` 输出；IXDCGM 字段暂作为增强候选，不影响主线通过/失败。
5. 如需展示不同 GPU 负载下的输出差异，运行 `analyzer-load-demo`。它会依次制造 idle、memory pressure、D2D copy、compute kernel、mixed 负载，并对 Prefill、Decode、GEMM/MLP、KV Cache、AllReduce trace 输出 phase / bottleneck / goal / live resource 表格。
6. 如果现场 runtime 对单进程显存分配有限制，`memory_pressure` 会输出实际持有显存，不伪造高水位；后续 `Resource Replay` 会用显式资源快照验证高显存/高带宽/高通信下的决策切换。

### 沐曦 MetaX

1. `mx-smi` 确认驱动、设备、利用率和显存字段可见。
2. `scripts/test_metax_analyzer.sh` 会检查核心 MXSML extension 符号：
   - 必需：`mxSmlExInit`、`mxSmlExGetDeviceHandleByIndex`、`mxSmlExDeviceGetName`、`mxSmlExDeviceGetUtilization`、`mxSmlExDeviceGetMemoryInfo`。
   - 增强：`mxSmlExGetPcieThroughput`、`mxSmlExGetPowerUsage`、`mxSmlExDeviceGetTemperature`、`mxSmlExDeviceGetComputeRunningProcesses`、`mxSmlExDeviceGetFieldValues`。
3. 同一个 `libmxsml.so` 还会检查 base MXSML 高阶字段：
   - `MxSmlGetMemoryInfo`、`MxSmlGetHbmBandWidth`、`MxSmlGetDmaBandwidth`、`MxSmlGetPcieThroughput`。
   - `MxSmlGetMetaXLinkBandwidth`、`MxSmlGetMetaXLinkTrafficStat`。
   - `MxSmlGetNumberOfProcess`、`MxSmlGetProcessInfo`、`MxSmlGetDeviceTopology`。
4. 当前实现验证仍看 extension utilization/memory、HCCL communication sampling、`analyzer-demo` 与 `analyzer-load-demo`；base MXSML 高阶字段暂作为增强候选。
5. 如需展示不同 GPU 负载下的输出差异，运行 `analyzer-load-demo`。它会依次制造 idle、memory pressure、D2D copy、compute kernel、mixed 负载，并对 Prefill、Decode、GEMM/MLP、KV Cache、AllReduce trace 输出 phase / bottleneck / goal / live resource 表格。

## MetaX C500 实测结果（2026-04-30）

测试命令：

```bash
cd /data/InfiniGraph/InfiniCore
METAX_USE_MC=1 bash scripts/test_metax_analyzer.sh
```

结论：

- 环境：`mx-smi 2.2.9`，MetaX C500 x1，MACA `3.2.1.10`，Kernel Mode Driver `3.0.11`。
- MXSML extension 与 base 高阶符号均可见，`mxSmlExDeviceGetUtilization` / `mxSmlExDeviceGetMemoryInfo` 可用于当前资源快照。
- `infinirt-test-analyzer-hw`：7 passed, 0 failed。
- `analyzer-demo`：6/6 phase 正确，资源模块成功采集 1 个 MetaX 加速器，P99 `19.81 ms`，小于 10s。
- `analyzer-load-demo`：真实负载 live 段 25/25 phase 正确，`Resource Replay` 段 25/25 phase 正确；整体最坏单次分析 `41.87 ms`，小于 10s。
- `analyzer-test`：42 passed, 0 failed。

`analyzer-load-demo` 的真实负载段显示 MetaX 管理库读数会随负载变化：

| 场景 | 实测资源变化 | 说明 |
| --- | --- | --- |
| idle | mem `1.3%`, gpu `0.0%`, bw `1.0%` | 空闲基线 |
| memory_pressure | held `13.23 GiB`, mem `21.9%`, bw `21.0%` | 当前 MACA runtime 对单进程可分配显存存在上限，demo 记录真实可达压力 |
| bandwidth_copy | mem `2.8%`, gpu `41.0%`, bw `2.0%` | D2D copy 在该驱动上主要反映到 GPU utilization |
| compute_kernel | mem `1.5%`, gpu `98.0%`, bw `1.0%` | MetaX `.maca` 计算压测能打满 compute utilization |
| mixed | held `9.73 GiB`, mem `18.2%`, gpu `91.0%`, bw `18.0%` | 显存持有 + copy + compute 的组合压力 |

`Resource Replay` 段不声称是真实 GPU 负载，但验证了需求分析规则在显式资源快照下会切换：

- high-memory snapshot：所有任务切到 `memory_bound + memory_safe`，本地瓶颈为 `memory_bound`。
- high-bandwidth snapshot：Prefill / Decode / GEMM / KV 等任务切到 `bandwidth_bound` 或相应 latency/throughput 目标，本地瓶颈为 `bandwidth_bound`。
- high-communication snapshot：所有任务切到 `communication_bound + stability_first`，本地瓶颈为 `communication_bound`。

仍需注意的资源语义边界：

- `kernel_time_ratio` 当前由 compute utilization 估计，不是独立 profiler 采样。
- 单卡环境下 `ALLREDUCE` 任务 trace 只验证任务侧 communication phase；真实通信字节/占比需要多卡 HCCL/IXCCL 事件采样。
- `Resource Replay` 用于规则覆盖和验收展示，不应写成真实 GPU 负载。

## 资料来源

- IXML / NVML-compatible API: https://gitee.com/deep-spark/go-ixml
- IXDCGM / DCGM-like API: https://gitee.com/deep-spark/go-ixdcgm
- MXSML extension / base API: https://github.com/MetaX-MACA/go-mxsml
- MetaX 部署与 `mx-smi` 现场工具参考: https://docs.opencloudos.org/en/OC9/ai-deployment/GPU-optimization-practice/metax-deployment/
