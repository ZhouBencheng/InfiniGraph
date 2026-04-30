# Iluvatar Analyzer Support — PR 说明与测试指南

> Historical note: this document is an older PR/debug guide for the initial Iluvatar enablement work.
> It records issues that existed before the current BI-V150 acceptance path. For current status,
> use `docs/resource-load-awareness-implementation.md` and `docs/iluvatar-10s-acceptance-status.md`.

## 1. 背景：为什么要改

Mutual Awareness Analyzer 模块在 NVIDIA GPU 上已完整实现并通过测试，但在天数智芯（Iluvatar）BI-V150 上存在两个阻塞性问题：

### 问题 A：编译失败 — 通信采样结构体被 NVIDIA 宏保护

`infinirt_cuda.cu` 中，`PendingCommunicationSample`、`CommunicationStatsStore` 等结构体和辅助函数全部在 `#if defined(ENABLE_NVIDIA_API) && !defined(_WIN32)` 保护之内。但 `populateCommunicationSnapshot()`（被 `getDeviceResourceSnapshot` 调用）和 `recordCommunicationSample()`（被 `infiniccl` 调用）在宏保护之外引用了这些符号。

当使用 `--iluvatar-gpu=y` 编译时，宏 `ENABLE_ILUVATAR_API` 被定义而非 `ENABLE_NVIDIA_API`，导致这些符号未定义，编译报错：

```
error: use of undeclared identifier 'communicationStatsStore'
error: use of undeclared identifier 'PendingCommunicationSample'
```

### 问题 B：`std::min` 与 Corex SDK 冲突

Corex SDK 4.3.0 的 `cuda_wrappers/algorithm` 覆盖了 `std::min` 模板，导致 `std::min(1.0, double_expr)` 出现 `float vs double` 类型推导冲突：

```
error: no matching function for call to 'min'
candidate template ignored: deduced conflicting types for parameter '_Tp' ('float' vs 'double')
```

### 问题 C：GPU 利用率数据缺失（非阻塞，但影响分析质量）

原代码中 NVML utilization 加载仅在 `ENABLE_NVIDIA_API` 下启用。调研发现天数的管理库 **IXML** (`libixml.so`) 采用了 NVML 兼容 API（函数名、结构体、返回值完全相同），可直接复用 NVML 动态加载逻辑来获取 GPU compute/bandwidth utilization。

---

## 2. 改了什么

### 文件 1：`InfiniCore/src/infinirt/cuda/infinirt_cuda.cu`

| 改动 | 目的 |
|------|------|
| 通信结构体（`PendingCommunicationSample` 等）和辅助函数移出 `#if ENABLE_NVIDIA_API` guard | 所有 CUDA-like 后端（Iluvatar/QY/Hygon/Ali）共享通信采样能力 |
| `populateCommunicationSnapshot()` 合并到同一个 anonymous namespace | 消除前向引用问题 |
| `std::min(1.0, ...)` 替换为三目运算符 | 规避 Corex `cuda_wrappers/algorithm` 与标准库的冲突 |
| `#include <dlfcn.h>` guard 从 `ENABLE_NVIDIA_API` 扩展为 `!defined(_WIN32)` | Iluvatar 也需要 `dlopen` |
| NVML/IXML 管理库加载 guard 从 `ENABLE_NVIDIA_API` 扩展为 `ENABLE_NVIDIA_API \|\| ENABLE_ILUVATAR_API` | Iluvatar 通过 `libixml.so` 获取 utilization |
| dlopen 候选列表根据宏切换：NVIDIA → `libnvidia-ml.so*`，Iluvatar → `libixml.so` | 加载正确的管理库 |

### 文件 2：`InfiniCore/src/infinirt-test/test_analyzer_hw.cc`（新增）

针对硬件层的测试程序，包含 7 个测试项，自动检测设备类型。

### 文件 3：`InfiniCore/xmake/test.lua`

新增 `infinirt-test-analyzer-hw` 构建目标。

---

## 3. 未完成事项

以下工作需要在天数服务器上实机验证后再确认：

| 项目 | 状态 | 说明 |
|------|------|------|
| `cudaMemGetInfo` on Iluvatar | 待验证 | Corex SDK 理论兼容，需实测 |
| `cudaEvent*` 系列（Create/Record/Query/ElapsedTime） | 待验证 | 通信采样依赖 event timing |
| `libixml.so` dlopen + nvml* 符号 | 待验证 | IXML 的 NVML 兼容性来自 go-ixml 开源项目推断，需确认天数服务器上 libixml.so 存在且符号可用 |
| `cudaMallocAsync` / `cudaFreeAsync` | 可能不支持 | 这是 CUDA 11.2+ 特性，Corex 基于 10.2，但不属于 analyzer 路径 |
| `cudaStreamWaitEvent` | 已知不支持 | 代码中已返回 `INFINI_STATUS_NOT_IMPLEMENTED`，不影响 analyzer |
| CUDA Graph APIs | 可能不支持 | 不属于 analyzer 路径 |
| `analyzer-test`（上层单元测试） | 待验证 | 需要 `--mutual-awareness=y` 编译，依赖 `infinicore_cpp_api` |

---

## 4. 测试指南

### 4.1 环境要求

- 天数智芯 BI-V150 GPU（或其他 Iluvatar GPU）
- Corex SDK 4.x（测试环境为 4.3.0）
- xmake >= 3.0

### 4.2 构建

```bash
cd /path/to/InfiniGraph/InfiniCore

# 配置（启用 Iluvatar + mutual awareness）
xmake f --iluvatar-gpu=y --mutual-awareness=y -c

# 构建硬件测试
xmake build infinirt-test-analyzer-hw

# 构建上层 analyzer 单元测试
xmake build analyzer-test
```

### 4.3 运行测试

#### 硬件层测试（infinirt 层直测）

```bash
xmake run infinirt-test-analyzer-hw
```

预期输出示例：

```
========================================
 Analyzer HW Tests
 Device: Iluvatar x 2
========================================

[TEST] getMemInfo                                    (free=31.9 GiB, total=32.0 GiB) PASSED
[TEST] snapshot_memory                               (used=64.0 MiB, free=31936.0 MiB, total=32768.0 MiB) PASSED
[TEST] snapshot_device_name                          (name=Iluvatar BI-V150) PASSED
[TEST] snapshot_utilization                          (compute=0.0%, mem_bw=0.0%) PASSED
[TEST] snapshot_communication                        (comm_ratio=0.000, comm_bytes=0) PASSED
[TEST] event_timing                                  (elapsed=0.012 ms, status=0) PASSED
[TEST] malloc_memcpy                                 (256 floats OK) PASSED
[TEST] multi_device_snapshot                         [dev0: 32.0GiB] [dev1: 32.0GiB] PASSED

========================================
 Results: 8 passed, 0 failed
========================================
```

**重点关注项：**

| 测试 | 如果 FAILED | 排查方向 |
|------|------------|---------|
| `getMemInfo` | `cudaMemGetInfo` 在 Corex 上不兼容 | 检查 Corex SDK 版本，确认 cuda_runtime.h 中有该函数 |
| `snapshot_utilization` | `libixml.so` 加载失败或符号不匹配 | 见 4.4 节 |
| `event_timing` | `cudaEvent` 系列 API 不兼容 | 通信采样功能将不可用，需降级处理 |
| `malloc_memcpy` | 基础 CUDA 内存 API 不兼容 | 这是根本性问题，需检查 Corex 安装 |

#### 上层 analyzer 单元测试

```bash
xmake run analyzer-test
```

这个测试覆盖 OpTraceRing、PhaseDetector、ResourceSensor、IntentGenerator、MutualAwarenessAnalyzer 全链路，大部分是纯逻辑测试不依赖 GPU，但 `mutual_awareness_analyzer_auto_collect_memory_stats` 等少数测试需要 runtime 支持。

### 4.4 排查 libixml.so

如果 `snapshot_utilization` 测试失败，按以下步骤排查：

```bash
# 1. 确认 libixml.so 存在
find /usr/local/corex* -name "libixml.so*" 2>/dev/null
ls -la /usr/local/corex/lib/libixml.so 2>/dev/null

# 2. 确认 nvml* 符号可用
nm -D /usr/local/corex/lib/libixml.so 2>/dev/null | grep nvmlInit
nm -D /usr/local/corex/lib/libixml.so 2>/dev/null | grep nvmlDeviceGetUtilizationRates

# 3. 确认运行时能找到（可能需要设置 LD_LIBRARY_PATH）
export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
xmake run infinirt-test-analyzer-hw

# 4. 如果 libixml.so 完全不存在
#    → 说明该 Corex 版本不附带管理库
#    → utilization 测试预期 FAILED（不影响其他功能）
#    → analyzer 会退化为仅依赖 memory capacity 进行瓶颈判断
```

### 4.5 手动验证脚本

将以下脚本保存为 `test_iluvatar_analyzer.sh` 在服务器上运行：

```bash
#!/bin/bash
set -e

echo "=== 1. Environment Check ==="
echo "Corex SDK:"
ls -d /usr/local/corex* 2>/dev/null || echo "  NOT FOUND"
echo "ixsmi:"
which ixsmi && ixsmi | head -15 || echo "  NOT FOUND"
echo "libixml.so:"
find /usr/local/corex* -name "libixml.so*" 2>/dev/null || echo "  NOT FOUND"

echo ""
echo "=== 2. Build ==="
cd "$(dirname "$0")"  # 假设脚本在 InfiniCore 目录
xmake f --iluvatar-gpu=y --mutual-awareness=y -c
xmake build infinirt-test-analyzer-hw 2>&1
echo "Build: OK"

echo ""
echo "=== 3. libixml.so symbol check ==="
IXML=$(find /usr/local/corex* -name "libixml.so" 2>/dev/null | head -1)
if [ -n "$IXML" ]; then
    echo "Found: $IXML"
    echo "nvmlInit_v2:                   $(nm -D "$IXML" | grep -c nvmlInit_v2)"
    echo "nvmlDeviceGetUtilizationRates: $(nm -D "$IXML" | grep -c nvmlDeviceGetUtilizationRates)"
    echo "nvmlDeviceGetHandleByIndex_v2: $(nm -D "$IXML" | grep -c nvmlDeviceGetHandleByIndex_v2)"
    echo "nvmlShutdown:                  $(nm -D "$IXML" | grep -c nvmlShutdown)"
    export LD_LIBRARY_PATH="$(dirname "$IXML"):$LD_LIBRARY_PATH"
else
    echo "libixml.so not found — utilization test will fail (expected)"
fi

echo ""
echo "=== 4. Run HW Tests ==="
xmake run infinirt-test-analyzer-hw

echo ""
echo "=== 5. Run Analyzer Unit Tests ==="
xmake build analyzer-test 2>&1 && xmake run analyzer-test || echo "analyzer-test build/run failed (may need investigation)"

echo ""
echo "=== Done ==="
```

---

## 5. 技术背景

### IXML 是什么

天数智芯的 GPU 管理库，等价于 NVIDIA 的 NVML。关键特征：
- 动态库位于 `/usr/local/corex/lib/libixml.so`
- **函数签名与 NVML 完全兼容**（`nvmlInit_v2`、`nvmlDeviceGetUtilizationRates` 等）
- 结构体也兼容（`nvmlUtilization_t { gpu, memory }`）
- 额外提供 `ixml*` 前缀的天数专有扩展

来源：[github.com/Deep-Spark/go-ixml](https://github.com/Deep-Spark/go-ixml)（Go 绑定，揭示了 C API 接口）

### 天数 Corex SDK 已知限制

| 特性 | 状态 |
|------|------|
| WARP_SIZE | 64（NVIDIA 为 32） |
| Max Thread Block | 4096（NVIDIA 为 1024） |
| FP64 | 受限，不建议使用 |
| `cudaStreamWaitEvent` | 不支持（代码已处理） |
| `cudaMallocAsync` / `cudaFreeAsync` | 可能不支持（CUDA 11.2+ 特性） |
| CUDA Graph | 可能不支持 |

### 代码结构变更图

```
infinirt_cuda.cu (修改前)
├── #if ENABLE_NVIDIA_API        ← 仅 NVIDIA
│   ├── Communication structs    ← 被锁死在这里
│   ├── NVML loader
│   └── tryPopulateNvmlUtilization
├── populateCommunicationSnapshot ← 引用上面的符号，但在 guard 外 → 编译失败
└── namespace infinirt::iluvatar   ← recordCommunicationSample 也引用 → 编译失败

infinirt_cuda.cu (修改后)
├── Communication structs         ← 移到全局，所有后端共享
├── populateCommunicationSnapshot ← 同一 namespace，无前向引用问题
├── #if ENABLE_NVIDIA_API || ENABLE_ILUVATAR_API  ← 扩展
│   ├── NVML/IXML loader (dlopen libnvidia-ml.so / libixml.so)
│   └── tryPopulateNvmlUtilization
└── namespace infinirt::iluvatar   ← recordCommunicationSample 可正常编译
```
