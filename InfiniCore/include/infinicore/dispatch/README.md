# KernelDispatcher — 设备感知的 Kernel 调度模块

## 背景与动机

InfiniCore 支持多种 GPU 后端（NVIDIA、Iluvatar、MetaX、ALI 等），paged attention prefill 算子有 **8 种 kernel 变体**（warp、warpcta、warpcta8、warpcta8pipe、warpcta8mma、warpcta8n128、warpcta16、ref），在不同硬件上性能差异显著。

**问题**：kernel 变体选择原先硬编码在各 `.cu` / `.maca` 文件的 `default_prefill_kernel()` 函数中，调度逻辑与算子实现强耦合，导致：
- 新增 GPU 硬件需要修改 `.cu` 文件
- 无法根据运行时负载特征动态调整 kernel 选择
- 未能利用已有的 `MutualAwarenessAnalyzer`（它已能检测 prefill/decode 阶段并输出 `OptimizationGoal`）

**Benchmark 数据驱动**（2026-03-31）：
- **NVIDIA A100**：`warpcta8pipe` 在 fp16/bf16 + page_block_size=256 + head_size=128 下赢得 297/300 个配置
- **Iluvatar BI-V150**：`warp` 和 `warpcta8pipe` 各约 50%，取决于 head_size 和 batch size：
  - head_size <= 64：`warpcta8pipe` 始终更优
  - head_size = 128, batch >= 4 或 q_per_seq >= 64：`warp` 更优
  - head_size = 128, 小 batch：`warpcta8pipe` 更优

不同硬件需要不同的调度策略，这促使我们将调度逻辑抽离为独立模块，新增硬件时无需改动算子代码。

## 架构

```
MutualAwarenessAnalyzer                KernelDispatcher（本模块）
├── PhaseDetector                      ├── 三维查找表:
├── ResourceSensor                     │   (OpType, DeviceType, OptimizationGoal)
├── IntentGenerator                    │       -> KernelSelectFn
│   -> OptimizationGoal                │
└──────────────┬───────────────────────┘
               |
        .cu / .maca 文件（消费者）
        default_prefill_kernel(info) {
            KernelDispatcher::instance()
                .selectKernel(op, device, &info)
                -> 查询 analyzer 获取当前 goal
                -> 查表获取注册的规则
                -> 返回 kernel 名称（或 nullptr 走 fallback）
        }
```

### 调用链

```
DISPATCH_KERNEL 宏
  -> default_prefill_kernel(info)
       -> KernelDispatcher::selectKernel(op, device, &info)
            1. 查询 MutualAwarenessAnalyzer 获取当前 OptimizationGoal
               (LATENCY_FIRST / THROUGHPUT_FIRST / MEMORY_SAFE / STABILITY_FIRST)
            2. 用 (OpType, DeviceType, Goal) 查表
            3. 调用注册的 KernelSelectFn(info) -> 返回 kernel 名称
       -> 若返回 nullptr：使用原有硬编码启发式作为 fallback
```

### 条件编译

- `ENABLE_MUTUAL_AWARENESS`：关闭时整个调度模块被跳过，`.cu` 文件使用原有硬编码逻辑，行为完全不变。
- `ENABLE_*_API`（NVIDIA/ILUVATAR/METAX/ALI）：决定静态初始化时注册哪个设备的规则。

## 文件结构

```
include/infinicore/dispatch/
├── kernel_dispatcher.hpp        # KernelDispatcher 类定义（单例，线程安全）
├── prefill_dispatch_rules.h     # Benchmark 驱动的调度规则 + 静态注册
└── README.md                    # 本文件

src/infinicore/dispatch/
├── kernel_dispatcher.cc         # selectKernel() 实现
└── prefill_dispatch_rules.cc    # 占位（规则从 .cu 编译单元通过 header 注册）
```

### 为什么规则放在 header 而非 .cc 文件

`PagedAttentionPrefillInfo` 定义在 `src/infiniop/ops/paged_attention_prefill/info.h`，该路径不在 `src/infinicore/` 的 include path 中。`.cu` / `.maca` 文件通过自身的头文件链已经 include 了 `info.h`，所以 `prefill_dispatch_rules.h` 在那里被 include 时可以直接使用该结构体。

## 新增 GPU 支持

1. 在 `prefill_dispatch_rules.h` 中添加 `#elif defined(ENABLE_<VENDOR>_API)` 块
2. 基于新硬件的 benchmark 数据实现 `KernelSelectFn` 函数
3. 在 `registerPrefillRules()` 中注册
4. 无需修改 `.cu` 文件或 `kernel_dispatcher.cc`

## TODO

### 必须完成（合入主线前）

- [ ] **编译验证**：分别用 `ENABLE_MUTUAL_AWARENESS` 开/关 × `ENABLE_NVIDIA_API` / `ENABLE_ILUVATAR_API` / `ENABLE_METAX_API` 在对应平台编译
- [ ] **回退验证**：确认 `ENABLE_MUTUAL_AWARENESS=OFF` 时行为与改动前完全一致（无回归）
- [ ] **BI-V150 远程 benchmark**：验证 Iluvatar 调度规则产生正确的 kernel 选择，性能不低于原先硬编码的 `"warp"` 默认值
- [ ] **A100 benchmark 验证**：确认 LATENCY_FIRST 目标下 `warpcta8pipe` 选择与此前 benchmark 结果一致

### 建议完成

- [ ] **BI-200 benchmark 与规则**：Iluvatar BI-200 性能特征可能不同，需要新的 benchmark 数据和对应规则
- [ ] **调试日志**：支持 `INFINIOP_DEBUG_PREFILL_DISPATCH=1` 环境变量，打印选择了哪个 kernel 及原因（goal、device、规则匹配情况）
- [ ] **线程安全审计**：`selectKernel()` 读取 `table_[]` 时未加锁（静态初始化保证写入先于读取，`std::array` 并发读安全）。需确认无延迟注册导致的竞态

### 可选优化

- [ ] **Python 绑定**：`infinicore.dispatch.get_dispatcher()` / `override_kernel()` 用于调试和手动覆盖
- [ ] **JSON 规则加载**：从配置文件加载调度规则（实现 benchmark -> 自动生成规则的流水线）
- [ ] **模型配置覆盖**：允许模型部署配置中指定固定 kernel
- [ ] **扩展到其他算子**：将 KernelDispatcher 模式应用到 paged_attention_prefill 之外的算子（如 decode attention、GEMM）
