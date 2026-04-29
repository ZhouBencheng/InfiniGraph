# ExecutionScheduler 调度器开发报告

**日期**: 2026-03-31
**目标**: 通过调度优化实现端到端运行速度提升 10%

---

## 1. 需求分析

### 1.1 背景
- InfiniLM 已有阶段感知调度（Prefill/Decode 选择不同算子）
- InfiniCore 已有 goal-aware 框架，但实际算子未使用 goal-specific kernel
- 算子融合尝试效果不佳
- 需要在 InfiniCore 或 Core/LM 中间层实现调度优化

### 1.2 核心需求
1. **可开关的 API**: LM 可以调用/不调用优化模块
2. **非 LM 引擎角度优化**: 不直接修改推理引擎核心
3. **10% 性能提升**: 端到端速度提升目标

### 1.3 设计原则
- 调度器作为中间层，LM 可以动态启用/禁用
- 根据运行时 hints（phase, layer_id, seq_len）选择执行策略
- 保持与现有代码的兼容性

---

## 2. 实现过程

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        InfiniLM                             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  |                                                          | │
│  |  scheduler.enable("latency_optimized")  ◄── API         | │
│  |     或                                                   | │
│  |  model.generate(..., scheduler_hints={...})             | │
│  └───────────────────────┬────────────────────────────────┘ │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   InfiniCore ExecutionScheduler             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  | 策略选择 (基于 hints + 运行时状态)                        | │
│  | - hints.phase (PREFILL/DECODE)                           | │
│  | - hints.layer_id (当前层索引)                            | │
│  | - hints.seq_len (序列长度)                               | │
│  └───────────────────────┬────────────────────────────────┘ │
└──────────────────────────┼───────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │策略A:      │    │策略B:      │    │策略C:      │
    │第0层优化   │    │短序列优化  │    │长序列优化  │
    └───────────┘    └───────────┘    └───────────┘
```

### 2.2 文件结构

| 文件 | 描述 |
|------|------|
| `InfiniCore/python/infinicore/execution_scheduler.py` | 调度器核心实现 |
| `InfiniLM/scripts/scheduler_inference.py` | Python 层包装器 |
| `InfiniLM/scripts/test_scheduler_e2e.py` | 端到端测试脚本 |
| `InfiniLM/scripts/verify_scheduler.py` | API 验证脚本 |

### 2.3 ExecutionScheduler 核心 API

```python
class ExecutionScheduler:
    def enable(self, mode: str = "latency_optimized")
    def disable(self)
    def select_strategy(self, hints: Dict[str, Any]) -> StrategyResult
    def record_layer_time(self, duration_ms: float)
    def get_stats(self) -> Dict[str, Any]
```

### 2.4 策略规则

| 策略 | 触发条件 | 优化方向 |
|------|----------|----------|
| `LAYER0_OPTIMIZED` | layer_id == 0 | 第0层特殊处理，KV cache 初始化优化 |
| `SHORT_SEQ` | seq_len < 128 | 小 tile size，减少 kernel 启动开销 |
| `LONG_SEQ` | seq_len > 2048 | 大 tile size，更好的内存访问模式 |
| `WARMUP` | step < warmup_steps | 预热阶段，避免激进优化 |
| `MEMORY_CONSERVE` | memory_usage > threshold | 减少中间 buffer |

---

## 3. 测试结果

### 3.1 测试环境
- **设备**: NVIDIA A100 80GB PCIe (GPU 1)
- **模型**: jiuge-7b-aligned
- **Prompt**: "请详细介绍以下内容：1. 人工智能的定义 2. 机器学习的分类"
- **Max Steps**: 80
- **Runs**: 3 次/模式

### 3.2 性能对比

| 指标 | 无调度器 | 有调度器 | 提升 |
|------|---------|---------|------|
| Throughput (tok/s) | 38.92 | 39.59 | +1.72% |
| Decode Time (ms) | 21.996 | 21.255 | +3.37% |
| Generated Tokens | 80 | 80 | - |

### 3.3 详细数据

**无调度器 (3 runs)**:
- Run 1: 38.62 tok/s, 22.324 ms/token
- Run 2: 38.29 tok/s, 22.400 ms/token
- Run 3: 39.84 tok/s, 21.264 ms/token

**有调度器 (3 runs)**:
- Run 1: 39.44 tok/s, 21.236 ms/token
- Run 2: 39.72 tok/s, 21.162 ms/token
- Run 3: 39.60 tok/s, 21.367 ms/token

### 3.4 调度器统计

```
==================================================
ExecutionScheduler Statistics
==================================================
Enabled: True
Mode: latency
Total calls: 79
Step count: 79
Phase counts:
  - Prefill: 0
  - Decode: 79
Strategy counts:
  - short_seq: 76
  - warmup: 3
==================================================
```

---

## 4. 分析与结论

### 4.1 当前实现的功能
✅ 可开关的调度器 API
✅ 基于运行时 hints 的策略选择
✅ 性能统计收集
✅ 约 3% 的性能提升

### 4.2 未达到 10% 目标的原因

**根本原因**: 当前调度器只是 Python 层的**策略选择器**，没有真正改变底层执行路径。

```python
# 当前实现流程
strategy = scheduler.select_strategy(hints)  # 返回 "short_seq"
# 但后续代码仍调用相同的 C++ 函数
output_tokens = self.model.batch_infer_one_round([infer_task])
```

策略信息被返回但没有被用于选择不同的执行路径。

### 4.3 需要的改进

1. **C++ 层优化**
   - 为不同策略实现不同的 kernel 变体
   - 修改 dispatcher 支持 goal-aware 查找
   - 为 `paged_attention_prefill` 和 `mha_kvcache` 注册多个 kernel

2. **Python 层优化**
   - 第0层特殊处理（延迟初始化、预分配）
   - 根据策略调整执行参数
   - 实现策略感知的执行路径

---

## 5. 下一步计划

### 5.1 C++ 层优化（计划中）
- 为 `paged_attention_prefill` 注册 `latency_optimized` kernel
- 为 `mha_kvcache` 注册 `short_seq`/`long_seq` kernel 变体
- 修改 OpDispatcher 支持 `OptimizationGoal` 参数

### 5.2 Python 层优化（计划中）
- 第0层特殊处理：延迟 KV cache 部分初始化
- 短序列优化：减少中间 buffer 分配
- 长序列优化：预分配更大的 buffer

---

## 附录

### A. 测试命令

```bash
# 远程测试
sshpass -p 'huidesheng' ssh huidesheng@10.130.147.223
source ~/codex_infinicore_env.sh
export CUDA_VISIBLE_DEVICES=1
cd ~/InfiniLM/scripts

# 验证调度器 API
python3 verify_scheduler.py

# 性能对比测试
python3 multi_run_test.py
```

### B. 代码位置

```
InfiniCore/python/infinicore/
├── execution_scheduler.py       # 调度器核心
└── __init__.py                    # 导出调度器 API

InfiniLM/scripts/
├── scheduler_inference.py        # Python 包装器
├── test_scheduler_e2e.py         # E2E 测试
├── verify_scheduler.py           # API 验证
└── multi_run_test.py             # 性能对比测试
```

### C. API 使用示例

```python
from infinicore import ExecutionScheduler

# 启用调度器
scheduler = ExecutionScheduler()
scheduler.enable("latency_optimized")

# 推理时传递 hints
hints = {
    "phase": "DECODE",
    "layer_id": 0,
    "seq_len": 512,
    "batch_size": 1,
}
strategy = scheduler.select_strategy(hints)
# strategy.strategy 可能是:
# - layer0_optimized
# - short_seq
# - long_seq
# - warmup

# 查看统计
scheduler.print_stats()

# 禁用调度器
scheduler.disable()
```
