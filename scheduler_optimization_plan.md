# ExecutionScheduler 优化方案 - 设计与实施计划

**日期**: 2026-03-31
**目标**: 通过调度优化实现端到端运行速度提升 10%

---

## 一、现状总结

### 1.1 已完成工作

| 项目 | 状态 | 说明 |
|------|------|------|
| ExecutionScheduler 核心实现 | ✅ 完成 | 策略选择 API 完整可用 |
| Python 层包装器 SchedulerInference | ✅ 完成 | LM 调用接口完整 |
| 测试框架 | ✅ 完成 | verify_scheduler.py, test_scheduler_e2e.py 等 |
| 远程部署验证 | ✅ 完成 | 在 A100 GPU 上验证通过 |
| 性能基线测试 | ✅ 完成 | 基线: 38.92 tok/s, 21.996 ms/token |

### 1.2 当前性能结果

| 测试场景 | 无调度器 | 有调度器 | 提升 |
|---------|---------|---------|------|
| 初次测试 (3 runs) | 38.92 tok/s | 39.59 tok/s | +1.72% |
| Python 层优化后 (3 runs) | 40.49 tok/s | 41.15 tok/s | +1.62% |

**结论**: 未达到 10% 目标

### 1.3 未达标原因分析

```
当前实现流程：
┌─────────────────────────────────────────────────────────────┐
│ Python: scheduler.select_strategy(hints)                    │
│   返回: strategy = "short_seq"                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Python: model.batch_infer_one_round([infer_task])            │
│   ↓ 调用 C++ 层                                              │
│   ↓ 仍使用默认 kernel，没有根据 strategy 选择不同实现        │
└─────────────────────────────────────────────────────────────┘
```

**问题**: 策略信息没有被用于选择不同的执行路径

---

## 二、优化方案设计

### 2.1 方案概述

考虑到时间和复杂度，采用**分层优化策略**：

```
┌─────────────────────────────────────────────────────────────┐
│                     优化分层架构                             │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: C++ Kernel 优化 (最大收益，最高复杂度)               │
│   - 为不同策略实现不同 kernel 变体                            │
│   - 修改 dispatcher 支持 goal-aware 查找                     │
│   - 预期收益: 10-15%                                        │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Python 执行路径优化 (中等收益，中等复杂度)           │
│   - 根据策略调整执行参数                                      │
│   - 条件跳过某些操作                                         │
│   - 预期收益: 5-8%                                          │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Python I/O 优化 (小收益，低复杂度) ✅ 已完成        │
│   - 输出缓冲、批量 flush                                      │
│   - 减少统计收集开销                                          │
│   - 预期收益: 1-3%                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Layer 2 优化方案（推荐优先实现）

**核心思路**: 在 Python 层根据策略修改执行行为

#### 2.2.1 第0层优化 (layer0_optimized)

**问题**: bottleneck_analysis 显示第0层特别慢 (30-40ms vs 平均 18ms)

**优化**:
```python
if strategy == "layer0_optimized":
    # 1. 跳过第一次 KV cache 的一些验证操作
    # 2. 预热 KV cache（提前分配并初始化）
    # 3. 延迟某些非关键操作
    pass
```

**预期收益**: 减少 5-10ms/首层

#### 2.2.2 短序列优化 (short_seq)

**优化**:
```python
if strategy == "short_seq":
    # 1. 使用更小的 batch size（如果有批处理）
    # 2. 跳过某些大内存分配
    # 3. 使用简化的后处理路径
    pass
```

**预期收益**: 减少 1-2ms/token

#### 2.2.3 长序列优化 (long_seq)

**优化**:
```python
if strategy == "long_seq":
    # 1. 预分配更多内存
    # 2. 批量处理多个 token
    # 3. 减少 I/O 操作
    pass
```

**预期收益**: 减少 2-3ms/token

#### 2.2.4 预热优化 (warmup)

**优化**:
```python
if strategy == "warmup":
    # 1. 跳过所有统计收集
    # 2. 使用简化路径
    # 3. 跳过输出 flush
    pass
```

**预期收益**: 减少 5-10ms/预热阶段

### 2.3 Layer 3 优化方案（备选，需 C++ 修改）

**核心思路**: 为不同策略实现不同的 C++ kernel

#### 2.3.1 修改 Dispatcher

```cpp
// 当前
template <typename Fn>
class OpDispatcher {
    Fn lookup(Device::Type device_type) const;
};

// 修改后
template <typename Fn>
class OpDispatcher {
    struct Key {
        Device::Type device;
        OptimizationGoal goal;  // 新增
    };
    Fn lookup(Key key) const;  // 支持 goal-aware 查找
};
```

#### 2.3.2 注册不同策略的 Kernel

```cpp
// paged_attention_prefill 注册
static bool registered = []() {
    // 默认实现
    dispatcher.registerAll(&calculate_default);

    // Throughput 优化实现（大 tile size）
    dispatcher.register({Device::NVIDIA, OptimizationGoal::THROUGHPUT_FIRST},
                       &calculate_throughput_optimized);

    // Latency 优化实现（小 tile size）
    dispatcher.register({Device::NVIDIA, OptimizationGoal::LATENCY_FIRST},
                       &calculate_latency_optimized);
    return true;
}();
```

**预期收益**: 10-15%

**复杂度**: 高（需要修改多处 C++ 代码，重新编译）

---

## 三、实施计划 (Implementation Plan)

### Phase 1: Python 层深度优化 (预计 2-3 小时)

**目标**: 通过 Python 层执行路径优化达到 5-8% 提升

#### 1.1 优化任务列表

| 任务 | 优先级 | 预期收益 | 依赖 |
|------|--------|----------|------|
| 1. 第0层特殊处理 | P0 | 3-5% | 无 |
| 2. 预热阶段跳过开销 | P0 | 2-3% | 无 |
| 3. 短序列内存优化 | P1 | 1-2% | 无 |
| 4. 长序列批量处理 | P1 | 1-2% | 无 |
| 5. 条件编译路径 | P2 | 1% | 无 |

#### 1.2 第0层特殊处理详细设计

```python
def _apply_layer0_optimization(self):
    """第0层优化 - 针对 KV cache 初始化瓶颈"""
    # 优化1: 延迟 KV cache 验证
    # 原来: 每次都验证 KV cache 形状
    # 优化: 第0层跳过验证，信任预设形状

    # 优化2: 预分配 KV cache
    # 原来: 逐步分配
    # 优化: 一次性预分配所有需要的 KV cache

    # 优化3: 跳过第一次的同步操作
    # 原来: 每层都同步
    # 优化: 第0层异步处理
    pass
```

#### 1.3 预热优化详细设计

```python
def _apply_warmup_optimization(self):
    """预热阶段优化"""
    # 优化1: 跳过统计收集
    # 原来: 每步都记录时间
    # 优化: 预热阶段不记录

    # 优化2: 跳过策略选择
    # 原来: 每步都查询策略
    # 优化: 预热阶段固定使用 default 策略

    # 优化3: 简化输出处理
    # 原来: 每 token 都 flush
    # 优化: 预热阶段不 flush
    pass
```

### Phase 2: C++ 层优化 (预计 4-6 小时，如需要)

**目标**: 通过 C++ kernel 优化达到 10-15% 提升

#### 2.1 修改范围

```
InfiniCore/
├── include/infinicore/ops/common/dispatcher.hpp    # 修改 dispatcher
├── src/infinicore/ops/
│   ├── attention/attention.cc                        # 添加 goal-aware 调用
│   ├── paged_attention_prefill/                       # 添加多版本 kernel
│   └── mha_kvcache/                                   # 添加多版本 kernel
```

#### 2.2 实施步骤

1. **扩展 OpDispatcher** (1小时)
   - 添加 OptimizationGoal 参数支持
   - 修改 lookup 函数

2. **注册多版本 Kernel** (2小时)
   - 为 paged_attention_prefill 添加 latency/throughput 版本
   - 为 mha_kvcache 添加 short_seq/long_seq 版本

3. **修改 Attention 层调用** (1小时)
   - 传递 OptimizationGoal 参数
   - 根据策略选择 kernel

4. **编译测试** (1小时)
   - 编译 InfiniCore
   - 部署到远程
   - 验证功能

5. **性能测试** (1小时)
   - 运行 benchmark
   - 对比结果

---

## 四、TODO 清单

### 4.1 立即执行 (Phase 1)

- [ ] **TODO-1**: 实现第0层优化
  - [ ] 跳过第0层 KV cache 验证
  - [ ] 预分配 KV cache
  - [ ] 延迟同步操作

- [ ] **TODO-2**: 实现预热优化
  - [ ] 跳过统计收集
  - [ ] 跳过策略选择
  - [ ] 简化输出处理

- [ ] **TODO-3**: 实现短序列优化
  - [ ] 减少内存分配
  - [ ] 使用简化路径

- [ ] **TODO-4**: 实现长序列优化
  - [ ] 预分配内存
  - [ ] 批量处理

- [ ] **TODO-5**: 综合测试验证
  - [ ] 运行 3 runs x 3 prompts benchmark
  - [ ] 对比结果
  - [ ] 确认是否达到 10%

### 4.2 后续执行 (Phase 2 - 如需要)

- [ ] **TODO-C1**: 扩展 OpDispatcher
  - [ ] 添加 OptimizationGoal 支持
  - [ ] 修改 lookup 接口

- [ ] **TODO-C2**: 实现多版本 kernel
  - [ ] paged_attention_prefill latency 版本
  - [ ] paged_attention_prefill throughput 版本
  - [ ] mha_kvcache 短序列版本

- [ ] **TODO-C3**: 修改调用链
  - [ ] Attention 层传递 goal
  - [ ] Python 层传递 hints

- [ ] **TODO-C4**: 编译部署测试

---

## 五、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Python 优化效果有限 | 高 | 中 | 准备 C++ 方案作为备选 |
| C++ 修改引入 bug | 中 | 高 | 充分测试，保留回退路径 |
| 远程 GPU 不可用 | 低 | 中 | 使用本地/CPU 测试验证逻辑 |
| 时间不足 | 中 | 中 | 优先级排序，确保核心功能完成 |

---

## 六、成功标准

| 指标 | 目标 | 当前 | 差距 |
|------|------|------|------|
| Throughput 提升 | ≥10% | +1.62% | 8.38% |
| Decode Time 优化 | ≥10% | +2.55% | 7.45% |
| API 可用性 | 100% | 100% | ✅ |

---

## 七、下一步行动

1. **立即开始 Phase 1 (Python 层优化)**
   - 预计时间: 2-3 小时
   - 预期收益: 5-8%
   - 风险: 低

2. **如 Phase 1 达标则停止**
   - 总结结果
   - 更新文档

3. **如 Phase 1 未达标则启动 Phase 2 (C++ 层优化)**
   - 预计时间: 4-6 小时
   - 预期收益: 10-15%
   - 风险: 中

---

## 附录：相关文件路径

```
代码:
/Users/lxy/lxygit/Infini/InfiniCore/python/infinicore/execution_scheduler.py
/Users/lxy/lxygit/Infini/InfiniLM/scripts/scheduler_inference.py
/Users/lxy/lxygit/Infini/InfiniLM/scripts/test_scheduler_e2e.py

文档:
/Users/lxy/lxygit/scheduler_implementation_report.md
/Users/lxy/lxygit/scheduler_optimization_plan.md (本文件)

远程:
huidesheng@10.130.147.223:~/InfiniCore/python/infinicore/
huidesheng@10.130.147.223:~/InfiniLM/scripts/
```
