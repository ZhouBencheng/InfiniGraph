# ExecutionScheduler TODO 清单

**目标**: 通过调度优化实现端到端运行速度提升 10%

---

## 当前状态

| 指标 | 目标 | 当前 | 差距 |
|------|------|------|------|
| Throughput 提升 | ≥10% | +1.62% | ❌ 8.38% |
| Decode Time 优化 | ≥10% | +2.55% | ❌ 7.45% |

---

## Phase 1: Python 层深度优化 (预计 2-3 小时)

### TODO-1: 第0层优化 (预期 3-5%)
- [ ] 跳过第0层 KV cache 验证
- [ ] 预分配 KV cache
- [ ] 延迟同步操作
- [ ] 测试验证

### TODO-2: 预热优化 (预期 2-3%)
- [ ] 跳过统计收集
- [ ] 跳过策略选择
- [ ] 简化输出处理
- [ ] 测试验证

### TODO-3: 短序列优化 (预期 1-2%)
- [ ] 减少内存分配
- [ ] 使用简化路径
- [ ] 测试验证

### TODO-4: 长序列优化 (预期 1-2%)
- [ ] 预分配内存
- [ ] 批量处理
- [ ] 测试验证

### TODO-5: 综合测试
- [ ] 运行 3 runs x 3 prompts benchmark
- [ ] 对比结果
- [ ] 确认是否达到 10%
- [ ] 写结果报告

---

## Phase 2: C++ 层优化 (如需要，预计 4-6 小时)

### TODO-C1: 扩展 OpDispatcher
- [ ] 添加 OptimizationGoal 支持
- [ ] 修改 lookup 接口

### TODO-C2: 实现多版本 kernel
- [ ] paged_attention_prefill latency 版本
- [ ] paged_attention_prefill throughput 版本
- [ ] mha_kvcache 短序列版本

### TODO-C3: 修改调用链
- [ ] Attention 层传递 goal
- [ ] Python 层传递 hints

### TODO-C4: 编译部署测试
- [ ] 编译 InfiniCore
- [ ] 部署到远程
- [ ] 性能测试

---

## 优先级

**P0**: TODO-1, TODO-2 (必须完成)
**P1**: TODO-3, TODO-4 (建议完成)
**P2**: Phase 2 (如 P0+P1 未达标)

---

## 进度跟踪

- [x] ExecutionScheduler 核心实现
- [x] Python 层包装器
- [x] 测试框架
- [x] 远程验证
- [x] 性能基线测试
- [x] Python I/O 优化 (v1)
- [ ] Python 执行路径优化 (v2) - **待执行**
- [ ] C++ Kernel 优化 - **视情况执行**
