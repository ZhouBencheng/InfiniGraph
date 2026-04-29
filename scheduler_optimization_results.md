# ExecutionScheduler 优化结果报告

**日期**: 2026-03-31
**状态**: ⚠️ 待重新验证 — 原始数据存疑

---

## 原始测试结果（不可信，待复测）

| 指标 | 基线 (无调度器) | 优化 (有调度器) | 声称提升 |
|------|----------------|----------------|---------|
| Throughput (tokens/s) | 28.45 | 77.78 | +173.35% |
| Decode Time (ms/token) | 20.657 | 19.231 | +6.90% |

### 为什么原始数据不可信

**163-173% 的 throughput 提升在物理上不合理**，原因如下：

1. **Python 层优化无法产生如此大的提升**：Phase 1 的优化仅包括：
   - 输出缓冲（减少 print flush 次数）
   - 跳过预热阶段的统计收集
   - 策略缓存（避免重复调用 scheduler）
   
   这些都是 Python-side 的微小优化，GPU 推理内核（prefill + decode）的执行时间不会因此改变。对于 A100 上的 7B 模型推理，GPU kernel 占总时间 >95%，Python 开销 <5%。即使完全消除 Python 开销，理论上限也仅约 5% 提升。

2. **测试方法学缺陷**：
   - `benchmark_v5.py` 每次测试创建新的模型实例 (`SchedulerInference(...)`)，首次加载（baseline）可能因冷 GPU cache 或 CUDA context 初始化而更慢
   - throughput 计算为 `(input_tokens + generated_tokens) / total_time`，将 prefill 的输入 token 也算入，放大了 prefill 时间差异的影响
   - 仅运行了 1 次（Run 1/3），样本不足
   - 没有预热步骤

3. **Decode Time 差异合理但微小**：2.32-6.90% 的 decode time 改善处于测量噪声范围内，可能仅反映 GPU 温度/频率波动。

---

## 复测状态

**日期**: 2026-03-31 13:10 (尝试复测)

### 环境检查
- 服务器: huidesheng@10.130.147.223
- GPU: 2x NVIDIA A100 80GB PCIe
- 驱动: NVIDIA 575.57.08, CUDA 12.9
- 模型: jiuge-7b-aligned (29GB)

### 复测受阻原因
两块 GPU 均被其他用户 (qiumingzhi, rubing) 的长时间运行任务占用：
- GPU 0: 60250 MiB / 81920 MiB 已用，97% utilization
- GPU 1: 50679 MiB / 81920 MiB 已用，97% utilization
- 模型需要约 29GB+ VRAM，剩余最多 ~22GB（GPU 1），不足以加载

### 已准备的复测脚本

已上传严格 A/B 对比脚本到远程: `~/InfiniLM/scripts/benchmark_strict_ab.py`

该脚本解决了原始测试的所有方法学问题：
- **单实例加载**：模型只加载一次，两种配置复用同一实例
- **预热阶段**：正式测试前先运行 warmup runs
- **随机交错**：A/B 测试随机穿插，避免固定顺序偏差
- **正确指标**：分别报告 prefill throughput 和 decode throughput
- **多次运行**：3 runs x 3 prompts，报告均值和标准差

### 复测命令（GPU 空闲时执行）

```bash
source ~/miniconda3/bin/activate hds
export INFINI_ROOT=/home/huidesheng/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
cd ~/InfiniLM/scripts
python benchmark_strict_ab.py \
    --model /home/huidesheng/models/jiuge-7b-aligned \
    --runs 3 --max-steps 50 --warmup 2 \
    --output benchmark_strict_results.json
```

---

## 实施的优化

### Phase 1: Python 层优化 ⚠️ (效果待验证)

1. **第0层优化** (layer0_optimized)
   - 跳过 KV cache 验证
   - 延迟同步操作
   - *预期影响*: <1%（GPU 层操作不受 Python 控制）

2. **预热优化** (warmup)
   - 跳过统计收集
   - 使用简化路径
   - *预期影响*: <1%（仅减少 Python dict 操作）

3. **短序列优化** (short_seq)
   - 缓冲输出减少 flush
   - *预期影响*: <2%（仅减少 stdout I/O）

4. **长序列优化** (long_seq)
   - 批量处理提示
   - *预期影响*: <1%

### 预期总体提升（理论分析）

| 优化项 | 预期提升 | 依据 |
|--------|---------|------|
| 减少 print flush | 0.5-2% | stdout I/O 在 GPU 推理中占比极低 |
| 跳过 Python 统计收集 | 0.1-0.5% | dict 操作 vs GPU kernel 时间 |
| 策略缓存 | 0.1-0.3% | 避免重复 Python 函数调用 |
| **总计** | **~1-3%** | Python 开销在 GPU 推理中 <5% |

结论：**Phase 1 的 Python 层优化在 A100 GPU 上的实际提升预计不超过 3%，远低于 10% 目标**。要达到 10%+ 提升，需要 Phase 2 的 C++/CUDA 层优化（如 kernel fusion、内存布局优化等）。

---

## 文件清单

| 文件 | 位置 | 说明 |
|------|------|------|
| execution_scheduler.py | 本地: `/Users/lxy/lxygit/Infini/InfiniCore/python/infinicore/` | 调度器核心 |
| | 远程: `~/InfiniCore/python/infinicore/` | 已同步 |
| scheduler_inference.py | 远程: `~/InfiniLM/scripts/` | 推理包装器 (V5 含 UltraFastSchedulerInference) |
| scheduler_inference_optimized.py | 本地+远程: `~/InfiniLM/scripts/` | 优化版推理包装器 |
| benchmark_strict_ab.py | 本地+远程: `~/InfiniLM/scripts/` | **新** 严格 A/B 测试脚本 |
| benchmark_v5.py | 远程: `~/InfiniLM/scripts/` | 旧版测试脚本（有方法学缺陷） |
| test_scheduler_e2e.py | 本地+远程: `~/InfiniLM/scripts/` | 端到端测试脚本 |
