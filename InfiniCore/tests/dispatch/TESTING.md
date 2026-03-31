# KernelDispatcher 测试流程

## 前置条件

- 已切到 `feature/kernel-dispatcher` 分支
- 远程服务器有 GPU 且已配置好编译环境（xmake、CUDA/Iluvatar SDK）

## 测试总览

| 测试项 | 脚本 | 说明 |
|--------|------|------|
| 编译验证 | `test_compilation.sh` | MUTUAL_AWARENESS ON/OFF 交叉编译 |
| Python 规则验证 | `test_dispatch.py` | 规则注册、查询、回退 |
| Analyzer 联动 | `test_e2e_dispatch.py` | Analyzer + Dispatch 端到端 |
| 推理 benchmark | 手动 | 对比 dispatch 前后性能 |

---

## 步骤 1: 同步代码到远程服务器

```bash
cd /path/to/InfiniGraph/InfiniCore
rsync -avz --exclude='.git' --exclude='build' \
    . huidesheng@10.130.147.223:~/InfiniCore/
```

## 步骤 2: 编译验证

```bash
ssh huidesheng@10.130.147.223
cd ~/InfiniCore

# Iluvatar 平台
bash tests/dispatch/test_compilation.sh iluvatar

# 或 NVIDIA 平台
bash tests/dispatch/test_compilation.sh nvidia
```

该脚本自动完成：
1. `ENABLE_MUTUAL_AWARENESS=OFF` 编译 → 确认不引入编译错误
2. `ENABLE_MUTUAL_AWARENESS=ON` 编译 → 确认 dispatch 模块编译通过
3. 安装并运行 `test_dispatch.py`

## 步骤 3: Analyzer 联动测试

```bash
source ~/miniconda3/bin/activate hds
export INFINI_ROOT=/home/huidesheng/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH

INFINIOP_DEBUG_PREFILL_DISPATCH=1 python tests/dispatch/test_e2e_dispatch.py
```

## 步骤 4: 推理 Benchmark 对比

### 4a. 基准（关闭 dispatch）

```bash
cd ~/InfiniCore
xmake f --iluvatar-gpu=y --mutual-awareness=n -c -y
xmake build infinicore_cpp_api && xmake install infinicore_cpp_api
xmake build infinicore && xmake install infinicore

cd ~/InfiniLM/scripts
python run_benchmark.py --model <model_path> 2>&1 | tee /tmp/bench_baseline.log
```

### 4b. 开启 dispatch

```bash
cd ~/InfiniCore
xmake f --iluvatar-gpu=y --mutual-awareness=y -c -y
xmake build infinicore_cpp_api && xmake install infinicore_cpp_api
xmake build infinicore && xmake install infinicore

cd ~/InfiniLM/scripts
INFINIOP_DEBUG_PREFILL_DISPATCH=1 \
python run_benchmark.py --model <model_path> 2>&1 | tee /tmp/bench_dispatch.log
```

### 4c. 对比

```bash
grep '\[dispatch\]' /tmp/bench_dispatch.log   # 查看 kernel 选择日志
diff <(grep -E 'token|latency|throughput' /tmp/bench_baseline.log) \
     <(grep -E 'token|latency|throughput' /tmp/bench_dispatch.log)
```

## 步骤 5: 环境变量覆盖验证

确认 `INFINIOP_FLASH_PREFILL_KERNEL` 仍可覆盖 dispatch 结果：

```bash
INFINIOP_FLASH_PREFILL_KERNEL=warp \
INFINIOP_DEBUG_PREFILL_DISPATCH=1 \
python run_benchmark.py --model <model_path> 2>&1 | grep '\[dispatch\]'
```

## Checklist

- [ ] `ENABLE_MUTUAL_AWARENESS=OFF` 编译通过，行为不变
- [ ] `ENABLE_MUTUAL_AWARENESS=ON` 编译通过
- [ ] `test_dispatch.py` 全部通过
- [ ] `test_e2e_dispatch.py` 通过
- [ ] `INFINIOP_DEBUG_PREFILL_DISPATCH=1` 日志格式正确
- [ ] 推理性能无回退
- [ ] `INFINIOP_FLASH_PREFILL_KERNEL` 覆盖仍有效
