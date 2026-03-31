#!/bin/bash
# ============================================================
# Iluvatar Analyzer 实机验证脚本
#
# 用法：在天数服务器的 InfiniCore 目录下运行
#   cd /path/to/InfiniGraph/InfiniCore
#   bash scripts/test_iluvatar_analyzer.sh
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}OK${NC}: $1"; }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " Iluvatar Analyzer 实机验证"
echo " 目录: $PROJECT_DIR"
echo " 时间: $(date)"
echo "============================================"
echo ""

# ---- Step 1: 环境检查 ----
echo "--- Step 1: 环境检查 ---"

# Corex SDK
COREX_DIR=$(ls -d /usr/local/corex* 2>/dev/null | head -1)
if [ -n "$COREX_DIR" ]; then
    pass "Corex SDK: $COREX_DIR"
else
    fail "Corex SDK 未找到"
    exit 1
fi

# ixsmi
if command -v ixsmi &>/dev/null; then
    GPU_COUNT=$(ixsmi -L 2>/dev/null | grep -c "GPU" || echo 0)
    pass "ixsmi 可用, 检测到 $GPU_COUNT 个 GPU"
    ixsmi 2>/dev/null | head -15
else
    fail "ixsmi 不在 PATH 中"
fi

# libixml.so
IXML_PATH=$(find /usr/local/corex* -name "libixml.so" 2>/dev/null | head -1)
if [ -n "$IXML_PATH" ]; then
    pass "libixml.so: $IXML_PATH"
    # 检查关键符号
    NVML_INIT=$(nm -D "$IXML_PATH" 2>/dev/null | grep -c "nvmlInit_v2" || echo 0)
    NVML_UTIL=$(nm -D "$IXML_PATH" 2>/dev/null | grep -c "nvmlDeviceGetUtilizationRates" || echo 0)
    NVML_HANDLE=$(nm -D "$IXML_PATH" 2>/dev/null | grep -c "nvmlDeviceGetHandleByIndex_v2" || echo 0)
    NVML_SHUTDOWN=$(nm -D "$IXML_PATH" 2>/dev/null | grep -c "nvmlShutdown" || echo 0)

    echo "  符号检查:"
    echo "    nvmlInit_v2:                   $NVML_INIT"
    echo "    nvmlDeviceGetUtilizationRates: $NVML_UTIL"
    echo "    nvmlDeviceGetHandleByIndex_v2: $NVML_HANDLE"
    echo "    nvmlShutdown:                  $NVML_SHUTDOWN"

    if [ "$NVML_INIT" -gt 0 ] && [ "$NVML_UTIL" -gt 0 ] && [ "$NVML_HANDLE" -gt 0 ] && [ "$NVML_SHUTDOWN" -gt 0 ]; then
        pass "所有 NVML 兼容符号存在"
    else
        warn "部分符号缺失 — utilization 查询可能不可用"
    fi

    # 确保运行时能找到
    export LD_LIBRARY_PATH="$(dirname "$IXML_PATH"):${LD_LIBRARY_PATH:-}"
else
    warn "libixml.so 未找到 — GPU utilization 测试预期失败（不影响其他功能）"
fi

echo ""

# ---- Step 2: 构建 ----
echo "--- Step 2: 构建 ---"

echo "  配置 xmake ..."
xmake f --iluvatar-gpu=y --mutual-awareness=y -c 2>&1 | tail -3
echo ""

echo "  构建 infinirt-test-analyzer-hw ..."
if xmake build infinirt-test-analyzer-hw 2>&1; then
    pass "infinirt-test-analyzer-hw 构建成功"
else
    fail "infinirt-test-analyzer-hw 构建失败"
    echo "  请检查编译错误并修复后重试"
    exit 1
fi

echo ""
echo "  构建 analyzer-test ..."
if xmake build analyzer-test 2>&1; then
    pass "analyzer-test 构建成功"
else
    warn "analyzer-test 构建失败（可能依赖未满足）"
fi

echo ""

# ---- Step 3: 运行硬件层测试 ----
echo "--- Step 3: 运行硬件层测试 ---"
echo ""
xmake run infinirt-test-analyzer-hw
HW_EXIT=$?
echo ""
if [ $HW_EXIT -eq 0 ]; then
    pass "硬件层测试全部通过"
else
    fail "硬件层测试有失败项 (exit code: $HW_EXIT)"
fi

echo ""

# ---- Step 4: 运行 analyzer 单元测试 ----
echo "--- Step 4: 运行 analyzer 单元测试 ---"
echo ""
if xmake run analyzer-test 2>&1; then
    ANALYZER_EXIT=$?
else
    ANALYZER_EXIT=$?
fi
echo ""
if [ ${ANALYZER_EXIT:-1} -eq 0 ]; then
    pass "analyzer 单元测试全部通过"
else
    warn "analyzer 单元测试有失败项 (exit code: ${ANALYZER_EXIT:-1})"
fi

echo ""
echo "============================================"
echo " 验证完成"
echo "============================================"
