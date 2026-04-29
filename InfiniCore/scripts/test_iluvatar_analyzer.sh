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
symbol_count() {
    local lib="$1"
    local symbol="$2"
    local symbols
    symbols=$(nm -D "$lib" 2>/dev/null || true)
    printf '%s\n' "$symbols" | awk -v sym="$symbol" 'index($0, sym) > 0 { count++ } END { print count + 0 }'
}

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
    NVML_INIT=$(symbol_count "$IXML_PATH" "nvmlInit_v2")
    NVML_UTIL=$(symbol_count "$IXML_PATH" "nvmlDeviceGetUtilizationRates")
    NVML_HANDLE=$(symbol_count "$IXML_PATH" "nvmlDeviceGetHandleByIndex_v2")
    NVML_MEM=$(symbol_count "$IXML_PATH" "nvmlDeviceGetMemoryInfo")
    NVML_PCIE=$(symbol_count "$IXML_PATH" "nvmlDeviceGetPcieThroughput")
    NVML_POWER=$(symbol_count "$IXML_PATH" "nvmlDeviceGetPowerUsage")
    NVML_TEMP=$(symbol_count "$IXML_PATH" "nvmlDeviceGetTemperature")
    NVML_SHUTDOWN=$(symbol_count "$IXML_PATH" "nvmlShutdown")

    echo "  IXML / NVML-compatible 符号检查:"
    echo "    nvmlInit_v2:                   $NVML_INIT"
    echo "    nvmlDeviceGetHandleByIndex_v2: $NVML_HANDLE"
    echo "    nvmlDeviceGetUtilizationRates: $NVML_UTIL"
    echo "    nvmlDeviceGetMemoryInfo:       $NVML_MEM"
    echo "    nvmlDeviceGetPcieThroughput:   $NVML_PCIE"
    echo "    nvmlDeviceGetPowerUsage:       $NVML_POWER"
    echo "    nvmlDeviceGetTemperature:      $NVML_TEMP"
    echo "    nvmlShutdown:                  $NVML_SHUTDOWN"

    if [ "$NVML_INIT" -gt 0 ] && [ "$NVML_HANDLE" -gt 0 ] && \
       [ "$NVML_UTIL" -gt 0 ] && [ "$NVML_MEM" -gt 0 ] && [ "$NVML_SHUTDOWN" -gt 0 ]; then
        pass "IXML 核心资源符号完整"
    else
        warn "IXML 核心符号缺失 — memory / utilization 查询可能降级"
    fi

    if [ "$NVML_PCIE" -gt 0 ] && [ "$NVML_POWER" -gt 0 ] && [ "$NVML_TEMP" -gt 0 ]; then
        pass "IXML 增强监控符号完整"
    else
        warn "IXML 增强监控符号不完整 — PCIe / power / temperature 仅作为可选增强"
    fi

    # 确保运行时能找到
    export LD_LIBRARY_PATH="$(dirname "$IXML_PATH"):${LD_LIBRARY_PATH:-}"
else
    warn "libixml.so 未找到 — GPU utilization 测试预期失败（不影响其他功能）"
fi

# libixdcgm.so 是天数 CoreX 的可选增强管理栈，用于 SM/Dram active、进程和诊断类指标。
IXDCGM_PATH=$(find /usr/local/corex* /usr/lib /usr/lib64 /opt -name "libixdcgm.so*" 2>/dev/null | head -1 || true)
if [ -n "$IXDCGM_PATH" ]; then
    pass "libixdcgm.so: $IXDCGM_PATH"
    export LD_LIBRARY_PATH="$(dirname "$IXDCGM_PATH"):${LD_LIBRARY_PATH:-}"

    IXDCGM_INIT=$(symbol_count "$IXDCGM_PATH" "dcgmInit")
    IXDCGM_SHUTDOWN=$(symbol_count "$IXDCGM_PATH" "dcgmShutdown")
    IXDCGM_EMBEDDED=$(symbol_count "$IXDCGM_PATH" "dcgmStartEmbedded")
    IXDCGM_CONNECT=$(symbol_count "$IXDCGM_PATH" "dcgmConnect")
    IXDCGM_VALUES=$(symbol_count "$IXDCGM_PATH" "dcgmGetLatestValuesForFields")
    IXDCGM_WATCH=$(symbol_count "$IXDCGM_PATH" "dcgmWatchFields")
    IXDCGM_ATTR=$(symbol_count "$IXDCGM_PATH" "dcgmGetDeviceAttributes")

    echo "  IXDCGM / DCGM-like 可选符号检查:"
    echo "    dcgmInit:                     $IXDCGM_INIT"
    echo "    dcgmShutdown:                 $IXDCGM_SHUTDOWN"
    echo "    dcgmStartEmbedded*:           $IXDCGM_EMBEDDED"
    echo "    dcgmConnect:                  $IXDCGM_CONNECT"
    echo "    dcgmGetLatestValuesForFields: $IXDCGM_VALUES"
    echo "    dcgmWatchFields:              $IXDCGM_WATCH"
    echo "    dcgmGetDeviceAttributes:      $IXDCGM_ATTR"

    if [ "$IXDCGM_INIT" -gt 0 ] && [ "$IXDCGM_SHUTDOWN" -gt 0 ] && \
       { [ "$IXDCGM_EMBEDDED" -gt 0 ] || [ "$IXDCGM_CONNECT" -gt 0 ]; } && \
       [ "$IXDCGM_VALUES" -gt 0 ] && [ "$IXDCGM_WATCH" -gt 0 ]; then
        pass "IXDCGM 可用于后续增强资源采样"
    else
        warn "IXDCGM 符号不完整；当前主线仍使用 IXML"
    fi
else
    warn "libixdcgm.so 未找到；跳过 IXDCGM 增强符号检查"
fi

echo ""

# ---- Step 2: 构建 ----
echo "--- Step 2: 构建 ---"

echo "  配置 xmake ..."
xmake f --iluvatar-gpu=y --mutual-awareness=y --ccl=y -c -y
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

echo "  构建 analyzer-demo ..."
if xmake build analyzer-demo 2>&1; then
    pass "analyzer-demo 构建成功"
else
    fail "analyzer-demo 构建失败"
    exit 1
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

# ---- Step 4: 运行真实输出 demo ----
echo "--- Step 4: 运行真实输出 demo ---"
echo ""
xmake run analyzer-demo
DEMO_EXIT=$?
echo ""
if [ $DEMO_EXIT -eq 0 ]; then
    pass "真实 analyzer-demo 运行通过"
else
    fail "真实 analyzer-demo 运行失败 (exit code: $DEMO_EXIT)"
    exit $DEMO_EXIT
fi

echo ""

# ---- Step 5: 运行 analyzer 单元测试 ----
echo "--- Step 5: 运行 analyzer 单元测试 ---"
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
