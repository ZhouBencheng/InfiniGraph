#!/bin/bash
# ============================================================
# MetaX Analyzer 实机验证脚本
#
# 用法：在沐曦服务器的 InfiniCore 目录下运行
#   cd /path/to/InfiniGraph/InfiniCore
#   bash scripts/test_metax_analyzer.sh
#
# 可选：
#   METAX_USE_MC=1 bash scripts/test_metax_analyzer.sh
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
echo " MetaX Analyzer 实机验证"
echo " 目录: $PROJECT_DIR"
echo " 时间: $(date)"
echo "============================================"
echo ""

# ---- Step 1: 环境检查 ----
echo "--- Step 1: 环境检查 ---"

if [ -z "${MACA_PATH:-${MACA_HOME:-${MACA_ROOT:-}}}" ]; then
    for candidate in /opt/maca /opt/mxdriver /opt/mxn100; do
        if [ -d "$candidate/include" ] && [ -d "$candidate/lib" ]; then
            export MACA_PATH="$candidate"
            pass "自动设置 MACA_PATH=$MACA_PATH"
            break
        fi
    done
fi

if [ -n "${MACA_PATH:-${MACA_HOME:-${MACA_ROOT:-}}}" ]; then
    pass "MACA SDK: ${MACA_PATH:-${MACA_HOME:-${MACA_ROOT:-}}}"
else
    fail "MACA_PATH / MACA_HOME / MACA_ROOT 未设置，且未在 /opt 下发现 SDK"
    exit 1
fi

if command -v mx-smi &>/dev/null; then
    pass "mx-smi 可用"
    mx-smi 2>/dev/null | head -20 || true
else
    warn "mx-smi 不在 PATH 中；继续检查 libmxsml.so"
fi

MXSML_PATH=""
for candidate in \
    /opt/mxdriver/lib/libmxsml.so \
    /opt/maca/lib/libmxsml.so \
    /opt/mxn100/lib/libmxsml.so; do
    if [ -f "$candidate" ]; then
        MXSML_PATH="$candidate"
        break
    fi
done

if [ -z "$MXSML_PATH" ]; then
    MXSML_PATH="$(find /opt/mxdriver /opt/maca /opt/mxn100 -name libmxsml.so 2>/dev/null | head -1 || true)"
fi

if [ -n "$MXSML_PATH" ]; then
    pass "libmxsml.so: $MXSML_PATH"
    export LD_LIBRARY_PATH="$(dirname "$MXSML_PATH"):${LD_LIBRARY_PATH:-}"

    MXSML_INIT=$(nm -D "$MXSML_PATH" 2>/dev/null | grep -c "mxSmlExInit" || echo 0)
    MXSML_HANDLE=$(nm -D "$MXSML_PATH" 2>/dev/null | grep -c "mxSmlExGetDeviceHandleByIndex" || echo 0)
    MXSML_UTIL=$(nm -D "$MXSML_PATH" 2>/dev/null | grep -c "mxSmlExDeviceGetUtilization" || echo 0)
    MXSML_MEM=$(nm -D "$MXSML_PATH" 2>/dev/null | grep -c "mxSmlExDeviceGetMemoryInfo" || echo 0)

    echo "  符号检查:"
    echo "    mxSmlExInit:                   $MXSML_INIT"
    echo "    mxSmlExGetDeviceHandleByIndex: $MXSML_HANDLE"
    echo "    mxSmlExDeviceGetUtilization:   $MXSML_UTIL"
    echo "    mxSmlExDeviceGetMemoryInfo:    $MXSML_MEM"

    if [ "$MXSML_INIT" -gt 0 ] && [ "$MXSML_HANDLE" -gt 0 ] && \
       [ "$MXSML_UTIL" -gt 0 ] && [ "$MXSML_MEM" -gt 0 ]; then
        pass "MXSML 扩展资源符号完整"
    else
        warn "MXSML 扩展符号不完整；utilization / memory 快照可能降级"
    fi
else
    warn "libmxsml.so 未找到；将只能依赖 hcMemGetInfo 和事件通信采样"
fi

echo ""

# ---- Step 2: 构建 ----
echo "--- Step 2: 构建 ---"

CONFIG_ARGS=(--metax-gpu=y --mutual-awareness=y --ccl=y -c -y)
if [ "${METAX_USE_MC:-0}" = "1" ]; then
    CONFIG_ARGS+=(--use-mc=y)
fi

echo "  配置 xmake: ${CONFIG_ARGS[*]}"
xmake f "${CONFIG_ARGS[@]}"
echo ""

echo "  构建 infinirt-test-analyzer-hw ..."
xmake build infinirt-test-analyzer-hw
pass "infinirt-test-analyzer-hw 构建成功"

echo ""
echo "  构建 analyzer-demo ..."
xmake build analyzer-demo
pass "analyzer-demo 构建成功"

echo ""
echo "  构建 analyzer-test ..."
if xmake build analyzer-test; then
    ANALYZER_TEST_BUILT=1
    pass "analyzer-test 构建成功"
else
    ANALYZER_TEST_BUILT=0
    warn "analyzer-test 构建失败；继续运行硬件层与 demo"
fi

echo ""

# ---- Step 3: 运行硬件层测试 ----
echo "--- Step 3: 运行硬件层测试 ---"
xmake run infinirt-test-analyzer-hw
pass "硬件层测试通过"
echo ""

# ---- Step 4: 运行真实输出 demo ----
echo "--- Step 4: 运行真实输出 demo ---"
xmake run analyzer-demo
pass "真实 analyzer-demo 运行通过"
echo ""

# ---- Step 5: 运行 analyzer 单元测试 ----
echo "--- Step 5: 运行 analyzer 单元测试 ---"
if [ "$ANALYZER_TEST_BUILT" -eq 1 ]; then
    xmake run analyzer-test
    pass "analyzer 单元测试通过"
else
    warn "跳过 analyzer-test 运行，因为构建未通过"
fi

echo ""
echo "============================================"
echo " 验证完成"
echo "============================================"
