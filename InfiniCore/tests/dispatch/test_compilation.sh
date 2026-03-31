#!/bin/bash
# KernelDispatcher 编译验证
# 用法: cd InfiniCore && bash tests/dispatch/test_compilation.sh [nvidia|iluvatar|metax]
set -e

PLATFORM=${1:-nvidia}
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$PLATFORM" in
    nvidia)   GPU_FLAG="--nv-gpu=y" ;;
    iluvatar) GPU_FLAG="--iluvatar-gpu=y" ;;
    metax)    GPU_FLAG="--metax-gpu=y" ;;
    *)        echo "支持: nvidia, iluvatar, metax"; exit 1 ;;
esac

echo -e "${YELLOW}=== KernelDispatcher 编译验证 (${PLATFORM}) ===${NC}"

# 测试 1: MUTUAL_AWARENESS=OFF
echo -e "\n${YELLOW}[1/2] ENABLE_MUTUAL_AWARENESS=OFF${NC}"
xmake f $GPU_FLAG --mutual-awareness=n -c -y 2>&1 | tail -3
if xmake build infinicore_cpp_api 2>&1; then
    echo -e "${GREEN}  [PASS] 编译成功（无 analyzer/dispatch）${NC}"
else
    echo -e "${RED}  [FAIL]${NC}"; exit 1
fi

# 测试 2: MUTUAL_AWARENESS=ON
echo -e "\n${YELLOW}[2/2] ENABLE_MUTUAL_AWARENESS=ON${NC}"
xmake f $GPU_FLAG --mutual-awareness=y -c -y 2>&1 | tail -3
if xmake build infinicore_cpp_api 2>&1; then
    echo -e "${GREEN}  [PASS] 编译成功（含 dispatch 模块）${NC}"
else
    echo -e "${RED}  [FAIL]${NC}"; exit 1
fi

# 安装并运行 Python 测试
echo -e "\n${YELLOW}[+] 安装并运行 Python 测试${NC}"
xmake install infinicore_cpp_api 2>&1 | tail -2
xmake build infinicore 2>&1 && xmake install infinicore 2>&1 | tail -2

echo ""
INFINIOP_DEBUG_PREFILL_DISPATCH=1 python tests/dispatch/test_dispatch.py

echo -e "\n${GREEN}=== 全部编译验证通过 ===${NC}"
