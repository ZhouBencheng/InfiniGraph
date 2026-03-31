#!/usr/bin/env python3
"""
KernelDispatcher 调度模块测试脚本

用法:
    python test_dispatch.py
    INFINIOP_DEBUG_PREFILL_DISPATCH=1 python test_dispatch.py
"""
import sys
import os

def main():
    try:
        import infinicore
    except ImportError:
        print("[ERROR] infinicore 未安装，请先编译安装:")
        print("  cd InfiniCore && xmake f --<gpu>=y --mutual-awareness=y -c -y")
        print("  xmake build infinicore_cpp_api && xmake install infinicore_cpp_api")
        print("  xmake build infinicore && xmake install infinicore")
        sys.exit(1)

    if not hasattr(infinicore, 'dispatch'):
        print("[ERROR] infinicore.dispatch 不存在，需要 --mutual-awareness=y 编译")
        sys.exit(1)

    dispatch = infinicore.dispatch
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} {detail}")
            failed += 1

    # 测试 1: dump_rules
    print("\n=== 测试 1: dump_rules ===")
    rules = dispatch.dump_rules()
    print(f"  已注册规则数: {len(rules)}")
    for r in rules:
        print(f"    {r['op']:30s} | {r['device']:10s} | {r['goal']:12s} -> {r['kernel']}")
    check("至少有一条规则", len(rules) > 0)

    # 测试 2: has_rule
    print("\n=== 测试 2: has_rule ===")
    devices = ["nvidia", "iluvatar", "metax", "ali"]
    goals = ["latency", "throughput", "memory_safe", "stability"]
    found_any = False
    for dev in devices:
        for goal in goals:
            if dispatch.has_rule("paged_attention_prefill", dev, goal):
                found_any = True
                print(f"  [INFO] 有规则: {dev}/{goal}")
    check("至少有一个设备有规则", found_any)

    # 测试 3: select_kernel 静态规则
    print("\n=== 测试 3: select_kernel 静态规则 ===")
    for dev in devices:
        for goal in ["memory_safe", "throughput"]:
            if dispatch.has_rule("paged_attention_prefill", dev, goal):
                result = dispatch.select_kernel("paged_attention_prefill", dev, goal)
                print(f"  [INFO] {dev}/{goal} -> {result}")
                check(f"{dev}/{goal} 返回非 None", result is not None)

    # 测试 4: 无规则返回 None
    print("\n=== 测试 4: 回退验证 ===")
    result = dispatch.select_kernel("paged_attention_prefill", "cpu", "latency")
    check("CPU 无规则返回 None", result is None, f"got: {result}")

    result = dispatch.select_kernel("gemm", "nvidia", "latency")
    check("GEMM 无调度规则返回 None", result is None, f"got: {result}")

    # 汇总
    print(f"\n{'='*50}")
    print(f"通过: {passed}  失败: {failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
