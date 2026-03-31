#!/usr/bin/env python3
"""
KernelDispatcher 端到端验证（Analyzer 联动）

用法:
    INFINIOP_DEBUG_PREFILL_DISPATCH=1 python test_e2e_dispatch.py
"""
import sys

def main():
    import infinicore
    assert hasattr(infinicore, 'dispatch'), "dispatch 模块未加载"
    assert hasattr(infinicore, 'analyzer'), "analyzer 模块未加载"

    analyzer = infinicore.analyzer
    dispatch = infinicore.dispatch

    # 注入 prefill 模式 op trace
    print("=== Analyzer 联动测试 ===")
    analyzer.clear_trace()
    for _ in range(20):
        analyzer.trace_op_for_test(
            analyzer.OpType.PAGED_ATTENTION_PREFILL,
            [32, 2048, 128], dtype=0, device_type=1, device_id=0)
        analyzer.trace_op_for_test(
            analyzer.OpType.GEMM,
            [32, 2048, 4096], dtype=0, device_type=1, device_id=0)

    intent = analyzer.analyze()
    print(f"  phase={intent.global_intent.current_phase}")
    print(f"  goal={intent.global_intent.goal}")

    rules = dispatch.dump_rules()
    print(f"  dispatch 规则: {len(rules)} 条")
    for r in rules:
        print(f"    {r['op']} | {r['device']} | {r['goal']} -> {r['kernel']}")

    assert len(rules) > 0, "无规则注册"
    print("\n[PASS] 联动测试通过")

if __name__ == "__main__":
    main()
