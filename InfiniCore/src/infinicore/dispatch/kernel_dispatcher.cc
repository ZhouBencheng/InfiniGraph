#include "infinicore/dispatch/kernel_dispatcher.hpp"

#ifdef ENABLE_MUTUAL_AWARENESS
#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"
#endif

namespace infinicore::dispatch {

const char *KernelDispatcher::selectKernel(
    analyzer::OpType op,
    Device::Type device,
    const void *op_info) const {

    // Determine current optimization goal
    auto goal = analyzer::OptimizationGoal::LATENCY_FIRST;
#ifdef ENABLE_MUTUAL_AWARENESS
    goal = analyzer::MutualAwarenessAnalyzer::instance().getCurrentOptimizationGoal();
#endif

    // Look up the registered rule
    auto k = key(op, device, goal);
    KernelSelectFn fn = (k < kTableSize) ? table_[k] : nullptr;

    if (fn != nullptr) {
        return fn(op_info);
    }

    // No rule registered — caller should use its own fallback
    return nullptr;
}

} // namespace infinicore::dispatch
