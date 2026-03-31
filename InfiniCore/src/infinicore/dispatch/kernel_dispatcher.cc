#include "infinicore/dispatch/kernel_dispatcher.hpp"
#include "infinicore/analyzer/op_type.hpp"
#include "infinicore/analyzer/optimization_intent.hpp"
#include "infinicore/device.hpp"

#ifdef ENABLE_MUTUAL_AWARENESS
#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace infinicore::dispatch {

namespace {
bool debugEnabled() {
    static const bool enabled = []() {
        const char *v = std::getenv("INFINIOP_DEBUG_PREFILL_DISPATCH");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}

const char *deviceTypeStr(Device::Type d) {
    switch (d) {
    case Device::Type::CPU: return "CPU";
    case Device::Type::NVIDIA: return "NVIDIA";
    case Device::Type::ILUVATAR: return "ILUVATAR";
    case Device::Type::METAX: return "METAX";
    case Device::Type::ALI: return "ALI";
    default: return "OTHER";
    }
}
} // namespace

const char *KernelDispatcher::selectKernel(
    analyzer::OpType op,
    Device::Type device,
    const void *op_info) const {

    auto goal = analyzer::OptimizationGoal::LATENCY_FIRST;
#ifdef ENABLE_MUTUAL_AWARENESS
    goal = analyzer::MutualAwarenessAnalyzer::instance().getCurrentOptimizationGoal();
#endif

    const char *result = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto k = key(op, device, goal);
        KernelSelectFn fn = (k < kTableSize) ? table_[k] : nullptr;
        if (fn != nullptr) {
            result = fn(op_info);
        }
    }

    if (debugEnabled()) {
        std::fprintf(stderr,
                     "[dispatch] op=%s device=%s goal=%s -> kernel=%s\n",
                     analyzer::opTypeToString(op),
                     deviceTypeStr(device),
                     analyzer::optimizationGoalToString(goal),
                     result ? result : "(fallback)");
    }

    return result;
}

} // namespace infinicore::dispatch
