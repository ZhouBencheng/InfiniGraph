#include "infinicore/analyzer/op_trace.hpp"

namespace infinicore::analyzer {

namespace {

std::atomic_bool &opTraceEnabledFlag() {
    static std::atomic_bool enabled{true};
    return enabled;
}

} // namespace

OpTraceRing &getGlobalOpTrace() {
    static OpTraceRing instance(OpTraceRing::DEFAULT_CAPACITY);
    return instance;
}

bool isOpTraceEnabled() {
    return opTraceEnabledFlag().load(std::memory_order_relaxed);
}

void setOpTraceEnabled(bool enabled) {
    opTraceEnabledFlag().store(enabled, std::memory_order_relaxed);
}

} // namespace infinicore::analyzer
