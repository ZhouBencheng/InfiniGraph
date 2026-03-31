#pragma once

#include "../analyzer/op_type.hpp"
#include "../analyzer/optimization_intent.hpp"
#include "../device.hpp"

#include <array>
#include <cstddef>
#include <mutex>

namespace infinicore::dispatch {

/// Kernel selection function signature.
/// Receives op-specific info struct (e.g. PagedAttentionPrefillInfo*),
/// returns a kernel variant name (e.g. "warp", "warpcta8pipe").
/// Must return a string literal (static lifetime).
using KernelSelectFn = const char *(*)(const void *op_info);

/// Independent kernel variant dispatcher.
///
/// Decouples kernel variant selection (warp vs warpcta8pipe etc.)
/// from operator implementations (.cu / .maca files).
///
/// Three-dimensional lookup: (OpType, DeviceType, OptimizationGoal) → KernelSelectFn
///
/// Usage from .cu files:
///   const char *k = KernelDispatcher::instance().selectKernel(
///       OpType::PAGED_ATTENTION_PREFILL, Device::Type::ILUVATAR, &info);
///
/// Rules are registered via static initializers in separate .cc files,
/// one per hardware platform. Adding a new GPU only requires a new rules file.
class KernelDispatcher {
public:
    static KernelDispatcher &instance() {
        static KernelDispatcher inst;
        return inst;
    }

    /// Register a kernel selection rule for a specific (op, device, goal) triple.
    void registerRule(
        analyzer::OpType op,
        Device::Type device,
        analyzer::OptimizationGoal goal,
        KernelSelectFn fn) {
        std::lock_guard<std::mutex> lock(mu_);
        table_[key(op, device, goal)] = fn;
    }

    /// Register a default rule for all goals under (op, device).
    /// Does not overwrite existing per-goal rules.
    void registerDefault(
        analyzer::OpType op,
        Device::Type device,
        KernelSelectFn fn) {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint8_t g = 0; g < kGoalCount; ++g) {
            auto k = key(op, device, static_cast<analyzer::OptimizationGoal>(g));
            if (table_[k] == nullptr) {
                table_[k] = fn;
            }
        }
    }

    /// Select a kernel variant for the given op + device + current workload.
    ///
    /// Internally queries MutualAwarenessAnalyzer for the current OptimizationGoal.
    /// Falls back to LATENCY_FIRST if analyzer is unavailable.
    ///
    /// Returns nullptr if no rule is registered (caller should use its own fallback).
    const char *selectKernel(
        analyzer::OpType op,
        Device::Type device,
        const void *op_info) const;

private:
    KernelDispatcher() { table_.fill(nullptr); }

    static constexpr size_t kOpCount = static_cast<size_t>(analyzer::OpType::OP_TYPE_COUNT);
    static constexpr size_t kDeviceCount = static_cast<size_t>(Device::Type::COUNT);
    static constexpr size_t kGoalCount = 4; // LATENCY, THROUGHPUT, MEMORY_SAFE, STABILITY
    static constexpr size_t kTableSize = kOpCount * kDeviceCount * kGoalCount;

    static size_t key(analyzer::OpType op, Device::Type device, analyzer::OptimizationGoal goal) {
        return (static_cast<size_t>(op) * kDeviceCount + static_cast<size_t>(device)) * kGoalCount
               + static_cast<size_t>(goal);
    }

    std::array<KernelSelectFn, kTableSize> table_;
    mutable std::mutex mu_;
};

} // namespace infinicore::dispatch
