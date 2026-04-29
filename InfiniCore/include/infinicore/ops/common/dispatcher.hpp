#pragma once

#include "../../analyzer/optimization_intent.hpp"
#include "../../device.hpp"

#include <array>

namespace infinicore::op::common {
template <typename Fn>
class OpDispatcher {
public:
    void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true) {
        auto &entry = table_[(size_t)device_type];
        if (entry == nullptr || override_existing) {
            entry = fn;
        }
    }

    void registerDevice(Device::Type device_type,
                        Fn fn,
                        analyzer::OptimizationGoal goal,
                        bool override_existing = true) {
        auto &entry = goal_table_[(size_t)device_type][goalIndex(goal)];
        if (entry == nullptr || override_existing) {
            entry = fn;
        }
    }

    void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true) {
        for (auto device_type : device_types) {
            registerDevice(device_type, fn, override_existing);
        }
    }

    void registerAll(Fn fn, bool override_existing = true) {
        for (size_t device_type = 0; device_type < static_cast<size_t>(Device::Type::COUNT); ++device_type) {
            registerDevice((Device::Type)device_type, fn, override_existing);
        }
    }

    void registerAll(Fn fn, analyzer::OptimizationGoal goal, bool override_existing = true) {
        for (size_t device_type = 0; device_type < static_cast<size_t>(Device::Type::COUNT); ++device_type) {
            registerDevice((Device::Type)device_type, fn, goal, override_existing);
        }
    }

    Fn lookup(Device::Type device_type) const {
        return table_.at((size_t)device_type);
    }

    Fn lookup(Device::Type device_type, analyzer::OptimizationGoal goal) const {
        auto fn = goal_table_.at((size_t)device_type).at(goalIndex(goal));
        return fn != nullptr ? fn : lookup(device_type);
    }

private:
    static constexpr size_t kGoalCount = 4;

    static constexpr size_t goalIndex(analyzer::OptimizationGoal goal) {
        return static_cast<size_t>(goal);
    }

    std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_{};
    std::array<std::array<Fn, kGoalCount>, static_cast<size_t>(Device::Type::COUNT)> goal_table_{};
};
} // namespace infinicore::op::common
