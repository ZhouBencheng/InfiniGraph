#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/dispatch/kernel_dispatcher.hpp"
#include "infinicore/analyzer/op_type.hpp"
#include "infinicore/analyzer/optimization_intent.hpp"
#include "infinicore/device.hpp"

#include <string>

namespace py = pybind11;

namespace infinicore::dispatch::pybind {

inline void bind(py::module &m) {
    auto dispatch_mod = m.def_submodule("dispatch",
        "Device-Aware Kernel Dispatch Module");

    dispatch_mod.def("select_kernel",
        [](const std::string &op_name,
           const std::string &device_name,
           const std::string &goal_name) -> py::object {
            analyzer::OpType op = analyzer::OpType::UNKNOWN;
            if (op_name == "paged_attention_prefill")
                op = analyzer::OpType::PAGED_ATTENTION_PREFILL;
            else if (op_name == "paged_attention")
                op = analyzer::OpType::PAGED_ATTENTION;

            Device::Type dev = Device::Type::CPU;
            if (device_name == "nvidia") dev = Device::Type::NVIDIA;
            else if (device_name == "iluvatar") dev = Device::Type::ILUVATAR;
            else if (device_name == "metax") dev = Device::Type::METAX;
            else if (device_name == "ali") dev = Device::Type::ALI;

            analyzer::OptimizationGoal goal = analyzer::OptimizationGoal::LATENCY_FIRST;
            if (goal_name == "throughput") goal = analyzer::OptimizationGoal::THROUGHPUT_FIRST;
            else if (goal_name == "memory_safe") goal = analyzer::OptimizationGoal::MEMORY_SAFE;
            else if (goal_name == "stability") goal = analyzer::OptimizationGoal::STABILITY_FIRST;

            const char *result = KernelDispatcher::instance()
                .selectKernelWithGoal(op, dev, goal, nullptr);
            if (result) return py::cast(std::string(result));
            return py::none();
        },
        "Query kernel selection (without op_info, only works for static rules)",
        py::arg("op_name"),
        py::arg("device_name"),
        py::arg("goal_name") = "latency");

    dispatch_mod.def("dump_rules", []() -> py::list {
        py::list entries;
        auto &d = KernelDispatcher::instance();
        const char *dev_names[] = {"cpu", "nvidia", "cambricon", "ascend",
                                    "metax", "moore", "iluvatar", "kunlun",
                                    "hygon", "qy", "ali"};
        const char *goal_names[] = {"latency", "throughput", "memory_safe", "stability"};

        for (size_t oi = 0; oi < static_cast<size_t>(analyzer::OpType::OP_TYPE_COUNT); ++oi) {
            auto op = static_cast<analyzer::OpType>(oi);
            for (size_t di = 0; di < static_cast<size_t>(Device::Type::COUNT); ++di) {
                auto dev = static_cast<Device::Type>(di);
                for (size_t gi = 0; gi < 4; ++gi) {
                    auto goal = static_cast<analyzer::OptimizationGoal>(gi);
                    if (!d.hasRule(op, dev, goal)) continue;
                    const char *result = d.selectKernelWithGoal(op, dev, goal, nullptr);
                    py::dict entry;
                    entry["op"] = analyzer::opTypeToString(op);
                    entry["device"] = di < 11 ? dev_names[di] : "unknown";
                    entry["goal"] = gi < 4 ? goal_names[gi] : "unknown";
                    entry["kernel"] = result ? result : "(needs op_info)";
                    entries.append(entry);
                }
            }
        }
        return entries;
    }, "Dump all registered dispatch rules");

    dispatch_mod.def("has_rule",
        [](const std::string &op_name,
           const std::string &device_name,
           const std::string &goal_name) -> bool {
            analyzer::OpType op = analyzer::OpType::UNKNOWN;
            if (op_name == "paged_attention_prefill")
                op = analyzer::OpType::PAGED_ATTENTION_PREFILL;

            Device::Type dev = Device::Type::CPU;
            if (device_name == "nvidia") dev = Device::Type::NVIDIA;
            else if (device_name == "iluvatar") dev = Device::Type::ILUVATAR;
            else if (device_name == "metax") dev = Device::Type::METAX;
            else if (device_name == "ali") dev = Device::Type::ALI;

            analyzer::OptimizationGoal goal = analyzer::OptimizationGoal::LATENCY_FIRST;
            if (goal_name == "throughput") goal = analyzer::OptimizationGoal::THROUGHPUT_FIRST;
            else if (goal_name == "memory_safe") goal = analyzer::OptimizationGoal::MEMORY_SAFE;
            else if (goal_name == "stability") goal = analyzer::OptimizationGoal::STABILITY_FIRST;

            return KernelDispatcher::instance().hasRule(op, dev, goal);
        },
        "Check if a dispatch rule is registered for (op, device, goal)",
        py::arg("op_name"),
        py::arg("device_name"),
        py::arg("goal_name") = "latency");
}

} // namespace infinicore::dispatch::pybind
