// ============================================================
// Analyzer Load Demo
//
// Runs the mutual-awareness analyzer under several live GPU load
// scenarios. The workload trace is synthetic, but resource readings
// are collected from the real backend through infinirt/IXML.
// ============================================================

#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"
#include "infinicore/analyzer/op_trace.hpp"
#include "infinirt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace infinicore::analyzer;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;
constexpr size_t GiB = 1024ull * MiB;

#if defined(ANALYZER_LOAD_DEMO_HAS_COMPUTE_KERNEL)
extern "C" void analyzerLoadDemoLaunchCompute(float *data, size_t n, int rounds, void *stream);
#endif

struct DetectedDevice {
    infiniDevice_t type = INFINI_DEVICE_CPU;
    int count = 0;
    const char *name = "CPU";
};

const char *deviceTypeName(infiniDevice_t t) {
    switch (t) {
    case INFINI_DEVICE_ILUVATAR: return "Iluvatar";
    case INFINI_DEVICE_NVIDIA: return "NVIDIA";
    case INFINI_DEVICE_METAX: return "MetaX";
    default: return "CPU";
    }
}

DetectedDevice detectAccelerator() {
    infiniDevice_t order[] = {
        INFINI_DEVICE_ILUVATAR,
        INFINI_DEVICE_METAX,
        INFINI_DEVICE_NVIDIA,
    };
    for (auto t : order) {
        int count = 0;
        if (infinirtGetDeviceCount(t, &count) == INFINI_STATUS_SUCCESS && count > 0) {
            return {t, count, deviceTypeName(t)};
        }
    }
    return {};
}

void printRule(char ch, int n = 96) {
    for (int i = 0; i < n; ++i) {
        std::putchar(ch);
    }
    std::putchar('\n');
}

const char *phaseStr(PhaseType p) { return phaseTypeToString(p); }
const char *bottleneckStr(BottleneckType b) { return bottleneckTypeToString(b); }
const char *goalStr(OptimizationGoal g) { return optimizationGoalToString(g); }

void injectOp(OpType op, const std::vector<size_t> &shape, uint8_t device_type, int8_t device_id) {
    traceOp(op, shape.data(), shape.size(),
            /*dtype=*/2,
            device_type,
            device_id);
}

struct TaskCase {
    const char *name;
    const char *summary;
    PhaseType expected;
    size_t seq_len;
    std::vector<OpType> ops;
};

std::vector<TaskCase> buildTaskCases() {
    return {
        {
            "Prefill",
            "long-seq attention + GEMM",
            PhaseType::PREFILL,
            512,
            {OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::GEMM, OpType::RMS_NORM,
             OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::GEMM, OpType::ADD,
             OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::GEMM, OpType::RMS_NORM,
             OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::GEMM, OpType::ADD},
        },
        {
            "Decode",
            "single-token attention + MLP",
            PhaseType::DECODE,
            1,
            {OpType::ATTENTION, OpType::GEMM, OpType::SILU_AND_MUL, OpType::RMS_NORM,
             OpType::ATTENTION, OpType::GEMM, OpType::SILU_AND_MUL, OpType::ADD,
             OpType::ATTENTION, OpType::GEMM, OpType::SILU_AND_MUL, OpType::RMS_NORM,
             OpType::ATTENTION, OpType::GEMM, OpType::SILU_AND_MUL, OpType::ADD},
        },
        {
            "GEMM/MLP",
            "dense matmul + activation",
            PhaseType::GEMM_MLP_DENSE,
            128,
            {OpType::GEMM, OpType::LINEAR, OpType::SILU_AND_MUL, OpType::GELU,
             OpType::GEMM, OpType::LINEAR, OpType::SWIGLU, OpType::ADD,
             OpType::GEMM, OpType::LINEAR, OpType::SILU_AND_MUL, OpType::GELU,
             OpType::GEMM, OpType::LINEAR, OpType::SWIGLU, OpType::ADD},
        },
        {
            "KV Cache",
            "KV/paged-cache updates",
            PhaseType::KV_CACHE,
            128,
            {OpType::KV_CACHING, OpType::PAGED_CACHING, OpType::KV_CACHING, OpType::PAGED_CACHING,
             OpType::KV_CACHING, OpType::PAGED_CACHING, OpType::ADD, OpType::RMS_NORM,
             OpType::KV_CACHING, OpType::PAGED_CACHING, OpType::KV_CACHING, OpType::PAGED_CACHING,
             OpType::KV_CACHING, OpType::PAGED_CACHING, OpType::ADD, OpType::RMS_NORM},
        },
        {
            "AllReduce",
            "communication-heavy window",
            PhaseType::COMMUNICATION,
            128,
            {OpType::ALLREDUCE, OpType::ALLREDUCE, OpType::GEMM, OpType::ADD,
             OpType::ALLREDUCE, OpType::ALLREDUCE, OpType::RMS_NORM, OpType::ADD,
             OpType::ALLREDUCE, OpType::ALLREDUCE, OpType::GEMM, OpType::ADD,
             OpType::ALLREDUCE, OpType::ALLREDUCE, OpType::RMS_NORM, OpType::ADD},
        },
    };
}

enum class LoadMode {
    Idle,
    MemoryPressure,
    BandwidthCopy,
    ComputeKernel,
    Mixed,
};

struct LoadScenario {
    LoadMode mode;
    const char *name;
    const char *description;
};

std::vector<LoadScenario> buildLoadScenarios() {
    return {
        {LoadMode::Idle, "idle", "no artificial GPU load"},
        {LoadMode::MemoryPressure, "memory_pressure", "hold about 88% of device memory"},
        {LoadMode::BandwidthCopy, "bandwidth_copy", "continuous device-to-device memcpy"},
        {LoadMode::ComputeKernel, "compute_kernel", "continuous backend arithmetic kernel"},
        {LoadMode::Mixed, "mixed", "memory hold + copy + compute"},
    };
}

struct ResourceView {
    bool valid = false;
    float memory_usage = 0.0f;
    float compute = 0.0f;
    float bandwidth = 0.0f;
    float communication = 0.0f;
    float confidence = 0.0f;
    size_t free_bytes = 0;
};

ResourceView firstResource(const OptimizationIntent &intent) {
    ResourceView view;
    if (intent.per_device.empty()) {
        return view;
    }
    const auto &d = intent.per_device.front();
    view.valid = true;
    view.memory_usage = d.memory_usage_ratio;
    view.compute = d.compute_utilization;
    view.bandwidth = d.memory_bandwidth_utilization;
    view.communication = d.communication_time_ratio;
    view.confidence = d.resource_confidence;
    view.free_bytes = d.memory_available_bytes;
    return view;
}

size_t alignDown(size_t value, size_t alignment) {
    return value / alignment * alignment;
}

bool rtOk(infiniStatus_t status, const char *what) {
    if (status == INFINI_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[analyzer-load-demo] %s failed: status=%d\n", what, static_cast<int>(status));
    return false;
}

bool allocateWithBackoff(void **ptr, size_t &bytes) {
    *ptr = nullptr;
    bytes = alignDown(bytes, MiB);
    while (bytes >= 64 * MiB) {
        if (infinirtMalloc(ptr, bytes) == INFINI_STATUS_SUCCESS) {
            return true;
        }
        *ptr = nullptr;
        bytes /= 2;
        bytes = alignDown(bytes, MiB);
    }
    return false;
}

class ScopedGpuLoad {
public:
    ScopedGpuLoad() = default;
    ~ScopedGpuLoad() { stop(); }

    bool start(LoadMode mode, const DetectedDevice &device) {
        if (mode == LoadMode::Idle || device.type == INFINI_DEVICE_CPU) {
            return true;
        }
        device_ = device;

        if (!rtOk(infinirtSetDevice(device.type, 0), "infinirtSetDevice")) {
            return false;
        }
        if (!rtOk(infinirtStreamCreate(&stream_), "infinirtStreamCreate")) {
            return false;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        (void)infinirtGetMemInfo(device.type, 0, &free_bytes, &total_bytes);

        if (mode == LoadMode::MemoryPressure || mode == LoadMode::Mixed) {
            size_t current_used = total_bytes > free_bytes ? total_bytes - free_bytes : 0;
            size_t target_used = static_cast<size_t>(static_cast<double>(total_bytes) * (mode == LoadMode::Mixed ? 0.65 : 0.88));
            hold_bytes_ = target_used > current_used ? target_used - current_used : 0;
            hold_bytes_ = std::min(hold_bytes_, free_bytes > GiB ? free_bytes - GiB : free_bytes / 2);
            if (!allocateWithBackoff(&hold_, hold_bytes_)) {
                std::fprintf(stderr, "[analyzer-load-demo] memory pressure allocation unavailable\n");
                return failStart();
            }
        }

        if (mode == LoadMode::BandwidthCopy || mode == LoadMode::Mixed) {
            copy_bytes_ = std::min<size_t>(512 * MiB, free_bytes / 16);
            if (!allocateWithBackoff(&copy_a_, copy_bytes_)) {
                std::fprintf(stderr, "[analyzer-load-demo] first copy buffer allocation unavailable\n");
                return failStart();
            }
            size_t second = copy_bytes_;
            if (!allocateWithBackoff(&copy_b_, second)) {
                std::fprintf(stderr, "[analyzer-load-demo] second copy buffer allocation unavailable\n");
                return failStart();
            }
            copy_bytes_ = std::min(copy_bytes_, second);
        }

        if (mode == LoadMode::ComputeKernel || mode == LoadMode::Mixed) {
#if !defined(ANALYZER_LOAD_DEMO_HAS_COMPUTE_KERNEL)
            std::fprintf(stderr, "[analyzer-load-demo] compute kernel is not compiled for this backend\n");
            return failStart();
#else
            compute_bytes_ = 128 * MiB;
            if (!allocateWithBackoff(&compute_, compute_bytes_)) {
                std::fprintf(stderr, "[analyzer-load-demo] compute buffer allocation unavailable\n");
                return failStart();
            }
#endif
        }

        if (copy_a_ != nullptr || compute_ != nullptr) {
            running_.store(true);
            worker_ = std::thread([this]() { this->run(); });
        }
        return true;
    }

    void stop() {
        running_.store(false);
        if (worker_.joinable()) {
            worker_.join();
        }
        if (stream_ != nullptr) {
            (void)infinirtStreamSynchronize(stream_);
        }
        if (copy_a_) (void)infinirtFree(copy_a_);
        if (copy_b_) (void)infinirtFree(copy_b_);
        if (compute_) (void)infinirtFree(compute_);
        if (hold_) (void)infinirtFree(hold_);
        copy_a_ = nullptr;
        copy_b_ = nullptr;
        compute_ = nullptr;
        hold_ = nullptr;
        if (stream_ != nullptr) {
            (void)infinirtStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    size_t heldBytes() const { return hold_bytes_; }
    size_t copyBytes() const { return copy_bytes_; }
    size_t computeBytes() const { return compute_bytes_; }

private:
    bool failStart() {
        stop();
        return false;
    }

    void run() {
        (void)infinirtSetDevice(device_.type, 0);
        int iter = 0;
        while (running_.load()) {
            if (copy_a_ != nullptr && copy_b_ != nullptr && copy_bytes_ > 0) {
                (void)infinirtMemcpyAsync(copy_b_, copy_a_, copy_bytes_, INFINIRT_MEMCPY_D2D, stream_);
                (void)infinirtMemcpyAsync(copy_a_, copy_b_, copy_bytes_, INFINIRT_MEMCPY_D2D, stream_);
            }
#if defined(ANALYZER_LOAD_DEMO_HAS_COMPUTE_KERNEL)
            if (compute_ != nullptr && compute_bytes_ > 0) {
                auto *data = static_cast<float *>(compute_);
                size_t n = compute_bytes_ / sizeof(float);
                analyzerLoadDemoLaunchCompute(data, n, 1024, stream_);
            }
#endif
            if ((++iter % 4) == 0) {
                (void)infinirtStreamSynchronize(stream_);
            }
        }
        (void)infinirtStreamSynchronize(stream_);
    }

    DetectedDevice device_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    infinirtStream_t stream_ = nullptr;
    void *hold_ = nullptr;
    void *copy_a_ = nullptr;
    void *copy_b_ = nullptr;
    void *compute_ = nullptr;
    size_t hold_bytes_ = 0;
    size_t copy_bytes_ = 0;
    size_t compute_bytes_ = 0;
};

OptimizationIntent analyzeTask(const TaskCase &task, const DetectedDevice &dev, double *latency_ms) {
    auto &trace = getGlobalOpTrace();
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    trace.clear();
    analyzer.clearGraphCache();

    std::vector<size_t> shape = {1u, 32u, task.seq_len};
    for (auto op : task.ops) {
        injectOp(op, shape, static_cast<uint8_t>(dev.type), 0);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto intent = analyzer.analyze();
    auto t1 = std::chrono::steady_clock::now();
    *latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return intent;
}

void printHeader(const DetectedDevice &dev) {
    printRule('=');
    std::printf(" Analyzer Load Demo - different GPU load x different task trace\n");
    printRule('=');
    std::printf(" Device          : %d x %s\n", dev.count, dev.name);
    std::printf(" Resource source : infinirtGetDeviceResourceSnapshot() -> backend management fields\n");
    std::printf(" Task source     : synthetic OpTrace windows for phase/goal comparison\n");
    std::printf(" Requirement     : analyze() must finish within 10 seconds\n");
    printRule('=');
    std::putchar('\n');
}

void printScenarioIntro(const LoadScenario &scenario, const ScopedGpuLoad &load) {
    printRule('-');
    std::printf(" GPU Load: %s\n", scenario.name);
    std::printf(" Meaning : %s\n", scenario.description);
    if (load.heldBytes() > 0) {
        std::printf(" Held memory : %.2f GiB\n", static_cast<float>(load.heldBytes()) / GiB);
    }
    if (load.copyBytes() > 0) {
        std::printf(" Copy buffer : %.2f MiB x2\n", static_cast<float>(load.copyBytes()) / MiB);
    }
    if (load.computeBytes() > 0) {
        std::printf(" Compute buf : %.2f MiB\n", static_cast<float>(load.computeBytes()) / MiB);
    }
    printRule('-');
    std::printf("%-12s %-26s %-18s %-18s %-18s %7s %7s %7s %7s %9s\n",
                "task", "trace", "phase", "bottleneck", "goal",
                "mem%", "gpu%", "bw%", "conf", "ms");
}

void printRow(const TaskCase &task, const OptimizationIntent &intent, double latency_ms) {
    auto r = firstResource(intent);
    bool phase_ok = intent.global.current_phase == task.expected;
    std::printf("%-12s %-26s %-18s %-18s %-18s %6.1f%% %6.1f%% %6.1f%% %7.2f %9.2f%s\n",
                task.name,
                task.summary,
                phaseStr(intent.global.current_phase),
                bottleneckStr(intent.global.primary_bottleneck),
                goalStr(intent.global.goal),
                r.valid ? r.memory_usage * 100.0f : 0.0f,
                r.valid ? r.compute * 100.0f : 0.0f,
                r.valid ? r.bandwidth * 100.0f : 0.0f,
                r.valid ? r.confidence : 0.0f,
                static_cast<float>(latency_ms),
                phase_ok ? "" : "  phase-mismatch");
}

void printInterpretation() {
    printRule('=');
    std::printf(" Reading the table\n");
    printRule('=');
    std::printf(" - Rows under the same GPU Load show how task semantics change phase/bottleneck/goal.\n");
    std::printf(" - Columns mem%%/gpu%%/bw%% are live resource readings from the accelerator.\n");
    std::printf(" - memory_pressure should push most tasks toward memory_bound + memory_safe.\n");
    std::printf(" - compute/copy/mixed loads expose whether the backend reports high compute or bandwidth pressure.\n");
    std::printf(" - ms is end-to-end MutualAwarenessAnalyzer::analyze() latency; requirement is <= 10000 ms.\n");
    printRule('=');
}

} // namespace

int main(int argc, char **argv) {
    int warmup_ms = 1500;
    if (argc >= 2) {
        warmup_ms = std::max(0, std::atoi(argv[1]));
    }

    if (infinirtInit() != INFINI_STATUS_SUCCESS) {
        std::fprintf(stderr, "[analyzer-load-demo] infinirtInit() failed\n");
        return 1;
    }

    auto dev = detectAccelerator();
    if (dev.type == INFINI_DEVICE_CPU || dev.count <= 0) {
        std::fprintf(stderr, "[analyzer-load-demo] no accelerator device detected\n");
        return 1;
    }

    (void)infinirtSetDevice(dev.type, 0);

    MutualAwarenessAnalyzer::instance().setEnabled(true);
    printHeader(dev);

    auto scenarios = buildLoadScenarios();
    auto tasks = buildTaskCases();
    double worst_ms = 0.0;
    int rows = 0;
    int phase_ok = 0;

    for (const auto &scenario : scenarios) {
        ScopedGpuLoad load;
        if (!load.start(scenario.mode, dev)) {
            std::fprintf(stderr, "[analyzer-load-demo] failed to start load scenario: %s\n", scenario.name);
            continue;
        }

        if (warmup_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(warmup_ms));
        }

        printScenarioIntro(scenario, load);
        for (const auto &task : tasks) {
            double latency_ms = 0.0;
            auto intent = analyzeTask(task, dev, &latency_ms);
            printRow(task, intent, latency_ms);
            worst_ms = std::max(worst_ms, latency_ms);
            ++rows;
            if (intent.global.current_phase == task.expected) {
                ++phase_ok;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
        std::putchar('\n');
        load.stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    printInterpretation();
    std::printf(" Summary: phase=%d/%d correct, worst analyze latency=%.2f ms, requirement=%s\n",
                phase_ok,
                rows,
                static_cast<float>(worst_ms),
                worst_ms <= 10000.0 ? "PASSED" : "FAILED");
    return worst_ms <= 10000.0 ? 0 : 2;
}
