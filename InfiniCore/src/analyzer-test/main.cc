// ============================================================
// Analyzer Module Unit Tests
//
// Analyzer-focused tests with a small runtime-backed B1 pilot.
// Tests: OpTraceRing, PhaseDetector, ResourceSensor,
//        IntentGenerator, MutualAwarenessAnalyzer, OpDispatcher
// ============================================================

#include "infinicore/analyzer/op_trace.hpp"
#include "infinicore/analyzer/op_type_registry.hpp"
#include "infinicore/analyzer/phase_detector.hpp"
#include "infinicore/analyzer/resource_sensor.hpp"
#include "infinicore/analyzer/intent_generator.hpp"
#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/attention.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/flash_attention.hpp"
#include "infinicore/ops/per_tensor_quant_i8.hpp"
#include "infinirt.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace infinicore::analyzer;

// ============================================================
// Test utilities
// ============================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_CASE(name) \
    static void test_##name(); \
    static bool _registered_##name = []() { \
        return true; \
    }(); \
    static void test_##name()

#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "  ❌ ASSERT_TRUE failed: " << #expr \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed: " #expr); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "  ❌ ASSERT_EQ failed: " << #a << " == " << #b \
                      << " (" << (a) << " != " << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while (0)

#define ASSERT_GT(a, b) \
    do { \
        if (!((a) > (b))) { \
            std::cerr << "  ❌ ASSERT_GT failed: " << #a << " > " << #b \
                      << " (" << (a) << " <= " << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while (0)

#define ASSERT_GE(a, b) \
    do { \
        if (!((a) >= (b))) { \
            std::cerr << "  ❌ ASSERT_GE failed: " << #a << " >= " << #b \
                      << " (" << (a) << " < " << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while (0)

#define RUN_TEST(name) \
    do { \
        std::cout << "[TEST] " << #name << " ... " << std::flush; \
        try { \
            test_##name(); \
            std::cout << "✅ PASSED" << std::endl; \
            tests_passed++; \
        } catch (const std::exception &e) { \
            std::cout << "❌ FAILED: " << e.what() << std::endl; \
            tests_failed++; \
        } \
    } while (0)

namespace {

using DispatcherProbeFn = int (*)(int);

int dispatcherDefaultProbe(int value) {
    return value + 1;
}

int dispatcherLatencyProbe(int value) {
    return value + 10;
}

int dispatcherThroughputProbe(int value) {
    return value + 100;
}

int attention_default_hits = 0;
int attention_throughput_hits = 0;

void resetAttentionDispatchCounters() {
    attention_default_hits = 0;
    attention_throughput_hits = 0;
}

void attentionDefaultStub(infinicore::Tensor out,
                          infinicore::Tensor q,
                          infinicore::Tensor k,
                          infinicore::Tensor v,
                          infinicore::Tensor k_cache,
                          infinicore::Tensor v_cache,
                          size_t pos) {
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)k_cache;
    (void)v_cache;
    (void)pos;
    attention_default_hits++;
}

void attentionThroughputStub(infinicore::Tensor out,
                             infinicore::Tensor q,
                             infinicore::Tensor k,
                             infinicore::Tensor v,
                             infinicore::Tensor k_cache,
                             infinicore::Tensor v_cache,
                             size_t pos) {
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)k_cache;
    (void)v_cache;
    (void)pos;
    attention_throughput_hits++;
}

infinicore::Tensor makeCpuBlobTensor(std::vector<uint16_t> &storage, const infinicore::Shape &shape) {
    return infinicore::Tensor::from_blob(storage.data(), shape, infinicore::DataType::F16, infinicore::Device::cpu());
}

} // namespace

// ============================================================
// OpTraceRing Tests
// ============================================================

void test_op_trace_ring_basic() {
    OpTraceRing ring(8);
    ASSERT_EQ(ring.size(), 0u);
    ASSERT_EQ(ring.capacity(), 8u);
    ASSERT_EQ(ring.totalCount(), 0u);

    // Write one entry
    OpTraceEntry entry;
    entry.op_type = OpType::GEMM;
    entry.timestamp_ns = OpTraceEntry::now();
    ring.write(entry);

    ASSERT_EQ(ring.size(), 1u);
    ASSERT_EQ(ring.totalCount(), 1u);

    auto entries = ring.getAllEntries();
    ASSERT_EQ(entries.size(), 1u);
    ASSERT_EQ(static_cast<int>(entries[0].op_type), static_cast<int>(OpType::GEMM));
}

void test_op_trace_ring_wrap() {
    OpTraceRing ring(4);

    // Write 6 entries to wrap around
    for (int i = 0; i < 6; i++) {
        OpTraceEntry entry;
        entry.op_type = static_cast<OpType>(i + 1);
        ring.write(entry);
    }

    ASSERT_EQ(ring.size(), 4u);     // Capacity is 4
    ASSERT_EQ(ring.totalCount(), 6u);

    auto entries = ring.getAllEntries();
    ASSERT_EQ(entries.size(), 4u);
    // Should contain entries 3, 4, 5, 6 (0-indexed: op types 3,4,5,6)
    ASSERT_EQ(static_cast<int>(entries[0].op_type), 3);
    ASSERT_EQ(static_cast<int>(entries[3].op_type), 6);
}

void test_op_trace_ring_recent() {
    OpTraceRing ring(16);

    for (int i = 0; i < 10; i++) {
        OpTraceEntry entry;
        entry.op_type = static_cast<OpType>(i + 1);
        ring.write(entry);
    }

    // Get last 3
    auto entries = ring.getRecentEntries(3);
    ASSERT_EQ(entries.size(), 3u);
    ASSERT_EQ(static_cast<int>(entries[0].op_type), 8);
    ASSERT_EQ(static_cast<int>(entries[2].op_type), 10);
}

void test_op_trace_ring_clear() {
    OpTraceRing ring(8);

    OpTraceEntry entry;
    entry.op_type = OpType::ADD;
    ring.write(entry);
    ring.write(entry);
    ASSERT_EQ(ring.size(), 2u);

    ring.clear();
    ASSERT_EQ(ring.size(), 0u);
    ASSERT_EQ(ring.totalCount(), 0u);
}

void test_op_trace_entry_shape() {
    OpTraceEntry entry;
    size_t dims[] = {2, 128, 64, 32, 16}; // 5 dims, only first 4 stored
    entry.setShape(dims, 5);

    ASSERT_EQ(entry.ndim, 4u); // MAX_DIMS = 4
    ASSERT_EQ(entry.shape[0], 2u);
    ASSERT_EQ(entry.shape[1], 128u);
    ASSERT_EQ(entry.shape[2], 64u);
    ASSERT_EQ(entry.shape[3], 32u);
}

void test_op_trace_global_singleton() {
    auto &trace1 = getGlobalOpTrace();
    auto &trace2 = getGlobalOpTrace();
    ASSERT_TRUE(&trace1 == &trace2);  // Same instance
}

void test_op_trace_records_metadata() {
    auto &trace = getGlobalOpTrace();
    trace.clear();

    size_t dims[] = {2, 128, 64};
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::NVIDIA),
            3);

    auto entries = trace.getAllEntries();
    ASSERT_EQ(entries.size(), 1u);
    ASSERT_EQ(static_cast<int>(entries[0].op_type), static_cast<int>(OpType::FLASH_ATTENTION));
    ASSERT_EQ(entries[0].ndim, 3u);
    ASSERT_EQ(entries[0].shape[0], 2u);
    ASSERT_EQ(entries[0].shape[1], 128u);
    ASSERT_EQ(entries[0].shape[2], 64u);
    ASSERT_TRUE(entries[0].dtype != 0);
    ASSERT_TRUE(entries[0].device_type != 0);
    ASSERT_EQ(entries[0].device_id, 3);
    ASSERT_TRUE(entries[0].timestamp_ns > 0);

    trace.clear();
}

void test_op_trace_runtime_switch() {
    auto &trace = getGlobalOpTrace();
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    trace.clear();
    analyzer.clearGraphCache();

    analyzer.setEnabled(false);
    ASSERT_TRUE(!analyzer.isEnabled());
    ASSERT_TRUE(!isAnalyzerEnabled());
    traceOp(OpType::GEMM,
            nullptr,
            0,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::NVIDIA),
            0);
    ASSERT_EQ(trace.size(), 0u);
    ASSERT_EQ(static_cast<int>(analyzer.analyze().global.current_phase),
              static_cast<int>(PhaseType::UNKNOWN));

    analyzer.setEnabled(true);
    ASSERT_TRUE(analyzer.isEnabled());
    ASSERT_TRUE(isAnalyzerEnabled());
    traceOp(OpType::GEMM,
            nullptr,
            0,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::NVIDIA),
            0);
    ASSERT_EQ(trace.size(), 1u);

    trace.clear();
}

// ============================================================
// OpType Tests
// ============================================================

void test_op_type_classification() {
    ASSERT_TRUE(isAttentionOp(OpType::ATTENTION));
    ASSERT_TRUE(isAttentionOp(OpType::FLASH_ATTENTION));
    ASSERT_TRUE(isAttentionOp(OpType::PAGED_ATTENTION));
    ASSERT_TRUE(!isAttentionOp(OpType::GEMM));
    ASSERT_TRUE(!isAttentionOp(OpType::SILU));

    ASSERT_TRUE(isGemmMlpOp(OpType::GEMM));
    ASSERT_TRUE(isGemmMlpOp(OpType::LINEAR));
    ASSERT_TRUE(isGemmMlpOp(OpType::MATMUL));
    ASSERT_TRUE(!isGemmMlpOp(OpType::ATTENTION));

    ASSERT_TRUE(isActivationOp(OpType::SILU));
    ASSERT_TRUE(isActivationOp(OpType::GELU));
    ASSERT_TRUE(!isActivationOp(OpType::GEMM));

    ASSERT_TRUE(isKvCacheOp(OpType::KV_CACHING));
    ASSERT_TRUE(isKvCacheOp(OpType::PAGED_CACHING));
    ASSERT_TRUE(!isKvCacheOp(OpType::ATTENTION));
}

void test_op_type_to_string() {
    ASSERT_TRUE(std::string(opTypeToString(OpType::GEMM)) == "gemm");
    ASSERT_TRUE(std::string(opTypeToString(OpType::ATTENTION)) == "attention");
    ASSERT_TRUE(std::string(opTypeToString(OpType::UNKNOWN)) == "unknown");
}

void test_graph_op_type_mapping() {
    ASSERT_EQ(static_cast<int>(opTypeFromName("Add")), static_cast<int>(OpType::ADD));
    ASSERT_EQ(static_cast<int>(opTypeFromName("FlashAttention")),
              static_cast<int>(OpType::FLASH_ATTENTION));
    ASSERT_EQ(static_cast<int>(opTypeFromName("PerTensorQuantI8")),
              static_cast<int>(OpType::PER_TENSOR_QUANT_I8));
    ASSERT_EQ(static_cast<int>(opTypeFromName("AllReduce")),
              static_cast<int>(OpType::ALLREDUCE));
}

// ============================================================
// OpDispatcher Tests
// ============================================================

void test_op_dispatcher_goal_exact_match() {
    infinicore::op::common::OpDispatcher<DispatcherProbeFn> dispatcher;
    dispatcher.registerDevice(infinicore::Device::Type::CPU, &dispatcherDefaultProbe);
    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &dispatcherThroughputProbe,
        OptimizationGoal::THROUGHPUT_FIRST);

    ASSERT_EQ(dispatcher.lookup(infinicore::Device::Type::CPU, OptimizationGoal::THROUGHPUT_FIRST)(5), 105);
}

void test_op_dispatcher_goal_fallback() {
    infinicore::op::common::OpDispatcher<DispatcherProbeFn> dispatcher;
    dispatcher.registerDevice(infinicore::Device::Type::CPU, &dispatcherDefaultProbe);

    ASSERT_EQ(dispatcher.lookup(infinicore::Device::Type::CPU, OptimizationGoal::MEMORY_SAFE)(7), 8);
}

void test_op_dispatcher_legacy_lookup() {
    infinicore::op::common::OpDispatcher<DispatcherProbeFn> dispatcher;
    dispatcher.registerDevice(infinicore::Device::Type::CPU, &dispatcherDefaultProbe);
    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &dispatcherThroughputProbe,
        OptimizationGoal::THROUGHPUT_FIRST);

    ASSERT_EQ(dispatcher.lookup(infinicore::Device::Type::CPU)(9), 10);
}

void test_op_dispatcher_goal_override_existing() {
    infinicore::op::common::OpDispatcher<DispatcherProbeFn> dispatcher;
    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &dispatcherThroughputProbe,
        OptimizationGoal::THROUGHPUT_FIRST,
        false);
    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &dispatcherLatencyProbe,
        OptimizationGoal::THROUGHPUT_FIRST,
        false);

    ASSERT_EQ(dispatcher.lookup(infinicore::Device::Type::CPU, OptimizationGoal::THROUGHPUT_FIRST)(3), 103);

    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &dispatcherLatencyProbe,
        OptimizationGoal::THROUGHPUT_FIRST,
        true);

    ASSERT_EQ(dispatcher.lookup(infinicore::Device::Type::CPU, OptimizationGoal::THROUGHPUT_FIRST)(3), 13);
}

// ============================================================
// PhaseDetector Tests
// ============================================================

static std::vector<OpTraceEntry> makeWindow(
    const std::vector<OpType> &types,
    uint32_t seq_len = 128) {

    std::vector<OpTraceEntry> window;
    for (auto t : types) {
        OpTraceEntry e;
        e.op_type = t;
        e.ndim = 3;
        e.shape[0] = 1;   // batch
        e.shape[1] = 32;  // heads
        e.shape[2] = seq_len;  // seq_len
        window.push_back(e);
    }
    return window;
}

void test_phase_detector_attention() {
    PhaseDetector detector;
    auto window = makeWindow({
        OpType::ATTENTION, OpType::CAUSAL_SOFTMAX, OpType::FLASH_ATTENTION,
        OpType::ATTENTION, OpType::RMS_NORM, OpType::ADD
    }, /*seq_len=*/16);

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::ATTENTION_DENSE));
}

void test_phase_detector_gemm_mlp() {
    PhaseDetector detector;
    auto window = makeWindow({
        OpType::LINEAR, OpType::SILU, OpType::LINEAR,
        OpType::GEMM, OpType::GELU, OpType::LINEAR
    });

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::GEMM_MLP_DENSE));
}

void test_phase_detector_kv_cache() {
    PhaseDetector detector;
    auto window = makeWindow({
        OpType::KV_CACHING, OpType::KV_CACHING, OpType::PAGED_CACHING,
        OpType::KV_CACHING, OpType::ADD
    });

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::KV_CACHE));
}

void test_phase_detector_communication() {
    PhaseDetector detector;
    auto window = makeWindow({
        OpType::ALLREDUCE, OpType::ALLREDUCE, OpType::ALLREDUCE,
        OpType::GEMM, OpType::ADD
    });

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::COMMUNICATION));
}

void test_phase_detector_decode() {
    PhaseDetector detector;
    // seq_len=1 → decode
    auto window = makeWindow({
        OpType::ATTENTION, OpType::CAUSAL_SOFTMAX, OpType::LINEAR,
        OpType::SILU, OpType::GEMM, OpType::ADD
    }, /*seq_len=*/1);

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::DECODE));
}

void test_phase_detector_prefill() {
    PhaseDetector detector;
    // seq_len=512 → prefill
    auto window = makeWindow({
        OpType::ATTENTION, OpType::FLASH_ATTENTION, OpType::CAUSAL_SOFTMAX,
        OpType::ATTENTION, OpType::RMS_NORM, OpType::ADD
    }, /*seq_len=*/512);

    auto phase = detector.detect(window);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::PREFILL));
}

void test_phase_detector_empty() {
    PhaseDetector detector;
    std::vector<OpTraceEntry> empty;
    auto phase = detector.detect(empty);
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::UNKNOWN));
}

// ============================================================
// ResourceSensor Tests
// ============================================================

void test_resource_sensor_high_memory() {
    ResourceSensor sensor;
    MemoryStats stats;
    stats.allocated_bytes = 900;
    stats.total_capacity = 1000;

    auto intent = sensor.sense(0, stats);
    ASSERT_EQ(intent.device_id, 0);
    ASSERT_GT(intent.memory_usage_ratio, 0.85f);
    ASSERT_EQ(static_cast<int>(intent.local_bottleneck),
              static_cast<int>(BottleneckType::MEMORY_BOUND));
}

void test_resource_sensor_low_memory() {
    ResourceSensor sensor;
    MemoryStats stats;
    stats.allocated_bytes = 100;
    stats.total_capacity = 1000;

    auto intent = sensor.sense(0, stats);
    ASSERT_EQ(static_cast<int>(intent.local_bottleneck),
              static_cast<int>(BottleneckType::COMPUTE_BOUND));
}

void test_resource_sensor_moderate_memory() {
    ResourceSensor sensor;
    MemoryStats stats;
    stats.allocated_bytes = 600;
    stats.total_capacity = 1000;

    auto intent = sensor.sense(0, stats);
    ASSERT_EQ(static_cast<int>(intent.local_bottleneck),
              static_cast<int>(BottleneckType::BALANCED));
}

void test_resource_sensor_bandwidth_snapshot() {
    ResourceSensor sensor;
    DeviceResourceSnapshot snapshot;
    snapshot.device_id = 0;
    snapshot.device_type = infinicore::Device::Type::NVIDIA;
    snapshot.has_memory_capacity = true;
    snapshot.total_bytes = 1000;
    snapshot.used_bytes = 300;
    snapshot.free_bytes = 700;
    snapshot.has_compute_utilization = true;
    snapshot.compute_utilization = 0.35f;
    snapshot.has_memory_bandwidth_utilization = true;
    snapshot.memory_bandwidth_utilization = 0.9f;

    auto intent = sensor.sense(snapshot);
    ASSERT_EQ(static_cast<int>(intent.local_bottleneck),
              static_cast<int>(BottleneckType::BANDWIDTH_BOUND));
    ASSERT_GT(intent.resource_confidence, 0.5f);
}

void test_resource_sensor_communication_snapshot() {
    ResourceSensor sensor;
    DeviceResourceSnapshot snapshot;
    snapshot.device_id = 1;
    snapshot.device_type = infinicore::Device::Type::NVIDIA;
    snapshot.has_memory_capacity = true;
    snapshot.total_bytes = 1000;
    snapshot.used_bytes = 200;
    snapshot.free_bytes = 800;
    snapshot.has_communication = true;
    snapshot.communication_time_ratio = 0.4f;
    snapshot.communication_bytes = 1024;

    auto intent = sensor.sense(snapshot);
    ASSERT_EQ(static_cast<int>(intent.local_bottleneck),
              static_cast<int>(BottleneckType::COMMUNICATION_BOUND));
}

// ============================================================
// IntentGenerator Tests
// ============================================================

void test_intent_generator_attention_phase() {
    IntentGenerator gen;
    auto window = makeWindow({
        OpType::ATTENTION, OpType::FLASH_ATTENTION
    });

    DeviceLocalIntent dev;
    dev.device_id = 0;
    dev.memory_usage_ratio = 0.3f;
    dev.local_bottleneck = BottleneckType::COMPUTE_BOUND;

    auto intent = gen.generate(PhaseType::ATTENTION_DENSE, window, {dev});

    ASSERT_EQ(static_cast<int>(intent.global.current_phase),
              static_cast<int>(PhaseType::ATTENTION_DENSE));
    ASSERT_EQ(static_cast<int>(intent.global.primary_bottleneck),
              static_cast<int>(BottleneckType::BANDWIDTH_BOUND));
    ASSERT_GT(intent.global.confidence, 0.0f);
    ASSERT_EQ(intent.per_device.size(), 1u);
    ASSERT_EQ(intent.per_device[0].device_id, 0);
}

void test_intent_generator_memory_pressure() {
    IntentGenerator gen;
    auto window = makeWindow({OpType::GEMM, OpType::LINEAR});

    DeviceLocalIntent dev;
    dev.device_id = 0;
    dev.memory_usage_ratio = 0.95f;
    dev.local_bottleneck = BottleneckType::MEMORY_BOUND;

    auto intent = gen.generate(PhaseType::GEMM_MLP_DENSE, window, {dev});

    // Under memory pressure, bottleneck should be MEMORY_BOUND
    ASSERT_EQ(static_cast<int>(intent.global.primary_bottleneck),
              static_cast<int>(BottleneckType::MEMORY_BOUND));
    // Goal should be MEMORY_SAFE
    ASSERT_EQ(static_cast<int>(intent.global.goal),
              static_cast<int>(OptimizationGoal::MEMORY_SAFE));
    // Should suggest in-place and recomputation
    ASSERT_TRUE(intent.global.strategy.prefer_in_place);
    ASSERT_TRUE(intent.global.strategy.prefer_recomputation);
}

void test_intent_generator_decode_latency() {
    IntentGenerator gen;
    auto window = makeWindow({OpType::ATTENTION}, /*seq_len=*/1);

    DeviceLocalIntent dev;
    dev.device_id = 0;
    dev.memory_usage_ratio = 0.3f;
    dev.local_bottleneck = BottleneckType::COMPUTE_BOUND;

    auto intent = gen.generate(PhaseType::DECODE, window, {dev});

    ASSERT_EQ(static_cast<int>(intent.global.goal),
              static_cast<int>(OptimizationGoal::LATENCY_FIRST));
    ASSERT_TRUE(intent.global.strategy.prefer_fused_ops);
}

void test_intent_generator_multi_device() {
    IntentGenerator gen;
    auto window = makeWindow({OpType::LINEAR, OpType::GEMM});

    DeviceLocalIntent dev0, dev1;
    dev0.device_id = 0;
    dev0.memory_usage_ratio = 0.3f;
    dev0.local_bottleneck = BottleneckType::COMPUTE_BOUND;
    dev1.device_id = 1;
    dev1.memory_usage_ratio = 0.9f;
    dev1.local_bottleneck = BottleneckType::MEMORY_BOUND;

    auto intent = gen.generate(PhaseType::GEMM_MLP_DENSE, window, {dev0, dev1});

    ASSERT_EQ(intent.per_device.size(), 2u);
    // Device 1 has high memory → global should detect memory bound
    ASSERT_EQ(static_cast<int>(intent.global.primary_bottleneck),
              static_cast<int>(BottleneckType::MEMORY_BOUND));
    // Should be able to look up per-device intent
    auto *d0 = intent.getDeviceIntent(0);
    auto *d1 = intent.getDeviceIntent(1);
    ASSERT_TRUE(d0 != nullptr);
    ASSERT_TRUE(d1 != nullptr);
    ASSERT_GT(d1->memory_usage_ratio, d0->memory_usage_ratio);
}

void test_intent_generator_communication_phase() {
    IntentGenerator gen;
    auto window = makeWindow({OpType::ALLREDUCE, OpType::ALLREDUCE});

    DeviceLocalIntent dev;
    dev.device_id = 0;
    dev.local_bottleneck = BottleneckType::COMMUNICATION_BOUND;
    dev.communication_time_ratio = 0.35f;

    auto intent = gen.generate(PhaseType::COMMUNICATION, window, {dev});

    ASSERT_EQ(static_cast<int>(intent.global.primary_bottleneck),
              static_cast<int>(BottleneckType::COMMUNICATION_BOUND));
    ASSERT_EQ(static_cast<int>(intent.global.goal),
              static_cast<int>(OptimizationGoal::STABILITY_FIRST));
}

// ============================================================
// OptimizationIntent Tests
// ============================================================

void test_optimization_intent_device_lookup() {
    OptimizationIntent intent;
    DeviceLocalIntent device0;
    device0.device_id = 0;
    device0.memory_usage_ratio = 0.3f;
    device0.memory_available_bytes = 700;
    device0.local_bottleneck = BottleneckType::COMPUTE_BOUND;

    DeviceLocalIntent device1;
    device1.device_id = 1;
    device1.memory_usage_ratio = 0.8f;
    device1.memory_available_bytes = 200;
    device1.local_bottleneck = BottleneckType::MEMORY_BOUND;

    intent.per_device.push_back(device0);
    intent.per_device.push_back(device1);

    ASSERT_TRUE(intent.getDeviceIntent(0) != nullptr);
    ASSERT_TRUE(intent.getDeviceIntent(1) != nullptr);
    ASSERT_TRUE(intent.getDeviceIntent(99) == nullptr);
}

void test_mutual_awareness_analyzer_phase_from_trace() {
    auto &trace = getGlobalOpTrace();
    trace.clear();

    size_t dims[] = {1, 32, 1};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    auto phase = MutualAwarenessAnalyzer::instance().getCurrentPhase();
    ASSERT_EQ(static_cast<int>(phase), static_cast<int>(PhaseType::DECODE));

    trace.clear();
    MutualAwarenessAnalyzer::instance().clearGraphCache();
}

void test_mutual_awareness_analyzer_goal_decode() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);

    size_t dims[] = {1, 32, 1};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    ASSERT_EQ(static_cast<int>(analyzer.getCurrentOptimizationGoal()),
              static_cast<int>(OptimizationGoal::LATENCY_FIRST));

    trace.clear();
    analyzer.clearGraphCache();
}

void test_mutual_awareness_analyzer_goal_prefill() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    ASSERT_EQ(static_cast<int>(analyzer.getCurrentOptimizationGoal()),
              static_cast<int>(OptimizationGoal::THROUGHPUT_FIRST));

    trace.clear();
    analyzer.clearGraphCache();
}

void test_mutual_awareness_analyzer_goal_memory_safe_with_stats() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    MemoryStats stats{950 * 1024 * 1024, 1024 * 1024 * 1024UL, 74 * 1024 * 1024, 10};
    auto intent = analyzer.analyze({{0, stats}});

    ASSERT_EQ(static_cast<int>(intent.global.goal),
              static_cast<int>(OptimizationGoal::MEMORY_SAFE));
    ASSERT_EQ(static_cast<int>(analyzer.lastIntent().global.goal),
              static_cast<int>(OptimizationGoal::MEMORY_SAFE));

    trace.clear();
    analyzer.clearGraphCache();
}

void test_mutual_awareness_analyzer_with_resource_snapshot() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::NVIDIA),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::NVIDIA),
            0);

    DeviceResourceSnapshot snapshot;
    snapshot.device_id = 0;
    snapshot.device_type = infinicore::Device::Type::NVIDIA;
    snapshot.has_memory_capacity = true;
    snapshot.total_bytes = 1000;
    snapshot.used_bytes = 300;
    snapshot.free_bytes = 700;
    snapshot.reserved_bytes = 800;
    snapshot.has_compute_utilization = true;
    snapshot.compute_utilization = 0.3f;
    snapshot.has_memory_bandwidth_utilization = true;
    snapshot.memory_bandwidth_utilization = 0.92f;
    snapshot.has_kernel_time_ratio = true;
    snapshot.kernel_time_ratio = 0.3f;

    auto intent = analyzer.analyze({snapshot});

    ASSERT_EQ(static_cast<int>(intent.global.current_phase),
              static_cast<int>(PhaseType::PREFILL));
    ASSERT_EQ(static_cast<int>(intent.global.primary_bottleneck),
              static_cast<int>(BottleneckType::BANDWIDTH_BOUND));
    ASSERT_EQ(static_cast<int>(intent.global.goal),
              static_cast<int>(OptimizationGoal::THROUGHPUT_FIRST));
    ASSERT_EQ(intent.per_device.size(), 1u);
    ASSERT_EQ(static_cast<int>(intent.per_device[0].local_bottleneck),
              static_cast<int>(BottleneckType::BANDWIDTH_BOUND));

    trace.clear();
    analyzer.clearGraphCache();
}

void test_mutual_awareness_analyzer_auto_collect_runtime_snapshots() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    auto intent = analyzer.analyze();

    int counts[INFINI_DEVICE_TYPE_COUNT] = {0};
    bool have_accelerator = false;
    if (infinirtGetAllDeviceCount(counts) == INFINI_STATUS_SUCCESS) {
        for (int dt = 0; dt < INFINI_DEVICE_TYPE_COUNT; ++dt) {
            if (dt != INFINI_DEVICE_CPU && counts[dt] > 0) {
                have_accelerator = true;
                break;
            }
        }
    }

    if (have_accelerator) {
        ASSERT_TRUE(!intent.per_device.empty());
        bool saw_resource_signal = false;
        for (auto const &dev : intent.per_device) {
            if (dev.resource_confidence > 0.0f) {
                saw_resource_signal = true;
                break;
            }
        }
        ASSERT_TRUE(saw_resource_signal);
    } else {
        ASSERT_TRUE(intent.per_device.empty());
        ASSERT_EQ(static_cast<int>(intent.global.goal),
                  static_cast<int>(OptimizationGoal::THROUGHPUT_FIRST));
    }

    bool valid_memory_ratios = true;
    for (auto const &dev : intent.per_device) {
        if (dev.memory_usage_ratio < 0.0f || dev.memory_usage_ratio > 1.0f) {
            valid_memory_ratios = false;
            break;
        }
    }
    ASSERT_TRUE(valid_memory_ratios);

    trace.clear();
    analyzer.clearGraphCache();
}

void test_attention_execute_consumes_goal_dispatch() {
    auto &analyzer = MutualAwarenessAnalyzer::instance();
    auto &trace = getGlobalOpTrace();
    trace.clear();
    analyzer.clearGraphCache();
    analyzer.setEnabled(true);
    resetAttentionDispatchCounters();

    auto &dispatcher = infinicore::op::Attention::dispatcher();
    auto default_cpu = dispatcher.lookup(infinicore::Device::Type::CPU);

    struct DispatcherRestoreGuard {
        infinicore::op::common::OpDispatcher<infinicore::op::Attention::schema> &dispatcher;
        infinicore::op::Attention::schema default_cpu;

        ~DispatcherRestoreGuard() {
            dispatcher.registerDevice(infinicore::Device::Type::CPU, default_cpu);
            dispatcher.registerDevice(
                infinicore::Device::Type::CPU,
                nullptr,
                OptimizationGoal::THROUGHPUT_FIRST);
            resetAttentionDispatchCounters();
            getGlobalOpTrace().clear();
            MutualAwarenessAnalyzer::instance().clearGraphCache();
        }
    } guard{dispatcher, default_cpu};

    dispatcher.registerDevice(infinicore::Device::Type::CPU, &attentionDefaultStub);
    dispatcher.registerDevice(
        infinicore::Device::Type::CPU,
        &attentionThroughputStub,
        OptimizationGoal::THROUGHPUT_FIRST);

    size_t dims[] = {1, 32, 512};
    traceOp(OpType::ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);
    traceOp(OpType::FLASH_ATTENTION,
            dims,
            3,
            static_cast<uint8_t>(infinicore::DataType::F16),
            static_cast<uint8_t>(infinicore::Device::Type::CPU),
            0);

    ASSERT_EQ(static_cast<int>(analyzer.getCurrentOptimizationGoal()),
              static_cast<int>(OptimizationGoal::THROUGHPUT_FIRST));

    infinicore::Shape shape = {1, 32, 512};
    std::vector<uint16_t> out_storage(1 * 32 * 512, 0);
    std::vector<uint16_t> q_storage(1 * 32 * 512, 0);
    std::vector<uint16_t> k_storage(1 * 32 * 512, 0);
    std::vector<uint16_t> v_storage(1 * 32 * 512, 0);
    std::vector<uint16_t> k_cache_storage(1 * 32 * 512, 0);
    std::vector<uint16_t> v_cache_storage(1 * 32 * 512, 0);

    auto out = makeCpuBlobTensor(out_storage, shape);
    auto q = makeCpuBlobTensor(q_storage, shape);
    auto k = makeCpuBlobTensor(k_storage, shape);
    auto v = makeCpuBlobTensor(v_storage, shape);
    auto k_cache = makeCpuBlobTensor(k_cache_storage, shape);
    auto v_cache = makeCpuBlobTensor(v_cache_storage, shape);

    infinicore::op::Attention::execute(out, q, k, v, k_cache, v_cache, 0);

    ASSERT_EQ(attention_default_hits, 0);
    ASSERT_EQ(attention_throughput_hits, 1);
}

// ============================================================
// End-to-end Scenario Test
// ============================================================

void test_e2e_llm_inference_sequence() {
    // Simulate a typical LLM transformer layer:
    // attention → add+norm → linear → silu → linear → add+norm
    OpTraceRing ring(64);

    auto trace = [&](OpType t, uint32_t seq_len = 128) {
        OpTraceEntry e;
        e.op_type = t;
        e.ndim = 3;
        e.shape[0] = 1; e.shape[1] = 32; e.shape[2] = seq_len;
        e.timestamp_ns = OpTraceEntry::now();
        ring.write(e);
    };

    // Simulate 2 transformer layers
    for (int layer = 0; layer < 2; layer++) {
        trace(OpType::RMS_NORM);
        trace(OpType::ATTENTION);
        trace(OpType::FLASH_ATTENTION);
        trace(OpType::ADD);
        trace(OpType::RMS_NORM);
        trace(OpType::LINEAR);
        trace(OpType::SILU_AND_MUL);
        trace(OpType::LINEAR);
        trace(OpType::ADD);
    }

    PhaseDetector detector;
    auto window = ring.getRecentEntries(detector.config().window_size);
    auto phase = detector.detect(window);

    // The last ops are MLP-dominated → should detect GEMM_MLP or at least not UNKNOWN
    ASSERT_TRUE(phase != PhaseType::UNKNOWN);

    // Generate intent with moderate memory
    ResourceSensor sensor;
    MemoryStats stats{500 * 1024 * 1024, 1024 * 1024 * 1024UL, 600 * 1024 * 1024, 42};
    auto dev_intent = sensor.sense(0, stats);

    IntentGenerator gen;
    auto intent = gen.generate(phase, window, {dev_intent});

    ASSERT_TRUE(intent.global.confidence > 0.0f);
    ASSERT_TRUE(intent.global.timestamp_ns > 0);
    ASSERT_EQ(intent.per_device.size(), 1u);
}

// ============================================================
// Performance Test
// ============================================================

void test_op_trace_performance() {
    OpTraceRing ring(256);

    auto start = std::chrono::high_resolution_clock::now();
    constexpr int N = 100000;

    for (int i = 0; i < N; i++) {
        OpTraceEntry entry;
        entry.op_type = OpType::GEMM;
        entry.timestamp_ns = OpTraceEntry::now();
        ring.write(entry);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double per_op_ns = static_cast<double>(ns) / N;

    std::cout << " (" << per_op_ns << " ns/op) " << std::flush;

    // Should be under 500ns per op (generous limit; typically ~50-100ns)
    ASSERT_TRUE(per_op_ns < 500.0);
}

void test_analysis_performance() {
    OpTraceRing ring(256);

    // Fill ring
    for (int i = 0; i < 256; i++) {
        OpTraceEntry entry;
        entry.op_type = (i % 3 == 0) ? OpType::ATTENTION : OpType::LINEAR;
        entry.ndim = 3;
        entry.shape[0] = 1; entry.shape[1] = 32; entry.shape[2] = 128;
        ring.write(entry);
    }

    PhaseDetector detector;
    ResourceSensor sensor;
    IntentGenerator gen;

    auto start = std::chrono::high_resolution_clock::now();
    constexpr int N = 10000;

    for (int i = 0; i < N; i++) {
        auto window = ring.getRecentEntries(16);
        auto phase = detector.detect(window);
        MemoryStats stats{500'000'000, 1'000'000'000, 600'000'000, 42};
        auto dev = sensor.sense(0, stats);
        auto intent = gen.generate(phase, window, {dev});
        (void)intent;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double per_analysis_us = static_cast<double>(us) / N;

    std::cout << " (" << per_analysis_us << " μs/analysis) " << std::flush;

    // Full analysis pipeline should be under 100μs (far below 10s constraint)
    ASSERT_TRUE(per_analysis_us < 100.0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "Mutual Awareness Analyzer — Unit Tests" << std::endl;
    std::cout << "==============================================" << std::endl;

    // OpTraceRing tests
    RUN_TEST(op_trace_ring_basic);
    RUN_TEST(op_trace_ring_wrap);
    RUN_TEST(op_trace_ring_recent);
    RUN_TEST(op_trace_ring_clear);
    RUN_TEST(op_trace_entry_shape);
    RUN_TEST(op_trace_global_singleton);
    RUN_TEST(op_trace_records_metadata);
    RUN_TEST(op_trace_runtime_switch);

    // OpType tests
    RUN_TEST(op_type_classification);
    RUN_TEST(op_type_to_string);
    RUN_TEST(graph_op_type_mapping);

    // OpDispatcher tests
    RUN_TEST(op_dispatcher_goal_exact_match);
    RUN_TEST(op_dispatcher_goal_fallback);
    RUN_TEST(op_dispatcher_legacy_lookup);
    RUN_TEST(op_dispatcher_goal_override_existing);

    // PhaseDetector tests
    RUN_TEST(phase_detector_attention);
    RUN_TEST(phase_detector_gemm_mlp);
    RUN_TEST(phase_detector_kv_cache);
    RUN_TEST(phase_detector_communication);
    RUN_TEST(phase_detector_decode);
    RUN_TEST(phase_detector_prefill);
    RUN_TEST(phase_detector_empty);

    // ResourceSensor tests
    RUN_TEST(resource_sensor_high_memory);
    RUN_TEST(resource_sensor_low_memory);
    RUN_TEST(resource_sensor_moderate_memory);
    RUN_TEST(resource_sensor_bandwidth_snapshot);
    RUN_TEST(resource_sensor_communication_snapshot);

    // IntentGenerator tests
    RUN_TEST(intent_generator_attention_phase);
    RUN_TEST(intent_generator_memory_pressure);
    RUN_TEST(intent_generator_decode_latency);
    RUN_TEST(intent_generator_multi_device);
    RUN_TEST(intent_generator_communication_phase);

    // OptimizationIntent tests
    RUN_TEST(optimization_intent_device_lookup);
    RUN_TEST(mutual_awareness_analyzer_phase_from_trace);
    RUN_TEST(mutual_awareness_analyzer_goal_decode);
    RUN_TEST(mutual_awareness_analyzer_goal_prefill);
    RUN_TEST(mutual_awareness_analyzer_goal_memory_safe_with_stats);
    RUN_TEST(mutual_awareness_analyzer_with_resource_snapshot);
    RUN_TEST(mutual_awareness_analyzer_auto_collect_runtime_snapshots);
    RUN_TEST(attention_execute_consumes_goal_dispatch);

    // End-to-end tests
    RUN_TEST(e2e_llm_inference_sequence);

    // Performance tests
    RUN_TEST(op_trace_performance);
    RUN_TEST(analysis_performance);

    // Summary
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "==============================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
