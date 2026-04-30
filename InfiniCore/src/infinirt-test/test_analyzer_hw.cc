// ============================================================
// Hardware-level analyzer tests for Iluvatar, MetaX, and NVIDIA backends
//
// Verifies the infinirt layer: getMemInfo, getDeviceResourceSnapshot,
// event timing, and management library (NVML/IXML/MXSML) loading.
//
// Build & run:
//   xmake build infinirt-test-analyzer-hw
//   xmake run infinirt-test-analyzer-hw
// ============================================================

#include <infinirt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN(name) \
    do { \
        printf("[TEST] %-50s ", #name); \
        fflush(stdout); \
        if (test_##name()) { \
            printf("PASSED\n"); \
            tests_passed++; \
        } else { \
            printf("FAILED\n"); \
            tests_failed++; \
        } \
    } while (0)

// ============================================================
// Detect device type and count at runtime
// ============================================================

static infiniDevice_t g_device_type = INFINI_DEVICE_CPU;
static int g_device_count = 0;

static void detectDevice() {
    infiniDevice_t try_types[] = {INFINI_DEVICE_ILUVATAR, INFINI_DEVICE_METAX, INFINI_DEVICE_NVIDIA};
    for (auto dt : try_types) {
        int count = 0;
        if (infinirtGetDeviceCount(dt, &count) == INFINI_STATUS_SUCCESS && count > 0) {
            g_device_type = dt;
            g_device_count = count;
            return;
        }
    }
}

static const char *deviceTypeName(infiniDevice_t dt) {
    switch (dt) {
    case INFINI_DEVICE_NVIDIA: return "NVIDIA";
    case INFINI_DEVICE_ILUVATAR: return "Iluvatar";
    case INFINI_DEVICE_METAX: return "MetaX";
    default: return "Unknown";
    }
}

// ============================================================
// Test 1: getMemInfo
// ============================================================

static bool test_getMemInfo() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    size_t free_bytes = 0, total_bytes = 0;
    auto status = infinirtGetMemInfo(g_device_type, 0, &free_bytes, &total_bytes);
    if (status != INFINI_STATUS_SUCCESS) {
        printf("(infinirtGetMemInfo returned %d) ", status);
        return false;
    }
    if (total_bytes == 0) { printf("(total_bytes == 0) "); return false; }
    if (free_bytes > total_bytes) { printf("(free > total) "); return false; }

    printf("(free=%.1f GiB, total=%.1f GiB) ",
           free_bytes / (1024.0 * 1024.0 * 1024.0),
           total_bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

// ============================================================
// Test 2: getDeviceResourceSnapshot — memory fields
// ============================================================

static bool test_snapshot_memory() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    infinirtDeviceResourceSnapshot_t snap{};
    auto status = infinirtGetDeviceResourceSnapshot(g_device_type, 0, &snap);
    if (status != INFINI_STATUS_SUCCESS) {
        printf("(snapshot returned %d) ", status);
        return false;
    }
    if (!(snap.valid_fields & INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY)) {
        printf("(MEMORY_CAPACITY flag not set) ");
        return false;
    }
    if (snap.total_bytes == 0) { printf("(total_bytes == 0) "); return false; }

    printf("(used=%.1f MiB, free=%.1f MiB, total=%.1f MiB) ",
           snap.used_bytes / (1024.0 * 1024.0),
           snap.free_bytes / (1024.0 * 1024.0),
           snap.total_bytes / (1024.0 * 1024.0));
    return true;
}

// ============================================================
// Test 3: getDeviceResourceSnapshot — real device/product name
// ============================================================

static bool test_snapshot_device_name() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    infinirtDeviceResourceSnapshot_t snap{};
    auto status = infinirtGetDeviceResourceSnapshot(g_device_type, 0, &snap);
    if (status != INFINI_STATUS_SUCCESS) {
        printf("(snapshot returned %d) ", status);
        return false;
    }

    if (!(snap.valid_fields & INFINIRT_RESOURCE_FIELD_DEVICE_NAME)) {
        printf("(DEVICE_NAME flag not set) ");
        return false;
    }
    if (snap.device_name[0] == '\0') {
        printf("(device_name empty) ");
        return false;
    }
    if (std::strlen(snap.device_name) >= INFINIRT_DEVICE_NAME_MAX) {
        printf("(device_name not bounded) ");
        return false;
    }

    printf("(name=%s) ", snap.device_name);
    return true;
}

// ============================================================
// Test 4: getDeviceResourceSnapshot — utilization (NVML/IXML/MXSML)
// ============================================================

static bool test_snapshot_utilization() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    infinirtDeviceResourceSnapshot_t snap{};
    infinirtGetDeviceResourceSnapshot(g_device_type, 0, &snap);

    bool has_compute = (snap.valid_fields & INFINIRT_RESOURCE_FIELD_COMPUTE_UTILIZATION) != 0;
    bool has_bw = (snap.valid_fields & INFINIRT_RESOURCE_FIELD_MEMORY_BANDWIDTH_UTILIZATION) != 0;

    if (!has_compute || !has_bw) {
        printf("(utilization unavailable — NVML/IXML/MXSML not loaded? compute=%s, bw=%s) ",
               has_compute ? "yes" : "no", has_bw ? "yes" : "no");
        return false;
    }

    if (snap.compute_utilization < 0.0f || snap.compute_utilization > 1.0f) {
        printf("(compute out of range: %f) ", snap.compute_utilization);
        return false;
    }

    printf("(compute=%.1f%%, mem_bw=%.1f%%) ",
           snap.compute_utilization * 100.0f,
           snap.memory_bandwidth_utilization * 100.0f);
    return true;
}

// ============================================================
// Test 5: getDeviceResourceSnapshot — communication baseline
// ============================================================

static bool test_snapshot_communication() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    infinirtDeviceResourceSnapshot_t snap{};
    infinirtGetDeviceResourceSnapshot(g_device_type, 0, &snap);

    if (!(snap.valid_fields & INFINIRT_RESOURCE_FIELD_COMMUNICATION)) {
        printf("(COMMUNICATION flag not set) ");
        return false;
    }

    printf("(comm_ratio=%.3f, comm_bytes=%llu) ",
           snap.communication_time_ratio,
           (unsigned long long)snap.communication_bytes);
    return true;
}

// ============================================================
// Test 6: event create / record / elapsed time
// ============================================================

static bool test_event_timing() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }

    // Set current device first
    infinirtSetDevice(g_device_type, 0);

    infinirtStream_t stream = nullptr;
    if (infinirtStreamCreate(&stream) != INFINI_STATUS_SUCCESS) {
        printf("(streamCreate failed) ");
        return false;
    }

    infinirtEvent_t start_evt = nullptr, end_evt = nullptr;
    if (infinirtEventCreate(&start_evt) != INFINI_STATUS_SUCCESS) {
        printf("(eventCreate start failed) ");
        infinirtStreamDestroy(stream);
        return false;
    }
    if (infinirtEventCreate(&end_evt) != INFINI_STATUS_SUCCESS) {
        printf("(eventCreate end failed) ");
        infinirtEventDestroy(start_evt);
        infinirtStreamDestroy(stream);
        return false;
    }

    // Record start
    infinirtEventRecord(start_evt, stream);

    // Do some GPU work: alloc + memset via D2D copy
    void *tmp = nullptr;
    infinirtMalloc(&tmp, 1024 * 1024);
    if (tmp) infinirtFree(tmp);

    // Record end
    infinirtEventRecord(end_evt, stream);
    infinirtStreamSynchronize(stream);

    float elapsed_ms = -1.0f;
    auto s = infinirtEventElapsedTime(&elapsed_ms, start_evt, end_evt);

    printf("(elapsed=%.3f ms, status=%d) ", elapsed_ms, s);

    infinirtEventDestroy(end_evt);
    infinirtEventDestroy(start_evt);
    infinirtStreamDestroy(stream);

    return s == INFINI_STATUS_SUCCESS && elapsed_ms >= 0.0f;
}

// ============================================================
// Test 7: malloc / memcpy round-trip
// ============================================================

static bool test_malloc_memcpy() {
    if (g_device_count == 0) { printf("(skip: no GPU) "); return true; }
    infinirtSetDevice(g_device_type, 0);

    const size_t N = 256;
    float host_src[N], host_dst[N];
    for (size_t i = 0; i < N; i++) host_src[i] = static_cast<float>(i);
    memset(host_dst, 0, sizeof(host_dst));

    void *dev = nullptr;
    if (infinirtMalloc(&dev, N * sizeof(float)) != INFINI_STATUS_SUCCESS) {
        printf("(malloc failed) ");
        return false;
    }

    if (infinirtMemcpy(dev, host_src, N * sizeof(float), INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
        printf("(H2D failed) ");
        infinirtFree(dev);
        return false;
    }
    if (infinirtMemcpy(host_dst, dev, N * sizeof(float), INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
        printf("(D2H failed) ");
        infinirtFree(dev);
        return false;
    }

    infinirtFree(dev);

    for (size_t i = 0; i < N; i++) {
        if (host_dst[i] != host_src[i]) {
            printf("(mismatch at %zu: %f != %f) ", i, host_dst[i], host_src[i]);
            return false;
        }
    }

    printf("(256 floats OK) ");
    return true;
}

// ============================================================
// Test 8: multi-device snapshot
// ============================================================

static bool test_multi_device_snapshot() {
    if (g_device_count < 2) {
        printf("(skip: need 2+ GPUs, have %d) ", g_device_count);
        return true;
    }

    for (int i = 0; i < g_device_count; i++) {
        infinirtSetDevice(g_device_type, i);
        infinirtDeviceResourceSnapshot_t snap{};
        auto status = infinirtGetDeviceResourceSnapshot(g_device_type, i, &snap);
        if (status != INFINI_STATUS_SUCCESS) {
            printf("(device %d failed: %d) ", i, status);
            return false;
        }
        if (snap.total_bytes == 0) {
            printf("(device %d total == 0) ", i);
            return false;
        }
        printf("[dev%d: %.1fGiB] ", i, snap.total_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return true;
}

// ============================================================
// main
// ============================================================

int main() {
    detectDevice();

    printf("========================================\n");
    printf(" Analyzer HW Tests\n");
    printf(" Device: %s x %d\n", deviceTypeName(g_device_type), g_device_count);
    printf("========================================\n\n");

    if (g_device_count > 0) {
        infinirtSetDevice(g_device_type, 0);
    }

    RUN(getMemInfo);
    RUN(snapshot_memory);
    RUN(snapshot_device_name);
    RUN(snapshot_utilization);
    RUN(snapshot_communication);
    RUN(event_timing);
    RUN(malloc_memcpy);
    RUN(multi_device_snapshot);

    printf("\n========================================\n");
    printf(" Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
