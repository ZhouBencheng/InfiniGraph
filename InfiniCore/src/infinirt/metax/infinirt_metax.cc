#include "infinirt_metax.h"
#include "../../utils.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#ifdef ENABLE_METAX_MC_API
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#else
#include <hcr/hc_runtime.h>
#include <hcr/hc_runtime_api.h>
#endif

#define CHECK_MACART(RT_API) CHECK_INTERNAL(RT_API, hcSuccess)

namespace {

struct PendingCommunicationSample {
    hcEvent_t start_event = nullptr;
    hcEvent_t end_event = nullptr;
    uint64_t bytes = 0;
};

struct CompletedCommunicationSample {
    std::chrono::steady_clock::time_point completed_at;
    double duration_ms = 0.0;
    uint64_t bytes = 0;
};

struct DeviceCommunicationState {
    std::deque<PendingCommunicationSample> pending;
    std::deque<CompletedCommunicationSample> recent;
};

struct CommunicationStatsStore {
    std::mutex mutex;
    std::unordered_map<int, DeviceCommunicationState> per_device;
};

CommunicationStatsStore &communicationStatsStore() {
    static CommunicationStatsStore store;
    return store;
}

constexpr auto kCommunicationWindow = std::chrono::seconds(1);

void destroyCommunicationSample(const PendingCommunicationSample &sample) {
    if (sample.start_event != nullptr) {
        hcEventDestroy(sample.start_event);
    }
    if (sample.end_event != nullptr) {
        hcEventDestroy(sample.end_event);
    }
}

template <typename DeviceFn>
hcError_t withDeviceGuard(int device_id, DeviceFn &&fn) {
    int previous_device = 0;
    auto status = hcGetDevice(&previous_device);
    if (status != hcSuccess) {
        return status;
    }

    if (previous_device != device_id) {
        status = hcSetDevice(device_id);
        if (status != hcSuccess) {
            return status;
        }
    }

    auto fn_status = fn();

    if (previous_device != device_id) {
        auto restore_status = hcSetDevice(previous_device);
        if (fn_status == hcSuccess && restore_status != hcSuccess) {
            fn_status = restore_status;
        }
    }
    return fn_status;
}

void pruneCommunicationWindow(DeviceCommunicationState &state, std::chrono::steady_clock::time_point now) {
    while (!state.recent.empty() && now - state.recent.front().completed_at > kCommunicationWindow) {
        state.recent.pop_front();
    }
}

void flushCompletedCommunicationSamples(int device_id, DeviceCommunicationState &state) {
    std::deque<PendingCommunicationSample> remaining;

    auto status = withDeviceGuard(device_id, [&]() {
        for (auto &sample : state.pending) {
            auto query_status = hcEventQuery(sample.end_event);
            if (query_status == hcSuccess) {
                float elapsed_ms = 0.0f;
                auto elapsed_status = hcEventElapsedTime(&elapsed_ms, sample.start_event, sample.end_event);
                if (elapsed_status == hcSuccess) {
                    state.recent.push_back({
                        std::chrono::steady_clock::now(),
                        static_cast<double>(elapsed_ms),
                        sample.bytes});
                }
                destroyCommunicationSample(sample);
            } else {
                remaining.push_back(sample);
            }
        }
        return hcSuccess;
    });

    if (status == hcSuccess) {
        state.pending.swap(remaining);
    }
}

void populateCommunicationSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &store = communicationStatsStore();
    std::lock_guard<std::mutex> lock(store.mutex);
    auto &state = store.per_device[device_id];

    flushCompletedCommunicationSamples(device_id, state);
    auto now = std::chrono::steady_clock::now();
    pruneCommunicationWindow(state, now);

    double total_comm_ms = 0.0;
    uint64_t total_comm_bytes = 0;
    for (auto const &sample : state.recent) {
        total_comm_ms += sample.duration_ms;
        total_comm_bytes += sample.bytes;
    }

    double window_ms = std::chrono::duration<double, std::milli>(kCommunicationWindow).count();
    double ratio = total_comm_ms / window_ms;
    snapshot->communication_bytes = total_comm_bytes;
    snapshot->communication_time_ratio = static_cast<float>(ratio > 1.0 ? 1.0 : ratio);
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_COMMUNICATION;
}

#if !defined(_WIN32)
using MxsmlReturn = int;
using MxsmlDevice = void *;

struct MxsmlExUtilization {
    unsigned int gpu = 0;
    unsigned int memory = 0;
};

struct MxsmlExMemory {
    unsigned long long free = 0;
    unsigned long long total = 0;
    unsigned long long used = 0;
};

constexpr MxsmlReturn MXSML_SUCCESS = 0;

using MxsmlExInitFn = MxsmlReturn (*)();
using MxsmlExShutdownFn = MxsmlReturn (*)();
using MxsmlExGetDeviceHandleByIndexFn = MxsmlReturn (*)(unsigned int, MxsmlDevice *);
using MxsmlExDeviceGetUtilizationFn = MxsmlReturn (*)(MxsmlDevice, MxsmlExUtilization *);
using MxsmlExDeviceGetMemoryInfoFn = MxsmlReturn (*)(MxsmlDevice, MxsmlExMemory *);
using MxsmlExDeviceGetNameFn = MxsmlReturn (*)(MxsmlDevice, char *, unsigned int);

struct MxsmlApi {
    void *handle = nullptr;
    MxsmlExInitFn init = nullptr;
    MxsmlExShutdownFn shutdown = nullptr;
    MxsmlExGetDeviceHandleByIndexFn get_handle_by_index = nullptr;
    MxsmlExDeviceGetUtilizationFn get_utilization = nullptr;
    MxsmlExDeviceGetMemoryInfoFn get_memory_info = nullptr;
    MxsmlExDeviceGetNameFn get_name = nullptr;
    bool available = false;
    bool initialized = false;
};

MxsmlApi &mxsmlApi() {
    static MxsmlApi api = []() {
        MxsmlApi loaded;
        const char *candidates[] = {
            "libmxsml.so",
            "/opt/mxdriver/lib/libmxsml.so",
            "/opt/maca/lib/libmxsml.so",
            "/opt/mxn100/lib/libmxsml.so",
        };

        for (auto candidate : candidates) {
            loaded.handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
            if (loaded.handle != nullptr) {
                break;
            }
        }

        if (loaded.handle == nullptr) {
            return loaded;
        }

        loaded.init = reinterpret_cast<MxsmlExInitFn>(dlsym(loaded.handle, "mxSmlExInit"));
        loaded.shutdown = reinterpret_cast<MxsmlExShutdownFn>(dlsym(loaded.handle, "mxSmlExShutdown"));
        loaded.get_handle_by_index = reinterpret_cast<MxsmlExGetDeviceHandleByIndexFn>(dlsym(loaded.handle, "mxSmlExGetDeviceHandleByIndex"));
        loaded.get_utilization = reinterpret_cast<MxsmlExDeviceGetUtilizationFn>(dlsym(loaded.handle, "mxSmlExDeviceGetUtilization"));
        loaded.get_memory_info = reinterpret_cast<MxsmlExDeviceGetMemoryInfoFn>(dlsym(loaded.handle, "mxSmlExDeviceGetMemoryInfo"));
        loaded.get_name = reinterpret_cast<MxsmlExDeviceGetNameFn>(dlsym(loaded.handle, "mxSmlExDeviceGetName"));

        loaded.available = loaded.init != nullptr
                           && loaded.get_handle_by_index != nullptr;
        return loaded;
    }();
    return api;
}

bool ensureMxsmlInitialized(MxsmlApi &api) {
    if (!api.available) {
        return false;
    }

    if (!api.initialized) {
        if (api.init() != MXSML_SUCCESS) {
            return false;
        }
        api.initialized = true;
    }
    return true;
}

bool getMxsmlDevice(int device_id, MxsmlDevice *device) {
    auto &api = mxsmlApi();
    if (device == nullptr || !ensureMxsmlInitialized(api)) {
        return false;
    }
    return api.get_handle_by_index(static_cast<unsigned int>(device_id), device) == MXSML_SUCCESS;
}

bool tryPopulateMxsmlDeviceName(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &api = mxsmlApi();
    if (api.get_name == nullptr) {
        return false;
    }

    MxsmlDevice device = nullptr;
    if (!getMxsmlDevice(device_id, &device)) {
        return false;
    }

    char name[INFINIRT_DEVICE_NAME_MAX] = {};
    if (api.get_name(device, name, static_cast<unsigned int>(sizeof(name))) != MXSML_SUCCESS || name[0] == '\0') {
        return false;
    }

    std::snprintf(snapshot->device_name, sizeof(snapshot->device_name), "%s", name);
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_DEVICE_NAME;
    return true;
}

bool tryPopulateMxsmlMemory(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &api = mxsmlApi();
    if (api.get_memory_info == nullptr) {
        return false;
    }

    MxsmlDevice device = nullptr;
    if (!getMxsmlDevice(device_id, &device)) {
        return false;
    }

    MxsmlExMemory memory{};
    if (api.get_memory_info(device, &memory) != MXSML_SUCCESS || memory.total == 0) {
        return false;
    }

    snapshot->free_bytes = static_cast<size_t>(memory.free);
    snapshot->total_bytes = static_cast<size_t>(memory.total);
    snapshot->used_bytes = static_cast<size_t>(memory.used);
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY;
    return true;
}

bool tryPopulateMxsmlUtilization(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &api = mxsmlApi();
    if (api.get_utilization == nullptr) {
        return false;
    }

    MxsmlDevice device = nullptr;
    if (!getMxsmlDevice(device_id, &device)) {
        return false;
    }

    MxsmlExUtilization util{};
    if (api.get_utilization(device, &util) != MXSML_SUCCESS) {
        return false;
    }

    snapshot->compute_utilization = static_cast<float>(util.gpu) / 100.0f;
    snapshot->memory_bandwidth_utilization = static_cast<float>(util.memory) / 100.0f;
    snapshot->kernel_time_ratio = snapshot->compute_utilization;
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_COMPUTE_UTILIZATION
                              | INFINIRT_RESOURCE_FIELD_MEMORY_BANDWIDTH_UTILIZATION
                              | INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO;
    snapshot->estimated_fields |= INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO;
    return true;
}
#else
bool tryPopulateMxsmlMemory(int, infinirtDeviceResourceSnapshot_t *) {
    return false;
}

bool tryPopulateMxsmlUtilization(int, infinirtDeviceResourceSnapshot_t *) {
    return false;
}

bool tryPopulateMxsmlDeviceName(int, infinirtDeviceResourceSnapshot_t *) {
    return false;
}
#endif

} // namespace

namespace infinirt::metax {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_MACART(hcGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_MACART(hcSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_MACART(hcDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    hcStream_t stream;
    CHECK_MACART(hcStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_MACART(hcStreamDestroy((hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_MACART(hcStreamSynchronize((hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_MACART(hcStreamWaitEvent((hcStream_t)stream, (hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    hcEvent_t event;
    CHECK_MACART(hcEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    hcEvent_t event;
    unsigned int hc_flags = hcEventDefault;

    // 映射 InfiniCore 的 flags 到 HC Runtime flags
    if (flags & INFINIRT_EVENT_DISABLE_TIMING) {
        hc_flags |= hcEventDisableTiming;
    }
    if (flags & INFINIRT_EVENT_BLOCKING_SYNC) {
        hc_flags |= hcEventBlockingSync;
    }

    CHECK_MACART(hcEventCreateWithFlags(&event, hc_flags));

    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_MACART(hcEventRecord((hcEvent_t)event, (hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    if (status_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = hcEventQuery((hcEvent_t)event);
    *status_ptr = status == hcSuccess
        ? INFINIRT_EVENT_COMPLETE
        : INFINIRT_EVENT_NOT_READY;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_MACART(hcEventSynchronize((hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_MACART(hcEventDestroy((hcEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    CHECK_MACART(hcEventElapsedTime(ms_ptr, (hcEvent_t)start, (hcEvent_t)end));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getMemInfo(int device_id, size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    int previous_device = 0;
    CHECK_MACART(hcGetDevice(&previous_device));

    if (previous_device != device_id) {
        CHECK_MACART(hcSetDevice(device_id));
    }

    auto query_status = hcMemGetInfo(free_bytes, total_bytes);

    if (previous_device != device_id) {
        auto restore_status = hcSetDevice(previous_device);
        if (query_status != hcSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        if (restore_status != hcSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    } else if (query_status != hcSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getDeviceResourceSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    if (snapshot == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    std::memset(snapshot, 0, sizeof(*snapshot));
    snapshot->device_id = device_id;
    snapshot->device_type = INFINI_DEVICE_METAX;

    auto status = getMemInfo(device_id, &snapshot->free_bytes, &snapshot->total_bytes);
    if (status == INFINI_STATUS_SUCCESS && snapshot->total_bytes > 0) {
        if (snapshot->total_bytes >= snapshot->free_bytes) {
            snapshot->used_bytes = snapshot->total_bytes - snapshot->free_bytes;
        }
        snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY;
    }

    (void)tryPopulateMxsmlMemory(device_id, snapshot);
    if (!(snapshot->valid_fields & INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY)) {
        return status;
    }

    (void)tryPopulateMxsmlDeviceName(device_id, snapshot);
    (void)tryPopulateMxsmlUtilization(device_id, snapshot);
    populateCommunicationSnapshot(device_id, snapshot);

    return INFINI_STATUS_SUCCESS;
}

void recordCommunicationSample(int device_id, infinirtEvent_t start_event, infinirtEvent_t end_event, uint64_t bytes) {
    if (start_event == nullptr || end_event == nullptr) {
        return;
    }
    if (bytes == 0) {
        destroyCommunicationSample(PendingCommunicationSample{
            static_cast<hcEvent_t>(start_event),
            static_cast<hcEvent_t>(end_event),
            bytes});
        return;
    }

    auto &store = communicationStatsStore();
    std::lock_guard<std::mutex> lock(store.mutex);
    store.per_device[device_id].pending.push_back(
        PendingCommunicationSample{
            static_cast<hcEvent_t>(start_event),
            static_cast<hcEvent_t>(end_event),
            bytes});
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_MACART(hcMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_MACART(hcMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_MACART(hcFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_MACART(hcFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

hcMemcpyKind toMacaMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return hcMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return hcMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return hcMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return hcMemcpyHostToHost;
    default:
        return hcMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_MACART(hcMemcpy(dst, src, size, toMacaMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_MACART(hcMemcpyAsync(dst, src, size, toMacaMemcpyKind(kind), (hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_MACART(hcMallocAsync(p_ptr, size, (hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_MACART(hcFreeAsync(ptr, (hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    hcStreamCaptureMode graph_mode;
    if (mode == INFINIRT_STREAM_CAPTURE_MODE_GLOBAL) {
        graph_mode = hcStreamCaptureModeGlobal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL) {
        graph_mode = hcStreamCaptureModeThreadLocal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_RELAXED) {
        graph_mode = hcStreamCaptureModeRelaxed;
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    CHECK_MACART(hcStreamBeginCapture((hcStream_t)stream, graph_mode));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    hcGraph_t graph;
    CHECK_MACART(hcStreamEndCapture((hcStream_t)stream, &graph));
    *graph_ptr = graph;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    CHECK_MACART(hcGraphDestroy((hcGraph_t)graph));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    CHECK_MACART(hcGraphInstantiate(
        (hcGraphExec_t *)graph_exec_ptr,
        (hcGraph_t)graph,
        (hcGraphNode_t *)node_ptr,
        log_buffer,
        buffer_size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    CHECK_MACART(hcGraphExecDestroy((hcGraphExec_t)graph_exec));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    CHECK_MACART(hcGraphLaunch((hcGraphExec_t)graph_exec, (hcStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace infinirt::metax
