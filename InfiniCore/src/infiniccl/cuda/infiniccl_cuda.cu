#include "infiniccl_cuda.h"

#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>

#include "../../utils.h"

#define CHECK_NCCL(API__) CHECK_INTERNAL(API__, ncclSuccess)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream);
}

inline ncclDataType_t getNcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return ncclFloat;
    case INFINI_DTYPE_F16:
        return ncclHalf;
    case INFINI_DTYPE_BF16:
        return ncclBfloat16;
    default:
        std::abort();
        return ncclHalf;
    }
}

inline ncclRedOp_t getNcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return ncclSum;
    case INFINICCL_PROD:
        return ncclProd;
    case INFINICCL_MAX:
        return ncclMax;
    case INFINICCL_MIN:
        return ncclMin;
    case INFINICCL_AVG:
        return ncclAvg;
    default:
        std::abort();
        return ncclSum;
    }
}

inline ncclComm_t getNcclComm(infinicclComm_t comm) {
    return static_cast<ncclComm_t>(comm->comm);
}

inline infiniDevice_t currentCudaFamilyDeviceType() {
#if defined(ENABLE_NVIDIA_API)
    return INFINI_DEVICE_NVIDIA;
#elif defined(ENABLE_ILUVATAR_API)
    return INFINI_DEVICE_ILUVATAR;
#elif defined(ENABLE_QY_API)
    return INFINI_DEVICE_QY;
#elif defined(ENABLE_HYGON_API)
    return INFINI_DEVICE_HYGON;
#elif defined(ENABLE_ALI_API)
    return INFINI_DEVICE_ALI;
#else
    return INFINI_DEVICE_NVIDIA;
#endif
}

inline void recordCudaFamilyCommunicationSample(
    int device_id,
    infinirtEvent_t start_event,
    infinirtEvent_t end_event,
    uint64_t bytes) {
#if defined(ENABLE_NVIDIA_API)
    infinirt::cuda::recordCommunicationSample(device_id, start_event, end_event, bytes);
#elif defined(ENABLE_ILUVATAR_API)
    infinirt::iluvatar::recordCommunicationSample(device_id, start_event, end_event, bytes);
#elif defined(ENABLE_QY_API)
    infinirt::qy::recordCommunicationSample(device_id, start_event, end_event, bytes);
#elif defined(ENABLE_HYGON_API)
    infinirt::hygon::recordCommunicationSample(device_id, start_event, end_event, bytes);
#elif defined(ENABLE_ALI_API)
    infinirt::ali::recordCommunicationSample(device_id, start_event, end_event, bytes);
#else
    (void)device_id;
    (void)start_event;
    (void)end_event;
    (void)bytes;
#endif
}

inline uint64_t estimateAllReduceCommunicationBytes(infinicclComm_t comm, size_t count, infiniDtype_t datatype) {
    auto data_bytes = static_cast<uint64_t>(count) * static_cast<uint64_t>(infiniSizeOf(datatype));
    if (comm == nullptr || comm->world_size <= 1) {
        return 0;
    }

    auto world_size = static_cast<uint64_t>(comm->world_size);
    return (2ull * (world_size - 1ull) * data_bytes) / world_size;
}

namespace infiniccl::cuda {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<ncclComm_t> nccl_comms(ndevice);
    CHECK_NCCL(ncclCommInitAll(nccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{currentCudaFamilyDeviceType(), device_ids[i], ndevice, (void *)(nccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_NCCL(ncclCommDestroy(getNcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    CHECK_INTERNAL(cudaEventCreate(&start_event), cudaSuccess);
    CHECK_INTERNAL(cudaEventCreate(&end_event), cudaSuccess);
    CHECK_INTERNAL(cudaEventRecord(start_event, getCudaStream(stream)), cudaSuccess);

    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, getNcclDtype(datatype),
                             getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));
    CHECK_INTERNAL(cudaEventRecord(end_event, getCudaStream(stream)), cudaSuccess);

    auto comm_bytes = estimateAllReduceCommunicationBytes(comm, count, datatype);
    recordCudaFamilyCommunicationSample(
        comm->device_id,
        static_cast<infinirtEvent_t>(start_event),
        static_cast<infinirtEvent_t>(end_event),
        comm_bytes);

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::cuda
