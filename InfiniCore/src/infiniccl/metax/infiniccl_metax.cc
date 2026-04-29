#include "infiniccl_metax.h"

#include "../../utils.h"
#include "../../infinirt/metax/infinirt_metax.h"

#ifdef ENABLE_METAX_MC_API
#include <mccl.h>
#include <mcr/mc_runtime_api.h>
#else
#include <hccl.h>
#include <hcr/hc_runtime_api.h>
#endif

#include <cstdint>
#include <iostream>
#include <vector>

#define CHECK_HCCL(API__) CHECK_INTERNAL(API__, hcclSuccess)

inline hcStream_t getMacaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<hcStream_t>(stream);
}

inline hcclDataType_t getHcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return hcclFloat;
    case INFINI_DTYPE_F16:
        return hcclHalf;
    case INFINI_DTYPE_BF16:
        return hcclBfloat16;
    default:
        std::abort();
        return hcclHalf;
    }
}

inline hcclRedOp_t getHcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return hcclSum;
    case INFINICCL_PROD:
        return hcclProd;
    case INFINICCL_MAX:
        return hcclMax;
    case INFINICCL_MIN:
        return hcclMin;
    case INFINICCL_AVG:
        return hcclAvg;
    default:
        std::abort();
        return hcclSum;
    }
}

inline hcclComm_t getHcclComm(infinicclComm_t comm) {
    return static_cast<hcclComm_t>(comm->comm);
}

inline uint64_t estimateAllReduceCommunicationBytes(infinicclComm_t comm, size_t count, infiniDtype_t datatype) {
    auto data_bytes = static_cast<uint64_t>(count) * static_cast<uint64_t>(infiniSizeOf(datatype));
    if (comm == nullptr || comm->world_size <= 1) {
        return 0;
    }

    auto world_size = static_cast<uint64_t>(comm->world_size);
    return (2ull * (world_size - 1ull) * data_bytes) / world_size;
}

namespace infiniccl::metax {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<hcclComm_t> hccl_comms(ndevice);
    CHECK_HCCL(hcclCommInitAll(hccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_METAX, device_ids[i], ndevice, (void *)(hccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_HCCL(hcclCommDestroy(getHcclComm(comm)));
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

    hcEvent_t start_event = nullptr;
    hcEvent_t end_event = nullptr;
    CHECK_INTERNAL(hcEventCreate(&start_event), hcSuccess);
    CHECK_INTERNAL(hcEventCreate(&end_event), hcSuccess);
    CHECK_INTERNAL(hcEventRecord(start_event, getMacaStream(stream)), hcSuccess);

    CHECK_HCCL(hcclAllReduce(sendbuf, recvbuf, count, getHcclDtype(datatype),
                             getHcclRedOp(op), getHcclComm(comm), getMacaStream(stream)));
    CHECK_INTERNAL(hcEventRecord(end_event, getMacaStream(stream)), hcSuccess);

    auto comm_bytes = estimateAllReduceCommunicationBytes(comm, count, datatype);
    infinirt::metax::recordCommunicationSample(
        comm->device_id,
        static_cast<infinirtEvent_t>(start_event),
        static_cast<infinirtEvent_t>(end_event),
        comm_bytes);

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::metax
