#ifndef __INFINIRT_MACA_H__
#define __INFINIRT_MACA_H__
#include "../../infiniop/devices/metax/metax_ht2mc.h"
#include "../infinirt_impl.h"

#include <cstdint>

namespace infinirt::metax {
#ifdef ENABLE_METAX_API
INFINIRT_DEVICE_API_IMPL

void recordCommunicationSample(
    int device_id,
    infinirtEvent_t start_event,
    infinirtEvent_t end_event,
    uint64_t bytes);
#else
INFINIRT_DEVICE_API_NOOP
inline void recordCommunicationSample(int, infinirtEvent_t, infinirtEvent_t, uint64_t) {}
#endif
} // namespace infinirt::metax

#endif // __INFINIRT_MACA_H__
