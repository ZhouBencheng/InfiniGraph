#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(KVCaching,
                          infinicore::analyzer::OpType::KV_CACHING,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

void kv_caching_(Tensor k_cache,
                 Tensor v_cache,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &past_kv_lengths);
} // namespace infinicore::op
