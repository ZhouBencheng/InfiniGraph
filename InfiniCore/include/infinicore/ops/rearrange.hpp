#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Rearrange, infinicore::analyzer::OpType::REARRANGE, Tensor, const Tensor &);

Tensor rearrange(const Tensor &x);
void rearrange_(Tensor y, const Tensor &x);

} // namespace infinicore::op
