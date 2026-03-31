#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SwiGLU, infinicore::analyzer::OpType::SWIGLU, Tensor, const Tensor &, const Tensor &);

Tensor swiglu(const Tensor &a, const Tensor &b);
void swiglu_(Tensor c, const Tensor &a, const Tensor &b);

} // namespace infinicore::op
