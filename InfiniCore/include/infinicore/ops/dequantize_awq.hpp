#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(
    DequantizeAWQ,
    infinicore::analyzer::OpType::DEQUANTIZE_AWQ,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &);

void dequantize_awq_(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zeros);
} // namespace infinicore::op
