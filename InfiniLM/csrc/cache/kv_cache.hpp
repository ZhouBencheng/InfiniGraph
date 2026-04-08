#pragma once

#include "cache.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include "../engine/distributed/distributed.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::cache {

class Cache {
public:
    virtual ~Cache() = default;
};

class StaticKVCache : public Cache {
public:
    StaticKVCache(size_t k_head_dim,
                  size_t v_head_dim,
                  size_t num_k_heads,
                  size_t num_v_heads,
                  size_t num_layers,
                  size_t max_seq_len,
                  infinicore::DataType dtype,
                  const StaticKVCacheConfig &config,
                  const engine::distributed::RankInfo &rank_info)
        : num_layers_(num_layers),
          max_seq_len_(max_seq_len),
          k_head_dim_(k_head_dim),
          v_head_dim_(v_head_dim),
          num_k_heads_(num_k_heads),
          num_v_heads_(num_v_heads) {
        size_t batch = config.max_batch_size();
        if (config.max_cache_len() > 0) {
            max_seq_len_ = config.max_cache_len();
        }

        int tp_size = rank_info.tp_size;
        size_t local_k_heads = num_k_heads / tp_size;
        size_t local_v_heads = num_v_heads / tp_size;

        auto device = rank_info.device;
        k_caches_.reserve(num_layers);
        v_caches_.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            k_caches_.push_back(
                infinicore::Tensor::zeros({batch, local_k_heads, max_seq_len_, k_head_dim}, dtype, device));
            v_caches_.push_back(
                infinicore::Tensor::zeros({batch, local_v_heads, max_seq_len_, v_head_dim}, dtype, device));
        }
    }

    std::pair<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &past_kv_lengths) {
        infinicore::op::kv_caching_(k_caches_[layer_idx], v_caches_[layer_idx], k, v, past_kv_lengths);
        return {k_caches_[layer_idx], v_caches_[layer_idx]};
    }

private:
    size_t num_layers_;
    size_t max_seq_len_;
    size_t k_head_dim_;
    size_t v_head_dim_;
    size_t num_k_heads_;
    size_t num_v_heads_;
    std::vector<infinicore::Tensor> k_caches_;
    std::vector<infinicore::Tensor> v_caches_;
};

class PagedKVCache : public Cache {
public:
    PagedKVCache(size_t k_head_dim,
                 size_t v_head_dim,
                 size_t num_k_heads,
                 size_t num_v_heads,
                 size_t num_layers,
                 infinicore::DataType dtype,
                 const PagedKVCacheConfig &config,
                 const engine::distributed::RankInfo &rank_info)
        : num_layers_(num_layers),
          block_size_(config.block_size()),
          num_blocks_(config.num_blocks()) {
        int tp_size = rank_info.tp_size;
        size_t local_k_heads = num_k_heads / tp_size;
        size_t local_v_heads = num_v_heads / tp_size;

        auto device = rank_info.device;
        k_caches_.reserve(num_layers);
        v_caches_.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            k_caches_.push_back(
                infinicore::Tensor::zeros({num_blocks_, local_k_heads, block_size_, k_head_dim}, dtype, device));
            v_caches_.push_back(
                infinicore::Tensor::zeros({num_blocks_, local_v_heads, block_size_, v_head_dim}, dtype, device));
        }
    }

    std::pair<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &slot_mapping) {
        infinicore::op::paged_caching_(k_caches_[layer_idx], v_caches_[layer_idx], k, v, slot_mapping);
        return {k_caches_[layer_idx], v_caches_[layer_idx]};
    }

private:
    size_t num_layers_;
    size_t block_size_;
    size_t num_blocks_;
    std::vector<infinicore::Tensor> k_caches_;
    std::vector<infinicore::Tensor> v_caches_;
};

} // namespace infinilm::cache
