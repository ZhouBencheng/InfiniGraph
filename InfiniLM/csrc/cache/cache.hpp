#pragma once

#include <cstddef>
#include <memory>

namespace infinilm::cache {

class Cache;

class CacheConfig {
public:
    virtual ~CacheConfig() = default;
    virtual std::unique_ptr<CacheConfig> unique_copy() const = 0;
};

class StaticKVCacheConfig : public CacheConfig {
public:
    explicit StaticKVCacheConfig(size_t max_batch_size = 1,
                                 size_t max_cache_len = 0)
        : max_batch_size_(max_batch_size),
          max_cache_len_(max_cache_len) {}

    size_t max_batch_size() const { return max_batch_size_; }
    size_t max_cache_len() const { return max_cache_len_; }

    std::unique_ptr<CacheConfig> unique_copy() const override {
        return std::make_unique<StaticKVCacheConfig>(*this);
    }

private:
    size_t max_batch_size_;
    size_t max_cache_len_;
};

class PagedKVCacheConfig : public CacheConfig {
public:
    explicit PagedKVCacheConfig(size_t num_blocks, size_t block_size = 16)
        : num_blocks_(num_blocks), block_size_(block_size) {}

    size_t num_blocks() const { return num_blocks_; }
    size_t block_size() const { return block_size_; }

    std::unique_ptr<CacheConfig> unique_copy() const override {
        return std::make_unique<PagedKVCacheConfig>(*this);
    }

private:
    size_t num_blocks_;
    size_t block_size_;
};

} // namespace infinilm::cache
