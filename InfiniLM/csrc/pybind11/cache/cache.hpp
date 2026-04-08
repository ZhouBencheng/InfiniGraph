#pragma once

#include "../../cache/cache.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinilm::cache {

inline void bind_cache(py::module &m) {
    py::class_<CacheConfig, std::shared_ptr<CacheConfig>>(m, "CacheConfig");

    py::class_<StaticKVCacheConfig, CacheConfig, std::shared_ptr<StaticKVCacheConfig>>(m, "StaticKVCacheConfig")
        .def(py::init<size_t, size_t>(),
             py::arg("max_batch_size") = 1,
             py::arg("max_cache_len") = 0)
        .def_property_readonly("max_batch_size", &StaticKVCacheConfig::max_batch_size)
        .def_property_readonly("max_cache_len", &StaticKVCacheConfig::max_cache_len)
        .def("__repr__", [](const StaticKVCacheConfig &cfg) {
            return "<StaticKVCacheConfig batch=" + std::to_string(cfg.max_batch_size()) +
                   " cache_len=" + std::to_string(cfg.max_cache_len()) + ">";
        });

    py::class_<PagedKVCacheConfig, CacheConfig, std::shared_ptr<PagedKVCacheConfig>>(m, "PagedKVCacheConfig")
        .def(py::init<size_t, size_t>(),
             py::arg("num_blocks"),
             py::arg("block_size") = 16)
        .def_property_readonly("num_blocks", &PagedKVCacheConfig::num_blocks)
        .def_property_readonly("block_size", &PagedKVCacheConfig::block_size)
        .def("__repr__", [](const PagedKVCacheConfig &cfg) {
            return "<PagedKVCacheConfig blocks=" + std::to_string(cfg.num_blocks()) +
                   " block_size=" + std::to_string(cfg.block_size()) + ">";
        });
}

} // namespace infinilm::cache
