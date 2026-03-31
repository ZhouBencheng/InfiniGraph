#pragma once

/**
 * Paged Attention Prefill — Benchmark-driven kernel dispatch rules.
 *
 * IMPORTANT: This header must be included AFTER "info.h" in .cu / .maca files,
 * because it references op::paged_attention_prefill::PagedAttentionPrefillInfo.
 *
 * It registers device-specific rules with KernelDispatcher via a static
 * initializer. Each .cu / .maca compilation unit picks up only the rules
 * matching its ENABLE_*_API define.
 *
 * Adding a new GPU platform:
 *   1. Add an #elif block below for ENABLE_<VENDOR>_API
 *   2. Implement the KernelSelectFn functions using benchmark data
 *   3. Register them in the static initializer
 */

#include "kernel_dispatcher.hpp"

namespace infinicore::dispatch::prefill_rules {

// ─── Iluvatar BI-V150 rules (benchmark 2026-03-31) ──────────────────
// hd64: warpcta8pipe always wins
// hd128, batch>=4 or q_per_seq>=64: warp wins
// hd128, low batch: warpcta8pipe wins
// THROUGHPUT/MEMORY_SAFE: warp (conservative, lower register pressure)

inline const char *iluvatar_latency(const void *info_) {
    auto &info = *static_cast<
        const op::paged_attention_prefill::PagedAttentionPrefillInfo *>(info_);
    size_t q_per_seq = info.num_seqs > 0
                           ? info.total_q_tokens / info.num_seqs
                           : 1;
    if (info.head_size <= 64) return "warpcta8pipe";
    // hd128: high-batch or long-sequence favors warp
    if (info.num_seqs >= 4 || q_per_seq >= 64) return "warp";
    return "warpcta8pipe";
}

inline const char *iluvatar_throughput(const void *) {
    return "warp";
}

inline const char *iluvatar_memory_safe(const void *) {
    return "warp";
}

// ─── NVIDIA A100 rules (benchmark 300 configs) ──────────────────────
// warpcta8pipe wins 297/300 for fp16/bf16 + page_block_size=256 + hd128
// Otherwise warpcta8 is the safe default.
// MEMORY_SAFE: warp (lowest register pressure)

inline const char *nvidia_default(const void *info_) {
    auto &info = *static_cast<
        const op::paged_attention_prefill::PagedAttentionPrefillInfo *>(info_);
    if (info.page_block_size == 256
        && (info.dtype == INFINI_DTYPE_F16 || info.dtype == INFINI_DTYPE_BF16)
        && info.head_size == 128) {
        return "warpcta8pipe";
    }
    return "warpcta8";
}

inline const char *nvidia_memory_safe(const void *) {
    return "warp";
}

// ─── MetaX rules (preliminary — mirrors NVIDIA) ─────────────────────

inline const char *metax_default(const void *info_) {
    auto &info = *static_cast<
        const op::paged_attention_prefill::PagedAttentionPrefillInfo *>(info_);
    if (info.page_block_size == 256
        && (info.dtype == INFINI_DTYPE_F16 || info.dtype == INFINI_DTYPE_BF16)
        && info.head_size == 128) {
        return "warpcta8pipe";
    }
    return "warpcta8";
}

inline const char *metax_memory_safe(const void *) {
    return "warp";
}

// ─── Static registration ────────────────────────────────────────────
// Each .cu / .maca compilation unit registers only its own device rules.

namespace detail {

inline bool registerPrefillRules() {
    auto &d = KernelDispatcher::instance();
    using Op = infinicore::analyzer::OpType;
    using Dev = infinicore::Device::Type;
    using Goal = infinicore::analyzer::OptimizationGoal;

#if defined(ENABLE_ILUVATAR_API)
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::ILUVATAR,
                   Goal::LATENCY_FIRST, iluvatar_latency);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::ILUVATAR,
                   Goal::THROUGHPUT_FIRST, iluvatar_throughput);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::ILUVATAR,
                   Goal::MEMORY_SAFE, iluvatar_memory_safe);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::ILUVATAR,
                   Goal::STABILITY_FIRST, iluvatar_latency); // same as latency

#elif defined(ENABLE_METAX_API) || defined(ENABLE_METAX_MC_API)
    d.registerDefault(Op::PAGED_ATTENTION_PREFILL, Dev::METAX, metax_default);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::METAX,
                   Goal::MEMORY_SAFE, metax_memory_safe);

#elif defined(ENABLE_ALI_API)
    // ALI PPU: use NVIDIA rules mapped to ALI device type
    d.registerDefault(Op::PAGED_ATTENTION_PREFILL, Dev::ALI, nvidia_default);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::ALI,
                   Goal::MEMORY_SAFE, nvidia_memory_safe);

#else
    // NVIDIA (default)
    d.registerDefault(Op::PAGED_ATTENTION_PREFILL, Dev::NVIDIA, nvidia_default);
    d.registerRule(Op::PAGED_ATTENTION_PREFILL, Dev::NVIDIA,
                   Goal::MEMORY_SAFE, nvidia_memory_safe);
#endif

    return true;
}

// Trigger registration at static-init time in each compilation unit.
static const bool registered_ = registerPrefillRules();

} // namespace detail
} // namespace infinicore::dispatch::prefill_rules
