/**
 * Paged Attention Prefill — Kernel dispatch rules (stub).
 *
 * Actual dispatch rules are registered via static initializers in
 * prefill_dispatch_rules.h, which is included by device-specific
 * .cu / .maca files (those files have access to PagedAttentionPrefillInfo).
 *
 * This .cc file exists so that the dispatch/*.cc glob in xmake.lua
 * always has at least one source file to compile alongside kernel_dispatcher.cc.
 *
 * See: include/infinicore/dispatch/prefill_dispatch_rules.h
 *      nvidia/paged_attention_prefill_nvidia.cu
 *      metax/paged_attention_prefill_metax.maca
 */
