# Fused FFN End-to-End Benchmark Report

- **Date:** 2026-04-13 05:53:18
- **Model:** `/home/huids25/models/9G7B_MHA/`
- **Device:** iluvatar (x1)
- **Warmup:** 5 rounds per mode
- **Measured rounds:** 20 per mode (ordering: interleaved)

> Only metric is end-to-end wall clock latency, measured with `time.perf_counter` around the `inferBatchJiuge` C call. Fused and non-fused runs share byte-identical inputs; rounds are interleaved so that thermal drift and system jitter affect both modes equally and cancel in the comparison.

## Correctness Verification

| Metric | Value |
|--------|-------|
| Max abs diff | `0.000000e+00` |
| Mean abs diff | `0.000000e+00` |
| Cosine similarity | `0.99998942` |
| Status | **PASS** (threshold 0.999) |

## Per-Scenario Results

### Decode (bs=1, seq=1)

`batch_size=1, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 55.577 | 31.486 | +43.35% |
| trimmed mean (ms) | 55.574 | 31.433 | |
| median (ms) | 55.583 | 31.401 | |
| stdev (ms) | 0.095 | 0.295 | |
| min (ms) | 55.444 | 31.287 | |
| p99 (ms) | 55.772 | 32.632 | |
| throughput (tok/s) | 17.99 | 31.76 | |
| **speedup ratio** | | | **1.765×** |

### Batched Decode (bs=4, seq=1)

`batch_size=4, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 63.577 | 37.318 | +41.30% |
| trimmed mean (ms) | 62.905 | 37.296 | |
| median (ms) | 62.247 | 37.054 | |
| stdev (ms) | 3.900 | 0.712 | |
| min (ms) | 61.487 | 36.155 | |
| p99 (ms) | 77.759 | 38.876 | |
| throughput (tok/s) | 62.92 | 107.19 | |
| **speedup ratio** | | | **1.704×** |

### Batched Decode (bs=8, seq=1)

`batch_size=8, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 69.087 | 45.128 | +34.68% |
| trimmed mean (ms) | 68.981 | 45.065 | |
| median (ms) | 68.400 | 44.016 | |
| stdev (ms) | 1.443 | 2.454 | |
| min (ms) | 67.557 | 42.096 | |
| p99 (ms) | 72.517 | 49.296 | |
| throughput (tok/s) | 115.80 | 177.27 | |
| **speedup ratio** | | | **1.531×** |

### Batched Decode (bs=16, seq=1)

`batch_size=16, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 92.275 | 80.009 | +13.29% |
| trimmed mean (ms) | 91.997 | 80.175 | |
| median (ms) | 91.497 | 80.257 | |
| stdev (ms) | 4.285 | 5.372 | |
| min (ms) | 86.243 | 67.171 | |
| p99 (ms) | 103.321 | 89.851 | |
| throughput (tok/s) | 173.39 | 199.98 | |
| **speedup ratio** | | | **1.153×** |

### Batched Decode (bs=32, seq=1)

`batch_size=32, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 138.598 | 131.145 | +5.38% |
| trimmed mean (ms) | 138.967 | 132.277 | |
| median (ms) | 138.960 | 134.076 | |
| stdev (ms) | 11.380 | 16.864 | |
| min (ms) | 105.356 | 86.177 | |
| p99 (ms) | 165.209 | 155.731 | |
| throughput (tok/s) | 230.88 | 244.00 | |
| **speedup ratio** | | | **1.057×** |

### Batched Prefill (bs=2, seq=256)

`batch_size=2, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 297.588 | 297.747 | -0.05% |
| trimmed mean (ms) | 297.457 | 297.752 | |
| median (ms) | 297.425 | 297.726 | |
| stdev (ms) | 0.761 | 0.211 | |
| min (ms) | 296.959 | 297.240 | |
| p99 (ms) | 300.575 | 298.166 | |
| throughput (tok/s) | 1720.50 | 1719.58 | |
| **speedup ratio** | | | **0.999×** |

### Batched Prefill (bs=4, seq=256)

`batch_size=4, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 577.971 | 562.194 | +2.73% |
| trimmed mean (ms) | 577.760 | 562.181 | |
| median (ms) | 577.599 | 562.121 | |
| stdev (ms) | 1.333 | 0.371 | |
| min (ms) | 576.790 | 561.527 | |
| p99 (ms) | 582.956 | 563.102 | |
| throughput (tok/s) | 1771.71 | 1821.43 | |
| **speedup ratio** | | | **1.028×** |

### Batched Prefill (bs=4, seq=512)

`batch_size=4, input_tokens=512, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1135.528 | 1116.424 | +1.68% |
| trimmed mean (ms) | 1135.491 | 1116.320 | |
| median (ms) | 1135.446 | 1116.485 | |
| stdev (ms) | 0.614 | 1.708 | |
| min (ms) | 1134.590 | 1113.749 | |
| p99 (ms) | 1137.123 | 1120.967 | |
| throughput (tok/s) | 1803.57 | 1834.43 | |
| **speedup ratio** | | | **1.017×** |

### Batched Prefill (bs=8, seq=512)

`batch_size=8, input_tokens=512, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 2238.975 | 2211.828 | +1.21% |
| trimmed mean (ms) | 2239.056 | 2211.827 | |
| median (ms) | 2239.335 | 2211.729 | |
| stdev (ms) | 1.378 | 2.132 | |
| min (ms) | 2235.778 | 2208.203 | |
| p99 (ms) | 2240.725 | 2215.485 | |
| throughput (tok/s) | 1829.41 | 1851.86 | |
| **speedup ratio** | | | **1.012×** |

### Prefill (bs=1, seq=32)

`batch_size=1, input_tokens=32, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 70.838 | 43.538 | +38.54% |
| trimmed mean (ms) | 70.814 | 43.516 | |
| median (ms) | 70.809 | 43.496 | |
| stdev (ms) | 0.136 | 0.181 | |
| min (ms) | 70.729 | 43.336 | |
| p99 (ms) | 71.373 | 44.145 | |
| throughput (tok/s) | 451.74 | 734.99 | |
| **speedup ratio** | | | **1.627×** |

### Prefill (bs=1, seq=64)

`batch_size=1, input_tokens=64, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 69.615 | 58.799 | +15.54% |
| trimmed mean (ms) | 69.602 | 58.773 | |
| median (ms) | 69.560 | 58.747 | |
| stdev (ms) | 0.152 | 0.166 | |
| min (ms) | 69.437 | 58.656 | |
| p99 (ms) | 70.026 | 59.413 | |
| throughput (tok/s) | 919.34 | 1088.45 | |
| **speedup ratio** | | | **1.184×** |

### Prefill (bs=1, seq=128)

`batch_size=1, input_tokens=128, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 98.681 | 83.155 | +15.73% |
| trimmed mean (ms) | 98.683 | 83.157 | |
| median (ms) | 98.688 | 83.157 | |
| stdev (ms) | 0.061 | 0.097 | |
| min (ms) | 98.558 | 82.927 | |
| p99 (ms) | 98.768 | 83.337 | |
| throughput (tok/s) | 1297.11 | 1539.30 | |
| **speedup ratio** | | | **1.187×** |

### Prefill (bs=1, seq=256)

`batch_size=1, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 177.876 | 149.711 | +15.83% |
| trimmed mean (ms) | 177.839 | 149.684 | |
| median (ms) | 177.830 | 149.696 | |
| stdev (ms) | 0.233 | 0.196 | |
| min (ms) | 177.651 | 149.464 | |
| p99 (ms) | 178.755 | 150.461 | |
| throughput (tok/s) | 1439.21 | 1709.96 | |
| **speedup ratio** | | | **1.188×** |

### Prefill (bs=1, seq=512)

`batch_size=1, input_tokens=512, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 298.175 | 298.159 | +0.01% |
| trimmed mean (ms) | 298.158 | 298.160 | |
| median (ms) | 298.148 | 298.135 | |
| stdev (ms) | 0.441 | 0.247 | |
| min (ms) | 297.406 | 297.711 | |
| p99 (ms) | 299.244 | 298.601 | |
| throughput (tok/s) | 1717.11 | 1717.20 | |
| **speedup ratio** | | | **1.000×** |

### Prefill (bs=1, seq=1024)

`batch_size=1, input_tokens=1024, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 585.342 | 570.119 | +2.60% |
| trimmed mean (ms) | 585.313 | 570.134 | |
| median (ms) | 585.355 | 570.224 | |
| stdev (ms) | 0.400 | 0.557 | |
| min (ms) | 584.674 | 568.856 | |
| p99 (ms) | 586.526 | 571.120 | |
| throughput (tok/s) | 1749.40 | 1796.12 | |
| **speedup ratio** | | | **1.027×** |

### Prefill (bs=1, seq=2048)

`batch_size=1, input_tokens=2048, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1194.909 | 1176.700 | +1.52% |
| trimmed mean (ms) | 1194.887 | 1176.706 | |
| median (ms) | 1194.814 | 1176.799 | |
| stdev (ms) | 0.784 | 1.464 | |
| min (ms) | 1193.771 | 1173.894 | |
| p99 (ms) | 1196.445 | 1179.396 | |
| throughput (tok/s) | 1713.94 | 1740.46 | |
| **speedup ratio** | | | **1.015×** |

### Prefill (bs=1, seq=4096)

`batch_size=1, input_tokens=4096, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 2628.598 | 2605.055 | +0.90% |
| trimmed mean (ms) | 2628.910 | 2605.372 | |
| median (ms) | 2629.727 | 2611.907 | |
| stdev (ms) | 19.122 | 17.606 | |
| min (ms) | 2599.866 | 2573.774 | |
| p99 (ms) | 2651.707 | 2630.621 | |
| throughput (tok/s) | 1558.25 | 1572.33 | |
| **speedup ratio** | | | **1.009×** |

## Summary

| Scenario | NF mean (ms) | F mean (ms) | Speedup | Ratio |
|----------|-------------|------------|---------|-------|
| Decode (bs=1, seq=1) | 55.577 | 31.486 | +43.35% | 1.765× |
| Batched Decode (bs=4, seq=1) | 63.577 | 37.318 | +41.30% | 1.704× |
| Batched Decode (bs=8, seq=1) | 69.087 | 45.128 | +34.68% | 1.531× |
| Batched Decode (bs=16, seq=1) | 92.275 | 80.009 | +13.29% | 1.153× |
| Batched Decode (bs=32, seq=1) | 138.598 | 131.145 | +5.38% | 1.057× |
| Batched Prefill (bs=2, seq=256) | 297.588 | 297.747 | -0.05% | 0.999× |
| Batched Prefill (bs=4, seq=256) | 577.971 | 562.194 | +2.73% | 1.028× |
| Batched Prefill (bs=4, seq=512) | 1135.528 | 1116.424 | +1.68% | 1.017× |
| Batched Prefill (bs=8, seq=512) | 2238.975 | 2211.828 | +1.21% | 1.012× |
| Prefill (bs=1, seq=32) | 70.838 | 43.538 | +38.54% | 1.627× |
| Prefill (bs=1, seq=64) | 69.615 | 58.799 | +15.54% | 1.184× |
| Prefill (bs=1, seq=128) | 98.681 | 83.155 | +15.73% | 1.187× |
| Prefill (bs=1, seq=256) | 177.876 | 149.711 | +15.83% | 1.188× |
| Prefill (bs=1, seq=512) | 298.175 | 298.159 | +0.01% | 1.000× |
| Prefill (bs=1, seq=1024) | 585.342 | 570.119 | +2.60% | 1.027× |
| Prefill (bs=1, seq=2048) | 1194.909 | 1176.700 | +1.52% | 1.015× |
| Prefill (bs=1, seq=4096) | 2628.598 | 2605.055 | +0.90% | 1.009× |

