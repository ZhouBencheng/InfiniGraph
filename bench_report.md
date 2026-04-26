# Fused FFN End-to-End Benchmark Report

- **Date:** 2026-04-26 07:14:49
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
| Cosine similarity | `0.99999464` |
| Status | **PASS** (threshold 0.999) |

## Per-Scenario Results

### bs=1,  seq=1

`batch_size=1, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 55.457 | 31.388 | +43.40% |
| trimmed mean (ms) | 55.457 | 31.390 | |
| median (ms) | 55.455 | 31.400 | |
| stdev (ms) | 0.047 | 0.103 | |
| min (ms) | 55.371 | 31.189 | |
| p99 (ms) | 55.541 | 31.559 | |
| throughput (tok/s) | 18.03 | 31.86 | |
| **speedup ratio** | | | **1.767×** |

### bs=4,  seq=1

`batch_size=4, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 62.847 | 37.726 | +39.97% |
| trimmed mean (ms) | 62.879 | 37.757 | |
| median (ms) | 63.149 | 37.796 | |
| stdev (ms) | 0.776 | 0.641 | |
| min (ms) | 61.090 | 36.126 | |
| p99 (ms) | 64.032 | 38.758 | |
| throughput (tok/s) | 63.65 | 106.03 | |
| **speedup ratio** | | | **1.666×** |

### bs=8,  seq=1

`batch_size=8, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 69.781 | 44.734 | +35.89% |
| trimmed mean (ms) | 69.703 | 44.577 | |
| median (ms) | 69.755 | 44.238 | |
| stdev (ms) | 1.497 | 2.056 | |
| min (ms) | 67.620 | 42.126 | |
| p99 (ms) | 73.343 | 50.156 | |
| throughput (tok/s) | 114.64 | 178.84 | |
| **speedup ratio** | | | **1.560×** |

### bs=16, seq=1

`batch_size=16, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 90.414 | 80.079 | +11.43% |
| trimmed mean (ms) | 90.639 | 80.698 | |
| median (ms) | 91.379 | 80.923 | |
| stdev (ms) | 4.009 | 6.006 | |
| min (ms) | 79.966 | 59.449 | |
| p99 (ms) | 96.809 | 89.568 | |
| throughput (tok/s) | 176.96 | 199.80 | |
| **speedup ratio** | | | **1.129×** |

### bs=32, seq=1

`batch_size=32, input_tokens=1, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 144.892 | 134.856 | +6.93% |
| trimmed mean (ms) | 143.989 | 135.178 | |
| median (ms) | 143.553 | 135.353 | |
| stdev (ms) | 6.786 | 6.492 | |
| min (ms) | 137.634 | 115.247 | |
| p99 (ms) | 168.402 | 148.679 | |
| throughput (tok/s) | 220.85 | 237.29 | |
| **speedup ratio** | | | **1.074×** |

### bs=2,  seq=64

`batch_size=2, input_tokens=64, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 100.168 | 84.569 | +15.57% |
| trimmed mean (ms) | 100.131 | 84.491 | |
| median (ms) | 100.020 | 84.473 | |
| stdev (ms) | 0.444 | 0.503 | |
| min (ms) | 99.634 | 84.129 | |
| p99 (ms) | 101.360 | 86.407 | |
| throughput (tok/s) | 1277.86 | 1513.56 | |
| **speedup ratio** | | | **1.184×** |

### bs=2,  seq=128

`batch_size=2, input_tokens=128, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 179.743 | 151.099 | +15.94% |
| trimmed mean (ms) | 179.721 | 151.060 | |
| median (ms) | 179.703 | 151.004 | |
| stdev (ms) | 0.484 | 0.646 | |
| min (ms) | 178.937 | 150.276 | |
| p99 (ms) | 180.931 | 152.633 | |
| throughput (tok/s) | 1424.26 | 1694.25 | |
| **speedup ratio** | | | **1.190×** |

### bs=2,  seq=256

`batch_size=2, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 297.664 | 297.758 | -0.03% |
| trimmed mean (ms) | 297.651 | 297.734 | |
| median (ms) | 297.526 | 297.731 | |
| stdev (ms) | 0.504 | 0.393 | |
| min (ms) | 296.916 | 297.193 | |
| p99 (ms) | 298.634 | 298.751 | |
| throughput (tok/s) | 1720.06 | 1719.52 | |
| **speedup ratio** | | | **1.000×** |

### bs=4,  seq=64

`batch_size=4, input_tokens=64, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 182.454 | 154.371 | +15.39% |
| trimmed mean (ms) | 182.371 | 154.283 | |
| median (ms) | 182.007 | 153.810 | |
| stdev (ms) | 0.952 | 1.060 | |
| min (ms) | 181.607 | 153.361 | |
| p99 (ms) | 184.812 | 156.962 | |
| throughput (tok/s) | 1403.09 | 1658.34 | |
| **speedup ratio** | | | **1.182×** |

### bs=4,  seq=128

`batch_size=4, input_tokens=128, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 300.405 | 300.701 | -0.10% |
| trimmed mean (ms) | 300.362 | 300.656 | |
| median (ms) | 300.350 | 300.485 | |
| stdev (ms) | 0.572 | 0.763 | |
| min (ms) | 299.775 | 299.689 | |
| p99 (ms) | 301.796 | 302.519 | |
| throughput (tok/s) | 1704.37 | 1702.69 | |
| **speedup ratio** | | | **0.999×** |

### bs=4,  seq=256

`batch_size=4, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 577.600 | 561.768 | +2.74% |
| trimmed mean (ms) | 577.526 | 561.776 | |
| median (ms) | 577.559 | 561.969 | |
| stdev (ms) | 0.748 | 0.711 | |
| min (ms) | 576.481 | 560.520 | |
| p99 (ms) | 580.046 | 562.865 | |
| throughput (tok/s) | 1772.85 | 1822.82 | |
| **speedup ratio** | | | **1.028×** |

### bs=4,  seq=512

`batch_size=4, input_tokens=512, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1135.410 | 1117.827 | +1.55% |
| trimmed mean (ms) | 1135.457 | 1117.852 | |
| median (ms) | 1135.691 | 1117.833 | |
| stdev (ms) | 0.945 | 1.228 | |
| min (ms) | 1133.345 | 1115.126 | |
| p99 (ms) | 1136.630 | 1120.073 | |
| throughput (tok/s) | 1803.75 | 1832.13 | |
| **speedup ratio** | | | **1.016×** |

### bs=8,  seq=64

`batch_size=8, input_tokens=64, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 306.671 | 306.775 | -0.03% |
| trimmed mean (ms) | 306.579 | 306.643 | |
| median (ms) | 306.358 | 306.220 | |
| stdev (ms) | 1.256 | 1.344 | |
| min (ms) | 305.394 | 305.730 | |
| p99 (ms) | 309.599 | 310.193 | |
| throughput (tok/s) | 1669.54 | 1668.98 | |
| **speedup ratio** | | | **1.000×** |

### bs=8,  seq=128

`batch_size=8, input_tokens=128, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 584.208 | 569.449 | +2.53% |
| trimmed mean (ms) | 584.130 | 569.440 | |
| median (ms) | 583.881 | 569.336 | |
| stdev (ms) | 1.266 | 1.346 | |
| min (ms) | 582.684 | 567.276 | |
| p99 (ms) | 587.150 | 571.790 | |
| throughput (tok/s) | 1752.80 | 1798.23 | |
| **speedup ratio** | | | **1.026×** |

### bs=8,  seq=256

`batch_size=8, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1134.392 | 1116.028 | +1.62% |
| trimmed mean (ms) | 1134.487 | 1116.083 | |
| median (ms) | 1134.286 | 1116.302 | |
| stdev (ms) | 1.397 | 1.752 | |
| min (ms) | 1130.818 | 1112.277 | |
| p99 (ms) | 1136.245 | 1118.777 | |
| throughput (tok/s) | 1805.37 | 1835.08 | |
| **speedup ratio** | | | **1.016×** |

### bs=8,  seq=512

`batch_size=8, input_tokens=512, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 2240.138 | 2213.741 | +1.18% |
| trimmed mean (ms) | 2240.195 | 2213.742 | |
| median (ms) | 2240.270 | 2213.572 | |
| stdev (ms) | 2.206 | 2.003 | |
| min (ms) | 2235.210 | 2210.140 | |
| p99 (ms) | 2244.052 | 2217.318 | |
| throughput (tok/s) | 1828.46 | 1850.26 | |
| **speedup ratio** | | | **1.012×** |

### bs=16, seq=64

`batch_size=16, input_tokens=64, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 598.278 | 582.598 | +2.62% |
| trimmed mean (ms) | 598.161 | 582.615 | |
| median (ms) | 598.598 | 582.856 | |
| stdev (ms) | 2.335 | 2.716 | |
| min (ms) | 594.506 | 577.898 | |
| p99 (ms) | 604.165 | 586.994 | |
| throughput (tok/s) | 1711.58 | 1757.64 | |
| **speedup ratio** | | | **1.027×** |

### bs=16, seq=128

`batch_size=16, input_tokens=128, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1146.993 | 1127.432 | +1.71% |
| trimmed mean (ms) | 1147.017 | 1127.437 | |
| median (ms) | 1147.572 | 1127.183 | |
| stdev (ms) | 1.892 | 2.705 | |
| min (ms) | 1143.414 | 1123.248 | |
| p99 (ms) | 1150.129 | 1131.536 | |
| throughput (tok/s) | 1785.54 | 1816.52 | |
| **speedup ratio** | | | **1.017×** |

### bs=16, seq=256

`batch_size=16, input_tokens=256, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 2235.496 | 2208.950 | +1.19% |
| trimmed mean (ms) | 2235.503 | 2208.693 | |
| median (ms) | 2234.466 | 2208.662 | |
| stdev (ms) | 2.917 | 2.556 | |
| min (ms) | 2230.734 | 2205.289 | |
| p99 (ms) | 2240.140 | 2217.238 | |
| throughput (tok/s) | 1832.26 | 1854.27 | |
| **speedup ratio** | | | **1.012×** |

### bs=1,  seq=32

`batch_size=1, input_tokens=32, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 70.803 | 43.530 | +38.52% |
| trimmed mean (ms) | 70.801 | 43.529 | |
| median (ms) | 70.810 | 43.523 | |
| stdev (ms) | 0.069 | 0.058 | |
| min (ms) | 70.659 | 43.407 | |
| p99 (ms) | 70.986 | 43.664 | |
| throughput (tok/s) | 451.96 | 735.13 | |
| **speedup ratio** | | | **1.627×** |

### bs=1,  seq=64

`batch_size=1, input_tokens=64, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 69.635 | 58.839 | +15.50% |
| trimmed mean (ms) | 69.626 | 58.826 | |
| median (ms) | 69.612 | 58.802 | |
| stdev (ms) | 0.115 | 0.145 | |
| min (ms) | 69.470 | 58.663 | |
| p99 (ms) | 69.967 | 59.249 | |
| throughput (tok/s) | 919.07 | 1087.71 | |
| **speedup ratio** | | | **1.183×** |

### bs=1,  seq=128

`batch_size=1, input_tokens=128, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 98.972 | 83.499 | +15.63% |
| trimmed mean (ms) | 98.921 | 83.473 | |
| median (ms) | 98.773 | 83.374 | |
| stdev (ms) | 0.487 | 0.333 | |
| min (ms) | 98.555 | 83.190 | |
| p99 (ms) | 100.305 | 84.264 | |
| throughput (tok/s) | 1293.29 | 1532.96 | |
| **speedup ratio** | | | **1.185×** |

### bs=1,  seq=256

`batch_size=1, input_tokens=256, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 177.838 | 149.807 | +15.76% |
| trimmed mean (ms) | 177.839 | 149.804 | |
| median (ms) | 177.858 | 149.790 | |
| stdev (ms) | 0.107 | 0.078 | |
| min (ms) | 177.631 | 149.675 | |
| p99 (ms) | 178.030 | 149.983 | |
| throughput (tok/s) | 1439.51 | 1708.87 | |
| **speedup ratio** | | | **1.187×** |

### bs=1,  seq=512

`batch_size=1, input_tokens=512, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 297.982 | 298.028 | -0.02% |
| trimmed mean (ms) | 297.982 | 298.033 | |
| median (ms) | 298.001 | 298.058 | |
| stdev (ms) | 0.313 | 0.219 | |
| min (ms) | 297.484 | 297.535 | |
| p99 (ms) | 298.490 | 298.425 | |
| throughput (tok/s) | 1718.22 | 1717.96 | |
| **speedup ratio** | | | **1.000×** |

### bs=1,  seq=1024

`batch_size=1, input_tokens=1024, output_tokens=32`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 585.349 | 569.823 | +2.65% |
| trimmed mean (ms) | 585.307 | 569.821 | |
| median (ms) | 585.254 | 570.013 | |
| stdev (ms) | 0.493 | 0.751 | |
| min (ms) | 584.646 | 568.547 | |
| p99 (ms) | 586.800 | 571.135 | |
| throughput (tok/s) | 1749.38 | 1797.05 | |
| **speedup ratio** | | | **1.027×** |

### bs=1,  seq=2048

`batch_size=1, input_tokens=2048, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 1195.638 | 1175.543 | +1.68% |
| trimmed mean (ms) | 1195.628 | 1175.551 | |
| median (ms) | 1195.404 | 1175.706 | |
| stdev (ms) | 0.895 | 1.479 | |
| min (ms) | 1194.237 | 1172.853 | |
| p99 (ms) | 1197.213 | 1178.103 | |
| throughput (tok/s) | 1712.89 | 1742.17 | |
| **speedup ratio** | | | **1.017×** |

### bs=1,  seq=4096

`batch_size=1, input_tokens=4096, output_tokens=1`

| Metric | Non-Fused | Fused | Δ |
|--------|-----------|-------|---|
| mean latency (ms) | 2608.918 | 2601.993 | +0.27% |
| trimmed mean (ms) | 2608.117 | 2602.635 | |
| median (ms) | 2603.593 | 2603.828 | |
| stdev (ms) | 11.743 | 11.735 | |
| min (ms) | 2597.438 | 2573.509 | |
| p99 (ms) | 2634.810 | 2618.917 | |
| throughput (tok/s) | 1570.00 | 1574.18 | |
| **speedup ratio** | | | **1.003×** |

## Summary

| Scenario | NF mean (ms) | F mean (ms) | Speedup | Ratio |
|----------|-------------|------------|---------|-------|
| bs=1,  seq=1 | 55.457 | 31.388 | +43.40% | 1.767× |
| bs=4,  seq=1 | 62.847 | 37.726 | +39.97% | 1.666× |
| bs=8,  seq=1 | 69.781 | 44.734 | +35.89% | 1.560× |
| bs=16, seq=1 | 90.414 | 80.079 | +11.43% | 1.129× |
| bs=32, seq=1 | 144.892 | 134.856 | +6.93% | 1.074× |
| bs=2,  seq=64 | 100.168 | 84.569 | +15.57% | 1.184× |
| bs=2,  seq=128 | 179.743 | 151.099 | +15.94% | 1.190× |
| bs=2,  seq=256 | 297.664 | 297.758 | -0.03% | 1.000× |
| bs=4,  seq=64 | 182.454 | 154.371 | +15.39% | 1.182× |
| bs=4,  seq=128 | 300.405 | 300.701 | -0.10% | 0.999× |
| bs=4,  seq=256 | 577.600 | 561.768 | +2.74% | 1.028× |
| bs=4,  seq=512 | 1135.410 | 1117.827 | +1.55% | 1.016× |
| bs=8,  seq=64 | 306.671 | 306.775 | -0.03% | 1.000× |
| bs=8,  seq=128 | 584.208 | 569.449 | +2.53% | 1.026× |
| bs=8,  seq=256 | 1134.392 | 1116.028 | +1.62% | 1.016× |
| bs=8,  seq=512 | 2240.138 | 2213.741 | +1.18% | 1.012× |
| bs=16, seq=64 | 598.278 | 582.598 | +2.62% | 1.027× |
| bs=16, seq=128 | 1146.993 | 1127.432 | +1.71% | 1.017× |
| bs=16, seq=256 | 2235.496 | 2208.950 | +1.19% | 1.012× |
| bs=1,  seq=32 | 70.803 | 43.530 | +38.52% | 1.627× |
| bs=1,  seq=64 | 69.635 | 58.839 | +15.50% | 1.183× |
| bs=1,  seq=128 | 98.972 | 83.499 | +15.63% | 1.185× |
| bs=1,  seq=256 | 177.838 | 149.807 | +15.76% | 1.187× |
| bs=1,  seq=512 | 297.982 | 298.028 | -0.02% | 1.000× |
| bs=1,  seq=1024 | 585.349 | 569.823 | +2.65% | 1.027× |
| bs=1,  seq=2048 | 1195.638 | 1175.543 | +1.68% | 1.017× |
| bs=1,  seq=4096 | 2608.918 | 2601.993 | +0.27% | 1.003× |

