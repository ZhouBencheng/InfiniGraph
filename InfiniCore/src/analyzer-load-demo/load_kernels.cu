#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void computeLoadKernel(float *data, size_t n, int rounds) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        float x = data[i];
        for (int r = 0; r < rounds; ++r) {
            x = x * 1.000001f + 0.000001f;
            x = x * 0.999999f + 0.000002f;
        }
        data[i] = x;
    }
}

} // namespace

extern "C" void analyzerLoadDemoLaunchCompute(float *data, size_t n, int rounds, void *stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    computeLoadKernel<<<4096, 256, 0, cuda_stream>>>(data, n, rounds);
}
