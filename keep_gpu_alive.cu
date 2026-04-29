// keep_gpu_alive.cu
// Continuously runs cuBLAS GEMM to keep GPU active and prevent sleep.
// Compile: nvcc -o keep_gpu_alive keep_gpu_alive.cu -lcublas
// Run:     nohup ./keep_gpu_alive &

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <signal.h>
#include <unistd.h>

static volatile int running = 1;

void handle_signal(int sig) {
    printf("\nCaught signal %d, stopping...\n", sig);
    running = 0;
}

int main() {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    const int N = 2048;  // matrix size: 2048x2048
    size_t bytes = N * N * sizeof(float);

    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Matrix size: %d x %d (FP32)\n", N, N);
    printf("Running continuous GEMM to keep GPU alive...\n");
    printf("Press Ctrl+C or kill this process to stop.\n\n");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Initialize with random values
    float *h_buf = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_buf[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_A, h_buf, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_buf, bytes, cudaMemcpyHostToDevice);
    free(h_buf);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    long long iter = 0;

    while (running) {
        // Run a batch of GEMMs
        for (int i = 0; i < 10 && running; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
            iter++;
        }
        cudaDeviceSynchronize();

        // Print status every 100 iterations
        if (iter % 100 == 0) {
            printf("Iterations: %lld\n", iter);
            fflush(stdout);
        }

        // Small sleep to avoid 100% GPU but still keep it awake
        usleep(100000);  // 100ms pause between batches
    }

    printf("Total iterations: %lld\n", iter);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("GPU resources freed. Exiting.\n");
    return 0;
}
