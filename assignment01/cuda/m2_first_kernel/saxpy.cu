#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err_ = (call);                                                                                     \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));                                                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                                            \
    do {                                                                                                               \
        CUDA_CHECK(cudaGetLastError());                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                                           \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void my_saxpy_kernel(const float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n == 0) {
        printf("SUM=0");
        return 0;
    }
    float *x, *y;
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    // init x and y
    for (int i = 0; i < n; i++) {
        x[i] = ((i % 2048) - 1024) * 0.5f;
    }
    for (int i = 0; i < n; i++) {
        y[i] = (i % 1024) - 512;
    }

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    GpuTimer timer;
    timer.start();
    my_saxpy_kernel<<<blocks, threads>>>(x, y, n);
    float kernel_time = timer.stop_ms();
    CUDA_CHECK_KERNEL();  // remember to check kernel

    double sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += (double)y[i];
    }

    printf("SUM=%.0f, n=%d, kernel time=%f ms\n", sum, n, kernel_time);

    return 0;
}
