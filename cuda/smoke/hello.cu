#include <cstdio>
#include <cuda_runtime.h>

__global__ void axpy(const float* x, const float* y, float* out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *hx = (float*)malloc(bytes), *hy = (float*)malloc(bytes), *ho = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }

    float *dx, *dy, *doo;
    cudaMalloc(&dx, bytes); cudaMalloc(&dy, bytes); cudaMalloc(&doo, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;
    axpy<<<blocks, threads>>>(dx, dy, doo, 3.0f, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("kernel launch error: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(ho, doo, bytes, cudaMemcpyDeviceToHost);

    // expected: 3*1 + 2 = 5
    bool ok = true;
    for (int i = 0; i < n; ++i) if (ho[i] != 5.0f) { ok = false; break; }

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  compute capability %d.%d\n", prop.name, prop.major, prop.minor);
    printf("axpy result check: %s (ho[0]=%f)\n", ok ? "PASS" : "FAIL", ho[0]);

    cudaFree(dx); cudaFree(dy); cudaFree(doo);
    free(hx); free(hy); free(ho);
    return ok ? 0 : 2;
}
