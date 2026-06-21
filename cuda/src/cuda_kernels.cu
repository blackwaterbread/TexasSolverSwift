#include "cuda_kernels.cuh"

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace texgpu {

__global__ void saxpy(const float* x, const float* y, float* out, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

bool print_device_info() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  compute capability %d.%d  (%zu MB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    return true;
}

__device__ __forceinline__ bool disjoint(int a1, int a2, int b1, int b2) {
    return a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2;
}

__global__ void k_showdown(
    const int* pc1, const int* pc2, const int* prank, int pn,
    const int* oc1, const int* oc2, const int* orank, const float* oreach, int on,
    float win, float lose, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pn) return;
    int a1 = pc1[i], a2 = pc2[i], ri = prank[i];
    float acc = 0.0f;
    for (int j = 0; j < on; ++j) {
        if (!disjoint(a1, a2, oc1[j], oc2[j])) continue;
        int rj = orank[j];
        if (ri < rj) acc += win * oreach[j];
        else if (ri > rj) acc += lose * oreach[j];
    }
    out[i] = acc;
}

__global__ void k_terminal(
    const int* pc1, const int* pc2, int pn,
    const int* oc1, const int* oc2, const float* oreach, int on,
    float payoff, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pn) return;
    int a1 = pc1[i], a2 = pc2[i];
    float acc = 0.0f;
    for (int j = 0; j < on; ++j) {
        if (disjoint(a1, a2, oc1[j], oc2[j])) acc += oreach[j];
    }
    out[i] = payoff * acc;
}

// ---- host wrappers (allocate/copy/launch; Phase 5 will keep data resident) ----
template <typename T>
static T* dev_copy(const T* host, int n) {
    T* d = nullptr;
    cudaMalloc(&d, (size_t)n * sizeof(T));
    cudaMemcpy(d, host, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

void showdown_payoff_gpu(
    const int* p_c1, const int* p_c2, const int* p_rank, int pn,
    const int* o_c1, const int* o_c2, const int* o_rank, const float* o_reach, int on,
    float win_payoff, float lose_payoff, float* out_host) {
    int *dpc1 = dev_copy(p_c1, pn), *dpc2 = dev_copy(p_c2, pn), *dprank = dev_copy(p_rank, pn);
    int *doc1 = dev_copy(o_c1, on), *doc2 = dev_copy(o_c2, on), *dorank = dev_copy(o_rank, on);
    float* dreach = dev_copy(o_reach, on);
    float* dout = nullptr; cudaMalloc(&dout, (size_t)pn * sizeof(float));

    int threads = 128, blocks = (pn + threads - 1) / threads;
    k_showdown<<<blocks, threads>>>(dpc1, dpc2, dprank, pn, doc1, doc2, dorank, dreach, on,
                                    win_payoff, lose_payoff, dout);
    cudaDeviceSynchronize();
    cudaMemcpy(out_host, dout, (size_t)pn * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dpc1); cudaFree(dpc2); cudaFree(dprank);
    cudaFree(doc1); cudaFree(doc2); cudaFree(dorank);
    cudaFree(dreach); cudaFree(dout);
}

void terminal_payoff_gpu(
    const int* p_c1, const int* p_c2, int pn,
    const int* o_c1, const int* o_c2, const float* o_reach, int on,
    float payoff, float* out_host) {
    int *dpc1 = dev_copy(p_c1, pn), *dpc2 = dev_copy(p_c2, pn);
    int *doc1 = dev_copy(o_c1, on), *doc2 = dev_copy(o_c2, on);
    float* dreach = dev_copy(o_reach, on);
    float* dout = nullptr; cudaMalloc(&dout, (size_t)pn * sizeof(float));

    int threads = 128, blocks = (pn + threads - 1) / threads;
    k_terminal<<<blocks, threads>>>(dpc1, dpc2, pn, doc1, doc2, dreach, on, payoff, dout);
    cudaDeviceSynchronize();
    cudaMemcpy(out_host, dout, (size_t)pn * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dpc1); cudaFree(dpc2);
    cudaFree(doc1); cudaFree(doc2);
    cudaFree(dreach); cudaFree(dout);
}

// ---- Discounted-CFR trainable kernels (Phase 4), one thread per hand ----
// Layout: index = action * ncards + hand.
__global__ void k_update_regrets(const float* regrets, float* r_plus, float* cum,
                                 int nact, int ncards,
                                 float alpha_coef, float beta, float strat_coef, float theta) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= ncards) return;

    // accumulate + discount r_plus, then rebuild positive sum for this hand
    float rsum = 0.0f;
    for (int a = 0; a < nact; ++a) {
        int idx = a * ncards + h;
        float v = regrets[idx] + r_plus[idx];
        v = (v > 0.0f) ? v * alpha_coef : v * beta;
        r_plus[idx] = v;
        if (v > 0.0f) rsum += v;
    }
    // current strategy (regret matching) and cumulative-strategy update
    for (int a = 0; a < nact; ++a) {
        int idx = a * ncards + h;
        float s = (rsum > 0.0f) ? (r_plus[idx] > 0.0f ? r_plus[idx] / rsum : 0.0f)
                                : (1.0f / nact);
        cum[idx] = cum[idx] * theta + s * strat_coef;
    }
}

__global__ void k_average_strategy(const float* cum, float* avg, int nact, int ncards) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= ncards) return;
    float csum = 0.0f;
    for (int a = 0; a < nact; ++a) csum += cum[a * ncards + h];
    for (int a = 0; a < nact; ++a) {
        int idx = a * ncards + h;
        avg[idx] = (csum > 0.0f) ? cum[idx] / csum : (1.0f / nact);
    }
}

void dcfr_run_gpu(int nact, int ncards, int iters,
                  const float* regret_seq, float* out_avg_host) {
    const float beta = 0.5f, gamma = 2.0f, theta = 0.9f, alpha = 1.5f;
    int sz = nact * ncards;
    float *d_rplus = nullptr, *d_cum = nullptr, *d_reg = nullptr, *d_avg = nullptr;
    cudaMalloc(&d_rplus, sz * sizeof(float));
    cudaMalloc(&d_cum, sz * sizeof(float));
    cudaMalloc(&d_reg, sz * sizeof(float));
    cudaMalloc(&d_avg, sz * sizeof(float));
    cudaMemset(d_rplus, 0, sz * sizeof(float));
    cudaMemset(d_cum, 0, sz * sizeof(float));

    int threads = 128, blocks = (ncards + threads - 1) / threads;
    for (int it = 0; it < iters; ++it) {
        int t = it + 1; // CPU passes iter+1 as iteration_number
        float alpha_coef = powf((float)t, alpha);
        alpha_coef = alpha_coef / (1.0f + alpha_coef);
        float strat_coef = powf((float)t / (float)(t + 1), gamma);
        cudaMemcpy(d_reg, regret_seq + (size_t)it * sz, sz * sizeof(float), cudaMemcpyHostToDevice);
        k_update_regrets<<<blocks, threads>>>(d_reg, d_rplus, d_cum, nact, ncards,
                                              alpha_coef, beta, strat_coef, theta);
    }
    k_average_strategy<<<blocks, threads>>>(d_cum, d_avg, nact, ncards);
    cudaDeviceSynchronize();
    cudaMemcpy(out_avg_host, d_avg, sz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rplus); cudaFree(d_cum); cudaFree(d_reg); cudaFree(d_avg);
}

bool saxpy_check(float a, int n) {
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f), ho(n, 0.0f);
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *dx = nullptr, *dy = nullptr, *doo = nullptr;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&doo, bytes);
    cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(dx, dy, doo, a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch error: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(ho.data(), doo, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(doo);

    float expected = a * 1.0f + 2.0f;
    for (int i = 0; i < n; ++i) {
        if (ho[i] != expected) return false;
    }
    return true;
}

} // namespace texgpu
