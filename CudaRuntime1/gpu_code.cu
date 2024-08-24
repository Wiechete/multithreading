#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

// Efektywny dostêp do danych
__global__ void computeSumKernelEfficient(const float* TAB, float* OUT, int N, int R, int OUT_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < OUT_size && j < OUT_size) {
        float sum = 0.0f;
        for (int di = -R; di <= R; ++di) {
            for (int dj = -R; dj <= R; ++dj) {
                int ni = i + R + di;
                int nj = j + R + dj;
                if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                    sum += TAB[ni * N + nj];
                }
            }
        }
        OUT[i * OUT_size + j] = sum;
    }
}

__global__ void computeSumKernelInefficient(const float* TAB, float* OUT, int N, int R, int OUT_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < OUT_size && j < OUT_size) {
        float sum = 0.0f;
        for (int di = -R; di <= R; ++di) {
            for (int dj = -R; dj <= R; ++dj) {
                // Efektywny dostêp: przeskakiwanie przez elementy
                int ni = i + R + di;
                int nj = j + R + dj;
                if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                    sum += TAB[ni * N + nj];
                }
            }
        }
        OUT[i * OUT_size + j] = sum;
    }
}


cudaError_t computeSumGPU(const float* TAB, float* OUT, int N, int R, int BS, bool efficient, int k) {
    int OUT_size = N - 2 * R;
    size_t size_TAB = N * N * sizeof(float);
    size_t size_OUT = OUT_size * OUT_size * sizeof(float);

    float* d_TAB = nullptr;
    float* d_OUT = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_TAB, size_TAB);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_OUT, size_OUT);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_TAB);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(d_TAB, TAB, size_TAB, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_TAB);
        cudaFree(d_OUT);
        return cudaStatus;
    }

    dim3 threadsPerBlock(BS, BS);
    dim3 blocksPerGrid((OUT_size + BS - 1) / BS, (OUT_size + BS - 1) / BS);

    if (efficient) {
        computeSumKernelEfficient << <blocksPerGrid, threadsPerBlock >> > (d_TAB, d_OUT, N, R, OUT_size);
    }
    else {
        computeSumKernelInefficient << <blocksPerGrid, threadsPerBlock >> > (d_TAB, d_OUT, N, R, OUT_size);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_TAB);
        cudaFree(d_OUT);
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_TAB);
        cudaFree(d_OUT);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(OUT, d_OUT, size_OUT, cudaMemcpyDeviceToHost);

    cudaFree(d_TAB);
    cudaFree(d_OUT);

    return cudaStatus;
}

