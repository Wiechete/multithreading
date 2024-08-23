#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void gpu_computation_ineffective(float* input, float* output, int N, int R, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * N + x;

    if (x < N && y < N) {
        for (int r = 0; r < R; ++r) {
            float value = input[(idx + r) % (N * N)];  // Nieefektywny dostêp
            float result = 1.0f;
            for (int i = 0; i < k; ++i) {
                result *= value;
            }
            output[idx] += result;
        }
    }
}

int main() {
    int N = 1024;  // Rozmiar tablicy
    int R = 10;    // Liczba iteracji
    int k = 2;     // Potêga

    size_t size = N * N * sizeof(float);

    // Alokacja pamiêci na CPU
    float* h_input = new float[N * N];
    float* h_output = new float[N * N]();

    // Wype³nienie tablicy wejœciowej przyk³adowymi danymi
    for (int i = 0; i < N * N; ++i) {
        h_input[i] = static_cast<float>(i % 100 + 1);
    }

    // Alokacja pamiêci na GPU
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    // Konfiguracja gridu i bloku
    dim3 block(16, 16);  // 16x16 w¹tki na blok
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Uruchomienie kernela CUDA
    gpu_computation_ineffective << <grid, block >> > (d_input, d_output, N, R, k);

    // Kopiowanie wyników z GPU do CPU
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Zwolnienie pamiêci
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
