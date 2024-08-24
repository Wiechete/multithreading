#include <iostream>
#include <vector>
#include <cmath> // Do funkcji fabs
#include "cpu_code.h"
#include "gpu_code.h"
#include <chrono>

int main() {
    const int N = 1024; // Przyk³adowa wiêksza wartoœæ N
    const int R1 = 2;
    const int R2 = 16;
    const int BS_values[] = { 8, 16, 32 };
    const int k_values[] = { 1, 2, 4 };

    // Przygotowanie danych wejœciowych
    std::vector<float> TAB(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            TAB[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    // Testowanie ró¿nych parametrów
    for (int BS : BS_values) {
        for (int R : {R1, R2}) {
            int OUT_size = N - 2 * R;
            std::vector<float> OUT_CPU(OUT_size * OUT_size, 0.0f);
            std::vector<float> OUT_GPU(OUT_size * OUT_size, 0.0f);

            // Obliczenia na CPU
            auto start_cpu = std::chrono::high_resolution_clock::now();
            computeSumCPU(TAB, OUT_CPU, N, R);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();

            // Obliczenia na GPU dla ró¿nych k
            for (int k : k_values) {
                for (bool efficient : {true, false}) {
                    auto start_gpu = std::chrono::high_resolution_clock::now();
                    cudaError_t cudaStatus = computeSumGPU(TAB.data(), OUT_GPU.data(), N, R, BS, efficient, k);
                    auto end_gpu = std::chrono::high_resolution_clock::now();

                    if (cudaStatus != cudaSuccess) {
                        std::cerr << "computeSumGPU failed!" << std::endl;
                        return 1;
                    }

                    double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
                    double flops_cpu = (double)(OUT_size * OUT_size * (2 * R + 1) * (2 * R + 1)) / time_cpu;
                    double flops_gpu = (double)(OUT_size * OUT_size * (2 * R + 1) * (2 * R + 1)) / time_gpu;

                    // Porównanie wyników CPU i GPU
                    bool resultsMatch = true;
                    for (int i = 0; i < OUT_CPU.size(); ++i) {
                        if (fabs(OUT_CPU[i] - OUT_GPU[i]) > 1e-5) { // Mo¿esz dostosowaæ tolerancjê b³êdu
                            resultsMatch = false;
                            break;
                        }
                    }

                    std::cout << "BS: " << BS << ", R: " << R << ", k: " << k
                        << ", Efficient: " << (efficient ? "Yes" : "No")
                        << ", Time CPU: " << time_cpu << " s"
                        << ", Time GPU: " << time_gpu << " s"
                        << ", CPU FLOP/s: " << flops_cpu
                        << ", GPU FLOP/s: " << flops_gpu
                        << ", Results Match: " << (resultsMatch ? "Yes" : "No") << std::endl;
                }
            }
        }
    }

    return 0;
}
