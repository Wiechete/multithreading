#include <iostream>
#include <vector>
#include <cmath> // Do funkcji fabs
#include "cpu_code.h"
#include "gpu_code.h"
#include <chrono>

// Testowanie nasycenia obliczeniami
void testSaturation() {
    const int BS = 16; // Przyk�adowa warto�� BS
    const int R_values[] = { 8, 32 }; // Przyk�adowe warto�ci R: mniejsze i wi�ksze od BS
    const int N_values[] = { 64, 128, 256, 512, 1024, 2048 }; // Przyk�adowe warto�ci N

    for (int R : R_values) {
        for (int N : N_values) {
            if (N > 2 * R) {
                std::vector<float> TAB(N * N, 0.0f);
                std::vector<float> OUT_GPU((N - 2 * R) * (N - 2 * R), 0.0f);

                // Przygotowanie danych wej�ciowych
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        TAB[i * N + j] = static_cast<float>(i * N + j);
                    }
                }

                auto start_gpu = std::chrono::high_resolution_clock::now();
                cudaError_t cudaStatus = computeSumGPU(TAB.data(), OUT_GPU.data(), N, R, BS, true, 1);
                auto end_gpu = std::chrono::high_resolution_clock::now();

                if (cudaStatus != cudaSuccess) {
                    std::cerr << "computeSumGPU failed!" << std::endl;
                    return;
                }

                double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
                int OUT_size = N - 2 * R;
                double flops_gpu = (double)(OUT_size * OUT_size * (2 * R + 1) * (2 * R + 1)) / time_gpu;

                std::cout << "N: " << N << ", R: " << R
                    << ", Time GPU: " << time_gpu << " s"
                    << ", GPU FLOP/s: " << flops_gpu << std::endl;
            }
        }
    }
}

// Testowanie wp�ywu parametru k
void testImpactK() {
    const int N = 1024; // Przyk�adowa warto�� N
    const int R = 16; // Przyk�adowa warto�� R
    const int BS = 16; // Przyk�adowa warto�� BS
    const int k_values[] = { 1, 2, 4 }; // Warto�ci k

    for (int k : k_values) {
        std::vector<float> TAB(N * N, 0.0f);
        std::vector<float> OUT_GPU((N - 2 * R) * (N - 2 * R), 0.0f);

        // Przygotowanie danych wej�ciowych
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                TAB[i * N + j] = static_cast<float>(i * N + j);
            }
        }

        auto start_gpu = std::chrono::high_resolution_clock::now();
        cudaError_t cudaStatus = computeSumGPU(TAB.data(), OUT_GPU.data(), N, R, BS, true, k);
        auto end_gpu = std::chrono::high_resolution_clock::now();

        if (cudaStatus != cudaSuccess) {
            std::cerr << "computeSumGPU failed!" << std::endl;
            return;
        }

        double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
        int OUT_size = N - 2 * R;
        double flops_gpu = (double)(OUT_size * OUT_size * (2 * R + 1) * (2 * R + 1)) / time_gpu;

        std::cout << "N: " << N << ", R: " << R << ", k: " << k
            << ", Time GPU: " << time_gpu << " s"
            << ", GPU FLOP/s: " << flops_gpu << std::endl;
    }
}

int main() {
    const int N = 1024; // Przyk�adowa wi�ksza warto�� N
    const int R1 = 2;
    const int R2 = 16;
    const int BS_values[] = { 8, 16, 32 };
    const int k_values[] = { 1, 2, 4 };

    std::cout << "Testowanie nasycenia obliczeniami:" << std::endl;
    // Testowanie nasycenia obliczeniami
    testSaturation();
    std::cout << std::endl;

    std::cout << "Testowanie wplywu parametru k:" << std::endl;
    // Testowanie wp�ywu parametru k
    testImpactK();
    std::cout << std::endl;

    std::cout << "wyniki:" << std::endl;
    // Przygotowanie danych wej�ciowych
    std::vector<float> TAB(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            TAB[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    // Testowanie r�nych parametr�w
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

            // Obliczenia na GPU dla r�nych k
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

                    // Por�wnanie wynik�w CPU i GPU
                    bool resultsMatch = true;
                    for (int i = 0; i < OUT_CPU.size(); ++i) {
                        if (fabs(OUT_CPU[i] - OUT_GPU[i]) > 1e-5) { // Mo�esz dostosowa� tolerancj� b��du
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
