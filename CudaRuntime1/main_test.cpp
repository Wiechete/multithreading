#include <iostream>
#include <vector>
#include "cpu_code.h"
#include "gpu_code.h"

// Funkcja do wyœwietlania macierzy w terminalu
void printMatrix(const std::vector<float>& matrix, int N, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main() {
    const int N = 5;  // Rozmiar macierzy (NxN)
    const int R = 1;  // Promieñ do obliczeñ sum

    // Przygotowanie danych wejœciowych (macierz NxN)
    std::vector<float> TAB(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            TAB[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    // Wyœwietlanie danych wejœciowych
    printMatrix(TAB, N, "Input Matrix (TAB)");

    // Przygotowanie danych wyjœciowych
    int OUT_size = N - 2 * R;
    std::vector<float> OUT_CPU(OUT_size * OUT_size, 0.0f);
    std::vector<float> OUT_GPU(OUT_size * OUT_size, 0.0f);

    // Obliczenia na CPU
    computeSumCPU(TAB, OUT_CPU, N, R);

    // Obliczenia na GPU
    cudaError_t cudaStatus = computeSumGPU(TAB.data(), OUT_GPU.data(), N, R);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "computeSumGPU failed!" << std::endl;
        return 1;
    }

    // Wyœwietlanie wyników z CPU i GPU
    printMatrix(OUT_CPU, OUT_size, "Output Matrix (CPU)");
    printMatrix(OUT_GPU, OUT_size, "Output Matrix (GPU)");

    // Porównanie wyników
    bool success = true;
    for (int i = 0; i < OUT_size; ++i) {
        for (int j = 0; j < OUT_size; ++j) {
            if (OUT_CPU[i * OUT_size + j] != OUT_GPU[i * OUT_size + j]) {
                std::cerr << "Mismatch at (" << i << "," << j << "): CPU = "
                    << OUT_CPU[i * OUT_size + j] << ", GPU = "
                    << OUT_GPU[i * OUT_size + j] << std::endl;
                success = false;
            }
        }
    }

    if (success) {
        std::cout << "Results match!" << std::endl;
    }
    else {
        std::cerr << "Results do not match!" << std::endl;
    }

    return 0;
}
