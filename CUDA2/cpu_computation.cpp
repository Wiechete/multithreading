// CUDA2.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

#include <iostream>
#include <cmath>

void cpu_computation(float* input, float* output, int N, int R, int k) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = y * N + x;
            for (int r = 0; r < R; ++r) {
                output[idx] += std::pow(input[idx], k);
            }
        }
    }
}

int main() {
    int N = 1024;  // Rozmiar tablicy
    int R = 10;    // Liczba iteracji
    int k = 2;     // Potęga

    // Alokacja pamięci
    float* input = new float[N * N];
    float* output = new float[N * N]();

    // Wypełnienie tablicy wejściowej przykładowymi danymi
    for (int i = 0; i < N * N; ++i) {
        input[i] = static_cast<float>(i % 100 + 1);
    }

    // Wykonanie obliczeń na CPU
    cpu_computation(input, output, N, R, k);

    // Zwolnienie pamięci
    delete[] input;
    delete[] output;

    return 0;
}


// Uruchomienie programu: Ctrl + F5 lub menu Debugowanie > Uruchom bez debugowania
// Debugowanie programu: F5 lub menu Debugowanie > Rozpocznij debugowanie

// Porady dotyczące rozpoczynania pracy:
//   1. Użyj okna Eksploratora rozwiązań, aby dodać pliki i zarządzać nimi
//   2. Użyj okna programu Team Explorer, aby nawiązać połączenie z kontrolą źródła
//   3. Użyj okna Dane wyjściowe, aby sprawdzić dane wyjściowe kompilacji i inne komunikaty
//   4. Użyj okna Lista błędów, aby zobaczyć błędy
//   5. Wybierz pozycję Projekt > Dodaj nowy element, aby utworzyć nowe pliki kodu, lub wybierz pozycję Projekt > Dodaj istniejący element, aby dodać istniejące pliku kodu do projektu
//   6. Aby w przyszłości ponownie otworzyć ten projekt, przejdź do pozycji Plik > Otwórz > Projekt i wybierz plik sln
