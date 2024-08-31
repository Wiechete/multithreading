#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void initialize_matrices(float **A, float **B, float **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (float)(rand() % 10);
            B[i][j] = (float)(rand() % 10);
            C[i][j] = 0.0f;
        }
    }
}

float** allocate_matrix(int n) {
    float **matrix = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float *)malloc(n * sizeof(float));
    }
    return matrix;
}

void free_matrix(float **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void version_A(float **A, float **B, float **C, int n) {
    for (int i = 0; i < n; i++) { // Przechodzi przez wiersze macierzy A
        for (int k = 0; k < n; k++) { // Przechodzi przez kolumny macierzy B
            for (int j = 0; j < n; j++) { // Mno¿y i sumuje odpowiednie elementy wiersza z A i kolumny z B
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    float **A = allocate_matrix(n);
    float **B = allocate_matrix(n);
    float **C = allocate_matrix(n);

    initialize_matrices(A, B, C, n);

    double start_time = omp_get_wtime();
    version_A(A, B, C, n);
    double end_time = omp_get_wtime();
    printf("Version A Time: %f seconds\n", end_time - start_time);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}

