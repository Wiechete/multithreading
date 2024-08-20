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

void version_C(float **A, float **B, float **C, int n, int r) {
    #pragma omp parallel
    {
        for (int i = 0; i < n; i += r) { // wszystkie wiersze blok�w
            for (int j = 0; j < n; j += r) { // kolejny wiersz blok�w
                for (int k = 0; k < n; k += r) { // wynik bloku RxR
                    #pragma omp for
                    for (int ii = i; ii < i + r; ii++) { // wynik cz�ciowy blok
                        for (int kk = k; kk < k + r; kk++) {
                            for (int jj = j; jj < j + r; jj++) {
                                C[ii][jj] += A[ii][kk] * B[kk][jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <block_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int r = atoi(argv[2]);

    float **A = allocate_matrix(n);
    float **B = allocate_matrix(n);
    float **C = allocate_matrix(n);

    initialize_matrices(A, B, C, n);

    double start_time = omp_get_wtime();
    version_C(A, B, C, n, r);
    double end_time = omp_get_wtime();
    printf("Version C Time: %f seconds\n", end_time - start_time);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}

