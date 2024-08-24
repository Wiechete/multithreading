#include "cpu_code.h"

void computeSumCPU(const std::vector<float>& TAB, std::vector<float>& OUT, int N, int R) {
    int OUT_size = N - 2 * R;

    for (int i = 0; i < OUT_size; ++i) {
        for (int j = 0; j < OUT_size; ++j) {
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
}
