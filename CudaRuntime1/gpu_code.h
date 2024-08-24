#pragma once
#ifndef GPU_CODE_H
#define GPU_CODE_H

#include <cuda_runtime.h>

cudaError_t computeSumGPU(const float* TAB, float* OUT, int N, int R);

#endif
