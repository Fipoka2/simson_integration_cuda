//
// Created by fipoka2 on 16.01.19.
//

#ifndef SIMSON_INTEGRATION_CUDA_INTEGRATION_H
#define SIMSON_INTEGRATION_CUDA_INTEGRATION_H

#include <device_functions.h>
#include <stdio.h>
#include <cuda_runtime.h>

struct Result {
    float time;
    float value;
};


Result integrateOnGpu(float left, float right, int segments, float step);
void setGPUConstants (float left, float right, int segments, float step);
#endif //SIMSON_INTEGRATION_CUDA_INTEGRATION_H
