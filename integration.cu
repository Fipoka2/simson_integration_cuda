#include "integration.h"
#include <math_functions.h>
#include <math.h>

namespace {
    const unsigned int MAX_THREADS = 1024;
    const unsigned int MAX_BLOCKS = 2147483647;
};

__constant__  float LEFT;
__constant__  float RIGHT;
__constant__  int SEGMENTS;
__constant__  float STEP;

float formula(float x) {
    return log10f(x);
};


__global__ void reduce(float *g_odata) {

    __shared__ float sdata[1024];

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= SEGMENTS) {
        sdata[threadIdx.x] = 0;
    }
    if (i < SEGMENTS ) {

        float val = log10f(LEFT + STEP * (i+1));
        sdata[threadIdx.x] =  (i+1)%2 == 0 ? 2 * val : 4 * val;

        __syncthreads();
        // do reduction in shared mem
        for (int s=1; s < blockDim.x; s *=2)
        {
            int index = 2 * s * threadIdx.x;

            if (index < blockDim.x)
            {
                sdata[index] += sdata[index + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem
        if (threadIdx.x == 0)
            atomicAdd(g_odata,sdata[0]);
    }
}

Result integrateOnGpu(float left, float right, int segments, float step) {

    cudaEvent_t start, stop;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    cudaEventRecord ( start, 0 );

    float* dev_result;
//    int seg1 = segments-1;

    cudaMalloc((void **) &dev_result, sizeof(float));
//    cudaMemcpyToSymbol(LEFT, &left, sizeof(float));
//    cudaMemcpyToSymbol(RIGHT, &right, sizeof(float));
//    cudaMemcpyToSymbol(SEGMENTS, &seg1, sizeof(int));
//    cudaMemcpyToSymbol(STEP, &step, sizeof(float));
    cudaSetDevice(0);
    const unsigned int numBlocks = (segments-1) / MAX_THREADS + 1;
    reduce <<< numBlocks, MAX_THREADS >>> (dev_result);

    float* result = new float(0);

    cudaMemcpy(result, dev_result, sizeof(float) , cudaMemcpyDeviceToHost);
    cudaFree(dev_result);
    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );
    cudaDeviceSynchronize();

    Result r;
    cudaEventElapsedTime ( &(r.time), start, stop );
    r.value = (step / 3) * (formula(left) + formula(right) + (*result));

    free(result);
    return r;
}

void setGPUConstants (float left, float right, int segments, float step) {
    int seg1 = segments-1;

    cudaMemcpyToSymbol(LEFT, &left, sizeof(float));
    cudaMemcpyToSymbol(RIGHT, &right, sizeof(float));
    cudaMemcpyToSymbol(SEGMENTS, &seg1, sizeof(int));
    cudaMemcpyToSymbol(STEP, &step, sizeof(float));
}


