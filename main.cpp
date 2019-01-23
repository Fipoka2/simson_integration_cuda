#include <iostream>
#include <omp.h>
#include "integration.h"
#include <math.h>

using namespace std;
const int DECISION_VALUE = 40000;

float cpuFormula(float x) {
    return log10f(x);
};


float cpuIntegrate(float left, float right, int segments, float step) {
    float evenSegments = 0;
    float oddSegments = 0;
    #pragma omp parallel for reduction(+: evenSegments, oddSegments) num_threads(2)
    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += cpuFormula(left + step * i) : oddSegments += cpuFormula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (cpuFormula(left) + cpuFormula(right) + evenSegments + oddSegments);
}

Result runOnGPU(float left, float right, int segments, float step);
Result runONCPU(float left, float right, int segments, float step);
Result optimizedVersion(float left, float right, int segments, float step);

int main() {

    const int SEGMENTS = 400000; // should be even number, also named as a 2n
    const float LEFT = 2;
    const float RIGHT = 1202;
    const float STEP = (RIGHT - LEFT) / SEGMENTS;
    setGPUConstants(LEFT,RIGHT,SEGMENTS, STEP);

    Result cpu = runONCPU(LEFT, RIGHT, SEGMENTS, STEP);
    Result gpu = runOnGPU(LEFT, RIGHT, SEGMENTS, STEP);
    Result comb = optimizedVersion(LEFT, RIGHT, SEGMENTS, STEP);

    cout << "CPU. Value: " << cpu.value <<" Time: " << cpu.time << "ms"<< endl;
    cout << "GPU. Value: " << gpu.value <<" Time: " << gpu.time << "ms"<< endl;
    cout << "Comb. Value: " << comb.value <<" Time: " << comb.time << "ms"<< endl;






    return 0;
}

Result runOnGPU(float left, float right, int segments, float step) {
    return integrateOnGpu(left, right, segments, step);
}

Result runONCPU(float left, float right, int segments, float step) {
    double start = omp_get_wtime();
    float value = cpuIntegrate(left,right,segments,step);
    double end = omp_get_wtime();
    auto time = static_cast<float>((end - start) * 1000);
    return {time, value};
}

Result optimizedVersion(float left, float right, int segments, float step) {
    return segments < DECISION_VALUE ?
        runONCPU(left, right, segments, step) : runOnGPU(left, right, segments, step);
}
