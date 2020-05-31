// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU
    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}

inline int factorial(int n) {
  int product = 1;
  for(int i = 1; i < n; ++i ) {
      product *= i;
  }
  return product;
}

kernel void maclaurin_cos(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
  
    float approximate = 0;
    for(int i = 0; i < 10; i++) {
        float x = inVector[id];
        float coef = pow(float(-1), i);
        int num = pow(x, 2*i);
        int denom = factorial(2*i);
        approximate += coef * (num/denom);
        i++;
    }
    outVector[id] = approximate;
}

inline float f(const float x) {
    float approximate = 0;
    for(int coeff = 1; coeff < 100; coeff+=2) {
        approximate += ((float)1/coeff)*sin(coeff*x);
    }
    return approximate;
}

constant float delta = 1e-4;

kernel void differential(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    
    float x = inVector[id];
    outVector[id] = (f(x+delta) - f(x-delta)) / 2.0f*delta;
}