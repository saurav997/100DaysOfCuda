#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// 🔥 Define Function Types
enum FunctionType { RASTRIGIN, SPHERE };

// 🔥 Device Rastrigin Function
__device__ double rastriginFunction(double x) {
    return 10.0 + x * x - 10.0 * cos(2.0 * M_PI * x);
}

// 🔥 Device Sphere Function (Alternative Function)
__device__ double sphereFunction(double x) {
    return x * x;
}

// 🔥 Function Selector
__device__ double evaluateFunction(double x, FunctionType funcType) {
    switch (funcType) {
        case RASTRIGIN: return rastriginFunction(x);
        case SPHERE: return sphereFunction(x);
        default: return 0.0;
    }
}

// 🔥 CUDA Kernel for Gradient Computation
__global__ void computeGradient(const double* x, double* grad, int n, FunctionType funcType) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    double h = 1e-5;
    double x_orig = x[index];

    double x_forward = x_orig + h;
    double x_backward = x_orig - h;

    // ✅ Use Function Selector
    grad[index] = (evaluateFunction(x_forward, funcType) - evaluateFunction(x_backward, funcType)) / (2.0 * h);

    printf("Thread %d | x: %f | grad: %f\n", index, x_orig, grad[index]);
}

int main() {
    int n = 20;

    // ✅ Allocate Unified Memory
    double* x;
    cudaMallocManaged(&x, n * sizeof(double));

    double* grad;
    cudaMallocManaged(&grad, n * sizeof(double));

    // ✅ Initialize Random Values
    srand(time(0));
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(rand()) / RAND_MAX * 10.0 - 5.0; // [-5,5] Range
    }

    // ✅ Compute Grid and Block Sizes
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    // ✅ Choose Function Type (RASTRIGIN or SPHERE)
    FunctionType selectedFunction = RASTRIGIN;

    // ✅ Launch CUDA Kernel with Function Type
    computeGradient<<<gridSize, blockSize>>>(x, grad, n, selectedFunction);
    cudaDeviceSynchronize();  // Ensure all GPU threads finish

    // ✅ Print Results
    std::cout << "\n🔥 Calculated Gradients:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "Gradient[" << i << "] = " << grad[i] << std::endl;
    }

    // ✅ Free GPU Memory
    cudaFree(x);
    cudaFree(grad);

    return 0;
}
