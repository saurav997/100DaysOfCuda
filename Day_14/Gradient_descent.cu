#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define LEARNING_RATE 0.001  // Learning rate α
#define MAX_ITER 100000      // Number of iterations
#define EPSILON 1e-7         // Convergence threshold

//  Define Function Types
enum FunctionType { RASTRIGIN, SPHERE };

//  Device Rastrigin Function
__device__ double rastriginFunction(double x) {
    return 10.0 + x * x - 10.0 * cos(2.0 * M_PI * x);
}

//  Device Sphere Function (Alternative Function)
__device__ double sphereFunction(double x) {
    return x * x;
}

//  Function Selector
__device__ double evaluateFunction(double x, FunctionType funcType) {
    switch (funcType) {
        case RASTRIGIN: return rastriginFunction(x);
        case SPHERE: return sphereFunction(x);
        default: return 0.0;
    }
}

//  CUDA Kernel for Gradient Computation
__global__ void computeGradient(const double* x, double* grad, int n, FunctionType funcType) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    double h = 1e-5;
    double x_orig = x[index];

    double x_forward = x_orig + h;
    double x_backward = x_orig - h;

    //  Use Function Selector
    grad[index] = (evaluateFunction(x_forward, funcType) - evaluateFunction(x_backward, funcType)) / (2.0 * h);
}

// CUDA Kernel for Gradient Descent Step
__global__ void updateX(double* x, const double* grad, int n, double alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    //  Update rule: x = x - α * grad
    x[index] -= alpha * grad[index];
}

// Corrected CUDA Kernel for L2 Norm Computation (Parallel Reduction)
__global__ void L2_Norm(const double* x, int n, double* x_norm) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    //  Use atomicAdd to prevent race conditions
    atomicAdd(x_norm, x[index] * x[index]);  // Sum of squares
}

//  Gradient Descent Optimization
void gradientDescent(double* x, int n, FunctionType funcType) {
    double* grad;
    cudaMallocManaged(&grad, n * sizeof(double));

    double* d_gradNorm;
    cudaMallocManaged(&d_gradNorm, sizeof(double));

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        //  Compute Gradient
        computeGradient<<<gridSize, blockSize>>>(x, grad, n, funcType);
        cudaDeviceSynchronize();

        //  Compute L2 Norm of Gradient (Stopping Criteria)
        *d_gradNorm = 0.0;  // Reset before launching kernel
        L2_Norm<<<gridSize, blockSize>>>(grad, n, d_gradNorm);
        cudaDeviceSynchronize();

        double gradNorm = sqrt(*d_gradNorm);  // Compute √sum(grad²)

        //  Stop if Converged
        if (gradNorm < EPSILON) {
            std::cout << " Converged at iteration " << iter << " with gradNorm = " << gradNorm << std::endl;
            break;
        }

        //  Update x using Gradient Descent Rule
        updateX<<<gridSize, blockSize>>>(x, grad, n, LEARNING_RATE);
        cudaDeviceSynchronize();

        //  Print Intermediate Values
        if (iter % 100 == 0) {  // Print every 100 iterations
            std::cout << "Iteration " << iter << ": x[0] = " << x[0] << ", GradNorm = " << gradNorm << std::endl;
        }
    }

    cudaFree(grad);
    cudaFree(d_gradNorm);
}

int main() {
    int n = 20;

    // Allocate Unified Memory for x
    double* x;
    cudaMallocManaged(&x, n * sizeof(double));

    // Initialize Random Values
    srand(time(0));
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(rand()) / RAND_MAX * 10.0 - 5.0; // [-5,5] Range
    }

    // Choose Function Type (RASTRIGIN or SPHERE)
    FunctionType selectedFunction = SPHERE;

    // Run Gradient Descent Optimization
    gradientDescent(x, n, selectedFunction);

    // Print Final Values
    std::cout << "\n Optimized x values:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    // Free GPU Memory
    cudaFree(x);

    return 0;
}
