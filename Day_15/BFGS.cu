#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define MAX_ITER 1000
#define EPSILON 1e-6
#define TAU 0.5   // Step size reduction factor in BTLS
#define C1 1e-4   // Armijo condition constant
#define C2 0.9    // Wolfe condition constant
#define M_PI 3.14159265358979323846

// 🔥 Device Function for Rastrigin's Function
__device__ double rastriginFunctionDevice(const double* x, int n) {
    double sum = 10.0 * n;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

// 🔥 CUDA Kernel for Function Evaluation
__global__ void evaluateFunction(const double* x, double* f_val, int n) {
    *f_val = rastriginFunctionDevice(x, n);
}

// 🔥 CUDA Kernel for Gradient Computation
__global__ void computeGradient(const double* x, double* grad, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    double h = 1e-5;
    double x_forward = x[index] + h;
    double x_backward = x[index] - h;

    grad[index] = (rastriginFunctionDevice(&x_forward, 1) - rastriginFunctionDevice(&x_backward, 1)) / (2.0 * h);
}

// 🔥 CUDA Kernel to Initialize Random Points
__global__ void initializePoints(double* x, int n, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    curandState state;
    curand_init(seed, index, 0, &state);
    x[index] = curand_uniform(&state) * 10.0 - 5.0; // Random values in range [-5,5]
}

// 🔥 CUDA Kernel to Update Hessian Approximation
__global__ void updateHessian(double* H, const double* s, const double* y, double rho, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    double term1 = (i == j) ? 1.0 : 0.0;
    double term2 = rho * s[i] * y[j];

    H[i * n + j] = term1 - term2;
}

// 🔥 BFGS Optimization (host function)
void bfgsOptimization(double* x, double* H, int n) {
    double *grad, *d, *x_new, *grad_new, *s, *y, *d_f_x, *d_f_x_new;
    
    cudaMallocManaged(&grad, n * sizeof(double));
    cudaMallocManaged(&d, n * sizeof(double));
    cudaMallocManaged(&x_new, n * sizeof(double));
    cudaMallocManaged(&grad_new, n * sizeof(double));
    cudaMallocManaged(&s, n * sizeof(double));
    cudaMallocManaged(&y, n * sizeof(double));
    cudaMallocManaged(&d_f_x, sizeof(double));
    cudaMallocManaged(&d_f_x_new, sizeof(double));

    // Initialize Hessian approximation to identity matrix
    for (int i = 0; i < n * n; i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Compute Gradient
        computeGradient<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, grad, n);
        cudaDeviceSynchronize();

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) grad_norm += grad[i] * grad[i];
        if (sqrt(grad_norm) < EPSILON) break;

        // Compute Search Direction: d = -H * grad
        for (int i = 0; i < n; i++) {
            d[i] = 0.0;
            for (int j = 0; j < n; j++) {
                d[i] -= H[i * n + j] * grad[j];
            }
        }

        // Backtracking Line Search (BTLS)
        double alpha = 1.0;
        while (true) {
            for (int i = 0; i < n; i++) x_new[i] = x[i] + alpha * d[i];

            evaluateFunction<<<1, 1>>>(x, d_f_x, n);
            evaluateFunction<<<1, 1>>>(x_new, d_f_x_new, n);
            cudaDeviceSynchronize();
            double f_x = *d_f_x;
            double f_x_new = *d_f_x_new;

            double gTd = 0.0;
            for (int i = 0; i < n; i++) gTd += grad[i] * d[i];

            if (f_x_new <= f_x + C1 * alpha * gTd) {
                double gTd_new = 0.0;
                for (int i = 0; i < n; i++) gTd_new += grad_new[i] * d[i];
                if (gTd_new >= C2 * gTd) break;
            }

            alpha *= TAU;
            if (alpha < 1e-8) break;
        }

        // Compute s_k = x_new - x
        for (int i = 0; i < n; i++) s[i] = x_new[i] - x[i];

        // Compute y_k = grad_new - grad
        for (int i = 0; i < n; i++) y[i] = grad_new[i] - grad[i];

        // Compute rho_k = 1 / (y_k^T * s_k)
        double ys = 0.0;
        for (int i = 0; i < n; i++) ys += y[i] * s[i];
        double rho = (ys != 0.0) ? (1.0 / ys) : 0.0;

        // Update Hessian
        dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        updateHessian<<<gridDim, blockDim>>>(H, s, y, rho, n);
        cudaDeviceSynchronize();

        // Update x
        for (int i = 0; i < n; i++) x[i] = x_new[i];
    }

    cudaFree(grad);
    cudaFree(d);
    cudaFree(x_new);
    cudaFree(grad_new);
    cudaFree(s);
    cudaFree(y);
    cudaFree(d_f_x);
    cudaFree(d_f_x_new);
}

int main() {
    int n = 10;
    double *x, *H;

    cudaMallocManaged(&x, n * sizeof(double));
    cudaMallocManaged(&H, n * n * sizeof(double));

    initializePoints<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, n, time(NULL));
    cudaDeviceSynchronize();

    bfgsOptimization(x, H, n);

    std::cout << "Optimized x: ";
    for (int i = 0; i < n; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    cudaFree(x);
    cudaFree(H);
    return 0;
}
