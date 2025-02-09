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

// 🔥 Device Rastrigin Function (for CUDA kernels)
__device__ double rastriginFunctionDevice(const double* x, int n) {
    printf("\n Inside rastrigin function");
    double sum = 10.0 * n;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

// 🔥 Host Rastrigin Function (for CPU computations)
__host__ double rastriginFunctionHost(const double* x, int n) {
    double sum = 10.0 * n;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    }
    return sum;
}

// CUDA Kernel for Gradient Computation
__global__ void computeGradient(const double* x, double* grad, int n) {
    printf("\n Inside computegrad function");
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    
    double h = 1e-5;
    double x_orig = x[index];

    double x_forward = x_orig + h;
    double x_backward = x_orig - h;

    grad[index] = (rastriginFunctionDevice(&x_forward, 1) - rastriginFunctionDevice(&x_backward, 1)) / (2.0 * h);
}

// CUDA Kernel to Initialize Random Points
__global__ void initializePoints(double* x, int n, unsigned long seed) {
    printf("\n Inside init function");
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    curandState state;
    curand_init(seed, index, 0, &state);
    x[index] = curand_uniform(&state) * 10.0 - 5.0; // Random values in range [-5,5]
}

// CUDA Kernel to Update Hessian Approximation
__global__ void updateHessian(double* H, const double* s, const double* y, double rho, int n) {
    printf("\n Inside update hessian function");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    double I = (i == j) ? 1.0 : 0.0;
    H[i * n + j] = I - rho * s[i] * y[j];
}

// 🔥 BFGS Optimization (host function)
void bfgsOptimization(double* x, double* H, int n) {
    printf("\n Inside BFGS function");
    double *grad, *d, *x_new, *grad_new, *s, *y;
    cudaMallocManaged(&grad, n * sizeof(double));
    cudaMallocManaged(&d, n * sizeof(double));
    cudaMallocManaged(&x_new, n * sizeof(double));
    cudaMallocManaged(&grad_new, n * sizeof(double));
    cudaMallocManaged(&s, n * sizeof(double));
    cudaMallocManaged(&y, n * sizeof(double));

    // Initialize Hessian approximation to identity matrix
    for (int i = 0; i < n * n; i++) {
        printf("\n Inside BFGS Loop1");
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        printf("\n Inside BFGS Loop2");
        // Compute Gradient
        computeGradient<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, grad, n);
        cudaDeviceSynchronize();

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) grad_norm += grad[i] * grad[i];
        if (sqrt(grad_norm) < EPSILON) break;

        // Compute Search Direction: d = -H * grad
        for (int i = 0; i < n; i++) {
            printf("\n Inside BFGS Loop3");
            d[i] = 0.0;
            for (int j = 0; j < n; j++) {
                d[i] -= H[i * n + j] * grad[j];
            }
        }

        // Backtracking Line Search (BTLS)
        double alpha = 1.0;
        while (true) {
            printf("\n Inside BFGS Loop4");
            for (int i = 0; i < n; i++) x_new[i] = x[i] + alpha * d[i];

            computeGradient<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x_new, grad_new, n);
            cudaDeviceSynchronize();

            // ✅ Use `rastriginFunctionHost` instead of `rastriginFunctionDevice`
            double f_x = rastriginFunctionHost(x, n);
            double f_x_new = rastriginFunctionHost(x_new, n);

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
        updateHessian<<<gridDim, dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(H, s, y, rho, n);
        cudaDeviceSynchronize();

        // ✅ Print intermediate `x` values
        std::cout << "Iteration " << iter + 1 << ": x = [ ";
        for (int i = 0; i < n; i++) std::cout << x_new[i] << " ";
        std::cout << "]\n";

        // Update x
        for (int i = 0; i < n; i++) x[i] = x_new[i];
    }

    cudaFree(grad);
    cudaFree(d);
    cudaFree(x_new);
    cudaFree(grad_new);
    cudaFree(s);
    cudaFree(y);
}


int main() {
    printf("\nStarted");
    int n = 10;
    double *x, *H;

    cudaMallocManaged(&x, n * sizeof(double));
    cudaMallocManaged(&H, n * n * sizeof(double));

    // Initialize x randomly
    initializePoints<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, n, time(NULL));
    cudaDeviceSynchronize();

    // Run BFGS Optimization
    bfgsOptimization(x, H, n);

    std::cout << "Optimized x: ";
    for (int i = 0; i < n; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    cudaFree(x);
    cudaFree(H);
    return 0;
}
