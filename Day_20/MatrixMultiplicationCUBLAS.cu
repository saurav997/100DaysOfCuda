#include <iostream>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding rand()
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 4  // Rows in A and C
#define K 3  // Columns in A, Rows in B
#define N 5  // Columns in B and C

// ✅ Function to Initialize Matrix with Random Values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand() % 10);  // Random values [0,9]
    }
}

// ✅ Function to Print Matrices
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << "\nMatrix " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // ✅ Step 1: Initialize cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ✅ Step 2: Allocate Host Memory & Initialize Random Matrices
    float A[M * K], B[K * N], C[M * N] = {0};  // Initialize C with 0
    srand(time(0));  // Seed for randomness
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // ✅ Step 3: Allocate GPU Memory (Device)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // ✅ Step 4: Copy Matrices from Host to Device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // ✅ Step 5: Define GEMM Parameters
    float alpha = 1.0f;  // Scaling factor for A * B
    float beta = 0.0f;   // Scaling factor for C (C = αAB + βC)

    // ✅ Step 6: Perform Matrix Multiplication using cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,  // Matrix dimensions
                &alpha,    // Scaling factor α
                d_A, M,    // A matrix and leading dimension
                d_B, K,    // B matrix and leading dimension
                &beta,     // Scaling factor β
                d_C, M);   // Output matrix and leading dimension

    // ✅ Step 7: Copy Result from Device to Host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ✅ Step 8: Print Matrices
    printMatrix(A, M, K, "A");
    printMatrix(B, K, N, "B");
    printMatrix(C, M, N, "C (Result of A * B)");

    // ✅ Step 9: Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
