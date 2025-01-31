#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Tiling factor for shared memory optimization

// CUDA kernel for optimized matrix multiplication (m × n) * (n × k) = (m × k)
__global__ void matrixMulOptimized(const float *A, const float *B, float *C, int m, int n, int k) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (n + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // Load tiles into shared memory (check for boundaries)
        if (row < m && (tileIdx * TILE_SIZE + threadIdx.x) < n)
            A_shared[threadIdx.y][threadIdx.x] = A[row * n + (tileIdx * TILE_SIZE + threadIdx.x)];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < k && (tileIdx * TILE_SIZE + threadIdx.y) < n)
            B_shared[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * k + col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Ensure all threads have loaded their data

        // Compute multiplication using tile
        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += A_shared[threadIdx.y][t] * B_shared[t][threadIdx.x];
        }

        __syncthreads();  // Ensure all computations are completed before loading the next tile
    }

    // Store the result
    if (row < m && col < k)
        C[row * k + col] = sum;
}

int main() {
    int m = 512, n = 1024, k = 512;  // Example matrix sizes
    int size_A = m * n * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);

    float *h_A = new float[m * n];
    float *h_B = new float[n * k];
    float *h_C = new float[m * k];

    // Initialize matrices with random values
    for (int i = 0; i < m * n; i++) {
        h_A[i] = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < n * k; i++) {
        h_B[i] = static_cast<float>(rand() % 10);
    }

    float *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy input matrices to device memory
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    // Launch optimized kernel
    matrixMulOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print a small part of the result matrix for verification
    std::cout << "Result Matrix C (partial output for verification):\n";
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << h_C[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU and CPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
