#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to add two matrices element-wise
__global__ void matrix_add(const float* a, const float* b, float* c, int rows, int cols) {
    // Calculate row and column indices for this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within matrix bounds
    if (row < rows && col < cols) {
        int index = row * cols + col; // Compute linear index for row-major order
        c[index] = a[index] + b[index]; // Perform element-wise addition
    }
}

int main() {
    // Define matrix dimensions and size in bytes
    int rows = 4, cols = 4;
    int size = rows * cols * sizeof(float);

    // Initialize host matrices A, B, and result matrix C
    float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[rows * cols];

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host matrices A and B to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions for kernel launch
    dim3 blockDim(2, 2); // 2x2 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y); // Grid covers entire matrix

    // Launch kernel to perform matrix addition
    matrix_add<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result matrix C
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_C[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}