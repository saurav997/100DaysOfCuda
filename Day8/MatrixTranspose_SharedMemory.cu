#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16  // Define block size for tiling

// CUDA Kernel for Matrix Transpose using Shared Memory
__global__ void matrixTransposeShared(const float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // Shared memory with padding to avoid bank conflicts

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data from global memory to shared memory (transpose index inside tile)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();  // Ensure all threads complete loading before writing

    // Compute transposed indices
    int transposed_x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Store transposed data from shared memory to global memory
    if (transposed_x < height && transposed_y < width) {
        output[transposed_x * height + transposed_y] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int width = 32, height = 32;  // Define matrix dimensions
    int size = width * height * sizeof(float);

    // Allocate memory on host
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];

    // Initialize matrix with values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate memory on device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixTransposeShared<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print a small part of the result for verification
    std::cout << "Original Matrix (First 5 Rows):\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_input[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTransposed Matrix (First 5 Rows):\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << h_output[i * height + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input
