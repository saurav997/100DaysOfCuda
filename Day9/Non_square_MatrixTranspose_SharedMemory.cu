#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For seeding rand()

#define BLOCK_SIZE 4  // Define block size for tiling (adjust as needed)

// CUDA Kernel for Matrix Transpose using Shared Memory
__global__ void matrixTransposeShared(const float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // Shared memory with padding to prevent bank conflicts

    // Compute global and local thread indices
    int x_in = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // Column index in input matrix
    int y_in = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // Row index in input matrix

    int x_out = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Column index in output (transposed)
    int y_out = blockIdx.x * BLOCK_SIZE + threadIdx.y; // Row index in output (transposed)

    // Load input into shared memory (coalesced read)
    if (x_in < width && y_in < height) {
        tile[threadIdx.y][threadIdx.x] = input[y_in * width + x_in];
    }

    __syncthreads();  // Synchronize threads before writing to global memory

    // Store transposed data from shared memory to global memory (coalesced write)
    if (x_out < height && y_out < width) {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int width = 8, height = 18;  // Define matrix dimensions
    int size = width * height * sizeof(float);

    // Seed random number generator
    std::srand(std::time(0));

    // Allocate memory on host
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];

    // Initialize matrix with random numbers
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(std::rand() % 100);  // Random values from 0 to 99
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
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << h_input[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTransposed Matrix (First 5 Rows):\n";
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            std::cout << h_output[i * height + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
