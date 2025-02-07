#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For seeding rand()

#define BLOCK_SIZE 4  // Define block size for tiling (adjust as needed)

__global__ void matrixTransposeBatched(const float* Input, float* Output, int rows, int cols, int batch_size)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // Shared memory for tiling

    int in_col = blockDim.x * blockIdx.x + threadIdx.x;
    int in_row = blockDim.y * blockIdx.y + threadIdx.y;
    int out_col = in_row;
    int out_row = in_col;
    int out_rows = cols;
    int out_cols = rows;
    int batch = blockIdx.z; // Each batch processes one matrix

    // Bounds check to prevent out-of-bounds memory access
    if (in_row >= rows || in_col >= cols || batch >= batch_size)
    {
        return;
    }

    // Load input matrix tile into shared memory
    tile[threadIdx.y][threadIdx.x] = Input[batch * rows * cols + in_row * cols + in_col];

    __syncthreads(); // Ensure all threads finish loading shared memory

    // Store transposed tile into global memory
    Output[batch * rows * cols + out_row*out_cols + out_col] = tile[threadIdx.y][threadIdx.x];

    // No need for __syncthreads() after writing to global memory
}



int main()
{
    int rows = 8;
    int cols = 10;
    int batch_size = 10;
    int out_rows = cols;
    int out_cols = rows;
    // Seed random number generator
    std::srand(std::time(0));

    int size = cols * rows * batch_size * sizeof(float);

    // Allocate memory on host
    float *h_input = new float[rows * cols * batch_size];
    float *h_output = new float[cols * rows * batch_size];

    // Initialize matrix batch with random numbers
    for (int i = 0; i < rows * cols * batch_size; i++) {
        h_input[i] = static_cast<float>(std::rand() % 100);  // Random values from 0 to 99
    }

    // Allocate memory on device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input batch to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);

    // Launch the kernel
    matrixTransposeBatched<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, batch_size);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print a small part of the result for verification
    for(int k=0;k<batch_size;k++)
    {
        std::cout << "Original Matrix Batch:\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << h_input[ k * rows * cols + i*cols + j] << "\t";
            }
            std::cout << std::endl;
        }

        std::cout << "\nTransposed Matrix Batch:\n";
        for (int i1 = 0; i1 < out_rows; i1++) {
            for (int j1 = 0; j1 < out_cols; j1++) {
                std::cout << h_output[k * rows * cols + i1*out_cols + j1] << "\t";
            }
            std::cout << std::endl;
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


₹