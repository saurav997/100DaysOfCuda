#include <iostream>
#include <cuda_runtime.h>

// Kernel function to perform element-wise vector addition
__global__ void vector_add(const float* a, const float* b, float* c, int length)
{
    // Calculate the global index of the current thread
    int len = blockDim.x * blockIdx.x + threadIdx.x;

    // Perform vector addition only if the index is within bounds
    if (len < length)
    {
        c[len] = a[len] + b[len];
    }
}

int main()
{
    // Length of the input vectors
    int length = 16;

    // Size in bytes for the vectors
    int size = length * sizeof(float);

    // Host input vectors (initialized with example values)
    float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[length]; // Host output vector

    // Device pointers for input and output vectors
    float* d_A, * d_B, * d_C;

    // Allocate memory on the device (GPU) for the input and output vectors
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input vectors from host memory (CPU) to device memory (GPU)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block (blockSize) and calculate the number of blocks
    int blockSize = 256;
    int numBlocks = 1 + length / blockSize; // Ensure all elements are covered

    // Launch the kernel to perform vector addition on the GPU
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, length);

    // Copy the result vector from device memory back to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result vector
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < length; ++i)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Deallocate dynamically allocated host memory (although `h_A` and `h_B` don't need `delete[]`)
    delete[] h_A; // Redundant since `h_A` is stack-allocated
    delete[] h_B; // Redundant since `h_B` is stack-allocated
    delete[] h_C; // Redundant since `h_C` is stack-allocated

    // Free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
w