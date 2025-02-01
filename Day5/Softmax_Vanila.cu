#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void Softmax(float* input, float* output, int rows, int cols) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < rows) {
        float max_val = -INFINITY;
        float sum = 0.0f;

        // Compute max value for numerical stability
        for (int i = 0; i < cols; i++) {
            max_val = fmaxf(max_val, input[row * cols + i]);
        }

        // Compute sum of exponentials
        for (int i = 0; i < cols; i++) {
            sum += expf(input[row * cols + i] - max_val);
        }

        // Compute final softmax values
        for (int i = 0; i < cols; i++) {
            output[row * cols + i] = expf(input[row * cols + i] - max_val) / sum;
        }
    }
}

int main() {
    int rows = 6;
    int cols = 3;
    int size = rows * cols * sizeof(float);

    float *h_input = new float[rows * cols] {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float *h_output = new float[rows * cols];

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x);

    Softmax<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print output for verification
    std::cout << "Softmax (Global Memory) - First 5 elements: ";
    for (int i = 0; i < 5; i++) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
