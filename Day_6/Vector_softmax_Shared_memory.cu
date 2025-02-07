#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void Softmax_Shared_Memory(const float* input, float* output, int size) {
    int numThreads = blockDim.x*gridDim.x;
    int numElementsPerThread = (size + numThreads - 1) / numThreads;
    int start_index = threadIdx.x * numElementsPerThread;
    int end_index = fmin(size, start_index + numElementsPerThread);

    if (start_index >= size) return;  // Avoid out-of-bounds threads

    // Use `extern __shared__` for dynamic shared memory allocation
    extern __shared__ float shared_data[];
    float* shared_max_val = shared_data;  // First half for max values
    float* shared_sum_exp = shared_data + numThreads;  // Second half for sum values

    float max_val = -INFINITY;
    float sum = 0.0f;

    // Step 1: Compute local max for numerical stability
    for (int i = start_index; i < end_index; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    shared_max_val[threadIdx.x] = max_val;
    __syncthreads();

    // Step 2: Find the global max using all threads
    if (threadIdx.x == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < numThreads; i++) {
            max_val = fmaxf(max_val, shared_max_val[i]);
        }
        shared_max_val[0] = max_val;  // Store global max in shared memory
    }
    __syncthreads();

    // Step 3: Compute exponentials and sum
    max_val = shared_max_val[0];  // All threads use the global max
    for (int i = start_index; i < end_index; i++) {
        sum += expf(input[i] - max_val);  // Use expf for single-precision
    }
    shared_sum_exp[threadIdx.x] = sum;
    __syncthreads();

    // Step 4: Compute total sum
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < numThreads; i++) {
            sum += shared_sum_exp[i];
        }
        shared_sum_exp[0] = sum;  // Store global sum in shared memory
    }
    __syncthreads();

    // Step 5: Compute final softmax values
    sum = shared_sum_exp[0];  // All threads use the global sum
    for (int i = start_index; i < end_index; i++) {
        output[i] = expf(input[i] - max_val) / sum;  // Use expf for single-precision
    }
}

int main() {
    int size = 8;
    int space = size * sizeof(float);
    float h_input[] = {2.23, 2.33, 3.14, 4.15, 5.6, 6.17, 7.8, 1.9};

    float* h_output = new float[size];  // Use dynamic allocation

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, space);
    cudaMalloc((void**)&d_output, space);

    cudaMemcpy(d_input, h_input, space, cudaMemcpyHostToDevice);

    dim3 blockDim(4);  // Number of threads per block
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);  // Compute correct grid size

    // Specify shared memory size in kernel launch
    Softmax_Shared_Memory<<<gridDim, blockDim, 2 * blockDim.x * sizeof(float)>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, space, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Results: " << std::endl;
    std::cout << "Original vector: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    long double sum = 0.0f;
    std::cout << "Softmax result: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_output[i] << " ";
        sum += h_output[i];
    }
    std::cout << std::endl;
    std::cout<<"the sum of Softmaxes: "<<sum<<std::endl;
    //Free memory
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
