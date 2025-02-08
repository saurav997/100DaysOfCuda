#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For seeding rand()

#define BLOCK_SIZE 8  // Define block size for CUDA kernel

// CUDA Kernel for Matrix Convolution
__global__ void MatrixConvolution(const float* Input, const float* Filter, float* Output, 
                                  int channels, int input_height, int input_width, 
                                  int filter_height, int filter_width, int stride) 
{
    channels = 1;
    // Compute output dimensions
    int Output_height = (input_height - filter_height) / stride + 1;
    int Output_width  = (input_width - filter_width) / stride + 1;

    extern __shared__ tile[];
    int tile_height = blockDim.y * stride + filter_height - 1;
    int tile_width  = blockDim.x * stride + filter_width - 1;
    // Compute global row and column indices
    int Output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int Output_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads stay within valid output bounds
    if (Output_row >= Output_height || Output_col >= Output_width) {
        return;
    }

    float sum = 0.0f;
    // Compute input starting index for this block
    int row_start = Output_row * stride;
    int col_start = Output_col * stride;

    // Perform convolution sum over all channels
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < filter_height; i++) {
            for (int j = 0; j < filter_width; j++) {
                int row_index = row_start + i;
                int col_index = col_start + j;
                
                // Ensure input indices are within bounds
                if (row_index < input_height && col_index < input_width) {
                    sum += Input[c * input_height * input_width + row_index * input_width + col_index] 
                           * Filter[c * filter_height * filter_width + i * filter_width + j];
                }
            }
        }
    }

    // Write the computed sum to the output tensor
    Output[Output_row * Output_width + Output_col] = sum;
}

int main()
{
    // Define matrix and filter sizes
    int channels = 3;      // Number of input channels
    int input_height = 8;  // Height of the input matrix
    int input_width = 8;   // Width of the input matrix
    int filter_height = 3; // Filter (kernel) height
    int filter_width = 3;  // Filter (kernel) width
    int stride = 2;        // Stride value

    // Compute output matrix size
    int output_height = (input_height - filter_height) / stride + 1;
    int output_width = (input_width - filter_width) / stride + 1;

    // Seed random number generator
    std::srand(std::time(0));

    // Allocate memory for input, filter, and output on the host
    float *h_input = new float[channels * input_height * input_width];
    float *h_filter = new float[channels * filter_height * filter_width];
    float *h_output = new float[output_height * output_width];

    // Initialize input matrix with random values
    for (int i = 0; i < channels * input_height * input_width; i++) {
        h_input[i] = static_cast<float>(std::rand() % 10);  // Random values between 0 and 9
    }

    // Initialize filter (kernel) with random values
    for (int i = 0; i < channels * filter_height * filter_width; i++) {
        h_filter[i] = static_cast<float>(std::rand() % 10);  // Random values between 0 and 9
    }

    // Allocate memory on GPU
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void**)&d_input, channels * input_height * input_width * sizeof(float));
    cudaMalloc((void**)&d_filter, channels * filter_height * filter_width * sizeof(float));
    cudaMalloc((void**)&d_output, output_height * output_width * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input, channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, channels * filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch CUDA kernel
    MatrixConvolution<<<gridDim, blockDim>>>(d_input, d_filter, d_output, 
                                             channels, input_height, input_width, 
                                             filter_height, filter_width, stride);

    // Copy result back to CPU
    cudaMemcpy(h_output, d_output, output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Input Matrix
    std::cout << "Input Matrix:\n";
    for (int i = 0; i < input_height; i++) {
        for (int j = 0; j < input_width; j++) {
            std::cout << h_input[i * input_width + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Print Filter (Kernel)
    std::cout << "\nFilter (Kernel):\n";
    for (int i = 0; i < filter_height; i++) {
        for (int j = 0; j < filter_width; j++) {
            std::cout << h_filter[i * filter_width + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Print Output Matrix
    std::cout << "\nOutput Matrix (Convolution Result):\n";
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            std::cout << h_output[i * output_width + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    // Free CPU memory
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;

    return 0;
}
