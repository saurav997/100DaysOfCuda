#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

// Define input size
#define INPUT_SIZE 5

// Error handling macro for CUDA and cuDNN
#define CHECK_CUDA(call)  { 
    cudaError_t err = call; 
    if (err != cudaSuccess) 
    { 
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    } 
}

#define CHECK_CUDNN(call) { 
    cudnnStatus_t err = call; 
    if (err != CUDNN_STATUS_SUCCESS) 
    { 
        std::cerr << "cuDNN error in " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE); 
    } 
}

int main() {
    // 1️⃣ Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 2️⃣ Define input data (ReLU will be applied to this)
    float h_input[INPUT_SIZE] = {-1.0f, 0.5f, 2.0f, -3.0f, 4.0f};  // Example values
    float h_output[INPUT_SIZE];  // To store the result

    // 3️⃣ Allocate GPU memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, INPUT_SIZE * sizeof(float)));

    // 4️⃣ Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // 5️⃣ Create Tensor Descriptors
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, INPUT_SIZE));

    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, INPUT_SIZE));

    // 6️⃣ Create Activation Descriptor for ReLU
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    // 7️⃣ Perform Activation Function (ReLU)
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));

    // 8️⃣ Copy Result Back to Host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // 9️⃣ Print Results
    std::cout << "Input:  ";
    for (int i = 0; i < INPUT_SIZE; i++) std::cout << h_input[i] << " ";
    std::cout << "\nOutput: ";
    for (int i = 0; i < INPUT_SIZE; i++) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    // 🔟 Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
