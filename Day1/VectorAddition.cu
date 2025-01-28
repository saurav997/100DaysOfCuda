#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float*c, int length)
{
    int len = blockDim.x*blockIdx.x +threadIdx.x;
    if(len<length)
    {
        c[len] = a[len] + b [len];
    }
}

int main()
{
    int length = 16;
    int size = length*sizeof(float);
    float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[length];

    float*d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = 1+length/blockSize;
    vector_add<<<numBlocks,blockSize>>>(d_A,d_B,d_C,length);
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    // print result
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < length; ++i) 
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}