#include <iostream>
#include <cuda_runtime.h>

#define Tile_Size 2

__global__ void SqMatMul(const float* A,const float* B,float* C,int N)
{
  __shared__ float A_shared[Tile_Size][Tile_Size];
  __shared__ float B_shared[Tile_Size][Tile_Size];
  int col = blockDim.x*blockIdx.x+threadIdx.x;
  int row = blockDim.y*blockIdx.y+threadIdx.y;

  float sum = 0;
  int col2, row2;

  for (int tileIdx = 0; tileIdx < (N + Tile_Size - 1) / Tile_Size; ++tileIdx)
  {
    col2 = tileIdx * Tile_Size + threadIdx.x;
    row2 = tileIdx * Tile_Size + threadIdx.y;
    if (row < N && col2 < N)
      A_shared[threadIdx.y][threadIdx.x] = A[row * N + col2];
    else
      A_shared[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && row2 < N)
        B_shared[threadIdx.y][threadIdx.x] = B[row2 * N + col];
    else
        B_shared[threadIdx.y][threadIdx.x] = 0.0f;
    // Synchronize threads before computation
    __syncthreads();  
    // Compute partial sum using shared memory
    for (int t = 0; t < Tile_Size; ++t) 
    {
        sum += A_shared[threadIdx.y][t] * B_shared[t][threadIdx.x];
    }
    // Synchronize threads before loading the next tile
    __syncthreads();  
  }

  // Store result in C
  if (row < N && col < N)
      C[row * N + col] = sum;
}

int main()
{
    int N = 3
    float h_a[] = {1.15,2.313,3.1415,6.28,7.1415,8.192,3,2,1};
    float h_b[] = {5.28,3.14,1.15,2.313,3.1415,6.28,7.1415,0,4};
    float h_c[9];
    int size = N*N*sizeof(float);

    float *d_a,*d_c,*d_b;
    cudaMalloc((void**)d_a,size);
    cudaMalloc((void**)d_b,size);
    cudaMalloc((void**)d_c,size);

    cudaMemCpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemCpy(d_b,h_b,size,cudaMemcpyHostToDevice);
    dim3 blockDim(Tile_Size,Tile_Size);
    dim3 gridDim((N+Tile_Size-1)/Tile_Size,(N+Tile_Size-1)/Tile_Size);
    SqMatMul<<<gridDim,blockDim>>>(d_a,d_b,d_c,N);
    cudaMemCpy(d_c,h_c,size,cudaMemcpyDeviceToHost);
    // Print the result matrix
    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << h_c[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}