#include <iostream>
#include <cuda_runtime.h>

__global__ void MatMul(const float* A,const float* B,float* C,int m,int n,int k)
{
  int col = blockDim.x*blockIdx.x+threadIdx.x;
  int row = blockDim.y*blockIdx.y+threadIdx.y;
  float sum = 0;
  if(row<m && col<k)
  {
    for(int i=0;i<n;i++)
    {
      sum += A[row*n + i]*B[i*k + col];  
    }
    C[row*k + col] = sum;
  }
}

int main()
{
  int m = 3;
  int n = 2;
  int k = 4;
  int size_A = m * n * sizeof(float);
  int size_B = n * k * sizeof(float);
  int size_C = m * k * sizeof(float);
  float h_a[] = {1.15,2.313,3.1415,6.28,7.1415,8.192};
  float h_b[] = {5.28,3.14,1.15,2.313,3.1415,6.28,7.1415,0};
  float h_c[m*k];

  float *d_a,*d_b,*d_c;

  cudaMalloc((void**)&d_a,size_A);
  cudaMalloc((void**)&d_b,size_B);
  cudaMalloc((void**)&d_c,size_C);

  cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,size_B,cudaMemcpyHostToDevice);
  dim3 blockDim(2,2);
  dim3 gridDim(((k/blockDim.x)+1),((m/blockDim.y)+1));

  MatMul<<<gridDim,blockDim>>>(d_a,d_b,d_c,m,n,k);
  cudaMemcpy(h_c,d_c,size_C,cudaMemcpyDeviceToHost);
  
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