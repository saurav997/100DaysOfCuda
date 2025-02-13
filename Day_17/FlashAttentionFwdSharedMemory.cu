#include <iostream>
#include <cuda_runtime.h>
#include <cmath>


#define BLOCK_SIZE 32  // Tile size for shared memory optimization

// CUDA Kernel for FlashAttention-2 Forward Pass with Shared Memory
__global__ void FlashAttention_Shared(const float* __restrict__ Q, 
                                      const float* __restrict__ K, 
                                      const float* __restrict__ V, 
                                      float* __restrict__ O, 
                                      float* __restrict__ L,
                                      int N, int d, int T_r, int T_c) {

    // Compute block sizes
    int B_r = (N + T_r - 1) / T_r;  // Rows per block
    int B_c = (N + T_c - 1) / T_c;  // Columns per block

    // Determine thread block
    int i = blockIdx.x;  // Each block computes one O_i
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Out-of-bounds check, applied once
    if (thread_x >= B_r || thread_y >= d) return;

    // Allocate Shared Memory
    __shared__ float Q_i[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float K_j[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_j[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float O_i[BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ float L_i[BLOCK_SIZE]; 
    __shared__ float m_i[BLOCK_SIZE];  

    // Initialize shared memory
    Q_i[thread_x][thread_y] = Q[i * B_r * d + thread_x * d + thread_y];
    O_i[thread_x][thread_y] = 0.0f;

    L_i[thread_x] = 0.0f;
    m_i[thread_x] = -INFINITY;
    __syncthreads();

    // Iterate over K_j, V_j blocks
    for (int j = 0; j < T_c; ++j) {
        K_j[thread_x][thread_y] = K[j * B_c * d + thread_x * d + thread_y];
        V_j[thread_x][thread_y] = V[j * B_c * d + thread_x * d + thread_y];
        __syncthreads();

        // Compute attention scores S_i_j = Q_i * K_j^T
        float S_ij = 0.0f;
        for (int k = 0; k < d; ++k) {
            S_ij += Q_i[thread_x][k] * K_j[thread_y][k];
        }
        __syncthreads();

        // Compute max for numerical stability
        float max_val = max(m_i[thread_x], S_ij);
        m_i[thread_x] = max_val;
        __syncthreads();

        // Compute softmax scaling
        float exp_Sij = expf(S_ij - max_val);
        float P_ij = exp_Sij;
        L_i[thread_x] = expf(m_i[thread_x] - max_val) * L_i[thread_x] + exp_Sij;
        __syncthreads();

        // Compute O_i update: O_i = diag(L_i^-1) * O_i + P_ij * V_j
        float O_update = 0.0f;
        for (int k = 0; k < d; ++k) {
            O_update += P_ij * V_j[thread_y][k];
        }
        O_i[thread_x][thread_y] += O_update;
        __syncthreads();
    }

    // Final scaling step
    O_i[thread_x][thread_y] *= (1.0f / L_i[thread_x]);
    __syncthreads();

    // Write back to global memory
    O[i * B_r * d + thread_x * d + thread_y] = O_i[thread_x][thread_y];
    L[i * B_r + thread_x] = m_i[thread_x] + logf(L_i[thread_x]);
}
