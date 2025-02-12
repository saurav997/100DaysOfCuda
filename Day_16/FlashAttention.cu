#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>  // For random number generation

#define BLOCK_SIZE 16  // Can be tuned for optimization

// ----------------- CUDA Kernel for FlashAttention -----------------
__global__ void FlashAttention(float* Q, float* K, float* V, float* O, float* L, 
                               int N, int d, int T_r, int T_c) 
{
    // Each thread computes one output block O_i (row-block)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T_r) return;

    // Compute block sizes
    int B_r = (N + T_r - 1) / T_r; // ceil(N / T_r)
    int B_c = (N + T_c - 1) / T_c; // ceil(N / T_c)

    // Load current block of Q, O, and L (B_r × d)
    float* Q_i = &Q[i * B_r * d];
    float* O_i = &O[i * B_r * d];
    float* L_i = &L[i * B_r];  // Row-wise logsumexp (B_r × 1)

    // Initialize `m_i` (Row-wise max tracking)
    float m_i[B_r];  
    for (int r = 0; r < B_r; r++) m_i[r] = -INFINITY;

    // Iterate Over Key Blocks (Column-wise Processing)
    for (int j = 0; j < T_c; j++) 
    {
        // Load current block of K and V (B_c × d)
        float* K_j = &K[j * B_c * d];
        float* V_j = &V[j * B_c * d];

        // Store old `m_i` values
        float m_old[B_r];
        for (int r = 0; r < B_r; r++) m_old[r] = m_i[r];

        // Compute Attention Scores S_i_j = Q_i × K_j^T (B_r × B_c)
        float S_i_j[B_r][B_c]; 
        for (int r = 0; r < B_r; r++) 
        {
            for (int c = 0; c < B_c; c++) 
            {
                S_i_j[r][c] = 0.0f;
                for (int k = 0; k < d; k++) 
                {
                    S_i_j[r][c] += Q_i[r * d + k] * K_j[c * d + k];
                }
            }
        }

        // Update m_i: Row-wise max update
        for (int r = 0; r < B_r; r++) 
        {
            float row_max_val = -INFINITY;
            for (int c = 0; c < B_c; c++) row_max_val = max(row_max_val, S_i_j[r][c]);
            m_i[r] = max(m_i[r], row_max_val);  
        }

        // Compute Softmax Scores P_i_j = exp(S_i_j - m_i)
        float P_i_j[B_r][B_c]; 
        for (int r = 0; r < B_r; r++) 
        {
            for (int c = 0; c < B_c; c++) 
            {
                P_i_j[r][c] = expf(S_i_j[r][c] - m_i[r]);  // Broadcasting m_i across columns
            }
        }

        // Compute L_i update: L_i = exp(m_old - m_i) * L_i + row_sum(P_i_j)
        for (int r = 0; r < B_r; r++) 
        {
            float row_sum_P = 0.0f;
            for (int c = 0; c < B_c; c++) row_sum_P += P_i_j[r][c];

            L_i[r] = expf(m_old[r] - m_i[r]) * L_i[r] + row_sum_P;
        }

        // Compute D_i_j (Diagonal scaling matrix)
        float D_i_j[B_r]; 
        for (int r = 0; r < B_r; r++) D_i_j[r] = expf(m_old[r] - m_i[r]);  

        // Update O_i: O_i = D_i_j * O_i + P_i_j * V_j
        float O_i_new[B_r][d]; 
        for (int r = 0; r < B_r; r++) 
        {
            for (int col = 0; col < d; col++) 
            {
                O_i_new[r][col] = D_i_j[r] * O_i[r * d + col];
            }

            for (int c = 0; c < B_c; c++) 
            {
                for (int col = 0; col < d; col++) 
                {
                    O_i_new[r][col] += P_i_j[r][c] * V_j[c * d + col];
                }
            }
        }

        // Copy new O_i values back
        for (int r = 0; r < B_r; r++) 
        {
            for (int col = 0; col < d; col++) 
            {
                O_i[r * d + col] = O_i_new[r][col];
            }
        }
    }

    // Final scaling O_i = L * O_i
    for (int r = 0; r < B_r; r++) 
    {
        for (int col = 0; col < d; col++) 
        {
            O_i[r * d + col] *= L_i[r];  
        }
    }

    // Final log update L_i = m_i + log(L_i)
    for (int r = 0; r < B_r; r++) 
    {
        L_i[r] = m_i[r] + logf(L_i[r]);
    }
}

// ----------------- Main Function -----------------
int main() {
    int N = 64;    // Sequence Length
    int d = 32;    // Embedding Dimension
    int T_r = 4;   // Number of Row Blocks
    int T_c = 4;   // Number of Column Blocks

    int size_Q = N * d * sizeof(float);
    int size_K = N * d * sizeof(float);
    int size_V = N * d * sizeof(float);
    int size_O = N * d * sizeof(float);
    int size_L = N * sizeof(float);

    // Allocate memory on host
    float* h_Q = new float[N * d];
    float* h_K = new float[N * d];
    float* h_V = new float[N * d];
    float* h_O = new float[N * d];
    float* h_L = new float[N];

    // Random Initialization
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, size_Q);
    cudaMalloc(&d_K, size_K);
    cudaMalloc(&d_V, size_V);
    cudaMalloc(&d_O, size_O);
    cudaMalloc(&d_L, size_L);

    // Copy data to GPU
    cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);

    // Launch kernel
    int gridSize = (T_r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    FlashAttention<<<gridSize, BLOCK_SIZE>>>(d_Q, d_K, d_V, d_O, d_L, N, d, T_r, T_c);
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O; delete[] h_L;

    return 0;
}
