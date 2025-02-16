import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def vector_sum_kernel(
    x_ptr, y_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute memory offsets
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask out-of-bounds indices
    mask = offset < N
    
    # Load data
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)

    # Compute sum
    out = x + y

    # Store result
    tl.store(out_ptr + offset, out, mask=mask)

# Python wrapper function
def vector_sum(x, y):
    assert x.shape == y.shape, "Vectors must have the same shape"
    
    N = x.shape[0]
    out = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE = 1024
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # Grid size to cover all elements

    # Launch Triton kernel
    vector_sum_kernel[grid_size](
        x, y, out, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Example usage
if __name__ == "__main__":
    N = 1 << 20  # 1M elements

    # Generate random input tensors
    x = torch.rand(N, dtype=torch.float32, device="cuda")
    y = torch.rand(N, dtype=torch.float32, device="cuda")

    # Compute sum using Triton
    out = vector_sum(x, y)

    # Verify correctness with PyTorch
    assert torch.allclose(out, x + y), "Mismatch in results!"

    print("Vector summation successful!")
