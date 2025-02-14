import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(X, output, stride_x, stride_y, N: tl.constexpr):
    row_idx = tl.program_id(0)  # Each row is handled by a different program
    row_start = X + row_idx * stride_x  # Starting point for row
    output_start = output + row_idx * stride_y  # Output row

    # Load row into registers
    x_vals = tl.load(row_start + tl.arange(0, N))  

    # Compute softmax
    x_max = tl.max(x_vals, axis=0)  # Row max for numerical stability
    safe_x = x_vals - x_max  # Subtract max
    numerator = tl.exp(safe_x)  # Compute exponentials
    denominator = tl.sum(numerator, axis=0)  # Sum across row
    sm_out = numerator / denominator  # Softmax output

    # Store back to output
    tl.store(output_start + tl.arange(0, N), sm_out)

# 🔥 Define Host Function to Launch Softmax Kernel
def softmax_triton(x):
    M, N = x.shape  # Shape of matrix
    output = torch.empty_like(x)  # Allocate output tensor

    # Launch kernel
    grid = (M,)  # One Triton program per row
    softmax_kernel[grid](x, output, x.stride(0), output.stride(0), N)

    return output

# ✅ Run Triton Softmax
sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')

# Reference Implementation in PyTorch
ref_out = torch.nn.functional.softmax(sample, dim=1)
print(f"ref_out=\n{ref_out}")

# Triton Softmax
triton_out = softmax_triton(sample)
print(f"triton_out=\n{triton_out}")
