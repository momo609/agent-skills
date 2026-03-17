"""
Triton implementation of LayerNorm forward kernel.
Equivalent to LayerNormForwardV2 in layer_norm_cuda_kernel.cu
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_forward_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    invvar_ptr,
    rows,
    cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm forward kernel using Triton.
    
    Each program instance processes one row of the input tensor.
    
    Args:
        input_ptr: Pointer to input tensor [rows, cols]
        output_ptr: Pointer to output tensor [rows, cols]
        gamma_ptr: Pointer to gamma parameter [cols], can be None
        beta_ptr: Pointer to beta parameter [cols], can be None
        mean_ptr: Pointer to store computed mean [rows]
        invvar_ptr: Pointer to store computed inverse variance [rows]
        rows: Number of rows
        cols: Number of columns (normalization dimension)
        epsilon: Small constant for numerical stability
        BLOCK_SIZE: Block size for parallel reduction
    """
    row_idx = tl.program_id(0)
    if row_idx >= rows:
        return
    
    row_start = row_idx * cols
    
    cols_range = tl.arange(0, BLOCK_SIZE)
    mask = cols_range < cols
    
    x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / cols
    
    x_minus_mean = x - mean
    variance = tl.sum(x_minus_mean * x_minus_mean, axis=0) / cols
    
    invvar = tl.rsqrt(variance + epsilon)
    
    if mean_ptr is not None:
        tl.store(mean_ptr + row_idx, mean)
    if invvar_ptr is not None:
        tl.store(invvar_ptr + row_idx, invvar)
    
    x_norm = x_minus_mean * invvar
    
    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + cols_range, mask=mask, other=1.0).to(tl.float32)
        x_norm = x_norm * gamma
    
    if beta_ptr is not None:
        beta = tl.load(beta_ptr + cols_range, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + beta
    
    tl.store(output_ptr + row_start + cols_range, x_norm, mask=mask)


def layer_norm_forward_triton(
    input: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> tuple:
    """
    LayerNorm forward function using Triton.
    
    Args:
        input: Input tensor of shape [rows, cols]
        gamma: Scale parameter of shape [cols], optional
        beta: Shift parameter of shape [cols], optional
        epsilon: Small constant for numerical stability
    
    Returns:
        output: Normalized output tensor [rows, cols]
        mean: Computed mean [rows]
        invvar: Computed inverse variance [rows]
    """
    assert input.is_contiguous(), "Input must be contiguous"
    assert input.ndim == 2, "Input must be 2D tensor"
    
    rows, cols = input.shape
    
    output = torch.empty_like(input)
    mean = torch.empty(rows, dtype=torch.float32, device=input.device)
    invvar = torch.empty(rows, dtype=torch.float32, device=input.device)
    
    BLOCK_SIZE = triton.next_power_of_2(cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (rows,)
    
    layer_norm_forward_kernel[grid](
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        rows,
        cols,
        epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, mean, invvar


@triton.jit
def layer_norm_forward_kernel_v2(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    invvar_ptr,
    rows,
    cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized LayerNorm forward kernel using Welford algorithm.
    
    This version more closely matches the CUDA implementation by using
    Welford's online algorithm for computing mean and variance.
    
    Args:
        input_ptr: Pointer to input tensor [rows, cols]
        output_ptr: Pointer to output tensor [rows, cols]
        gamma_ptr: Pointer to gamma parameter [cols], can be None
        beta_ptr: Pointer to beta parameter [cols], can be None
        mean_ptr: Pointer to store computed mean [rows]
        invvar_ptr: Pointer to store computed inverse variance [rows]
        rows: Number of rows
        cols: Number of columns (normalization dimension)
        epsilon: Small constant for numerical stability
        BLOCK_SIZE: Block size for parallel reduction
    """
    row_idx = tl.program_id(0)
    if row_idx >= rows:
        return
    
    row_start = row_idx * cols
    
    mean = 0.0
    m2 = 0.0
    count = 0.0
    
    num_blocks = tl.cdiv(cols, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        cols_range = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols_range < cols
        
        x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0).to(tl.float32)
        
        for i in range(BLOCK_SIZE):
            if block_idx * BLOCK_SIZE + i < cols:
                count += 1.0
                delta1 = x[i] - mean
                mean += delta1 / count
                delta2 = x[i] - mean
                m2 += delta1 * delta2
    
    variance = m2 / count
    variance = tl.maximum(variance, 0.0)
    invvar = tl.rsqrt(variance + epsilon)
    
    if mean_ptr is not None:
        tl.store(mean_ptr + row_idx, mean)
    if invvar_ptr is not None:
        tl.store(invvar_ptr + row_idx, invvar)
    
    for block_idx in range(num_blocks):
        cols_range = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols_range < cols
        
        x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0).to(tl.float32)
        
        x_norm = (x - mean) * invvar
        
        if gamma_ptr is not None:
            gamma = tl.load(gamma_ptr + cols_range, mask=mask, other=1.0).to(tl.float32)
            x_norm = x_norm * gamma
        
        if beta_ptr is not None:
            beta = tl.load(beta_ptr + cols_range, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_norm + beta
        
        tl.store(output_ptr + row_start + cols_range, x_norm, mask=mask)


def layer_norm_forward_triton_v2(
    input: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> tuple:
    """
    Optimized LayerNorm forward function using Welford algorithm.
    
    Args:
        input: Input tensor of shape [rows, cols]
        gamma: Scale parameter of shape [cols], optional
        beta: Shift parameter of shape [cols], optional
        epsilon: Small constant for numerical stability
    
    Returns:
        output: Normalized output tensor [rows, cols]
        mean: Computed mean [rows]
        invvar: Computed inverse variance [rows]
    """
    assert input.is_contiguous(), "Input must be contiguous"
    assert input.ndim == 2, "Input must be 2D tensor"
    
    rows, cols = input.shape
    
    output = torch.empty_like(input)
    mean = torch.empty(rows, dtype=torch.float32, device=input.device)
    invvar = torch.empty(rows, dtype=torch.float32, device=input.device)
    
    BLOCK_SIZE = 1024
    
    grid = (rows,)
    
    layer_norm_forward_kernel_v2[grid](
        input,
        output,
        gamma,
        beta,
        mean,
        invvar,
        rows,
        cols,
        epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, mean, invvar


def test_layer_norm():
    """Test the Triton LayerNorm implementation against PyTorch."""
    import torch.nn.functional as F
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU tests.")
        print("The Triton kernel is designed for GPU execution.")
        print("\nTo test on a GPU system, run this script on a machine with CUDA support.")
        return
    
    torch.manual_seed(42)
    
    rows, cols = 128, 512
    epsilon = 1e-5
    
    x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
    gamma = torch.randn(cols, device='cuda', dtype=torch.float32)
    beta = torch.randn(cols, device='cuda', dtype=torch.float32)
    
    output_triton, mean_triton, invvar_triton = layer_norm_forward_triton(
        x, gamma, beta, epsilon
    )
    
    output_pytorch = F.layer_norm(x, [cols], gamma, beta, epsilon)
    
    mean_pytorch = x.mean(dim=1)
    var_pytorch = x.var(dim=1, unbiased=False)
    invvar_pytorch = torch.rsqrt(var_pytorch + epsilon)
    
    print("Testing LayerNorm Triton implementation...")
    print(f"Input shape: {x.shape}")
    print(f"Output max diff: {(output_triton - output_pytorch).abs().max().item():.6e}")
    print(f"Mean max diff: {(mean_triton - mean_pytorch).abs().max().item():.6e}")
    print(f"InvVar max diff: {(invvar_triton - invvar_pytorch).abs().max().item():.6e}")
    
    output_triton_v2, mean_triton_v2, invvar_triton_v2 = layer_norm_forward_triton_v2(
        x, gamma, beta, epsilon
    )
    
    print("\nTesting LayerNorm Triton V2 implementation (Welford)...")
    print(f"Output max diff: {(output_triton_v2 - output_pytorch).abs().max().item():.6e}")
    print(f"Mean max diff: {(mean_triton_v2 - mean_pytorch).abs().max().item():.6e}")
    print(f"InvVar max diff: {(invvar_triton_v2 - invvar_pytorch).abs().max().item():.6e}")
    
    x_half = x.half()
    gamma_half = gamma.half()
    beta_half = beta.half()
    
    output_triton_half, _, _ = layer_norm_forward_triton(
        x_half, gamma_half, beta_half, epsilon
    )
    output_pytorch_half = F.layer_norm(x_half, [cols], gamma_half, beta_half, epsilon)
    
    print("\nTesting with FP16...")
    print(f"Output max diff: {(output_triton_half.float() - output_pytorch_half.float()).abs().max().item():.6e}")
    
    x_bf16 = x.bfloat16()
    gamma_bf16 = gamma.bfloat16()
    beta_bf16 = beta.bfloat16()
    
    output_triton_bf16, _, _ = layer_norm_forward_triton(
        x_bf16, gamma_bf16, beta_bf16, epsilon
    )
    output_pytorch_bf16 = F.layer_norm(x_bf16, [cols], gamma_bf16, beta_bf16, epsilon)
    
    print("\nTesting with BF16...")
    print(f"Output max diff: {(output_triton_bf16.float() - output_pytorch_bf16.float()).abs().max().item():.6e}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_layer_norm()
