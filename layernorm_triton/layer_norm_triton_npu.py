"""
Triton implementation of LayerNorm forward kernel for Ascend NPU.
Migrated from GPU version with NPU-specific optimizations.
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
    LayerNorm forward kernel using Triton for NPU.
    
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
    
    x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0, care_padding=False).to(tl.float32)
    
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
        gamma = tl.load(gamma_ptr + cols_range, mask=mask, other=1.0, care_padding=False).to(tl.float32)
        x_norm = x_norm * gamma
    
    if beta_ptr is not None:
        beta = tl.load(beta_ptr + cols_range, mask=mask, other=0.0, care_padding=False).to(tl.float32)
        x_norm = x_norm + beta
    
    tl.store(output_ptr + row_start + cols_range, x_norm, mask=mask)


def layer_norm_forward_triton(
    input: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> tuple:
    """
    LayerNorm forward function using Triton for NPU.
    
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
    Optimized LayerNorm forward kernel using block-wise processing for NPU.
    
    This version processes data in blocks to handle large dimensions efficiently.
    
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
        
        x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0, care_padding=False).to(tl.float32)
        
        block_count = tl.sum(mask.to(tl.float32), axis=0)
        block_mean = tl.sum(x, axis=0) / block_count
        block_m2 = tl.sum((x - block_mean) * (x - block_mean), axis=0)
        
        if count == 0.0:
            mean = block_mean
            m2 = block_m2
            count = block_count
        else:
            new_count = count + block_count
            nb_n = block_count / new_count
            delta = block_mean - mean
            mean = mean + delta * nb_n
            m2 = m2 + block_m2 + delta * delta * count * nb_n
            count = new_count
    
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
        
        x = tl.load(input_ptr + row_start + cols_range, mask=mask, other=0.0, care_padding=False).to(tl.float32)
        
        x_norm = (x - mean) * invvar
        
        if gamma_ptr is not None:
            gamma = tl.load(gamma_ptr + cols_range, mask=mask, other=1.0, care_padding=False).to(tl.float32)
            x_norm = x_norm * gamma
        
        if beta_ptr is not None:
            beta = tl.load(beta_ptr + cols_range, mask=mask, other=0.0, care_padding=False).to(tl.float32)
            x_norm = x_norm + beta
        
        tl.store(output_ptr + row_start + cols_range, x_norm, mask=mask)


def layer_norm_forward_triton_v2(
    input: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> tuple:
    """
    Optimized LayerNorm forward function using Welford algorithm for NPU.
    
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


def verify_accuracy(result, ref, dtype):
    """验证精度"""
    assert not torch.isnan(result).any(), "结果包含NaN"
    assert not torch.isinf(result).any(), "结果包含Inf"
    
    if dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 1e-3, 1e-3
    elif dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 0, 0
    
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)
    print(f"✅ 精度验证通过 (dtype={dtype}, rtol={rtol}, atol={atol})")


def test_layer_norm_npu():
    """Test the Triton LayerNorm implementation on NPU."""
    import torch.nn.functional as F
    
    if not torch.npu.is_available():
        print("NPU is not available. Skipping tests.")
        print("Please run this script on a system with Ascend NPU support.")
        return
    
    torch.manual_seed(42)
    
    print("=" * 60)
    print("Testing LayerNorm Triton implementation on NPU")
    print("=" * 60)
    
    rows, cols = 128, 512
    epsilon = 1e-5
    
    print(f"\n测试配置: rows={rows}, cols={cols}, epsilon={epsilon}")
    
    x = torch.randn(rows, cols, device='npu', dtype=torch.float32)
    gamma = torch.randn(cols, device='npu', dtype=torch.float32)
    beta = torch.randn(cols, device='npu', dtype=torch.float32)
    
    print("\n" + "=" * 60)
    print("测试基础版本 (layer_norm_forward_triton)")
    print("=" * 60)
    
    output_triton, mean_triton, invvar_triton = layer_norm_forward_triton(
        x, gamma, beta, epsilon
    )
    
    output_pytorch = F.layer_norm(x, [cols], gamma, beta, epsilon)
    
    mean_pytorch = x.mean(dim=1)
    var_pytorch = x.var(dim=1, unbiased=False)
    invvar_pytorch = torch.rsqrt(var_pytorch + epsilon)
    
    print(f"Input shape: {x.shape}")
    print(f"Output max diff: {(output_triton - output_pytorch).abs().max().item():.6e}")
    print(f"Mean max diff: {(mean_triton - mean_pytorch).abs().max().item():.6e}")
    print(f"InvVar max diff: {(invvar_triton - invvar_pytorch).abs().max().item():.6e}")
    
    verify_accuracy(output_triton, output_pytorch, torch.float32)
    
    print("\n" + "=" * 60)
    print("测试 FP16 精度")
    print("=" * 60)
    
    x_half = x.half()
    gamma_half = gamma.half()
    beta_half = beta.half()
    
    output_triton_half, _, _ = layer_norm_forward_triton(
        x_half, gamma_half, beta_half, epsilon
    )
    output_pytorch_half = F.layer_norm(x_half, [cols], gamma_half, beta_half, epsilon)
    
    print(f"Output max diff: {(output_triton_half.float() - output_pytorch_half.float()).abs().max().item():.6e}")
    verify_accuracy(output_triton_half, output_pytorch_half, torch.float16)
    
    print("\n" + "=" * 60)
    print("测试 BF16 精度")
    print("=" * 60)
    
    x_bf16 = x.bfloat16()
    gamma_bf16 = gamma.bfloat16()
    beta_bf16 = beta.bfloat16()
    
    output_triton_bf16, _, _ = layer_norm_forward_triton(
        x_bf16, gamma_bf16, beta_bf16, epsilon
    )
    output_pytorch_bf16 = F.layer_norm(x_bf16, [cols], gamma_bf16, beta_bf16, epsilon)
    
    print(f"Output max diff: {(output_triton_bf16.float() - output_pytorch_bf16.float()).abs().max().item():.6e}")
    verify_accuracy(output_triton_bf16, output_pytorch_bf16, torch.bfloat16)
    
    print("\n" + "=" * 60)
    print("测试无 gamma/beta 的情况")
    print("=" * 60)
    
    output_triton_no_param, mean_triton_no_param, _ = layer_norm_forward_triton(
        x, None, None, epsilon
    )
    output_pytorch_no_param = F.layer_norm(x, [cols], None, None, epsilon)
    
    print(f"Output max diff: {(output_triton_no_param - output_pytorch_no_param).abs().max().item():.6e}")
    verify_accuracy(output_triton_no_param, output_pytorch_no_param, torch.float32)
    
    print("\n" + "=" * 60)
    print("测试大维度 (cols=4096)")
    print("=" * 60)
    
    rows_large, cols_large = 32, 4096
    x_large = torch.randn(rows_large, cols_large, device='npu', dtype=torch.float32)
    gamma_large = torch.randn(cols_large, device='npu', dtype=torch.float32)
    beta_large = torch.randn(cols_large, device='npu', dtype=torch.float32)
    
    output_triton_large, _, _ = layer_norm_forward_triton(
        x_large, gamma_large, beta_large, epsilon
    )
    output_pytorch_large = F.layer_norm(x_large, [cols_large], gamma_large, beta_large, epsilon)
    
    print(f"Input shape: {x_large.shape}")
    print(f"Output max diff: {(output_triton_large - output_pytorch_large).abs().max().item():.6e}")
    verify_accuracy(output_triton_large, output_pytorch_large, torch.float32)
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_layer_norm_npu()
