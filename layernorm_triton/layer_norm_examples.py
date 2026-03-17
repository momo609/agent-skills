"""
LayerNorm Triton NPU 使用示例

这个示例展示如何在昇腾 NPU 上使用 LayerNorm Triton 算子。
"""

import torch
import torch.nn.functional as F
from layer_norm_triton_npu import layer_norm_forward_triton, layer_norm_forward_triton_v2


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)
    
    rows, cols = 1024, 512
    epsilon = 1e-5
    
    x = torch.randn(rows, cols, device='npu', dtype=torch.float32)
    gamma = torch.randn(cols, device='npu', dtype=torch.float32)
    beta = torch.randn(cols, device='npu', dtype=torch.float32)
    
    output, mean, invvar = layer_norm_forward_triton(x, gamma, beta, epsilon)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"均值形状: {mean.shape}")
    print(f"逆方差形状: {invvar.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print()


def example_no_parameters():
    """无参数示例"""
    print("=" * 60)
    print("示例 2: 无 gamma/beta 参数")
    print("=" * 60)
    
    rows, cols = 512, 256
    
    x = torch.randn(rows, cols, device='npu', dtype=torch.float32)
    
    output, mean, invvar = layer_norm_forward_triton(x, None, None)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")
    print()


def example_mixed_precision():
    """混合精度示例"""
    print("=" * 60)
    print("示例 3: 混合精度训练")
    print("=" * 60)
    
    rows, cols = 2048, 1024
    
    x_fp16 = torch.randn(rows, cols, device='npu', dtype=torch.float16)
    gamma_fp16 = torch.randn(cols, device='npu', dtype=torch.float16)
    beta_fp16 = torch.randn(cols, device='npu', dtype=torch.float16)
    
    output_fp16, _, _ = layer_norm_forward_triton(x_fp16, gamma_fp16, beta_fp16)
    
    print(f"FP16 输入形状: {x_fp16.shape}")
    print(f"FP16 输出形状: {output_fp16.shape}")
    print(f"FP16 输出范围: [{output_fp16.min().item():.4f}, {output_fp16.max().item():.4f}]")
    print()
    
    x_bf16 = torch.randn(rows, cols, device='npu', dtype=torch.bfloat16)
    gamma_bf16 = torch.randn(cols, device='npu', dtype=torch.bfloat16)
    beta_bf16 = torch.randn(cols, device='npu', dtype=torch.bfloat16)
    
    output_bf16, _, _ = layer_norm_forward_triton(x_bf16, gamma_bf16, beta_bf16)
    
    print(f"BF16 输入形状: {x_bf16.shape}")
    print(f"BF16 输出形状: {output_bf16.shape}")
    print(f"BF16 输出范围: [{output_bf16.min().item():.4f}, {output_bf16.max().item():.4f}]")
    print()


def example_large_dimension():
    """大维度示例"""
    print("=" * 60)
    print("示例 4: 大维度处理")
    print("=" * 60)
    
    rows, cols = 64, 8192
    
    x = torch.randn(rows, cols, device='npu', dtype=torch.float32)
    gamma = torch.randn(cols, device='npu', dtype=torch.float32)
    beta = torch.randn(cols, device='npu', dtype=torch.float32)
    
    print(f"使用 Welford 版本处理大维度...")
    output, mean, invvar = layer_norm_forward_triton_v2(x, gamma, beta)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")
    print()


def example_comparison_with_pytorch():
    """与 PyTorch 对比示例"""
    print("=" * 60)
    print("示例 5: 与 PyTorch LayerNorm 对比")
    print("=" * 60)
    
    rows, cols = 512, 512
    epsilon = 1e-5
    
    x = torch.randn(rows, cols, device='npu', dtype=torch.float32)
    gamma = torch.randn(cols, device='npu', dtype=torch.float32)
    beta = torch.randn(cols, device='npu', dtype=torch.float32)
    
    output_triton, mean_triton, invvar_triton = layer_norm_forward_triton(
        x, gamma, beta, epsilon
    )
    
    output_pytorch = F.layer_norm(x, [cols], gamma, beta, epsilon)
    
    diff = (output_triton - output_pytorch).abs()
    
    print(f"输入形状: {x.shape}")
    print(f"Triton 输出范围: [{output_triton.min().item():.4f}, {output_triton.max().item():.4f}]")
    print(f"PyTorch 输出范围: [{output_pytorch.min().item():.4f}, {output_pytorch.max().item():.4f}]")
    print(f"最大差异: {diff.max().item():.6e}")
    print(f"平均差异: {diff.mean().item():.6e}")
    print(f"相对误差: {(diff / (output_pytorch.abs() + 1e-6)).mean().item():.6e}")
    print()


def example_batch_processing():
    """批处理示例"""
    print("=" * 60)
    print("示例 6: 批处理场景")
    print("=" * 60)
    
    batch_size = 32
    seq_len = 128
    hidden_dim = 768
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device='npu', dtype=torch.float32)
    gamma = torch.randn(hidden_dim, device='npu', dtype=torch.float32)
    beta = torch.randn(hidden_dim, device='npu', dtype=torch.float32)
    
    x_2d = x.view(-1, hidden_dim)
    
    output_2d, mean, invvar = layer_norm_forward_triton(x_2d, gamma, beta)
    
    output = output_2d.view(batch_size, seq_len, hidden_dim)
    
    print(f"输入形状: {x.shape}")
    print(f"重塑后形状: {x_2d.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")
    print()


def main():
    """运行所有示例"""
    if not torch.npu.is_available():
        print("❌ NPU 不可用，无法运行示例")
        print("请在具有昇腾 NPU 的环境中运行此脚本")
        return
    
    print("\n" + "=" * 60)
    print("LayerNorm Triton NPU 使用示例")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_no_parameters()
    example_mixed_precision()
    example_large_dimension()
    example_comparison_with_pytorch()
    example_batch_processing()
    
    print("=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
