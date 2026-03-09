#!/usr/bin/env python3
"""
Triton算子调试脚本模板

使用方法：
1. 复制此文件并重命名为 debug_xxx_issue.py
2. 根据具体问题修改参数
3. 运行调试：python debug_xxx_issue.py
"""

import torch
import torch_npu
import numpy as np
import random
from typing import Optional, Tuple


def set_random_seed(seed: int = 42):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.npu.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_tensor_stats(tensor: torch.Tensor, name: str):
    """检查张量统计信息"""
    print(f"\n{name}:")
    print(f"  形状: {tensor.shape}")
    print(f"  数据类型: {tensor.dtype}")
    print(f"  设备: {tensor.device}")
    print(f"  最小值: {tensor.min().item():.6f}")
    print(f"  最大值: {tensor.max().item():.6f}")
    print(f"  平均值: {tensor.mean().item():.6f}")
    print(f"  标准差: {tensor.std().item():.6f}")
    print(f"  包含NaN: {torch.isnan(tensor).any().item()}")
    print(f"  包含Inf: {torch.isinf(tensor).any().item()}")
    
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        print(f"  NaN数量: {nan_count}")
    
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        print(f"  Inf数量: {inf_count}")


def find_first_nan_position(tensor: torch.Tensor) -> Optional[Tuple[int, ...]]:
    """找到第一个NaN的位置"""
    if not torch.isnan(tensor).any():
        return None
    
    nan_mask = torch.isnan(tensor)
    flat_idx = torch.argmax(nan_mask.flatten())
    return tuple(torch.unravel_index(flat_idx, tensor.shape))


def debug_basic():
    """
    基础调试：验证算子在简单输入下的行为
    
    适用场景：
    - 算子首次迁移
    - 验证基本功能
    """
    print("\n" + "="*60)
    print("基础调试")
    print("="*60)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建简单输入
    input_tensor = torch.randn(4, 8, 16, dtype=torch.float16, device='npu:0')
    
    # TODO: 根据算子修改参数
    dim = 1
    index = torch.tensor([0, 1, 2], device='npu:0')
    
    # 检查输入
    check_tensor_stats(input_tensor, "输入张量")
    
    # 运行昇腾实现
    # TODO: 替换为实际的昇腾实现
    # from xxx_ascend import xxx_ascend
    # output_ascend = xxx_ascend(input_tensor, dim, index)
    
    # 运行PyTorch参考实现
    output_ref = torch.index_select(input_tensor, dim, index)
    
    # 临时：直接使用PyTorch实现（需要替换）
    output_ascend = output_ref.clone()
    
    # 检查输出
    check_tensor_stats(output_ref, "PyTorch输出")
    check_tensor_stats(output_ascend, "昇腾输出")
    
    # 比较结果
    diff = torch.abs(output_ref - output_ascend)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\n误差分析:")
    print(f"  最大误差: {max_diff:.6e}")
    print(f"  平均误差: {mean_diff:.6e}")
    
    if max_diff < 1e-5:
        print("  ✓ 基础测试通过")
    else:
        print("  ✗ 基础测试失败")


def debug_nan_issue():
    """
    NaN问题调试：定位NaN产生的原因
    
    适用场景：
    - 输出包含NaN
    - 特定输入导致NaN
    """
    print("\n" + "="*60)
    print("NaN问题调试")
    print("="*60)
    
    # 测试多次随机输入
    for i in range(5):
        print(f"\n--- 测试 {i+1} ---")
        
        # 不固定随机种子
        input_tensor = torch.randn(4, 8, 16, dtype=torch.float16, device='npu:0')
        
        # TODO: 根据算子修改参数
        dim = 1
        dim_size = input_tensor.size(dim)
        index = torch.randperm(dim_size)[:3].to('npu:0')
        
        # 检查输入
        has_nan_input = torch.isnan(input_tensor).any().item()
        has_inf_input = torch.isinf(input_tensor).any().item()
        
        print(f"输入包含NaN: {has_nan_input}")
        print(f"输入包含Inf: {has_inf_input}")
        
        # 运行实现
        output_ref = torch.index_select(input_tensor, dim, index)
        output_ascend = output_ref.clone()  # TODO: 替换
        
        # 检查输出
        has_nan_ref = torch.isnan(output_ref).any().item()
        has_nan_ascend = torch.isnan(output_ascend).any().item()
        
        print(f"PyTorch输出包含NaN: {has_nan_ref}")
        print(f"昇腾输出包含NaN: {has_nan_ascend}")
        
        if has_nan_ascend:
            # 定位第一个NaN
            nan_pos = find_first_nan_position(output_ascend)
            print(f"第一个NaN位置: {nan_pos}")
            
            if nan_pos:
                print(f"参考值: {output_ref[nan_pos].item():.6f}")
                print(f"实际值: {output_ascend[nan_pos].item()}")


def debug_dtype_issue():
    """
    数据类型问题调试：对比不同数据类型的表现
    
    适用场景：
    - 特定数据类型失败
    - 精度问题
    """
    print("\n" + "="*60)
    print("数据类型问题调试")
    print("="*60)
    
    dtypes = [torch.float16, torch.float32]
    
    for dtype in dtypes:
        print(f"\n--- 测试 {dtype} ---")
        
        set_random_seed(42)
        input_tensor = torch.randn(4, 8, 16, dtype=dtype, device='npu:0')
        
        # TODO: 根据算子修改参数
        dim = 1
        index = torch.tensor([0, 1, 2], device='npu:0')
        
        # 运行实现
        output_ref = torch.index_select(input_tensor, dim, index)
        output_ascend = output_ref.clone()  # TODO: 替换
        
        # 检查输出
        check_tensor_stats(output_ref, f"PyTorch输出 ({dtype})")
        check_tensor_stats(output_ascend, f"昇腾输出 ({dtype})")
        
        # 比较结果
        diff = torch.abs(output_ref - output_ascend)
        max_diff = diff.max().item()
        
        print(f"最大误差: {max_diff:.6e}")
        
        if max_diff < 1e-5:
            print("✓ 测试通过")
        else:
            print("✗ 测试失败")


def debug_edge_case():
    """
    边界情况调试：测试极端输入
    
    适用场景：
    - 特定输入失败
    - 边界条件问题
    """
    print("\n" + "="*60)
    print("边界情况调试")
    print("="*60)
    
    edge_cases = [
        ("索引长度为1", (4, 8, 16), 1, [0]),
        ("索引等于维度大小", (4, 8, 16), 1, list(range(8))),
        ("重复索引", (4, 8, 16), 1, [0, 0, 1, 1, 2, 2]),
        ("负维度", (4, 8, 16), -1, [0, 1, 2]),
        ("大索引值", (4, 8, 16), 1, [7, 6, 5]),
        ("维度大小为1", (4, 1, 16), 1, [0]),
        ("最小输入", (1, 1, 1), 0, [0]),
    ]
    
    for desc, shape, dim, index_list in edge_cases:
        print(f"\n--- {desc} ---")
        print(f"形状: {shape}, 维度: {dim}, 索引: {index_list}")
        
        set_random_seed(42)
        input_tensor = torch.randn(shape, dtype=torch.float16, device='npu:0')
        index = torch.tensor(index_list, device='npu:0')
        
        # 检查输入
        print(f"输入形状: {input_tensor.shape}")
        print(f"输入数据类型: {input_tensor.dtype}")
        
        try:
            # 运行实现
            output_ref = torch.index_select(input_tensor, dim, index)
            output_ascend = output_ref.clone()  # TODO: 替换
            
            # 检查输出
            print(f"输出形状: {output_ref.shape}")
            
            # 比较结果
            diff = torch.abs(output_ref - output_ascend)
            max_diff = diff.max().item()
            
            print(f"最大误差: {max_diff:.6e}")
            
            if max_diff < 1e-5:
                print("✓ 测试通过")
            else:
                print("✗ 测试失败")
                
                # 定位误差位置
                max_idx = torch.argmax(diff.flatten())
                max_pos = tuple(torch.unravel_index(max_idx, diff.shape))
                print(f"误差最大位置: {max_pos}")
                print(f"参考值: {output_ref.flatten()[max_idx].item():.6f}")
                print(f"实际值: {output_ascend.flatten()[max_idx].item():.6f}")
                
        except Exception as e:
            print(f"✗ 异常: {e}")


def debug_comparison():
    """
    对比调试：详细对比PyTorch和昇腾实现
    
    适用场景：
    - 需要详细分析差异
    - 定位具体问题
    """
    print("\n" + "="*60)
    print("对比调试")
    print("="*60)
    
    set_random_seed(42)
    
    # 创建输入
    input_tensor = torch.randn(4, 8, 16, dtype=torch.float16, device='npu:0')
    
    # TODO: 根据算子修改参数
    dim = 1
    index = torch.tensor([0, 1, 2], device='npu:0')
    
    print(f"\n输入信息:")
    print(f"  形状: {input_tensor.shape}")
    print(f"  数据类型: {input_tensor.dtype}")
    print(f"  维度: {dim}")
    print(f"  索引: {index}")
    
    # 运行实现
    output_ref = torch.index_select(input_tensor, dim, index)
    output_ascend = output_ref.clone()  # TODO: 替换
    
    print(f"\n输出信息:")
    print(f"  参考输出形状: {output_ref.shape}")
    print(f"  昇腾输出形状: {output_ascend.shape}")
    
    # 详细对比
    print(f"\n详细对比:")
    
    # 1. 检查NaN/Inf
    print(f"\n1. NaN/Inf检查:")
    print(f"  参考输出包含NaN: {torch.isnan(output_ref).any().item()}")
    print(f"  参考输出包含Inf: {torch.isinf(output_ref).any().item()}")
    print(f"  昇腾输出包含NaN: {torch.isnan(output_ascend).any().item()}")
    print(f"  昇腾输出包含Inf: {torch.isinf(output_ascend).any().item()}")
    
    # 2. 统计信息
    print(f"\n2. 统计信息:")
    print(f"  参考输出: min={output_ref.min().item():.6f}, max={output_ref.max().item():.6f}, mean={output_ref.mean().item():.6f}")
    print(f"  昇腾输出: min={output_ascend.min().item():.6f}, max={output_ascend.max().item():.6f}, mean={output_ascend.mean().item():.6f}")
    
    # 3. 误差分析
    print(f"\n3. 误差分析:")
    diff = torch.abs(output_ref - output_ascend)
    print(f"  最大绝对误差: {diff.max().item():.6e}")
    print(f"  平均绝对误差: {diff.mean().item():.6e}")
    print(f"  误差标准差: {diff.std().item():.6e}")
    
    # 4. 相对误差
    ref_abs = torch.abs(output_ref)
    rel_diff = diff / (ref_abs + 1e-8)
    print(f"  最大相对误差: {rel_diff.max().item():.6e}")
    print(f"  平均相对误差: {rel_diff.mean().item():.6e}")
    
    # 5. 误差分布
    print(f"\n4. 误差分布:")
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for thresh in thresholds:
        count = (diff < thresh).sum().item()
        total = diff.numel()
        percentage = count / total * 100
        print(f"  误差 < {thresh:.0e}: {count}/{total} ({percentage:.2f}%)")
    
    # 6. 定位最大误差
    if diff.max().item() > 1e-5:
        print(f"\n5. 最大误差位置:")
        max_idx = torch.argmax(diff.flatten())
        max_pos = tuple(torch.unravel_index(max_idx, diff.shape))
        print(f"  位置: {max_pos}")
        print(f"  参考值: {output_ref.flatten()[max_idx].item():.6f}")
        print(f"  实际值: {output_ascend.flatten()[max_idx].item():.6f}")
        print(f"  误差: {diff.flatten()[max_idx].item():.6e}")


def debug_performance():
    """
    性能调试：对比性能差异
    
    适用场景：
    - 性能优化
    - 性能对比
    """
    print("\n" + "="*60)
    print("性能调试")
    print("="*60)
    
    import time
    
    shape = (32, 64, 128)
    dtype = torch.float16
    
    set_random_seed(42)
    input_tensor = torch.randn(shape, dtype=dtype, device='npu:0')
    
    # TODO: 根据算子修改参数
    dim = 1
    dim_size = input_tensor.size(dim)
    index = torch.randperm(dim_size)[:10].to('npu:0')
    
    # 预热
    print("\n预热...")
    for _ in range(10):
        _ = torch.index_select(input_tensor, dim, index)
    
    # PyTorch性能
    print("\n测试PyTorch性能...")
    torch.npu.synchronize()
    start = time.time()
    for _ in range(100):
        output_ref = torch.index_select(input_tensor, dim, index)
    torch.npu.synchronize()
    pytorch_time = (time.time() - start) / 100 * 1000
    
    # 昇腾性能
    print("测试昇腾性能...")
    torch.npu.synchronize()
    start = time.time()
    for _ in range(100):
        output_ascend = output_ref.clone()  # TODO: 替换
    torch.npu.synchronize()
    ascend_time = (time.time() - start) / 100 * 1000
    
    print(f"\n性能对比:")
    print(f"  PyTorch平均时间: {pytorch_time:.3f} ms")
    print(f"  昇腾平均时间: {ascend_time:.3f} ms")
    print(f"  加速比: {pytorch_time / ascend_time:.2f}x")


def main():
    """主函数"""
    # 检查NPU可用性
    if not torch.npu.is_available():
        print("错误：NPU不可用")
        return
    
    print(f"NPU设备数量: {torch.npu.device_count()}")
    print(f"当前NPU设备: {torch.npu.current_device()}")
    print(f"NPU设备名称: {torch.npu.get_device_name()}")
    
    # 运行调试
    # TODO: 根据需要选择调试函数
    
    # 1. 基础调试
    debug_basic()
    
    # 2. NaN问题调试
    # debug_nan_issue()
    
    # 3. 数据类型问题调试
    # debug_dtype_issue()
    
    # 4. 边界情况调试
    # debug_edge_case()
    
    # 5. 对比调试
    # debug_comparison()
    
    # 6. 性能调试
    # debug_performance()


if __name__ == "__main__":
    main()
