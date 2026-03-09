#!/usr/bin/env python3
"""
Triton算子迁移到昇腾NPU - 测试用例模板

使用方法：
1. 复制此文件并重命名为 test_xxx_ascend.py
2. 替换 xxx_operator 为实际算子名称
3. 根据算子特性修改测试用例
4. 运行测试：python test_xxx_ascend.py
"""

import torch
import torch_npu
import numpy as np
import random
from typing import List, Tuple, Optional


def set_random_seed(seed: int = 42):
    """设置所有随机种子，确保测试可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.npu.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def compare_outputs(ref_output: torch.Tensor, 
                   test_output: torch.Tensor, 
                   name: str = "",
                   atol: float = 1e-5,
                   rtol: float = 1e-5) -> bool:
    """
    比较参考输出和测试输出
    
    Args:
        ref_output: 参考输出（PyTorch原生实现）
        test_output: 测试输出（昇腾NPU实现）
        name: 测试名称
        atol: 绝对误差阈值
        rtol: 相对误差阈值
    
    Returns:
        是否通过测试
    """
    # 检查形状
    if ref_output.shape != test_output.shape:
        print(f"\n{name}")
        print(f"  ✗ 形状不匹配: {ref_output.shape} vs {test_output.shape}")
        return False
    
    # 检查NaN/Inf
    if torch.isnan(test_output).any():
        print(f"\n{name}")
        print(f"  ✗ 输出包含NaN")
        nan_count = torch.isnan(test_output).sum().item()
        print(f"  NaN数量: {nan_count}")
        return False
    
    if torch.isinf(test_output).any():
        print(f"\n{name}")
        print(f"  ✗ 输出包含Inf")
        inf_count = torch.isinf(test_output).sum().item()
        print(f"  Inf数量: {inf_count}")
        return False
    
    # 计算误差
    abs_diff = torch.abs(ref_output - test_output)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    # 计算相对误差（避免除零）
    ref_abs = torch.abs(ref_output)
    rel_diff = abs_diff / (ref_abs + 1e-8)
    max_rel_error = rel_diff.max().item()
    
    # 打印详细信息
    print(f"\n{name}")
    print(f"  最大绝对误差: {max_abs_error:.6e}")
    print(f"  平均绝对误差: {mean_abs_error:.6e}")
    print(f"  最大相对误差: {max_rel_error:.6e}")
    
    # 定位误差最大的位置
    if max_abs_error > atol:
        max_idx = torch.argmax(abs_diff.flatten())
        max_pos = tuple(torch.unravel_index(max_idx, abs_diff.shape))
        print(f"  误差最大位置: {max_pos}")
        print(f"  参考值: {ref_output.flatten()[max_idx]:.6f}")
        print(f"  实际值: {test_output.flatten()[max_idx]:.6f}")
    
    # 判断是否通过
    if max_abs_error < atol:
        print(f"  ✓ 测试通过")
        return True
    else:
        print(f"  ✗ 测试失败")
        return False


class TestXXXOperatorAscend:
    """XXX算子的昇腾NPU测试套件"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    # ==================== 基础精度测试 ====================
    
    def test_basic_accuracy(self):
        """
        基础精度测试：覆盖不同维度
        
        目的：验证算子在标准输入下的正确性
        """
        print("\n" + "="*60)
        print("基础精度测试")
        print("="*60)
        
        # 定义测试用例：根据算子特性修改
        test_cases = [
            # (输入形状, 维度, 其他参数)
            {"shape": (4, 8, 16), "dim": 0, "desc": "dim=0"},
            {"shape": (4, 8, 16), "dim": 1, "desc": "dim=1"},
            {"shape": (4, 8, 16), "dim": 2, "desc": "dim=2"},
            {"shape": (4, 8, 16, 32), "dim": 3, "desc": "dim=3"},
            {"shape": (4, 8, 16), "dim": -1, "desc": "dim=-1 (负维度)"},
            {"shape": (4, 8, 16), "dim": -2, "desc": "dim=-2 (负维度)"},
        ]
        
        for i, case in enumerate(test_cases, 1):
            set_random_seed(42)
            
            # 创建输入
            input_tensor = torch.randn(case["shape"], dtype=torch.float16, device='npu:0')
            
            # TODO: 根据算子修改参数
            # 示例：index_select
            dim = case["dim"]
            dim_size = input_tensor.size(dim)
            index_len = min(3, dim_size)
            index = torch.randperm(dim_size)[:index_len].to('npu:0')
            
            # 运行昇腾实现
            # TODO: 替换为实际的昇腾实现
            # from xxx_ascend import xxx_ascend
            # output_ascend = xxx_ascend(input_tensor, dim, index)
            
            # 运行PyTorch参考实现
            # TODO: 替换为实际的PyTorch实现
            output_ref = torch.index_select(input_tensor, dim, index)
            
            # 临时：直接使用PyTorch实现（需要替换）
            output_ascend = output_ref.clone()
            
            # 比较结果
            passed = compare_outputs(
                output_ref, output_ascend,
                f"测试用例{i}: {case['desc']}"
            )
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    # ==================== 不同规模测试 ====================
    
    def test_different_scales(self):
        """
        不同规模测试：验证性能和稳定性
        
        目的：确保算子在不同输入规模下都能正常工作
        """
        print("\n" + "="*60)
        print("不同规模测试")
        print("="*60)
        
        scales = [
            ("小规模", (2, 4, 8)),
            ("中规模", (8, 16, 32)),
            ("大规模", (32, 64, 128)),
            ("超大规模", (128, 256, 512)),
        ]
        
        for name, shape in scales:
            set_random_seed(42)
            
            # 创建输入
            input_tensor = torch.randn(shape, dtype=torch.float16, device='npu:0')
            
            # TODO: 根据算子修改参数
            dim = 1
            dim_size = input_tensor.size(dim)
            index_len = min(10, dim_size)
            index = torch.randperm(dim_size)[:index_len].to('npu:0')
            
            # 运行实现
            output_ref = torch.index_select(input_tensor, dim, index)
            output_ascend = output_ref.clone()  # TODO: 替换
            
            # 比较结果
            passed = compare_outputs(
                output_ref, output_ascend,
                f"{name}: 形状{shape}"
            )
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    # ==================== 边界情况测试 ====================
    
    def test_edge_cases(self):
        """
        边界情况测试：极端输入
        
        目的：验证算子对特殊输入的处理能力
        """
        print("\n" + "="*60)
        print("边界情况测试")
        print("="*60)
        
        edge_cases = [
            # (描述, 输入形状, 维度, 其他参数)
            ("索引长度为1", (4, 8, 16), 1, {"index": [0]}),
            ("索引等于维度大小", (4, 8, 16), 1, {"index_len": 8}),
            ("重复索引", (4, 8, 16), 1, {"index": [0, 0, 1, 1, 2, 2]}),
            ("负维度-2", (4, 8, 16), -2, {"index_len": 2}),
            ("大索引值", (4, 8, 16), 1, {"index": [7, 6, 5]}),
            ("维度大小为1", (4, 1, 16), 1, {"index": [0]}),
            ("最小维度", (1, 1, 1), 0, {"index": [0]}),
        ]
        
        for i, (desc, shape, dim, params) in enumerate(edge_cases, 1):
            set_random_seed(42)
            
            # 创建输入
            input_tensor = torch.randn(shape, dtype=torch.float16, device='npu:0')
            
            # 创建索引
            if "index" in params:
                index = torch.tensor(params["index"], device='npu:0')
            elif "index_len" in params:
                dim_size = input_tensor.size(dim)
                index = torch.randperm(dim_size)[:params["index_len"]].to('npu:0')
            else:
                dim_size = input_tensor.size(dim)
                index_len = min(3, dim_size)
                index = torch.randperm(dim_size)[:index_len].to('npu:0')
            
            # 运行实现
            output_ref = torch.index_select(input_tensor, dim, index)
            output_ascend = output_ref.clone()  # TODO: 替换
            
            # 比较结果
            passed = compare_outputs(
                output_ref, output_ascend,
                f"测试用例{i}: {desc}"
            )
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    # ==================== 不同数据类型测试 ====================
    
    def test_different_dtypes(self):
        """
        不同数据类型测试
        
        目的：验证算子对不同精度的支持
        """
        print("\n" + "="*60)
        print("不同数据类型测试")
        print("="*60)
        
        dtypes = [
            (torch.float16, 1e-3, "float16"),
            (torch.float32, 1e-5, "float32"),
            # (torch.bfloat16, 1e-2, "bfloat16"),  # 如果支持
        ]
        
        shape = (4, 8, 16)
        dim = 1
        
        for dtype, atol, dtype_name in dtypes:
            set_random_seed(42)
            
            # 创建输入
            input_tensor = torch.randn(shape, dtype=dtype, device='npu:0')
            dim_size = input_tensor.size(dim)
            index = torch.randperm(dim_size)[:3].to('npu:0')
            
            # 运行实现
            output_ref = torch.index_select(input_tensor, dim, index)
            output_ascend = output_ref.clone()  # TODO: 替换
            
            # 比较结果
            passed = compare_outputs(
                output_ref, output_ascend,
                f"{dtype_name}",
                atol=atol
            )
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    # ==================== 性能测试 ====================
    
    def test_performance(self):
        """
        性能基准测试
        
        目的：对比昇腾实现与PyTorch实现的性能
        """
        print("\n" + "="*60)
        print("性能基准测试")
        print("="*60)
        
        import time
        
        shape = (32, 64, 128)
        dim = 1
        dtype = torch.float16
        
        set_random_seed(42)
        input_tensor = torch.randn(shape, dtype=dtype, device='npu:0')
        dim_size = input_tensor.size(dim)
        index = torch.randperm(dim_size)[:10].to('npu:0')
        
        # 预热
        for _ in range(10):
            _ = torch.index_select(input_tensor, dim, index)
        
        # PyTorch性能
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            output_ref = torch.index_select(input_tensor, dim, index)
        torch.npu.synchronize()
        pytorch_time = (time.time() - start) / 100 * 1000
        
        # 昇腾性能（TODO: 替换为实际实现）
        torch.npu.synchronize()
        start = time.time()
        for _ in range(100):
            output_ascend = output_ref.clone()  # TODO: 替换
        torch.npu.synchronize()
        ascend_time = (time.time() - start) / 100 * 1000
        
        print(f"\nPyTorch平均时间: {pytorch_time:.3f} ms")
        print(f"昇腾平均时间: {ascend_time:.3f} ms")
        print(f"加速比: {pytorch_time / ascend_time:.2f}x")
    
    # ==================== 运行所有测试 ====================
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始测试 XXX 算子（昇腾NPU）")
        print("="*60)
        
        # 运行各项测试
        self.test_basic_accuracy()
        self.test_different_scales()
        self.test_edge_cases()
        self.test_different_dtypes()
        self.test_performance()
        
        # 打印总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"总计: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n✓ 所有测试通过！")
        else:
            print(f"\n✗ 有 {self.failed} 个测试失败，请检查实现")


def main():
    """主函数"""
    # 检查NPU可用性
    if not torch.npu.is_available():
        print("错误：NPU不可用")
        return
    
    print(f"NPU设备数量: {torch.npu.device_count()}")
    print(f"当前NPU设备: {torch.npu.current_device()}")
    print(f"NPU设备名称: {torch.npu.get_device_name()}")
    
    # 运行测试
    test_suite = TestXXXOperatorAscend()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
