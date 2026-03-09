# Triton算子迁移到昇腾NPU技能清单

**版本**: v1.0  
**更新日期**: 2025-01-18  
**用途**: 快速指导Triton算子从GPU迁移到昇腾NPU

---

## 🎯 核心技能速查

### 技能1: 设备适配（必须）

```python
# ❌ GPU代码
import torch
x = torch.randn(1024, device='cuda')

# ✅ NPU代码
import torch
import torch_npu
x = torch.randn(1024, device='npu')
```

**检查点**:
- [ ] 所有 `device='cuda'` 改为 `device='npu'`
- [ ] 删除 `torch.cuda.is_available()` 检查
- [ ] 导入 `torch_npu` 模块

---

### 技能2: Grid分核调整（关键）

**原理**: NPU的grid必须绑定物理核，不能像GPU那样自由定义逻辑维度。

```python
# 步骤1: 获取物理核数
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_CORES = properties["num_aicore"]  # 或 num_vectorcore

# 步骤2: 修改kernel启动方式
# ❌ GPU写法
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

# ✅ NPU写法（方案1: 固定核数 + 跨步分配）
@triton.jit
def kernel(..., NUM_CORE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # 跨步分配任务
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # 处理当前block
        ...

grid = (NUM_CORES,)
kernel[grid](..., NUM_CORE=NUM_CORES)

# ✅ NPU写法（方案2: 环境变量自动优化）
export TRITON_ALL_BLOCKS_PARALLEL=1
# 然后可以继续使用GPU的grid写法
```

**检查点**:
- [ ] 获取物理核数（num_aicore 或 num_vectorcore）
- [ ] Grid大小固定为物理核数
- [ ] Kernel内部使用跨步循环处理所有任务
- [ ] 或启用 `TRITON_ALL_BLOCKS_PARALLEL=1`

---

### 技能3: 逻辑运算符转换（最关键！）

**⚠️ 这是最常见的错误！**

```python
# ❌ 错误写法（导致精度误差或NaN）
mask1 = offsets < n_elements
mask2 = offsets >= 0
valid_mask = mask1 and mask2  # 错误！

# ✅ 正确写法
valid_mask = mask1 & mask2  # 正确！

# 其他运算符
# ❌ mask1 or mask2
# ✅ mask1 | mask2

# ❌ not mask
# ✅ ~mask
```

**检查点**:
- [ ] 搜索所有 `and` 关键字，改为 `&`
- [ ] 搜索所有 `or` 关键字，改为 `|`
- [ ] 搜索所有 `not` 关键字，改为 `~`
- [ ] 特别注意mask运算、条件表达式、where语句

**实际案例**: index_select算子迁移中，将 `and` 改为 `&` 后，精度从NaN恢复正常。

---

### 技能4: 内存对齐优化

**要求**:
- **VV算子**（仅Vector Core）: 尾轴大小需被 **32字节** 整除
- **CV算子**（Cube + Vector）: 尾轴大小需被 **512字节** 整除

```python
# 数据类型字节数
# float16/bfloat16: 2字节
# float32: 4字节
# int32: 4字节

# 示例：float16类型，VV算子
# 尾轴大小需满足: size * 2 % 32 == 0
# 即 size % 16 == 0

def check_alignment(tensor_shape, dtype, kernel_type='VV'):
    """检查尾轴对齐"""
    element_size = torch.tensor([], dtype=dtype).element_size()
    last_dim = tensor_shape[-1]
    
    if kernel_type == 'VV':
        alignment = 32
    else:  # CV
        alignment = 512
    
    required_alignment = alignment // element_size
    is_aligned = (last_dim % required_alignment) == 0
    
    if not is_aligned:
        print(f"⚠️ 尾轴{last_dim}不满足对齐要求（需被{required_alignment}整除）")
    
    return is_aligned

# 转置优化示例
# 原始: tensor [2048, 3], bfloat16
# 问题: 尾轴3不满足16的倍数
# 解决: 借轴转置
conv_state = tl.load(conv_state_ptr + ...)
conv_state_T = conv_state.reshape(128, 16*3).trans().reshape(16, 3*128).trans()
```

**检查点**:
- [ ] 检查输入张量尾轴是否满足对齐要求
- [ ] 不满足时考虑转置或padding
- [ ] VV算子: 尾轴 % (32/element_size) == 0
- [ ] CV算子: 尾轴 % (512/element_size) == 0

---

### 技能5: Autotune自动调优

```python
import triton

# 方案1: 手动配置（社区版）
def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 2048, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 4096, 'multibuffer': False}),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['n_elements'],  # 触发重新调优的参数
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    # kernel实现
    pass

# 方案2: 自动生成配置（进阶版）
import triton.backends.ascend.runtime  # 必须导入

@triton.autotune(
    configs=[],  # 空列表，自动生成
    key=['n_elements'],
    hints={
        'split_params': {'x': 'BLOCK_SIZE'},
        'tiling_params': {'x': 'BLOCK_SUB'},
        'low_dim_axes': ['x'],
        'reduction_axes': [],
    }
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SUB: tl.constexpr):
    pass
```

**检查点**:
- [ ] 为kernel添加 `@triton.autotune` 装饰器
- [ ] 定义候选配置或使用自动生成
- [ ] 设置 `key` 参数（触发重新调优的参数）
- [ ] 导入 `triton.backends.ascend.runtime`（进阶版）

---

### 技能6: Tiling分块优化

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SUB: tl.constexpr):
    """使用Tiling优化性能"""
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    
    # 计算子块数量
    num_sub_blocks = BLOCK_SIZE // BLOCK_SUB
    
    # 循环处理每个子块
    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SUB)
        mask = offsets < n_elements
        
        # 加载、计算、存储
        data = tl.load(in_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(out_ptr + offsets, result, mask=mask)
```

**适用场景**:
- UB空间溢出时
- 大BLOCK_SIZE性能不佳时
- 需要更细粒度的内存访问控制时

**检查点**:
- [ ] 识别是否需要Tiling（UB溢出、性能问题）
- [ ] 添加 `BLOCK_SUB` 参数
- [ ] 实现循环分块处理逻辑

---

### 技能7: 存算并行优化

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用 care_padding=False 提升并行度
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    
    result = compute(x)
    
    tl.store(out_ptr + offsets, result, mask=mask)
```

**检查点**:
- [ ] 在 `tl.load()` 中添加 `care_padding=False`
- [ ] 确保mask正确性
- [ ] 验证精度无影响

---

### 技能8: 数据类型优化

```python
# ❌ 避免使用int64（性能差）
@triton.jit
def slow_kernel(..., BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)  # 默认int64
    x = tl.load(x_ptr + offsets)  # Vector ADD退化为标量运算
    ...

# ✅ 使用int32
@triton.jit
def fast_kernel(..., BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int32)  # 转为int32
    x = tl.load(x_ptr + offsets)  # 高效向量运算
    ...
```

**检查点**:
- [ ] 检查 `tl.arange()` 返回值类型
- [ ] 添加 `.to(tl.int32)` 转换
- [ ] 验证性能提升

---

## 🔧 调试技能

### 调试1: NaN问题定位

```python
import torch
import torch_npu

def debug_nan():
    """NaN问题调试"""
    torch.manual_seed(42)
    x = torch.randn(1024, device='npu', dtype=torch.float16)
    
    result = triton_kernel(x)
    
    # 检查NaN
    if torch.isnan(result).any():
        nan_mask = torch.isnan(result)
        nan_indices = torch.where(nan_mask)
        print(f"❌ 发现NaN！位置: {nan_indices}")
        print(f"对应输入值: {x[nan_indices]}")
        
        # 检查输入
        print(f"输入是否包含NaN: {torch.isnan(x).any()}")
        print(f"输入是否包含Inf: {torch.isinf(x).any()}")
        
        # 检查逻辑运算符（最常见原因）
        print("⚠️ 请检查kernel中是否使用了 'and'/'or' 而非 '&'/|'")
    else:
        print("✅ 无NaN问题")
```

### 调试2: 解释器模式对比

```python
import os

def debug_with_interpreter():
    """使用CPU解释器模式对比结果"""
    x = torch.randn(1024, device='npu', dtype=torch.float16)
    
    # CPU解释器模式
    os.environ['TRITON_INTERPRET'] = '1'
    result_cpu = triton_kernel(x)
    
    # NPU模式
    os.environ['TRITON_INTERPRET'] = '0'
    result_npu = triton_kernel(x)
    
    # 对比
    diff = torch.abs(result_cpu - result_npu)
    print(f"最大差异: {diff.max()}")
    print(f"平均差异: {diff.mean()}")
    
    # 定位最大误差
    max_idx = torch.argmax(diff)
    print(f"最大误差位置: {max_idx}")
    print(f"CPU值: {result_cpu[max_idx]}")
    print(f"NPU值: {result_npu[max_idx]}")
```

### 调试3: IR文件分析

```bash
# 1. 启用调试输出
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1

# 2. 运行kernel
python your_kernel.py

# 3. 查看IR文件
cd ~/.triton/dump/<hash>/
cat kernel.ttir.mlir      # Triton IR
cat kernel.ttadapter.mlir # Ascend适配器IR
```

### 调试4: 打印调试

```python
@triton.jit
def debug_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 打印中间值
    tl.device_print("pid:", pid)
    tl.device_print("offsets:", offsets)
    
    x = tl.load(x_ptr + offsets, mask=offsets < n)
    tl.device_print("loaded x:", x)
    
    out = x * 2
    tl.store(out_ptr + offsets, out, mask=offsets < n)
```

---

## ✅ 迁移检查清单

### 阶段1: 准备工作

- [ ] 确认Triton-Ascend已正确安装
- [ ] 确认torch-npu可用
- [ ] 备份原始GPU代码
- [ ] 准备PyTorch参考实现
- [ ] 设计精度测试用例

### 阶段2: 代码修改

- [ ] **设备适配**: `device='cuda'` → `device='npu'`
- [ ] **导入模块**: 添加 `import torch_npu`
- [ ] **删除检查**: 删除 `torch.cuda.is_available()` 等
- [ ] **Grid分核**: 获取物理核数，修改grid逻辑
- [ ] **逻辑运算符**: `and` → `&`, `or` → `|`, `not` → `~`
- [ ] **内存对齐**: 检查尾轴是否满足对齐要求
- [ ] **Autotune**: 添加自动调优配置
- [ ] **Tiling**: 如需要，添加分块处理
- [ ] **存算并行**: 添加 `care_padding=False`
- [ ] **数据类型**: `tl.arange()` 添加 `.to(tl.int32)`

### 阶段3: 测试验证

- [ ] **基础功能**: 小规模数据测试
- [ ] **精度测试**: 多种数据类型（float16/32/bfloat16）
- [ ] **边界情况**: 空张量、单元素、大索引
- [ ] **数值稳定性**: NaN、Inf检查
- [ ] **性能测试**: 与GPU版本对比

### 阶段4: 性能优化

- [ ] 启用Autotune
- [ ] 使用Tiling分块
- [ ] 启用存算并行
- [ ] 优化数据类型
- [ ] 检查内存对齐
- [ ] 使用性能分析工具（msprof）

---

## 🚨 常见问题速查

### 问题1: coreDim超限

**错误**: `coreDim=xxxx can't be greater than UINT16_MAX`

**解决**:
```bash
# 方案1: 启用自动优化
export TRITON_ALL_BLOCKS_PARALLEL=1

# 方案2: 增大BLOCK_SIZE
BLOCK_SIZE = max(32768, triton.next_power_of_2(triton.cdiv(N, 65535)))
```

### 问题2: UB空间溢出

**错误**: `ub overflow, requires xxxx bits while 1572684 bits available!`

**解决**:
```python
# 方案1: 减小BLOCK_SIZE
BLOCK_SIZE = 1024  # 从2048减小

# 方案2: 使用Tiling
for i in range(BLOCK_SIZE // BLOCK_SUB):
    # 每次只处理BLOCK_SUB大小的数据
```

### 问题3: 精度误差或NaN

**症状**: 输出包含NaN或与参考结果差异大

**解决步骤**:
1. **检查逻辑运算符**（最常见）: `and` → `&`, `or` → `|`
2. **检查输入数据**: 是否包含NaN/Inf
3. **使用解释器模式**: 对比CPU和NPU结果
4. **查看IR文件**: 检查编译产物

### 问题4: 性能不佳

**解决步骤**:
1. 启用Autotune自动调优
2. 启用存算并行（`care_padding=False`）
3. 使用Tiling分块
4. 优化数据类型（避免int64）
5. 检查内存对齐

---

## 📚 环境变量速查

| 环境变量 | 作用 | 使用场景 |
|---------|------|---------|
| `TRITON_DEBUG=1` | 启用调试输出，生成IR文件 | 编译问题调试 |
| `TRITON_DISABLE_CACHE=1` | 禁用编译缓存 | 确保每次重新编译 |
| `TRITON_INTERPRET=1` | CPU解释器模式 | 精度问题调试 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动优化Grid分核 | coreDim超限问题 |
| `TRITON_BENCH_METHOD="npu"` | NPU性能采集方式 | Autotune性能测试 |

---

## 📖 参考资源

### 官方文档
- **Triton-Ascend文档**: `/home/w00664509/triton-ascend/docs/zh/`
  - 迁移指南: `migration_guide/migrate_from_gpu.md`
  - 架构差异: `migration_guide/architecture_difference.md`
  - 性能指南: `migration_guide/performance_guidelines.md`
  - 编程指南: `programming_guide.md`
  - 调试指南: `debug_guide/debugging.md`
  - FAQ: `FAQ.md`

### 实际案例
- **fused_recurrent_gated_delta_rule_fwd**: `/home/w00664509/vllm-ascend/vllm_ascend/ops/triton/`
- **index_select**: 同上目录
- **迁移总结**: `/home/w00664509/MIGRATION_SUMMARY.md`

### 工具模板
- **测试模板**: `/home/w00664509/vllm-ascend/vllm_ascend/ops/triton/test_template.py`
- **调试模板**: `/home/w00664509/vllm-ascend/vllm_ascend/ops/triton/debug_template.py`
- **完整文档**: `/home/w00664509/TRITON_AUTOMATED_MIGRATION_TO_ASCEND_NPU.md`

---

## 💡 最佳实践

### 1. 迁移顺序

```
设备适配 → Grid分核 → 逻辑运算符转换 → 内存对齐 → 精度测试 → 性能优化
```

### 2. 优先级排序

1. **🔴 必须修改**: 设备适配、逻辑运算符转换
2. **🟡 强烈建议**: Grid分核调整、内存对齐
3. **🟢 性能优化**: Autotune、Tiling、存算并行

### 3. 调试优先级

1. **逻辑运算符检查**（最常见问题）
2. **解释器模式对比**
3. **IR文件分析**
4. **打印调试**

### 4. 测试优先级

1. **基础精度测试**（小规模数据）
2. **边界情况测试**（空张量、大索引）
3. **数值稳定性测试**（NaN/Inf检查）
4. **性能测试**

---

**使用建议**:
- 新手: 按顺序完成"迁移检查清单"
- 老手: 直接查阅"核心技能速查"和"常见问题速查"
- 遇到问题: 查看"调试技能"和"常见问题速查"

**最后更新**: 2025-01-18
