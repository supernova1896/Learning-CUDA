# CUDA 算子实施与验证记录

## 1. 项目概况

本仓库是 InfiniTensor CUDA 方向的多平台 GPU 算子作业骨架，要求实现：

- `rmsNorm`：对二维张量最后一维执行 RMS 归一化并乘逐列权重；
- `flashAttention`：实现与 `torch.nn.functional.scaled_dot_product_attention` 对应的核心语义，支持 causal masking、GQA、`float` 和 `half`。

仓库当前公开源码包括：

| 文件 | 用途 |
| --- | --- |
| `src/kernels.cu` | NVIDIA 与 Iluvatar CoreX 共用的学生实现 |
| `src/kernels.maca` | MetaX 学生实现 |
| `src/kernels.mu` | Moore Threads 学生实现 |
| `tester/utils.h` | NVIDIA、Iluvatar、MetaX、Moore 运行时错误检查宏 |
| `tester/tester_*.o` | 各平台预编译测试器 |
| `Makefile` | 平台选择、编译、链接和运行入口 |

截至基线阶段，三个 kernel 文件中的 `rmsNorm` 和 `flashAttention` 都是 TODO。函数末尾提供的 `float`/`half` 显式模板实例是预编译测试器所需的固定 ABI，不应修改函数签名或删除实例。

## 2. 接口与数据布局

### 2.1 RMSNorm

输入和输出均为 row-major：

```text
input:  [rows, hidden_dim]
weight: [hidden_dim]
output: [rows, hidden_dim]
```

每行独立计算：

```text
mean_square = sum(input[i, j]^2) / hidden_dim
output[i, j] = input[i, j] * rsqrt(mean_square + eps) * weight[j]
```

`half` 输入的平方和、均值和归一化系数应使用 FP32 计算，最后转换回 `half`，避免低精度累加放大误差。

### 2.2 Attention

接口张量布局为：

```text
Q: [batch_size, target_seq_len, query_heads, head_dim]
K: [batch_size, src_seq_len, kv_heads, head_dim]
V: [batch_size, src_seq_len, kv_heads, head_dim]
O: [batch_size, target_seq_len, query_heads, head_dim]
```

计算包含 scaled QK、稳定 softmax 和 PV。GQA 要求 `query_heads` 可被 `kv_heads` 整除，query head 到 KV head 的映射为：

```text
kv_head = query_head / (query_heads / kv_heads)
```

causal mask 采用左上对齐的下三角可见区域，即 source 位置满足 `source_position <= target_position` 时可见。

## 3. 构建和测试体系

`Makefile` 根据 `PLATFORM` 选择工具链、学生源码和预编译测试对象：

| PLATFORM | 编译器 | 学生源码 | 测试对象 | C++ 标准 |
| --- | --- | --- | --- | --- |
| `nvidia` | `nvcc` | `src/kernels.cu` | `tester/tester_nv.o` | C++17 |
| `iluvatar` | `clang++` | `src/kernels.cu` | `tester/tester_iluvatar.o` | C++17 |
| `metax` | `mxcc` | `src/kernels.maca` | `tester/tester_metax.o` | C++17 |
| `moore` | `mcc` | `src/kernels.mu` | `tester/tester_moore.o` | C++11 |

测试器仅以二进制对象分发。符号检查确认 NVIDIA 测试器包含 RMSNorm CPU reference、Attention reference、`float`/`half` 测试入口，并引用学生代码的四个显式模板实例。测试器支持：

```bash
make VERBOSE=true
SKIP_ATTENTION=1 make
SKIP_RMS_NORM=1 make
```

二进制中的可见诊断表明测试覆盖不同 RMSNorm 维度，以及 Attention 的 causal、GQA、不同 head/sequence 组合。具体测试源码、完整尺寸表和容差并未公开，因此所有正确性结论必须来自实际运行结果。

## 4. 基线环境

基线采集日期：2026-08-05。

```text
Host architecture: aarch64
Kernel: Linux 6.11.0-1014-nvidia
GPU: NVIDIA GB10
Compute capability: 12.1
Driver: 580.82.09
CUDA Toolkit: 13.0
nvcc: V13.0.88
```

本机可找到 `nvcc`，未找到 Iluvatar `clang++`、MetaX `mxcc` 或 Moore `mcc` 工具链。因此本次只能尝试 NVIDIA 构建，不能宣称国产平台已经编译或运行验证。

## 5. 基线操作与结果

### 5.1 执行命令

```bash
make clean
make PLATFORM=nvidia build
SKIP_ATTENTION=1 make PLATFORM=nvidia run VERBOSE=true
SKIP_RMS_NORM=1 make PLATFORM=nvidia run VERBOSE=true
```

另外执行以下命令检查宿主和对象文件架构：

```bash
uname -m
file src/kernels.o tester/tester_nv.o
readelf -h tester/tester_nv.o
```

### 5.2 实际结果

`src/kernels.cu` 编译成功：

```text
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA -c src/kernels.cu -o src/kernels.o
```

链接失败，`ld` 重复报告：

```text
/usr/bin/ld: tester/tester_nv.o: Relocations in generic ELF (EM: 62)
collect2: error: ld returned 1 exit status
make: *** [Makefile:98：test_kernels] 错误 1
```

状态汇总：

```text
build=2
rms=2
attention=2
```

对象文件检查给出直接根因：

```text
src/kernels.o:      ELF 64-bit LSB relocatable, ARM aarch64
tester/tester_nv.o: ELF 64-bit LSB relocatable, x86-64
```

当前主机为 AArch64，CUDA 编译生成 ARM64 host object；仓库中的 NVIDIA 测试器是 x86-64 object。不同 host ISA 的可重定位对象不能链接，因此基线被测试器 ABI/架构不匹配阻塞，而不是被 TODO 函数的 CUDA 编译错误阻塞。

由于可执行文件未生成，两项选择性测试并未真正进入 tester，当前不能记录 RMSNorm 或 Attention 的运行正确性结果。

### 5.3 继续验证所需条件

后续实现可以在当前 AArch64 环境中做静态编译检查，但要满足“验证无误”并形成每阶段提交点，必须具备以下任一条件：

1. 获得由 AArch64 CUDA 工具链编译、且接口相同的 `tester/tester_nv.o`；或
2. 在带 NVIDIA GPU、CUDA 工具链的 x86-64 Linux 环境中使用仓库现有 `tester/tester_nv.o`；或
3. 由作业提供方提供 tester 源码，在当前 AArch64 环境重新编译。

不能通过链接器参数把 x86-64 host object 转换为 AArch64，也不应绕过预编译 tester 后声称官方测试通过。

## 6. 推荐实施顺序

### 阶段 1：分析与基线

- 写入本文件；
- 记录接口、测试结构、实际环境和当前阻塞；
- 不把未运行测试写成通过。

### 阶段 2：RMSNorm

- 一个 CUDA block 处理一行，线程跨列加载；
- FP32 平方和；
- warp shuffle 与 shared memory 两级归约；
- 支持非 2 的幂和非 block 倍数的 `hidden_dim`；
- 运行 `float`/`half` tester、`compute-sanitizer` memcheck 和 racecheck。

### 阶段 3：朴素 Attention 正确性版本

- 先计算完整 FP32 scores；
- 独立稳定 softmax；
- 独立 PV；
- 验证布局、causal mask、GQA、`float` 和 `half`；
- 保留为 Git 历史中的正确性检查点，并记录性能基线。

### 阶段 4：tiled online-softmax Attention

- 按 source tile 流式维护 running max、running sum 和 FP32 输出累加器；
- 不再分配完整 `[B,T,Hq,S]` scores；
- causal 时跳过不可见 source；
- 对比朴素版本的相同条件中位数性能；
- 将 NVIDIA 默认优化级别由 `-O0` 调整为 `-O3`，不硬编码当前 GPU 架构。

### 阶段 5：最终验证

- 分项和全量 tester；
- memcheck、racecheck、initcheck；
- 清理构建产物；
- 文档封板；
- 仅陈述实际验证过的平台、环境和结果。

## 7. 每阶段验证门槛

除分析基线阶段外，任一阶段出现以下情况均不得标记完成：

- 对应 `float` 或 `half` 测试失败；
- causal、non-causal 或 GQA 用例失败；
- kernel launch/runtime error；
- `compute-sanitizer` 报告非法访问、未初始化读取或真实数据竞争；
- 完整回归破坏之前已通过的算子；
- 文档记录与真实命令输出不一致。

当前阶段 1 已完成仓库分析和阻塞诊断，但 NVIDIA tester 的 AArch64 兼容对象尚未提供，因此功能阶段仍处于验证前阻塞状态。

## 8. 提交与产物规则

每个阶段都应在本文件中追加：

- 修改内容；
- 具体操作；
- 完整验证命令；
- 实际通过/失败结果；
- 环境、限制和未验证项。

只精确暂存对应源码和文档，不使用 `git add .` 或 `git add -A`。以下内容不得提交：

- `.claude/settings.local.json`；
- `test_kernels`；
- `src/kernels.o`；
- sanitizer 临时输出和其他本机构建产物。
