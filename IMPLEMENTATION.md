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

当前阶段 1 已完成仓库分析和阻塞诊断，但 NVIDIA tester 的 AArch64 兼容对象尚未提供，因此官方功能测试仍处于验证前阻塞状态。

## 8. 阶段 2：RMSNorm 实施记录

实施日期：2026-08-05。

### 8.1 具体实现

`src/kernels.cu` 已完成 NVIDIA RMSNorm：

- 固定使用 256 threads/block，一个 block 处理一行，并以 grid-stride 支持超过 65535 行；
- 每个线程跨步处理隐藏维度，不要求 `hidden_dim` 是 2 的幂或 block 大小的倍数；
- `float` 和 `half` 均转换为 FP32 计算平方和、均值及归一化；
- warp 内使用 `__shfl_down_sync` 归约，warp 间使用 shared memory 归约；
- block 共享 `inv_rms`，再将归一化结果乘权重并转换回目标类型；
- Host wrapper 检查空维度、`rows * hidden_dim` 溢出及输入尺寸，管理 H2D/D2H 和 device buffer；
- CUDA API、kernel launch 和同步 D2H 拷贝均使用 `RUNTIME_CHECK` 检查。

实现没有使用 CUB、Thrust、cuDNN 或其他库函数完成关键计算。

### 8.2 编译操作

由于官方 `tester_nv.o` 仍为 x86-64，本阶段先对 AArch64 CUDA 对象做独立编译：

```bash
make clean
nvcc -std=c++17 -O3 -DPLATFORM_NVIDIA \
  -Xcompiler=-Wall,-Wextra \
  -c src/kernels.cu -o src/kernels.o
```

结果：RMSNorm 代码编译成功。编译器只报告尚未实施的 `flashAttention` 参数未使用警告，没有 RMSNorm 编译警告或错误。

### 8.3 本机 smoke tester

为在当前 AArch64 GPU 环境验证计算结果，在 `/tmp/rms_smoke.cu` 编写了不纳入仓库的独立测试程序。该程序：

1. 生成确定性的 input 和 weight；
2. 调用仓库中的 `rmsNorm<float>` 或 `rmsNorm<half>`；
3. 在 Host 端以 FP32 公式计算独立 reference；
4. 逐元素计算最大绝对误差；
5. 覆盖非 2 的幂、跨 warp、跨 block 步进和超出 grid 上限的行数。

构建和运行命令：

```bash
nvcc -std=c++17 -O3 /tmp/rms_smoke.cu src/kernels.o -o /tmp/rms_smoke
/tmp/rms_smoke
```

实际结果：

| 类型 | rows | hidden_dim | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| float | 5 | 1 | 2.98023e-08 |
| half | 5 | 1 | 1.63913e-06 |
| float | 5 | 3 | 5.96046e-08 |
| half | 5 | 3 | 1.96099e-04 |
| float | 5 | 33 | 5.96046e-08 |
| half | 5 | 33 | 9.63211e-04 |
| float | 5 | 257 | 1.19209e-07 |
| half | 5 | 257 | 9.60588e-04 |
| float | 5 | 4096 | 2.08616e-07 |
| half | 5 | 4096 | 9.29117e-04 |
| float | 65537 | 1 | 2.98023e-08 |

测试最终输出：

```text
RMS_SMOKE_PASS
```

float 容差设为 `2e-5`，half 容差设为 `2e-3`，所有用例均通过。

### 8.4 Compute Sanitizer

执行：

```bash
compute-sanitizer --tool memcheck --error-exitcode=1 /tmp/rms_smoke
compute-sanitizer --tool racecheck --error-exitcode=1 /tmp/rms_smoke
```

实际结果：

```text
RMS_SMOKE_PASS
========= ERROR SUMMARY: 0 errors

RMS_SMOKE_PASS
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

### 8.5 独立代码审查

对归约、同步、half 数值路径、Host 尺寸处理和平台假设进行了独立审查。审查未发现计算正确性的阻塞问题，并确认：

- 当前固定 256-thread launch 下，两级归约覆盖 8 个 warp；
- grid-stride 每轮的共享状态由完整 block 同步隔离；
- half 在 FP32 中累加并在输出时转换；
- 后续补充了 `rows * hidden_dim` 的 `size_t` 溢出检查；
- warp size 32 和 `__shfl_down_sync` 是 NVIDIA 路径已验证的假设，Iluvatar 仍需对应工具链和硬件验证。

### 8.6 当前验证结论与阻塞

本阶段已经完成：

- AArch64 CUDA 13.0 编译；
- 自建 reference 的 float/half 正确性验证；
- 非 2 的幂和大隐藏维度验证；
- `rows > 65535` 的 grid-stride 验证；
- memcheck 和 racecheck。

尚未完成：

```bash
SKIP_ATTENTION=1 make PLATFORM=nvidia run VERBOSE=true
```

原因仍是仓库的 `tester/tester_nv.o` 为 x86-64，而当前主机为 AArch64。以上结果证明本机测试覆盖内的实现正确且无 sanitizer 报错，但不能替代官方 tester 的尺寸表和容差。阶段 2 在获得 AArch64 tester 或 x86-64 NVIDIA 运行环境前不标记为“官方验证完成”。


## 9. x86-64 环境的官方测试命令

当前仓库的 `tester/tester_nv.o` 是 x86-64 对象，因此必须在 **x86-64 Linux + NVIDIA GPU** 环境中执行以下命令。不要把当前 AArch64 环境生成的 `src/kernels.o` 或可执行文件复制到 x86-64，也不要把 x86-64 的 `tester_nv.o` 与 AArch64 对象混合链接。

### 9.1 环境检查

```bash
uname -m
file tester/tester_nv.o
nvcc --version
nvidia-smi
```

期望至少满足：

```text
uname -m                         -> x86_64
tester/tester_nv.o               -> ELF 64-bit LSB relocatable, x86-64
```

确认工作目录是仓库根目录，并确保使用当前仓库中的测试对象：

```bash
pwd
ls -l tester/tester_nv.o src/kernels.cu Makefile
```

### 9.2 构建

先清理其他平台或旧架构产生的构建产物，再构建 NVIDIA 版本：

```bash
make clean
make PLATFORM=nvidia build
```

如需查看详细编译命令：

```bash
make clean
make PLATFORM=nvidia VERBOSE=true build
```

### 9.3 RMSNorm 分项测试

```bash
SKIP_ATTENTION=1 make PLATFORM=nvidia run VERBOSE=true
```

或直接运行已经构建的测试器：

```bash
SKIP_ATTENTION=1 ./test_kernels --verbose
```

确认输出中的 RMSNorm 测试全部通过后，将实际输出中的测试数量、误差和耗时追加到本文件的“阶段 2：RMSNorm 实施记录”中。若测试失败，应保留失败用例、`Max Diff`、`Max Tolerance` 和完整命令，不要记录为通过。

### 9.4 Flash Attention 分项测试

当前阶段 3 或阶段 4 实现完成后运行：

```bash
SKIP_RMS_NORM=1 make PLATFORM=nvidia run VERBOSE=true
```

或：

```bash
SKIP_RMS_NORM=1 ./test_kernels --verbose
```

应确认 `float`、`half`、causal、non-causal 和 GQA 用例均通过。

### 9.5 完整回归测试

```bash
make PLATFORM=nvidia run VERBOSE=true
```

该命令同时运行 RMSNorm 和 Flash Attention。每次修改 kernel 后都应先执行：

```bash
make clean
make PLATFORM=nvidia build
```

避免旧的 `src/kernels.o` 影响结果。

### 9.6 NVIDIA Compute Sanitizer

在 x86-64 环境中，先完成普通测试，再运行 sanitizer：

```bash
compute-sanitizer --tool memcheck \
  --error-exitcode=1 \
  ./test_kernels --verbose

compute-sanitizer --tool racecheck \
  --error-exitcode=1 \
  ./test_kernels --verbose

compute-sanitizer --tool initcheck \
  --error-exitcode=1 \
  ./test_kernels --verbose
```

也可以只检查某个算子：

```bash
SKIP_ATTENTION=1 compute-sanitizer --tool memcheck \
  --error-exitcode=1 ./test_kernels --verbose

SKIP_RMS_NORM=1 compute-sanitizer --tool memcheck \
  --error-exitcode=1 ./test_kernels --verbose
```

判定标准：`memcheck` 不得报告非法访问，`racecheck` 不得报告真实竞争，`initcheck` 不得报告未初始化读取。sanitizer 运行时间通常明显长于普通测试，不应将其耗时用于性能比较。

### 9.7 性能记录

朴素 Attention 和 online-softmax Attention 必须使用相同的硬件、CUDA 版本、编译参数、测试对象和运行方式进行比较。建议先使用优化构建：

```bash
make clean
make PLATFORM=nvidia CFLAGS="-std=c++17 -O3" build
```

然后至少执行五次：

```bash
for run in 1 2 3 4 5; do
  echo "=== performance run ${run} ==="
  make PLATFORM=nvidia run VERBOSE=true
 done
```

记录 warm-up 规则、tester 输出的耗时、运行次数和中位数；不要使用 sanitizer 的耗时作为性能结果。若 tester 只输出聚合耗时，应明确注明无法从公开输出还原单个 shape 的独立耗时。



## 10. 提交与产物规则

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
