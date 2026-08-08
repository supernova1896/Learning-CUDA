# 从 AArch64 到 1660Ti 再到 RTX4090 的验证历程记录

## 1. 背景

这份报告记录的是我完成这个 CUDA 作业时，实际经历的验证路径、遇到的问题，以及为了解决这些问题做过的探索。

这个作业的目标不是只让代码“能编译”，而是让仓库里的官方 tester 能在可验证的环境里完整跑通，并且通过正确性和 sanitizer 检查。实际过程中，我先在本机 AArch64 主机上尝试，随后切到自己另一台 1660Ti 电脑的 WSL 环境，最终才在主办方提供的 RTX4090 服务器上完成闭环验证。

## 2. 第一阶段：先在 AArch64 主机上尝试

### 2.1 当时的判断

最开始我是在当前这台 AArch64 主机上做验证的。这个环境的好处是本地就能直接改代码、编译、观察错误；但很快就发现，它并不适合作为官方 tester 的最终验证环境。

### 2.2 遇到的问题

仓库里的 NVIDIA 官方 tester 是预编译的 `tester/tester_nv.o`，而这份对象文件是 **x86-64** 架构。AArch64 主机上用 `nvcc` 编出来的是 AArch64 host object，和 x86-64 tester 根本不能混合链接。

实际表现是链接阶段直接失败，典型错误包括：

```text
/usr/bin/ld: tester/tester_nv.o: Relocations in generic ELF (EM: 62)
/usr/bin/ld: tester/tester_nv.o: error adding symbols: file in wrong format
```

这说明问题不是 kernel 逻辑，而是 **主机架构与官方 tester ABI 不兼容**。

### 2.3 我做过的探索

我在这台机器上确认过：

- `nvcc` 能正常编译 `src/kernels.cu`；
- 本地源码本身没有语法问题；
- 但无法把生成的对象文件和官方 tester 正常链接到一起；
- 所以这里最多只能做“编译检查”，不能做“官方 tester 级别的最终验证”。

### 2.4 结论

AArch64 主机证明了代码可以进入编译阶段，但不能证明官方 tester 在这台机器上可运行。对这个作业来说，它不是最终验证环境。

## 3. 第二阶段：切到我自己的 1660Ti 电脑

### 3.1 为什么会换到这台机器

AArch64 主机走不通以后，我又尝试转到自己另一台 NVIDIA GTX 1660 Ti 的 WSL 电脑上，希望它能承担官方 tester 的验证任务。这个环境至少是 x86_64，而且也有 NVIDIA GPU，看上去更接近官方 tester 的运行条件。

### 3.2 遇到的问题

这台机器上最先遇到的不是 kernel 计算错误，而是 **运行时符号和链接 ABI 问题**。

最典型的错误是：

```text
undefined reference to `cudaGetDeviceProperties_v2'
```

我还检查过动态库里的导出符号，发现系统里的 CUDA runtime 导出的是：

```text
cudaGetDeviceProperties@@libcudart.so.13
```

而官方 tester 需要的是 `cudaGetDeviceProperties_v2`。这说明仅仅改链接参数、改库路径，并不能把它修好。

### 3.3 我做过的探索

为了让它跑起来，我尝试过很多方向：

- 显式指定 `-lcudart` 和不同的 `-L` 路径；
- 尝试静态链接 `cudart`；
- 补充 `-ldl`、`-lrt`、`-pthread` 等链接参数；
- 多次调整 `nvcc` 和 host linker 的参数组合；
- 反复确认 tester 本体和 CUDA runtime 的符号是否一致。

但这些探索最终都没有解决核心问题。

### 3.4 根因判断

我最后确认，这不是“少一个 flag”那么简单，而是 **官方预编译 tester 和这台 WSL 环境里的 CUDA runtime / 链接 ABI 组合不兼容**。

也就是说，这台 1660Ti 电脑可以做编译和局部 smoke test，但不能作为官方 tester 的最终判定环境。

### 3.5 结论

1660Ti 这一步的重要价值在于，它让我确认了：

- 问题不是我的 kernel 代码先天就不行；
- 也不是单纯再改几个链接参数就能解决；
- 需要找一个和官方 tester 更匹配的 x86_64 Linux + NVIDIA 环境。

## 4. 第三阶段：切到主办方提供的 RTX4090 服务器

### 4.1 为什么这一步能继续往下走

在前两个环境都跑不通以后，我切换到了主办方提供的 RTX4090 服务器。这一步才真正把验证工作推进到“官方 tester 可完整运行”的方向上。

这个服务器的环境是：

- `x86_64`
- NVIDIA GeForce RTX 4090
- CUDA 12.8
- `nvcc` 可用

它和官方 tester 的架构、运行方式都匹配，这一点非常关键。

### 4.2 一开始的现象

在 RTX4090 上，RMSNorm 很快通过，但 Attention 的 `float` 路径最初还有少量边界 case 失败，典型是 `#6` 和 `#14`。

也就是说，这一步已经不再是“能不能跑”，而是进入了真正的 **数值一致性调试** 阶段。

### 4.3 我做过的探索

我围绕 Attention 做过几轮探索：

1. **提高局部精度**
   - 例如把部分中间量改成 `double`；
   - 让某些累加更稳定；
   - 但边界失败没有消失。

2. **把整条 float 路径扩大成 double**
   - 结果反而更差；
   - 说明 tester 的 reference 更接近稳定的 float 舍入轨迹，而不是“全 double 再转回 float”。

3. **改成两遍 stable softmax**
   - 先求 `row_max`，再求 `row_sum`，最后做 `P * V`；
   - 这是比 online-softmax 更保守的数值路径；
   - 但我也发现，光靠这个还不够，最终还是要让 scale 的舍入轨迹和 CPU reference 对齐。

### 4.4 真正起作用的修复

最后真正让 RTX4090 官方 tester 通过的关键，是把 `float` 路径里的 scale 计算改成：

```cpp
const float inv_scale = static_cast<float>(
    1.0 / sqrt(static_cast<double>(head_dim)));
```

也就是说，问题不只是 softmax 的结构，而是 **scale 的数值轨迹必须和 tester 的 CPU reference 一致**。

修完以后：

- Attention `float` 通过了；
- `#6` 和 `#14` 的边界 case 也过了；
- `half` 本来就通过，这次也继续保持通过；
- `memcheck`、`racecheck`、`initcheck` 全部是 `0 errors` / `0 hazards`。

## 5. 这次经历里最重要的几个结论

### 5.1 环境比我想象得更重要

这个作业不是“任意一台有 CUDA 的机器都能完成验证”。

- AArch64 主机：能编译，但和官方 tester 架构不兼容；
- 1660Ti WSL：看起来更接近，但卡在 runtime / ABI 符号不一致；
- RTX4090 服务器：才是真正能完成官方验证闭环的环境。

### 5.2 预编译 tester 的兼容性不能靠猜

只要 tester 是预编译对象，架构、runtime、符号版本就都不能乱猜。很多时候错误不是 kernel 本身，而是测试器和环境不匹配。

### 5.3 数值问题和结构问题要分开看

Attention 的最终失败不是大逻辑错误，而是 float 路径的数值轨迹和 reference 有细微偏差。这个过程让我确认：

- 先解决环境问题；
- 再解决结构正确性；
- 最后才是数值一致性和性能。

## 6. 总结

这次作业的完成过程，实际上是一个从“环境不匹配”到“数值边界调试”再到“官方验证闭环”的过程。

我先在 AArch64 主机上确认了官方 tester 不能直接用；然后在自己的 1660Ti WSL 上又排除了 runtime / ABI 兼容性问题；最后在主办方提供的 RTX4090 服务器上完成了真正有效的官方测试，并把 Attention 的 float 边界问题修到了通过。

如果以后再做类似作业，我会优先把“官方 tester 是否与当前环境完全匹配”放在最前面，因为这一步会决定后面的大量工作是在正确轨道上推进，还是在不合适的环境里反复绕圈。
