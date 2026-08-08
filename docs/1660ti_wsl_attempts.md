# 1660Ti WSL 验证尝试与失败原因记录

## 1. 背景

这部分记录的是在另一台 NVIDIA GTX 1660 Ti 机器的 WSL 环境里，尝试使用仓库官方 tester 验证 `src/kernels.cu` 的过程。目标不是重新设计算法，而是确认这个环境能否承担“官方预编译 tester + 本地代码”的最终验证职责。

结论先写在前面：**这台 1660Ti WSL 机器不能作为官方 tester 的可靠验证环境**。它能做本地编译、部分 smoke test 和部分 CUDA sanitizer 检查，但在链接官方 `tester/tester_nv.o` 时遇到运行时符号/ABI 问题，无法形成可复现的官方验证闭环。

## 2. 环境与前置条件

在这台机器上确认到的信息包括：

- `uname -m`：`x86_64`
- GPU：`NVIDIA GeForce GTX 1660 Ti`
- CUDA Toolkit：`/usr/local/cuda-13.2`
- `nvcc`：`/usr/local/cuda-13.2/bin/nvcc`
- 仓库中的官方 tester：`tester/tester_nv.o` 为 `x86-64` ELF relocatable object

从架构上看，这台机器满足“宿主机是 x86_64”的基本要求；但这并不自动保证可以和预编译 tester 正常链接并运行。真正的问题出在 CUDA runtime、链接方式和测试器预编译对象之间的兼容性。

## 3. 多次尝试

### 3.1 直接构建

最初直接执行：

```bash
make PLATFORM=nvidia build
```

结果在链接阶段失败，报错围绕 `cudaGetDeviceProperties_v2`：

```text
undefined reference to `cudaGetDeviceProperties_v2'
```

这说明问题不在代码语法，而是在链接阶段找不到 tester 依赖的 CUDA runtime 符号。

### 3.2 尝试用系统默认 cudart

随后尝试显式指定 `-lcudart`，以及配合 `-L/usr/local/cuda/lib64`、`-L/usr/local/cuda-13.2/lib64` 等路径去找动态库。

同时通过：

```bash
nm -D /usr/local/cuda-13.2/targets/x86_64-linux/lib/libcudart.so | grep cudaGetDeviceProperties
```

确认动态库里导出的符号是：

```text
cudaGetDeviceProperties@@libcudart.so.13
```

但 tester 需要的却是 `cudaGetDeviceProperties_v2`。这表明仅靠切换路径并不能解决问题。

### 3.3 尝试静态链接 cudart

接着尝试把 runtime 改成静态链接，目的是绕开动态符号差异：

```bash
-cudart static
```

同时补充：

```bash
-ldl -lrt -pthread
```

这个方向里还踩过几个命令拼写问题，例如：

- 少空格导致 `staticEXTRA_LIBS=...` 被当作 `-cudart` 的参数；
- 把 `-pthread` 直接交给 `nvcc`，触发：

  ```text
  nvcc fatal: Unknown option '-pthread'
  ```

- 库参数拼接错误，写成了类似 `-L.../lib-lcudart_static -ldl-lrt`。

纠正后，仍然没有真正解决核心问题。

### 3.4 最终确认根因

即使把静态 `cudart`、`-Xcompiler=-pthread`、正确的 `-L... -lcudart_static -ldl -lrt` 都配齐，仍然报：

```text
undefined reference to `cudaGetDeviceProperties_v2'
```

这说明不是某个 flag 少写了，而是**官方 tester 的预编译对象和这台 WSL 环境中的 CUDA runtime / 链接 ABI 组合不兼容**。

## 4. 失败原因总结

失败不是单一原因，而是几个因素叠加：

1. **官方 tester 是预编译对象**，它的链接假设已经固定；
2. **当前 WSL 环境的 CUDA 13.2 runtime 导出符号和 tester 预期符号不一致**；
3. **静态链接并不能自动修复 ABI 兼容性**；
4. **1660 Ti 本身可以跑 CUDA，但这不代表能直接充当官方 tester 的最终验证机**。

一句话概括：这台机器能证明代码“可以编译”，但不能证明“官方 tester 在该环境下可完整运行”。

## 5. 采用的解决方案

最终采取的方案不是继续在这台机器上硬解，而是分层处理：

- 在 1660Ti WSL 上只做：
  - 本地编译检查；
  - 小规模 smoke test；
  - 必要时做 `compute-sanitizer` 检查。
- 把**官方 tester 的最终验证**迁移到和预编译对象更匹配的环境：
  - x86_64 Linux；
  - NVIDIA 官方 tester 可直接运行；
  - CUDA runtime 版本和 tester 兼容。

后续我把官方验证转到了 RTX4090 服务器上继续做。

## 6. 这次尝试给出的结论

- 1660Ti WSL **不是不可用**，但它**不适合作为官方 tester 的最终判定环境**；
- 遇到 `cudaGetDeviceProperties_v2` 这类 unresolved reference 时，优先怀疑 tester/runtime 兼容性，而不是先怀疑 kernel 逻辑；
- 如果目标是“官方 tester 结果”，应优先找和预编译对象匹配的 x86_64 Linux 环境，而不是继续在 WSL 里绕链接参数。
