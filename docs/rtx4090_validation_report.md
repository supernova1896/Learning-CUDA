# RTX4090 官方验证结果与修复记录

## 1. 背景

这份文档记录在 RTX4090 服务器上进行官方 tester 验证时的观察、修改方案和结果反馈。与 1660Ti WSL 不同，这台机器是我们最终真正用来跑仓库官方 `tester/tester_nv.o` 的环境，因此这里的结果具有更高的结论意义。

## 2. 远程环境

远程服务器通过 SSH 连接，配置为可复用的 `ControlMaster` 会话，避免重复输入密码。实际可用的环境信息如下：

- 架构：`x86_64`
- GPU：`NVIDIA GeForce RTX 4090`
- Driver：`570.124.06`
- CUDA Version：`12.8`
- 可用 `nvcc` 路径：`/usr/local/cuda-12.8/bin/nvcc`

在远程仓库 `/root/Learning-CUDA` 上，使用：

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH make clean
PATH=/usr/local/cuda-12.8/bin:$PATH make PLATFORM=nvidia build
```

可以成功完成构建。

## 3. 先前版本的结果反馈

在 RTX4090 上最早跑通的版本是本地已经合入的 online-softmax Attention 实现。这个版本的特点是：

- RMSNorm 已通过；
- Attention 的 `half` 用例已通过；
- Attention 的 `float` 只剩少量边界 case 失败。

当时官方 tester 的反馈里，出现了两个典型失败案例：

```text
Test Case #6 (Attention) float:
Max Diff: 0.0000178
Max Tolerance: 0.0000109
Verification: Failed
```

```text
Test Case #14 (Attention) float:
Max Diff: 0.0000143
Max Tolerance: 0.0000116
Verification: Failed
```

这说明问题非常集中：不是整体逻辑错，也不是内存错误，而是 **float 精度路径的数值偏差略高于 tester 容忍度**。

同时做过 `compute-sanitizer --tool memcheck`，结果是 0 errors，说明当时更像是数值问题，不是越界或非法访问。

## 4. 已尝试的修改方案

### 4.1 方案一：提高在线累加精度

先尝试把 online-softmax 里的部分中间量改成更高精度，包括：

- `row_sum` 改为 `double`
- 部分点积和输出累加使用 `fmaf`

这个方案的目标是尽量减少在线重缩放过程中的舍入误差。

实际结果：

- 改动后 `float` 的两个失败 case 仍然存在；
- 误差基本没有被显著压低；
- 说明主误差并不主要来自这几个局部变量的精度不够。

### 4.2 方案二：整条 float 路径改成 double

随后尝试把 Attention 的中间累加路径进一步扩大到 `double`，希望让整个 softmax/PV 链路更稳定。

实际结果反而更差：

- 原来只失败两个 case；
- 改成更大范围 double 路径后，更多 `float` case 开始失败。

这个结果说明 tester 的 reference 更接近“float stable softmax 的数值轨迹”，而不是“全 double 再回转成 float”的轨迹。过度改精度会让结果偏离 reference 的舍入方式。

### 4.3 方案三：改为两遍 stable softmax

当前采用的方案是把 Attention 从在线重缩放改成更直接的两遍 stable softmax：

1. 第一遍：整行求 `row_max`；
2. 第二遍：整行求 `row_sum`；
3. 第三步：按 tile 做 `P * V` 累加。

核心目标是：

- 避免 online rescale 带来的累计误差；
- 让数值路径更接近经典 stable softmax；
- 保持 `float` 和 `half` 的处理方式一致，只把中间统计量保留在 FP32。

同时把共享内存需求收回到只需要：

```cpp
kBlockSize * sizeof(float)
```

以匹配当前 tiled 读写方式。

## 5. 当前版本的结果反馈

这版代码已经完成了本地 `nvcc` 编译检查，说明语法和模板实例都没有问题。

我在本地已经清掉了两遍 softmax 改动后残留的未使用 double helper 和无用 include，使编译输出保持干净。

然后把当前版本同步到 RTX4090 服务器并重新运行官方 tester。到本文写作时，远程命令已经完成：

- `make clean`
- `make PLATFORM=nvidia build`
- `SKIP_RMS_NORM=1 make PLATFORM=nvidia run VERBOSE=true`

当前进度显示测试已经进入 `./test_kernels --verbose`，说明构建和链接都已通过，正在跑实际 case。

## 6. 这次修复方案的判断标准

这轮修改是否算成功，主要看下面几条：

- Attention `float` 是否把之前那两个边界失败 case 消掉；
- `half` 是否继续保持通过；
- RMSNorm 是否不受影响；
- `compute-sanitizer` 是否继续保持 0 errors；
- 远程官方 tester 是否在全部选择性用例中都通过。

如果最终两遍 stable softmax 版本通过，那么可以把它作为更保守的数值实现记录下来；如果仍然有 `float` case 边界失败，则继续针对 reference 的舍入路径做更细的定位，而不是盲目提升精度。

## 7. 目前可直接确认的结论

截至当前：

- 4090 环境已经成功用于官方构建与测试；
- 之前的 online-softmax 版本在 `float` 上有两个稳定失败 case；
- 单纯提高局部精度并不能解决问题；
- 两遍 stable softmax 是当前最合理的修复方向；
- 远程最终测试结果正在运行中，构建阶段已经通过。

这份文档后续如果官方 tester 最终给出全量结果，只需要在这一节补上具体通过/失败 case 即可。
