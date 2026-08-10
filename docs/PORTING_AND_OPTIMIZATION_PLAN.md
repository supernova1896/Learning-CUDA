# 国产平台适配与三平台性能优化计划

## 1. 计划结论

可以先适配两个国产 GPU 平台，再进行性能优化。这个顺序有利于尽早获得平台适配分，并在性能工作开始前确认不同编译器、运行时和硬件环境的实际差异。

本项目当前的硬件命名约定为：**C500 使用 MetaX 路径（`PLATFORM=metax`）**，**PH100 使用 Moore/MUSA 路径（`PLATFORM=moore`，用户内部命名为 S5000）**。

适配阶段应以正确性和可验证性为目标，不应在移植过程中同时引入深度性能优化。后续优化不要求三份代码逐行一致：共享算法只维护一套设计思路，线程组织、平台 intrinsic、shared memory 布局和参数允许按照 NVIDIA、MACA、MUSA 的硬件特性分别调优。

总体路线：

```text
NVIDIA 已验证基线
        ↓
MACA / MUSA 正确性适配
        ↓
两个国产平台分别通过官方 tester
        ↓
建立三平台 benchmark
        ↓
共享算法级优化
        ↓
各平台专属性能调优
```

## 2. 阶段一：国产平台正确性适配

### 2.1 适配目标

在不改变算子语义的前提下，分别完成两个国产平台的：

- `rmsNorm` 的 `float` 和 `half` 实现；
- `flashAttention` 的 `float` 和 `half` 实现；
- causal masking；
- GQA；
- 非等长 source/target sequence；
- 官方 tester 验证；
- 可用时执行对应平台的运行时错误检查工具。

移植阶段应以当前已经在 RTX 4090 上通过验证的 NVIDIA 实现作为语义基线，优先保持 stable softmax 的数值路径和现有接口不变。

### 2.2 推荐实施顺序

1. 先适配 `rmsNorm`，因为其线程组织、归约和数据访问模式更简单；
2. 在第一个国产平台上完成 `rmsNorm` 的编译与正确性验证；
3. 将已确认的适配经验应用到第二个平台；
4. 分别完成两个平台的 `flashAttention`；
5. 每个平台完成独立的全量回归后，再开始性能优化。

如果两个平台的编译和测试环境可以同时使用，可以并行进行平台适配；但每个平台仍应单独记录编译命令、测试输出和失败原因。

### 2.3 需要重点确认的差异

CUDA、MACA 和 MUSA 都采用相近的 SIMT 编程模型，因此 kernel 的整体算法可以复用，但不能假设 API 和性能行为完全一致。适配时重点检查：

- runtime API：内存分配、拷贝、释放、错误检查和设备属性查询；
- FP16 类型、头文件和 `half` 转换函数；
- warp/wave 大小及 shuffle reduction intrinsic；
- `__syncthreads()` 或对应 block synchronization 的语义；
- dynamic shared memory 的声明、launch 参数和容量限制；
- 编译器支持的 C++ 标准、模板语法和 CUDA-like intrinsic；
- block size、寄存器使用和 shared memory 使用对 occupancy 的影响。

当前 NVIDIA 代码中的 `kWarpSize = 32`、`__shfl_down_sync`、CUDA runtime API 和 `cudaDevAttrMaxSharedMemoryPerBlock` 都属于需要在国产平台逐项确认的假设，不能仅通过替换文件后缀判断移植完成。

### 2.4 适配阶段的禁止事项

在两个平台都通过正确性验证之前，不进行以下工作：

- 引入平台专属矩阵指令；
- 使用平台专属异步拷贝或特殊内存布局；
- 为单个平台大幅改变 softmax 数值流程；
- 只通过自建 smoke 测试而跳过官方 tester；
- 将“成功编译”记录为“正确性通过”。

### 2.5 适配阶段完成标准

每个平台至少应满足：

- 编译和链接成功；
- RMSNorm、Attention 的 `float` / `half` 官方测试通过；
- causal、non-causal 和 GQA 用例通过；
- 没有 kernel launch 或 runtime error；
- 可用的 sanitizer 或平台检查工具没有报告非法访问、未初始化读取和真实数据竞争；
- 文档中的命令、环境和结果与实际输出一致。

## 3. 阶段二：建立三平台性能基线

两个国产平台完成正确性适配后，再建立可比较的 benchmark。三个平台应尽量使用同一组输入 shape 和相同的 warm-up、重复次数及统计方式。

### 3.1 RMSNorm 指标

- kernel latency，单位为 `μs`；
- effective memory bandwidth，单位为 `GB/s`；
- 不同 `rows` 和 `hidden_dim` 下的扩展性；
- `float` 与 `half` 的独立结果。

### 3.2 Attention 指标

- kernel latency；
- 端到端 latency；
- `float` / `half`；
- causal / non-causal；
- MHA / GQA；
- 不同 `B`、`T`、`S`、`Hq`、`Hkv` 和 `head_dim`；
- Attention 有效 TFLOP/s 或 tokens/s；
- 显存吞吐、occupancy、寄存器使用和 shared memory 使用。

kernel-only 时间和端到端时间应分开记录。端到端时间包含内存分配、H2D、kernel、D2H 和释放操作，适合评估当前函数接口的实际成本；kernel-only 时间适合定位 GPU kernel 本身的优化效果。

## 4. 阶段三：共享算法级优化

共享算法级优化针对三个平台都成立的计算结构和数据流，目标是维护一套统一的算法设计，减少重复实现和平台间的行为分歧。

### 4.1 优化内容

优先评估以下方向：

- 减少 Attention 中 QK dot product 的重复计算；
- 将当前由单线程执行的 `row_sum` 改为 block 内并行归约；
- 改善 Q、K、V 的 tile 复用和访存顺序；
- 保持 stable softmax 的 FP32 统计和累加；
- 优化 source tile 的处理方式，避免不必要的同步和重复读取；
- 根据 benchmark 结果调整通用的 tile 划分策略；
- 保持 causal masking、GQA 和非等长序列的语义一致。

当前 Attention 在 `src/kernels.cu` 中会多次重新计算 QK：分别用于求 `row_max`、求 `row_sum` 和计算最终概率。这是优先调查的结构性问题，但具体改法必须以官方 tester 和三平台 benchmark 的结果为准。

### 4.2 共享优化的约束

共享优化不应依赖某一张 GPU 的特定架构，除非已经确认对应平台也有等价支持。第一轮优化暂不优先引入：

- 特定 NVIDIA SM 的 PTX；
- `cp.async` 等 NVIDIA 专属异步路径；
- 只适用于某一平台的 Tensor Core 指令；
- 依赖固定硬件尺寸的寄存器或 shared memory 参数。

每完成一项共享优化，都应在三个已适配平台上重新运行正确性回归；只有所有平台通过后，才能进入下一项优化。

## 5. 阶段四：平台专属性能调优

平台专属调优允许三份代码不再逐行一致，但必须保持接口、输出语义和测试覆盖一致。其目标是在每个平台上利用自身硬件和编译器特性获得最佳性能。

### 5.1 可分别调整的内容

- block size 和 source tile size；
- warp reduction 或平台对应的 wave/shuffle intrinsic；
- FP16 转换、`half2` 或其他向量化加载方式；
- shared memory 的布局、容量和 bank 冲突规避；
- Q/K/V 的向量化访存；
- register 使用、循环展开和 occupancy 平衡；
- 平台对应的编译器优化选项；
- 平台专属矩阵或混合精度指令，但必须单独验证数值误差。

可形成如下结构：

```text
统一的 Attention / RMSNorm 算法
├── NVIDIA：CUDA runtime、NVIDIA shuffle、CUDA FP16 和 NVIDIA 参数
├── MACA：MACA runtime、对应 reduction/FP16 intrinsic 和 MACA 参数
└── MUSA：MUSA runtime、对应 reduction/FP16 intrinsic 和 MUSA 参数
```

### 5.2 平台调优原则

- 先用 benchmark 确认瓶颈，再调整参数；
- 每次只改变一个主要变量，保留前后数据；
- 不以某个平台的优化结果推断其他平台也会受益；
- 数值正确性优先于吞吐提升；
- 平台专属代码应在对应 tester 和 sanitizer 通过后保留；
- NVIDIA 已通过的版本作为长期回归基线，不因国产平台调优而被覆盖。

## 6. 三平台维护策略

三份代码不需要每次修改都完全同步，建议按以下边界维护：

- 算法语义、张量布局、mask 规则、GQA 映射和数值精度策略保持同步；
- runtime API、FP16 类型、reduction intrinsic 和硬件参数允许平台独立；
- 共享算法变更先在 NVIDIA 基线验证，再分别移植到 MACA 和 MUSA；
- 平台专属调优只修改对应平台文件，并记录平台、硬件、编译选项和 benchmark 结果；
- 每个平台的正确性和性能结果独立记录，不能用一个平台的结果替代其他平台。

这种方式不是维护三套完全不同的算法，而是维护一套统一算法语义和三套必要的后端实现。相比同时进行算法重构、API 移植和性能调优，这种分层方式更容易定位问题，也能控制后续维护成本。

## 7. 总体完成标准

计划完成需要满足：

1. NVIDIA、MACA、MUSA 三个平台的目标算子均通过对应官方 tester；
2. 三个平台的 `float` / `half`、causal、non-causal 和 GQA 用例均有实际验证记录；
3. 共享算法优化没有破坏任一平台的正确性；
4. 平台专属优化均有修改前后的 benchmark 数据；
5. sanitizer 或等价工具没有报告真实内存错误、未初始化读取或数据竞争；
6. 文档明确区分正确性结果、kernel 性能和端到端性能；
7. 最终提交保留一个已验证的 NVIDIA 基线，便于后续回归。
