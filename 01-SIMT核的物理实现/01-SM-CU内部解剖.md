# SM/CU 内部解剖

> **所属模块**：模块 01 · SIMT 核的物理实现
> **状态**：🟨 学习中（第一次讲完，正文已成形，可回来补充）
> **一句话主旨**：SM 是一台"为吞吐而非延迟"设计的引擎——内部每个部件的尺寸和摆法，都是为了让海量 warp 驻留、随时切换、用"永远有活干"把访存延迟填满。

---

## 0. 这一节要回答的问题

- [x] 一个 SM（NVIDIA）/ CU（AMD）内部到底有哪些部件，各干什么？
  - [x] warp scheduler（发射单元）
  - [x] 执行单元：ALU / FMA lane
  - [x] 寄存器堆（register file）
  - [x] LSU（load/store unit，即 "LSM"）
  - [x] shared memory / L1（同一块 SRAM 的两种用法）
  - [x] Tensor Core 坐在这张图的哪个位置
- [x] 拿一个熟悉的算子（`y[i]=2*x[i]`）映射上去，warp / 锁步 / 执行单元 / 访存怎么第一次挂到同一张图上？

## 1. 核心概念

### 定盘星：吞吐 vs 延迟

- **CPU 核**：为「让一条指令尽快跑完」而生——深流水、乱序、大分支预测器，缩短单任务**延迟**。
- **GPU 的 SM**：不在乎单条指令多久跑完，只在乎单位时间吞掉多少条——靠**海量任务驻留 + 谁没数据就切走**，用"永远有活干"填满延迟，最大化**吞吐**。
- 👉 SM 内部每个部件的尺寸/摆法都是为这句话服务的。occupancy、锁步、同步全从这里长出来。

### 任务怎么摊到硬件：thread → warp → SM

- 每个数据元素配一个 **thread**。
- 硬件从不单独调度线程：连续 **32 个线程捆成一个 warp**（AMD：wavefront，64 宽），是**最小调度和执行单位**。
- 一个 warp 的 32 线程**共用一个 PC，同一拍执行同一条指令**，各作用在自己的数据上 = **SIMT**（Single Instruction Multiple Threads）= **锁步 lockstep**。

### 一个 SM 的解剖图（Hopper 量级）

```
┌──────────────────────── SM (Streaming Multiprocessor) ────────────────────────┐
│  ┌─ 子分区 0 (SMSP) ─┐ ┌─ 子分区 1 ─┐ ┌─ 子分区 2 ─┐ ┌─ 子分区 3 ─┐            │
│  │ Warp Scheduler    │ │   同左     │ │   同左     │ │   同左     │            │
│  │ Register File     │ │            │ │            │ │            │            │
│  │  (16K×32bit)      │ │            │ │            │ │            │            │
│  │ 32× FP32 ALU lane │ │            │ │            │ │            │            │
│  │ INT / SFU 单元    │ │            │ │            │ │            │            │
│  │ 1× Tensor Core    │ │            │ │            │ │            │            │
│  │ LSU (访存单元)    │ │            │ │            │ │            │            │
│  └───────────────────┘ └────────────┘ └────────────┘ └────────────┘            │
│  ┌──────────── Shared Memory / L1 (本 SM 共享, ~一两百 KB) ───────────────┐   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────────────┘
                                     ▼   L2 Cache (全芯片共享) ──► HBM (显存)
```

一个 SM = **4 个子分区（SMSP）**，每个是一套能独立发射指令的小引擎。

| 部件 | 数量（约） | 干什么 | 在 `y=2x` 里的角色 |
| --- | --- | --- | --- |
| **Warp Scheduler** | 每子分区 1 个 | 每拍挑一个"数据就绪"的 warp 发射它的下一条指令；warp 卡住就转身发别的 | 藏延迟的心脏；load 没回来就切走 |
| **FP32 ALU / FMA lane** | 每子分区 32 条（SM 共 128） | 与 warp 的 32 线程一一对齐，锁步计算；FMA=乘加融合 | 32 lane 同拍各算一个 `2*x` |
| **Register File** | 每 SM ~65536×32bit（256 KB） | 存每个驻留 warp 的上下文，**永久占着不动** | 使 warp 切换零开销的根源 |
| **LSU（=LSM）** | 每子分区若干 | 算访存地址、发 load/store、做 coalescing | 取 `x`、写 `y`，合并 32 请求 |
| **Shared Memory / L1** | 每 SM ~228 KB 可配 | 片上 SRAM，软管（shared）+ 硬管（L1）两用 | ⚠️ 本算子无复用，用不上 |
| **Tensor Core** | 每子分区 1 个 | 专算小矩阵块乘加的独立引擎（内含小脉动阵列） | ⚠️ 本算子全程空转 |

## 2. 关键机制 / 为什么这样设计

- **为什么寄存器堆大得离谱（比 L1 还大）**：warp 切换零开销的前提是**不做上下文保存/恢复**——每个驻留 warp 的寄存器一直躺在寄存器堆里，调度器只换索引。代价：单线程用寄存器越多 → 能驻留的 warp 越少 → 这就是下一节 **occupancy** 的命根子。
- **为什么 32 条 lane 恰好等于 warp 宽度**：锁步执行要求「一条指令喂满一排执行单元」，宽度对齐才不浪费。
- **为什么 element-wise 用不上 shared memory / Tensor Core**：每个元素只读一次、算一次、写一次，**没有数据复用**，也没有矩阵乘。这两个部件是给**有复用**（matmul/卷积/attention）的算子准备的。

### `y[i]=2*x[i]` 在一个 SM 里的一生

> 调度器挑 warp → LSU 发 32 个 load 取 `x`（合并成一两笔事务）→ **warp 卡住等数据，调度器立刻切别的 warp** → 数据回来后 32 条 ALU lane 锁步各算 `2*x` → LSU 写回 32 个结果 → warp 退休。
> 真正"算"的时间极短，大头在等访存 → element-wise 几乎总是 **memory-bound**（→ 引出下一节）。

## 3. 开发者视角：代码/工具里怎么摸到它

> **总框架——感知硬件只有三个通道**：① **源码拨盘**（源码里唯一朝硬件的旋钮：launch config、`__shared__`、dtype、调哪个库）② **编译器回执**（`-Xptxas -v` 的离线体检）③ **Profiler 探针**（Nsight Compute/Systems 的运行时示波器）。
> **硬件特性大多不是"写"出来的，是拨盘 + 访存模式"暗示"出来的，再靠 profiler 的症状反推。** ⇒ 学硬件的实用目的，一大半是为了看懂 profiler 在说什么。

| SM 部件 / 特性 | ① 源码拨盘 | ② 编译器回执 | ③ Profiler 探针（Nsight Compute） |
| --- | --- | --- | --- |
| warp 调度 / 藏延迟 / occupancy | launch config 的 **block 大小** | — | `Achieved Occupancy`；`Eligible Warps/Scheduler`；Stall `Long Scoreboard`（等显存） |
| ALU / FMA lane | 算术表达式 | SASS 里的 `FFMA` | `Compute (SM) Throughput %`；FMA pipe util |
| 寄存器堆 | 局部变量数；`__launch_bounds__`、`-maxrregcount` | `-Xptxas -v` → "Used N registers"、"**spill**"（溢出=坏信号） | `Registers Per Thread`；"limited by registers" |
| LSU / coalescing | **索引模式**（相邻线程是否相邻地址） | — | `Global Load/Store Efficiency`；`Sectors/Request`；`DRAM Throughput` |
| Shared Memory / L1 | `__shared__` 声明；padding；carveout 配置 | 静态 shared 用量 | `Shared Bank Conflicts`；`L1 Hit Rate` |
| Tensor Core | 用 `wmma`/cuBLAS/CUTLASS/Triton 而非裸 `*`；喂 fp16/bf16/tf32；形状对齐 | SASS 里的 `HMMA`（没有=没用上） | `Tensor Pipe Active %`（为 0 就是白买） |

`y=2x` 的闭环：`-Xptxas -v` 报"每线程 8 寄存器、0 spill" → `ncu` 看到 occupancy 高、Compute 低、`DRAM Throughput` ~90%、主 stall 是 `Long Scoreboard` → 三证齐指 **memory-bound**。

### 站位阶梯：你在栈的哪一层，"感官"完全不同

| 站位 | 拨盘 | 观测手段（感官） |
| --- | --- | --- |
| 框架层（PyTorch） | 算子选择、dtype、`torch.compile` | `torch.profiler`、`nvidia-smi`、看 kernel 名是否命中融合/TensorCore |
| Triton 层 | `BLOCK_SIZE`、`num_warps`、`num_stages` | `@triton.autotune` 结果、`do_bench` |
| CUDA C 层 | launch config、`__shared__`、`-maxrregcount` | 上面那张大表（Nsight Compute） |
| PTX/SASS 层 | 几乎不写，只读 | `cuobjdump -sass` / `nvdisasm` |

> 越往上拨盘越少越抽象、越难精确看见硬件；越往下旋钮越多、profiler 越精细、越费人。**算法侧日常主战场是上两层**，核心感官是 `torch.profiler` + Triton autotune，而非 SASS。这条阶梯就是**模块 04 软件栈**的骨架。

## 4. 和其它模块的挂钩

- 寄存器占用限制驻留 warp 数 → **本模块 02（occupancy）**
- 32 线程锁步是"分支发散"的前提 → **本模块 03**
- LSU 的 coalescing、shared/L1 的访存层次 → **本模块 04**
- Tensor Core 内部的小脉动阵列 → **模块 03**，对照大阵列 → **模块 02**
- shared memory 的显式复用、双缓冲 → 对照脉动阵列的 scratchpad/tiling → **模块 02 第 04**

## 5. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| SM | Streaming Multiprocessor | GPU 的基本计算核，内含 4 个子分区，为吞吐而设计 | 01 |
| CU | Compute Unit（AMD 对应） | AMD 侧对应 SM 的单元 | 01 |
| 子分区 | SMSP / sub-partition | SM 内能独立发射指令的小引擎，一个 SM 有 4 个 | 01 |
| warp | wavefront（AMD 64 宽） | 32 线程捆成一束，最小调度/执行单位，锁步执行 | 01 |
| SIMT / 锁步 | lockstep | 一条指令、多线程、共用 PC、同拍执行 | 00/01 |
| Warp Scheduler | | 每拍挑就绪 warp 发射，靠切换藏延迟 | 01 |
| FMA lane | | 乘加融合的执行 lane，每子分区 32 条 | 01 |
| 寄存器堆 | register file | 存驻留 warp 上下文、永久占用，使切换零开销 | 01 |
| LSU | Load/Store Unit | 算地址、发访存、做 coalescing | 01 |

> ✅ 已同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 6. 我的困惑 / 待深挖

- （待填）Hopper 之外的架构（Ampere/Ada/AMD CDNA）子分区数、寄存器量差异？
- （待填）"独立线程调度"（Volta 起）如何改变锁步的严格程度？→ 留到第 03 节发散

---

*最后更新：2026-07-05（第一次讲解）*
