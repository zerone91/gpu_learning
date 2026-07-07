# 片上存储层次与 tiling、双缓冲

> **所属模块**：模块 02 · 空间/脉动架构
> **状态**：🟨 学习中（第一版：本质/费曼 + 开发者视角 + 最新架构，知识截至 2026-01）
> **一句话主旨**：阵列本身零访存烦恼（复用在电线里），但**阵列的边缘就是它的"嘴"**——喂不上数据，再大的阵列也白搭。NPU 的解法是**scratchpad（软件管理的 SRAM，无 cache）+ tiling（把大问题切成装得进 SRAM 的块）+ 双缓冲（算这块、搬下块）**。这三件套 GPU 也全有（shared memory / BLOCK_* / cp.async 流水）——**区别只在"谁排时刻表"：NPU 是编译器排死，GPU 是运行时+软件流水**。喂料系统的账 = 边缘带宽 ≥ 阵列吞吐 ÷ 复用次数。

---

## 0. 这一节要回答的问题

- [x] NPU 的片上存储长什么样？为什么用 scratchpad 而不用 cache？
- [x] tiling 为什么是必然的？tile 尺寸怎么定（一道能算的不等式）？
- [x] 双缓冲怎么让"搬运"藏进"计算"？和 GPU 的 num_stages 什么关系？
- [x] "喂饱一个 128×128 阵列"需要多少边缘带宽？怎么算？
- [x] NPU 存储层次 vs GPU 存储层次，逐级对应关系是什么？

---

## 1. 核心概念

### 1.1 费曼起点：巨兽的嘴

§01-03 把阵列内部讲完了：一旦数据进入阵列，复用免费、时序确定、零同步。但有一个问题被悬置了——**数据怎么进来？**

一个 128×128 阵列在 1 GHz 下每拍要吃 **128 个激活 + 吐 128 个部分和**（WS 情形），一秒就是几百 GB 级的边缘流量。而数据源头在 DRAM（§04：慢、贵、200× 能量）。**如果每个数据都从 DRAM 直喂，任何 DRAM 都跟不上**——巨兽的胃口（阵列吞吐）和食堂的产能（DRAM 带宽）之间差着数量级。

解法和 GPU 完全同构（§01-04 的"复用放大器"）：**在阵列和 DRAM 之间垫一层大 SRAM，把数据先搬进来，让阵列从 SRAM 反复吃**。NPU 管这层叫 **global buffer / scratchpad**（TPU 的 Unified Buffer、DaVinci 的 L1/UB）。

### 1.2 scratchpad vs cache：为什么 NPU 不要 cache

同样一块 SRAM，两种管法（§01-04 提过 shared/L1 是"同一块料的两种用法"，NPU 直接砍掉了 cache 那一半）：

| | cache（GPU L1/L2） | scratchpad（NPU 主流） |
| --- | --- | --- |
| 谁决定放什么 | 硬件（LRU 等启发式） | **编译器/DMA 显式搬** |
| 命中与否 | **运行时才知道**（miss 是变长事件 → 要 MSHR/记分牌，§01-04） | **没有 miss 这个概念**——数据要么已被排定搬入，要么是编译器的 bug |
| 额外硬件 | tag 阵列、比较器、替换逻辑（面积/能耗 ~15-30%） | 几乎零管理开销，同面积**装更多数据** |
| 适合 | 访存模式不可预测 | 访存模式**编译期全知** |

> 🔑 **本质**：cache 是为"不可预测"设计的保险，而脉动阵列的世界里**访存模式是编译期全知的**（§02 的时序公式连"哪拍要哪个数"都写出来了）——为全知的访存买"不可预测保险"纯属浪费。**去掉 cache 不是省钱的妥协，而是确定性范式的逻辑必然。** 这与 §02"没有变长事件就不需要记分牌"是同一条推理链：**Groq 连 DRAM 都去了（纯 SRAM），是这条链的终点。**

### 1.3 tiling：把大问题切成"装得进 SRAM"的块

真实矩阵（K=N=上万）远大于 SRAM 容量 → 必然切块。**tiling 的账是一道不等式**：

> 选 tile 尺寸 `(Tm, Tk, Tn)`，使得：
> ① **装得下**：`Tm·Tk + Tk·Tn + Tm·Tn ≤ SRAM 容量`（A 块 + B 块 + C 块）
> ② **喂得饱**：切块后每字节的复用次数要足够高，使 `DRAM 流量 = 总计算量 ÷ 复用度 ≤ DRAM 带宽 × 计算时间`

这就是 A1 算术强度的另一面：**tiling 是"制造复用度"的手段**——tile 越大，块内复用越充分，DRAM 流量越低（GEMM 的 DRAM 流量 ~ `M·K·N / min(Tm,Tn)` 量级，tile 边长翻倍，流量减半）。但 tile 受 SRAM 容量封顶 → **SRAM 大小直接决定能制造多少复用度**，这就是 NPU 都堆大 SRAM 的原因（TPU UB 曾达 24-128MB 级，Cerebras 干脆 44GB 全 SRAM）。

**tile 的层级嵌套**：DRAM→global buffer 一层 tile，buffer→阵列边缘又一层（按阵列尺寸 128 对齐）——与 GPU 的 grid→block→warp tile 层级一一对应（§03 §1.4 说过：tiling=数据流选择的软件表达，此处是它的容量约束版）。

### 1.4 双缓冲：搬运藏进计算（乒乓）

切了块就有"算完这块、等下一块"的空档。解法是 SRAM 里开两份 tile 空间：

```
时间 →     t0          t1          t2
buffer A:  算 tile0    装 tile2    算 tile2
buffer B:  装 tile1    算 tile1    装 tile3
           （DMA 引擎搬运 ‖ 阵列计算，条件：搬运时间 ≤ 计算时间）
```

**完美重叠的条件**：`tile 搬运时间 ≤ tile 计算时间`，展开即 `tile字节数/DRAM带宽 ≤ tile FLOPs/阵列吞吐`——**又回到 A1 的机器平衡点**：tile 的算术强度必须 ≥ 硬件平衡点，双缓冲才藏得住搬运。藏不住 → 阵列等料 → 利用率塌（这正是 memory-bound 在 NPU 上的形态）。

> 🔑 **对照 GPU（这里两个范式几乎融合了）**：双缓冲 = §01-02 §2.6 的 N 级软件流水的 N=2 特例；DMA 引擎 = TMA；"装 tile1 时算 tile0" = `num_stages`。**唯一的区别**：NPU 上这套由**编译器静态排死**（哪一拍搬完是排定的，无需握手）；GPU 上由 **mbarrier 运行时握手**（因为 SIMT 世界里搬运完成时刻不可预测）。**同一个优化，两种"谁排时刻表"。**

### 1.5 边缘带宽账：喂饱阵列要多少？

以 128×128 WS 阵列 @1GHz 为例（bf16）：
- 峰值吞吐：128×128 MAC/拍 = **32.8 TFLOP/s**（16384 MAC×2）。
- 边缘需求：每拍进 128 激活 + 出 128 psum ≈ 每拍 ~768B（激活 2B、psum 4B）→ **~768 GB/s**——这是 **SRAM 必须提供**的带宽（片上 SRAM 轻松达到）。
- DRAM 需求 = 边缘流量 ÷ SRAM 内复用度。若 tile 让每个激活在 SRAM 里复用 ~100 次 → DRAM 只需 ~8 GB/s 级——**层层复用把带宽需求从"不可能"降到"平凡"**。

> 这就是"复用放大器"的定量版：**阵列吞吐固定，每加一层复用，上游带宽需求除以复用次数。** tiling/数据流/SRAM 容量三者共同决定这串除法能除多少。

---

## 2. 关键机制小结（一张对照表收拢）

| 概念 | GPU（模块 01） | NPU（本模块） | 差异本质 |
| --- | --- | --- | --- |
| 片上大 SRAM | shared memory（+L1 二用） | scratchpad / global buffer（无 cache） | 访存可预测性 |
| 切块 | BLOCK_M/N/K（软件） | tiling（编译器/mapper） | 同一道不等式 |
| 重叠搬算 | cp.async/TMA + mbarrier + num_stages | DMA + 双缓冲（静态排定） | 谁排时刻表 |
| 层级 | 寄存器→shared→L2→HBM | PE 寄存器→阵列→buffer→DRAM | 一一对应 |
| miss/等待 | 记分牌、Long Scoreboard stall | **不存在**（排定或 bug） | 确定性 |

## 3. 开发者视角

- **TPU/NPU 侧**：你不直接管 buffer——XLA/CANN 的**编译报告**里看：tile 选择、buffer 占用、DMA 与计算的重叠率（DaVinci 的 CANN profiling 有"Cube 利用率 vs MTE 搬运"通道占比，MTE=Memory Transfer Engine 即其 DMA）。**红旗信号：搬运通道满而 Cube 空转 = tile 太小/复用不够**——NPU 版的 memory-bound。
- **GPU 侧**：同一套思想全在你手里：`BLOCK_*`（tile 尺寸）、`num_stages`（缓冲深度）、shared 用量（§01-02 的三方抢预算）。**Triton autotune 搜的就是这道 tiling 不等式的最优解。**
- **通用心法**：不论哪个芯片，性能报告先问三件事——**tile 多大（复用够吗）、双缓冲开了吗（搬算重叠吗）、边缘/DRAM 带宽卡在哪级（A1 下钻）**。

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **TPU**：Unified Buffer/CMEM 路线延续，v5p/v6/v7 主要堆 HBM 带宽 + ICI；编译器（XLA）负责全部 tiling/双缓冲。
- **昇腾 DaVinci**：显式多级 scratchpad（L0A/L0B/L0C 贴 Cube，L1，UB），**层级全暴露给编译器/程序员**（TIK/AscendC 里手动指定数据搬到哪级）——比 TPU 更"手动挡"。
- **Groq**：无 DRAM、220MB 纯 SRAM——把"tiling 装得下"的约束变成"整个模型装得下"，装不下就多芯片流水。
- **Cerebras WSE-3**：44GB 片上 SRAM，权重全驻留，**取消 DRAM tiling 这一层**。
- **GPU 镜像**：Blackwell TMEM + 更大 shared + TMA 多播（cluster 内一次搬多 SM）——GPU 的"喂料系统"持续向 NPU 式显式管理靠拢。

## 5. 和其它模块的挂钩

- 复用放大器/机器平衡点/Little → **§01-04、A1**（同一套账，能量/带宽两种记法）
- 双缓冲=软件流水 N=2 → **§01-02 §2.6**；DMA=TMA、静态排定 vs mbarrier 握手 → **§01-05、02-02**
- tiling=制造复用度 → **§02-03 数据流**（谁驻留）+ **模块 04 映射问题**（统一收束）
- 无 cache 的逻辑必然 → **02-02 确定性链条**
- DaVinci 多级 scratchpad / TPU UB → **模块 05**

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| scratchpad | 便笺存储 / global buffer | 软件显式管理的 SRAM，无 tag 无 miss；确定性范式的必然 | 02 |
| Unified Buffer / UB | TPU | TPU 的大 scratchpad | 02/05 |
| L0A/L0B/L0C | DaVinci | 昇腾贴 Cube 的三块最内层 buffer（A/B/累加） | 02/05 |
| MTE | Memory Transfer Engine | DaVinci 的 DMA 引擎（≈NPU 版 TMA） | 02/05 |
| tiling 不等式 | | 装得下（≤SRAM）且喂得饱（复用≥平衡点） | 02/04 |
| 双缓冲 | double buffering / ping-pong | 两份 tile 空间轮换，搬运藏进计算；=num_stages 的 N=2 | 02 |
| 边缘带宽 | edge bandwidth | 阵列每拍进出的数据率；SRAM 必须扛住的量 | 02 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）多级 scratchpad（DaVinci L0/L1/UB）各级的容量/带宽比怎么定？有无类似"每级 10×"的经验律？
- （待填）编译器排静态 DMA 时刻表，遇到 DRAM 刷新/行冲突这类物理不确定性怎么兜底？
- （待填）TMA 多播（cluster 内一对多）在 GPU 上能省多少边缘流量？NPU 有无对应物？
- （待填）权重驻留型（Cerebras/Groq）在模型超过 SRAM 时的多芯片切分账怎么算？

---

*最后更新：2026-07-06（第一版）*
