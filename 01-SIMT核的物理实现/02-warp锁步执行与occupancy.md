# warp 锁步执行与 occupancy 的本质

> **所属模块**：模块 01 · SIMT 核的物理实现
> **状态**：🟨 学习中（第一版：本质 + 费曼 + 开发者视角 + 最新架构，知识截至 2026-01）
> **一句话主旨**：occupancy（占用率）**不是目的，只是手段**——它是"用线程级并行填满延迟"的一种方式；一旦延迟已被藏住，再堆占用率毫无收益，而且往往要以寄存器为代价反过来伤害性能。真正的目标是"让执行单元和访存管道不空转"，而这有三条路：占用率（TLP）、指令级并行（ILP）、显式异步流水。

---

## 0. 这一节要回答的问题

- [x] 一条指令怎么被一个 warp 锁步执行？调度器每一拍到底在做什么决策？
- [x] occupancy 精确定义是什么？理论占用率 vs 实测占用率差在哪？
- [x] 它被哪三条资源天花板卡住（寄存器 / shared memory / warp·block 槽位）？会算吗？
- [x] **高 occupancy 一定好吗？** 什么时候堆占用率没用、甚至有害？
- [x] 除了堆 warp，还有什么办法藏延迟？（ILP、异步流水）
- [x] 最新架构（Hopper/Blackwell 异步流水）如何进一步"架空"占用率？

---

## 1. 核心概念

### 1.1 锁步执行的微观机制：调度器每一拍在赌什么

接着第 01 节的车轮战。一个子分区（SMSP）里有一个 **warp 调度器**，它每个时钟拍做一件事：

> **从手里所有驻留 warp 中，挑一个「就绪（eligible）」的，发射它的下一条指令。**

什么叫"就绪"？—— 这条指令的**操作数都已备好、不在等任何未完成的结果**。判定靠 **记分牌（scoreboard）**：当一个 warp 发出一条长延迟指令（比如 load），硬件在记分牌上给目标寄存器记一笔"欠账"；任何要用这个寄存器的后续指令，在欠账还清前都**不就绪**，不能发射。

于是每一拍有三种局面：
- **有就绪 warp** → 发射，执行单元这一拍没白费。
- **有驻留 warp 但全都在等结果（无人就绪）** → **stall（气泡）**，这一拍执行单元空转。
- 调度策略在多个就绪 warp 间选谁：常见 GTO（greedy-then-oldest，先继续当前 warp，卡了再挑最老的）。

**锁步发生在"执行"这一层**：调度器发出**一条**指令，这条指令驱动 32 条 lane（一个子分区正好 32 条 FP32 lane）同拍各算一个线程的数据。注意区分两个词：
- **发射吞吐（issue throughput）**：FP32 指令 1 拍发完（32 lane 对齐 32 线程）；但窄管道（如 SFU 超越函数、FP64）一条 warp 指令要**跨多拍**发射。
- **指令延迟（latency）**：从发射到结果就绪。算术 ~4~6 拍（短），访存 ~400~800 拍（长）。

> 🔑 **这就是"为什么要海量 warp"的微观版**：调度器要每拍都找得到就绪 warp，才能让执行单元不 stall。一个 warp 发了 load 卡住的那几百拍里，得有**别的**就绪 warp 顶上。**占用率的全部意义，就是保证"每一拍都有牌可出"。**

### 1.2 occupancy 到底是什么：一个比值，两种口径

> **occupancy = 一个 SM 上实际驻留的 warp 数 ÷ 硬件允许的最大 warp 数。**（近几代 NV 上限是 64 warp = 2048 线程/SM。）

两种口径必须分清：
- **理论占用率（theoretical / achievable）**：由 kernel 的资源消耗在**编译/启动时**就定死——受寄存器、shared memory、block 尺寸三方约束能驻留几个 warp。是个上限。
- **实测占用率（achieved）**：运行时**平均**真正驻留了多少 warp。它 ≤ 理论值，因为有尾部效应（tail effect，最后几个 block 收尾时 SM 半空）、block 执行时间不均、负载不平衡。**profiler 里 `Achieved Occupancy` 常明显低于理论值，这本身就是信息。**

### 1.3 三条天花板：occupancy 是一道"取最紧约束"的除法

固定资源被瓜分，能驻留多少 warp = 三条约束里**最紧**的那条（以 Hopper H100 为例：65536 寄存器/SM，~228 KB shared/SM，最多 64 warp、32 block/SM，≤1024 线程/block）：

| 约束 | 算法 | 例（block=256 线程=8 warp） |
| --- | --- | --- |
| **寄存器** | 驻留线程 = 65536 ÷ 每线程寄存器数 | 32 reg→2048 线程(100%)；64 reg→1024(50%)；128 reg→512(25%) |
| **shared memory** | 驻留 block = 228KB ÷ 每 block 用量 | 每 block 用 100KB → 只 2 block = 16 warp(25%) |
| **warp/block 槽位** | ≤64 warp 且 ≤32 block | block 太小（32 线程=1 warp）→ 32 block×1 = 32 warp(50%) 封顶 |

三个例子各暴露一个坑：**寄存器用多了砍占用**、**shared 用多了砍占用**、**block 太小则被 block 槽位数卡住（占用率也上不去）**。真正的理论占用率 = 三者取 min。

> 一个能算的直觉：把每线程寄存器从 64 压到 32，占用率从 50% 跳到 100%——**一行代码多用几个寄存器，可能就把并发量砍半**。这是程序员和硬件之间最日常的一场谈判。

---

## 2. 关键机制 / 为什么这样设计：**高 occupancy 不一定好**

这是本节的题眼，也是新手最大的误区。占用率**不是越高越好**，理由有三层，层层递进。

### 2.1 占用率是"手段"，而手段会饱和

回到 Little 定律（§01 §4.1）：**藏住延迟所需的并发量 = 延迟 × 吞吐**。占用率的唯一作用，是提供足够的"在途工作量"去覆盖这个数。**一旦覆盖住了，就到头了。**

- 藏**算术**延迟（~4~6 拍）：每个调度器只要有 ~4~6 个就绪 warp 就够了 → **占用率 ~10~20% 就能把算术延迟藏满**。
- 再往上堆到 100%？**边际收益为零**——延迟早藏住了，多出来的 warp 只是排队等着，执行单元该忙还是忙、该闲还是闲。

> 从 50% 提到 100% 常常**一点不快**。如果你的 profiler 显示 `Eligible Warps Per Scheduler` 已经稳定 ≥1，占用率就已经够了，继续堆是白费力气。

### 2.2 堆占用率是有代价的：拿寄存器换，可能反噬

占用率不是免费涨的。要驻留更多 warp，就得**压低每线程寄存器数**（§1.3 那道除法）。而寄存器压太狠会触发 **register spilling**——现场装不下，编译器把变量踢到 "local memory"（其实在显存里）。于是荒诞的一幕：

> **你为了藏延迟而拉高占用率，却因为寄存器不够把数据挤到显存，凭空制造了新的访存延迟。** 得不偿失。

`-Xptxas -v` 报出 `spill stores/loads` 就是这个警报。很多时候，**宁可占用率 50% 且零 spill，也好过 100% 但疯狂 spill**。

### 2.3 ILP 可以替代占用率（Volkov 的经典反直觉结论）

关键洞察：藏延迟需要的是"足够多的**独立**在途工作"，而这份并行度有**两个来源**，彼此可替代：

- **TLP（线程级并行）= 占用率**：靠很多 warp，每个 warp 各挂一条独立指令。
- **ILP（指令级并行）**：靠**单个线程内部**若干条互不依赖的指令，一个 warp 也能同时挂着好几笔在途操作。

> 车轮战类比：你可以靠"很多盘棋"填满时间（TLP），也可以靠"每盘棋能一次走好几步互不影响的子"填满时间（ILP）。**两者都能让大师不空手。**

Volkov 2010 那篇著名的《**Better Performance at Lower Occupancy**》就是这个：让**每个线程算一个 8×8 的小块**（register blocking），单线程内有大量独立乘加（高 ILP），寄存器用得多、占用率故意压得**很低**，却比高占用率版本更快。**这正是高性能 GEMM（CUTLASS）的做法：低占用、高寄存器、高 ILP，是刻意的设计,不是失误。**

> 🔑 **本质**：占用率（TLP）和 ILP 是藏延迟的**两种可互换的燃料**。盯着占用率单一指标去优化，会错过"少而肥的线程"这条常常更优的路。

### 2.4 两个 regime：占用率什么时候真的重要

把上面收拢成一个判断框架——**取决于你受限于什么**：

| | 受限于什么 | 占用率的角色 |
| --- | --- | --- |
| **compute-bound** | 算术吞吐 / 藏算术短延迟 | 低占用 + 高 ILP 常常足够甚至更优（GEMM 型） |
| **memory-bound** | 显存带宽 / 藏访存长延迟 | 需要**很多在途访存请求**来喂满带宽（Little: 在途字节=BW×延迟）→ **占用率此时真的重要**，直到带宽饱和为止 |

所以第 01 节那个 `y=2x`（memory-bound、几乎无 ILP）**确实需要较高占用率**——它得靠很多 warp 同时挂着 load 才能把带宽喂满。而一个寄存器分块的 GEMM 则相反。**"高占用好不好"没有统一答案，先问"我卡在算力还是带宽"。**

### 2.5 现代趋势：异步流水把延迟"显式"藏了，占用率进一步靠边

接 §01 §4.1：Hopper/Blackwell 上还冒出第三种燃料——**显式异步流水**。用 `cp.async`/TMA 把访存从计算里剥离，配 mbarrier 做生产者-消费者，再加 **warp 专化**，就能用**很少的 warp**、很深的软件流水把访存延迟藏住。于是最新的高性能 kernel（FlashAttention-3、CUTLASS 3.x 的 warp-specialized GEMM）**常常是刻意低占用率**的。**"藏延迟"越来越不靠"堆 warp"，而靠"排好异步流水"。**（详见本模块第 05 节。）

---

## 3. 开发者视角：代码/工具里怎么摸到它

> 三通道回顾：① 源码拨盘 ② 编译器回执 ③ Profiler 探针。occupancy 恰恰是**编译器回执能算、profiler 能测**、但**源码只能间接拧**的典型。

| 你想控制/观察 | ① 源码拨盘 | ② 编译器回执 | ③ Profiler 探针（Nsight Compute） |
| --- | --- | --- | --- |
| 占用率上限 | block 尺寸；`__launch_bounds__`、`-maxrregcount=N`；shared 用量 | `-Xptxas -v` 的寄存器/shared 用量；`cudaOccupancyMaxActiveBlocksPerMultiprocessor` API / Occupancy Calculator | `Theoretical Occupancy` vs `Achieved Occupancy` |
| 延迟是否已藏住 | —（改上面几个拨盘间接影响） | — | **`Eligible Warps Per Scheduler`**（≥1 就够了）；`Issue Slot Utilization`；`Warp Cycles Per Issued Instruction` |
| 卡在哪种等待 | — | — | **Stall 原因分解**：`Long Scoreboard`(等显存)、`Short Scoreboard`(等算术/shared)、`Barrier`(等同步)、`MIO Throttle` 等 |
| 寄存器有没有反噬 | `-maxrregcount`、`__launch_bounds__` | **`spill stores/loads`**（大凶信号） | `Registers Per Thread`；"limited by registers" |
| 受限于算力还是带宽 | — | — | `Compute (SM) Throughput %` vs `DRAM Throughput %`（谁先顶到 ~90%） |

**关键读法**：不要一上来就冲高 `Achieved Occupancy`。先看 `Eligible Warps Per Scheduler`——若已稳定 ≥1，占用率够了，去别处找瓶颈；再看 Stall 分解定位真正的等待类型；`spill` 非零就先解决 spill。**占用率是过程指标，不是 KPI。**

### 站位阶梯

| 站位 | 拨盘 | 感官 |
| --- | --- | --- |
| PyTorch | 基本碰不到 occupancy（库 kernel 已调好） | `torch.profiler` 看 kernel 名/耗时 |
| **Triton** | **`num_warps`（≈直接定占用率）、`num_stages`（异步流水深度）**、BLOCK_SIZE | `@triton.autotune` 帮你把 num_warps/num_stages 搜出来——**你其实是在自动搜"占用率 vs ILP vs 流水深度"的最优点** |
| CUDA C | launch config、`__launch_bounds__`、`-maxrregcount` | 上面那张大表 |
| SASS | 只读 | 看寄存器分配、spill、指令调度 |

> 对算法侧的你，最常摸到 occupancy 的地方其实是 **Triton 的 `num_warps` / `num_stages` autotune**——这俩旋钮背后，正是本节 §2 那场"占用率 vs ILP vs 异步流水深度"的三方权衡。

---

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **Hopper / Blackwell — 占用率的地位被异步流水稀释**：`num_stages` 控制的软件流水深度（依赖 `cp.async`/TMA + mbarrier）成为藏访存延迟的**主力旋钮**，取代"无脑加 warp"。warp 专化、persistent kernel 让 SOTA kernel 常态低占用。
- **Blackwell TMEM（SM100）— 给"占用率 vs 寄存器"松绑**：MMA 累加器搬进专用张量内存后，寄存器压力下降，于是"想提占用率却因寄存器不够而 spill"的两难被缓解——可在**不 spill 的前提下**要么提占用、要么保留更多 ILP。
- **独立线程调度（Volta 起，延续至今）**：每个线程有了自己的 PC，warp 内不再保证严格同步推进（重收敛不再自动）——这对**发散**（第 03 节）影响最大，但也意味着"锁步"在最新架构上是"逻辑锁步、物理可微错开"。
- **跨厂商对照**：
  - **AMD CDNA（MI300X / MI350X）**：占用率讲的是"**每个 SIMD 上驻留几个 wavefront**"；资源约束换成 **VGPR（向量寄存器）/ LDS** 用量；wave 上限、Wave32/Wave64 都会影响。名字不同，除法一样。
  - **Groq LPU**：**没有动态 warp 调度器、没有 occupancy 这个概念**——一切在编译期静态排定。它是本节机制的正对面（→ 模块 02 空间架构会正面讲"为什么它不需要占用率来藏延迟"）。

---

## 5. 和其它模块的挂钩

- Little 定律、藏延迟的三种燃料 → 承接 **本模块 01 §4.1**
- 锁步是**分支发散**的前提；独立线程调度的松动 → **本模块 03**
- 异步流水、mbarrier、warp 专化、`num_stages` → **本模块 05**（同步全家桶）深讲
- shared memory 作为占用率约束之一 → **本模块 04**（访存通路）
- "Groq 无占用率概念""脉动阵列静态排时序" → **模块 02**（为什么空间架构不需要藏延迟）
- Triton 的 num_warps/num_stages autotune → **模块 04**（kernel 层自动调优）

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释（本质版） | 挂在哪个模块 |
| --- | --- | --- | --- |
| occupancy | 占用率 | 驻留 warp ÷ 硬件上限；藏延迟的**手段**而非目的 | 01 |
| 理论 vs 实测占用率 | theoretical / achieved | 前者资源算出的上限，后者运行时实测均值（含尾部效应） | 01 |
| 就绪 warp | eligible warp | 操作数已备、可被发射的 warp；每拍有 ≥1 个就不 stall | 01 |
| 记分牌 | scoreboard | 追踪长延迟结果何时就绪，决定指令是否就绪 | 01 |
| stall / 气泡 | | 某拍无就绪 warp，执行单元空转 | 01 |
| ILP | 指令级并行 | 单线程内独立指令；可**替代占用率**藏延迟 | 01 |
| register spilling | 寄存器溢出 | 现场装不下、变量被踢到显存，制造额外延迟 | 01 |
| num_warps / num_stages | | Triton 里定占用率 / 异步流水深度的两个旋钮 | 01/04 |
| VGPR / LDS | AMD 对应 | 向量寄存器 / 本地数据共享（≈寄存器 / shared） | 01 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）不同架构"藏满算术延迟"实际需要的 warp 数？memory-bound 时"喂满带宽"需要的 MLP（在途请求数）怎么估？
- （待填）`num_stages` 加深流水 vs 加 warp，两条路在什么形状下各自更优？有没有经验判据？
- （待填）Blackwell TMEM 之后，"低占用高 ILP" 的 GEMM 范式会不会变形？
- （待填）实测占用率被尾部效应压低时，grid-stride loop / persistent kernel 能补多少？

---

*最后更新：2026-07-05（第一版）*
