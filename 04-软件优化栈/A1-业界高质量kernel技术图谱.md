# 附录 A1（模块04）· 业界高质量 kernel 技术图谱：FA 全系 与 DeepSeek 开源栈

> **所属模块**：模块 04 · 软件优化栈（kernel 层的案例延伸）
> **状态**：🟨 学习中（第一版，知识截至 2026-01；FA4 细节据公开演讲/分析，可能不完整，已标注）
> **一句话主旨**：两条线看业界最强 kernel 的技术谱系——**FA1→FA4 是"单 kernel 调度"随硬件代际的演化史**（每代 GPU 出新异步器件，FA 就重写一次调度）；**DeepSeek 开源栈（DeepGEMM/FlashMLA/DeepEP + TBO/SBO）是"通信-计算重叠"把藏延迟哲学推上系统层**。两条线合起来印证全库总纲：**藏延迟这一个思想，在栈的每一层各现一次身**。

---

## 0. 这一节要回答的问题

- [x] FA1→FA2→FA3→FA4 每一代到底改了什么？各绑定哪代硬件的什么器件？
- [x] FA4 的两个招牌算法技巧（软件 exp / lazy rescale）是什么原理？
- [x] DeepEP 是什么？normal vs low-latency 两套 kernel 差在哪？IBGDA 是什么？
- [x] TBO / SBO 分别指什么、站在栈的哪一层、解决什么问题？
- [x] "藏延迟"在栈的各层怎么逐层重现？（统一视角）

---

## 1. FA 全系：一部"调度追着硬件跑"的演化史

**计算（数学）自 FA1 后基本没变**（tiling + online softmax，A2 §1.1），**变的全是调度**——每代硬件给出新的异步器件，FA 就把流水重排一次。这是 04-03"计算/调度分离"最好的活教材：

| 代 | 年份/硬件 | 核心变化（调度层面） | 对应本库知识点 |
| --- | --- | --- | --- |
| **FA1** | 2022 / A100 | 开创：tiling + online softmax，S 矩阵不落 HBM——**跨层走私的原型** | A2 §1.1、04-02 融合边界 |
| **FA2** | 2023 / A100 | ① 并行度重组：从"batch×head"扩到**序列维切块**（小 batch 长序列也能喂满 SM）② 减非 matmul FLOPs（rescale 挪到循环外）③ warp 间分工重排，减 shared 往返 | §01-02 占用率、A1 wave 量化 |
| **FA3** | 2024 / **Hopper** | 吃透新器件：**TMA 异步搬运 + wgmma 异步 MMA + warp 专化（生产者/消费者）+ ping-pong 调度**（softmax 藏进 GEMM）+ FP8 路径。H100 上 35%→75% 峰值 | A2 §1.4/§1.7、§01-02 §2.6 |
| **FA4** | 2025 / **Blackwell**（据 Hot Chips 2025 等公开资料，细节可能不全）| ① 改用 **CuTe-DSL（Python）**编写——③层工具本身的代际换代 ② **tcgen05 MMA + TMEM 累加**（§03-01 的落地）③ warp 专化角色更细（load/MMA/softmax/correction/epilogue 多角色深流水）④ **软件 exp**：用三次多项式在 **FFMA pipe 上近似 exp2**，绕开 MUFU 吞吐瓶颈 ⑤ **lazy softmax rescale**：running max 变化不大就跳过 correction。B200 上报 ~1.2 PFLOP/s，超 cuDNN ~20% | §03-01 tcgen05/TMEM、A2 §1.3 MUFU 瓶颈 |

**FA4 两个招牌技巧拆解**（都直指 A2 §1.3 那个"MUFU 卡在两个 matmul 之间"的老瓶颈）：

1. **软件 exp（多项式近似）**：exp2 用三次多项式算 → 跑在**FFMA pipe**（Blackwell 上极其充裕）而非 MUFU（吞吐只有 FMA 的零头，A1 表）。**本质：把负载从"贵且挤的 pipe"搬到"便宜且闲的 pipe"**——A1"搬瓶颈"思想的指令级版本。精度靠值域缩减 + 多项式阶数控制。
2. **lazy rescale**：online softmax 每块本要用新 max 重缩放累加器（correction）；实际上 **max 很少剧烈变化**——不到阈值就跳过 rescale，correction 工作量大减。**本质：把"每块必做"的保守同步变成"按需做"**——数学不变（最终仍精确归一化），只是延迟了修正时机。⚠️ 这是**改算法执行策略**而非改调度——又一次"编译器做不到、人能做到"（04-02 边界）。

> 🔑 **FA 系读法**：把四代放一起看，"**一份数学 × 四代调度**"——每代重写的驱动力都是硬件新器件（异步拷贝→TMA/wgmma→tcgen05/TMEM）。**追 FA 的版本号 = 追 NVIDIA 异步器件的版本号。**

---

## 2. DeepSeek 开源栈：通信-计算重叠登场

2025 年 DeepSeek 开源周放出的三件套 + 两个系统级策略，代表"极限工程"的另一条线——**MoE 时代，瓶颈从单卡算力转向 EP 通信（A2 §2 + 03-02）**：

### 2.1 三件套速览（各站在③层的什么位置）

| 组件 | 是什么 | 招牌技术 |
| --- | --- | --- |
| **DeepGEMM** | FP8 GEMM 库（含 Grouped GEMM，直指 A2 §2 的 MoE 场景） | **JIT 生成**（形状已知后现编，吃尽编译期信息——04-01 信息衰减的反向操作）；**细粒度 FP8 缩放**（per-128-block scale，两级累加保精度）；TMA 重度使用；代码刻意极简（~300 行核心） |
| **FlashMLA** | MLA（DeepSeek 的 attention 变体）decode kernel | paged KV、变长序列调度、针对 decode 的 seq 维并行——FA 思想在 MLA 结构上的重做 |
| **DeepEP** | **EP all-to-all 通信库**（dispatch/combine） | 见下 |

### 2.2 DeepEP：通信 kernel 也是 kernel

**通信也要 kernel**——把数据从本卡显存搬到别卡，本身就是在 SM 上跑的程序（或绕开 SM 的 RDMA）。DeepEP 提供两套：

| | **normal kernel**（prefill/训练） | **low-latency kernel**（decode） |
| --- | --- | --- |
| 目标 | 高吞吐 | 低延迟 |
| 路径 | **NVLink + RDMA 转发**（节点内 NVLink 中转跨节点流量，榨双层带宽——03-02 四层地图的实战） | **纯 RDMA**（跳过转发省 hop） |
| SM 占用 | **占用一部分 SM** 跑通信 kernel（warp 专化：不同 warp 管不同目标 rank/通道） | **零 SM 占用**：**IBGDA**（GPU 直接发起 RDMA、GPU 写 NIC 门铃，不劳驾 CPU 也不占 SM） |
| 重叠方式 | 与计算 kernel 分 SM 并行 | **hook 式**：发出后立刻返回，计算随后"钩"一下收尾——通信全程藏在计算背后 |
| 名场面 | 用了 `ld.global.nc.L1::no_allocate`（技术上未定义行为的 PTX 提示，实测更快）——**③层压榨到 ISA 边缘的例子**（§01-06） | FP8 dispatch 省带宽 |

### 2.3 TBO / SBO：重叠策略爬上系统层

**注意站位**：TBO/SBO **不是 DeepEP 里的 kernel 技术**，而是**架在 DeepEP 之上的调度策略**（系统/运行时层，⑤层往上）——这是听方案时最容易挂错位置的点：

- **TBO（Two-Batch Overlap，双批重叠）**：把一个 batch 切成两个 micro-batch，**A 算 attention 时，B 在做 MoE 的 all-to-all dispatch**——用一份计算盖住另一份通信。训练侧的 DualPipe（DeepSeek-V3）同思想；推理侧用于 prefill。**代价：两份激活并存，显存 ×~2；batch 要够大才能切。**
- **SBO（Single-Batch Overlap，单批重叠）**：不复制 batch，在**单个 batch 内部按算子阶段**做更细的重叠——比如 dispatch 通信与 shared-expert 计算重叠（SGLang 的 DeepSeek 部署用它）。**省掉 TBO 的显存翻倍，但重叠窗口更碎、对 kernel 拆分粒度要求更高**（DeepEP 的 hook 式接口正是为这种细粒度准备的）。

> 🔑 **统一视角（本附录的收束）——"藏延迟"在栈的每层各现一次身**：
>
> | 层 | 藏什么延迟 | 手段 |
> | --- | --- | --- |
> | 指令级 | 算术 4–6 拍 | ILP、控制位（§01-02） |
> | warp 级 | 访存几百拍 | occupancy 切换（§01-02） |
> | kernel 内 | HBM→shared 搬运 | num_stages 软件流水、warp 专化（§01-02 §2.6） |
> | kernel 间 | launch/依赖空隙 | 多流、CUDA Graph（04-01） |
> | **系统级** | **EP all-to-all 通信（μs–ms 级）** | **TBO/SBO、hook 式通信、DualPipe** |
>
> **每一层的配方都一样：找到两件独立的事，让一件的等待藏进另一件的忙碌。** 从 §01 的车轮战到 TBO，同一个思想爬完了整条栈。

## 3. 开发者视角

- **FA 系**：用户只管调库（`flash_attn`/cuDNN/FlexAttention）；**读它们的 release note = 免费的硬件新特性教材**。
- **DeepSeek 栈**：vLLM/SGLang 已集成（DeepEP+DeepGEMM 后端）；部署 MoE 时的关键旋钮就是 **TBO/SBO 模式选择**（prefill 大 batch→TBO，decode 显存紧→SBO/low-latency）。
- **判词训练**（收 00 判词法）：听到 wgmma/TMEM/ping-pong → 单 kernel 调度（③层）；听到 dispatch/combine/IBGDA → 通信 kernel（③层但对象是网络）；听到 TBO/SBO/DualPipe → 系统级重叠调度（⑤层以上）。

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **FA4**：Blackwell 专属（tcgen05/TMEM/CuTe-DSL）；细节以 Tri Dao 团队后续论文/代码为准（撰写时以公开演讲与第三方分析为据，**标注：可能不完整**）。
- **DeepEP**：面向 H800/受限带宽环境设计（NVLink+IB 混合），在全带宽 NVL72 域上的形态可能演化；IBGDA 依赖 NVSHMEM 生态。
- **通信-计算融合**是活跃前沿：NCCL 的 kernel 融合、Triton-distributed、各推理引擎的 overlap 调度——**"通信成为 kernel 层公民"是趋势**。

## 5. 和其它模块的挂钩

- FA1–3 细节 → **A2**（本篇只补 FA4 与谱系视角）；MUFU 瓶颈/搬瓶颈 → **A1、A2 §1.3**
- tcgen05/TMEM → **03-01**；NVLink/RDMA 四层地图 → **03-02**；EP 的 all-to-all 之痛 → **A2 §2**
- JIT 吃编译期信息 → **04-01 信息衰减**；lazy rescale=改算法非改调度 → **04-02 边界**
- "藏延迟爬全栈"总表 → 00 定盘星的最终延伸

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| FA4 | FlashAttention-4 | Blackwell 代：CuTe-DSL+tcgen05/TMEM+软件exp+lazy rescale | 04 |
| 软件 exp | polynomial exp2 | 多项式在 FFMA pipe 近似 exp，绕开 MUFU——指令级搬瓶颈 | 04 |
| lazy rescale | | max 变化不大就跳过 softmax correction——按需修正 | 04 |
| DeepGEMM | | FP8 JIT GEMM：形状已知后现编+细粒度缩放+两级累加 | 04 |
| FlashMLA | | MLA decode kernel：paged KV+变长调度 | 04 |
| DeepEP | | EP all-to-all 通信库：normal(NVLink+RDMA转发) / low-latency(纯RDMA) | 04 |
| IBGDA | GPU-initiated RDMA | GPU 直接写 NIC 门铃发 RDMA，零 CPU 零 SM 占用 | 04 |
| hook 式重叠 | | 通信发出即返回、计算随后钩收尾——细粒度重叠接口 | 04 |
| TBO | Two-Batch Overlap | 双 micro-batch 互相盖通信；显存×2 换重叠（DualPipe 同源） | 04 |
| SBO | Single-Batch Overlap | 单批内按算子阶段细粒度重叠；省显存但窗口碎 | 04 |
| 通信 kernel | | 通信也是 SM 上的程序（或 IBGDA 绕开 SM）——kernel 层新公民 | 04 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）FA4 软件 exp 的精度账：多项式阶数 vs ULP 误差 vs 吞吐的实测曲线？
- （待填）lazy rescale 的阈值怎么定？对数值稳定性的极端 case（长尾 logits）影响？
- （待填）IBGDA 的门铃机制细节；NVSHMEM 的对称堆在多租户下的约束？
- （待填）TBO 与 SBO 的收益边界：什么 batch/形状下 SBO 反超 TBO？重叠率实测怎么量？
- （待填）DeepGEMM 的 JIT 编译延迟怎么摊（缓存策略）？与 Triton autotune 缓存的对比？

---

*最后更新：2026-07-06（第一版）*
