# 附录 A2 · 案例：FlashAttention 与 MoE Group GEMM 的瓶颈剖析

> **所属模块**：模块 01 · SIMT（A1 方法论的实战案例）
> **状态**：🟨 学习中（第一版，知识截至 2026-01；性能数字为公开资料量级）
> **一句话主旨**：用 A1 的框架（算 AI → 分到具体 pipe/级 → Nsight 三级下钻）剖两个真实算子——**FlashAttention**（融合把瓶颈从 HBM 搬到 Tensor+MUFU；FA-3 再把 MUFU 藏进 Tensor）与 **MoE Group GEMM**（变长小 M 使 decode 落到权重带宽 bound，外加 tile/wave 量化与调度不均）。贯穿结论：**"复用度决定 AI，AI 决定 bound 在哪"，而 prefill vs decode 是同一算子的两种 bound。**

---

## 0. 这一节要回答的问题

- [x] FlashAttention 的瓶颈到底在哪？为什么 prefill 与 decode 完全不同？
- [x] softmax（exp）在里面扮演什么角色，为什么它能拖慢 Tensor Core？
- [x] FA-2 → FA-3 到底改了哪个 bound？（用 §02 §2.6 的异步流水解释）
- [x] MoE Group GEMM 为什么典型地"权重带宽 bound"？tile/wave 量化怎么伤害它？
- [x] 这两个案例怎么共同印证 A1 的"复用→AI→bound"主线？

---

## 1. FlashAttention（FA-2 / FA-3）

### 1.1 算子结构与形状

Attention：`O = softmax(QKᵀ / √d) · V`。形状 `Q,K,V ∈ [B, H, N, d]`（N=序列长，d=head dim，常见 64/128）。

FlashAttention 的核心：**不落 N×N 的分数矩阵**——按 K/V 分块，边扫边做 online softmax（维护 running max/sum 并重缩放），S 的 tile 只活在 shared/寄存器里。

### 1.2 AI 分析：prefill 为何 Tensor-bound、decode 为何 HBM-bound

**Prefill（长序列，N 大）**，每 head：
- FLOPs ≈ **4·N²·d**（两个 matmul：QKᵀ 与 PV，各 ~2N²d）。
- HBM 字节 ≈ **4·N·d·2**（bf16 读 Q/K/V + 写 O，都是 O(N·d) 而非 O(N²)——这正是 FA 的意义）。
- **AI ≈ 4N²d / (8Nd) = N/2**。N=4096 → AI≈2048 FLOP/byte ≫ 300（H100 FP16 Tensor 平衡点）→ **Tensor-pipe bound**。✅ 长序列 attention 的 matmul 部分是计算受限。

**Decode（自回归，query 只有 1 个 token，N_q=1）**：
- 每步读**整个 KV cache**（长度 N 的 K、V），FLOPs≈2·N·d，字节≈2·N·d·2。
- **AI ≈ 0.5 FLOP/byte → 深度 HBM-bound**（读 KV cache 的带宽封顶）。这就是 FlashDecoding / 分块并行 KV 的战场。

> 🔑 同一个 attention，**prefill 是 Tensor-bound、decode 是 HBM(KV cache)-bound**——和 A1 里"GEMM vs GEMV"是同一个道理。优化手段因此完全分家。

### 1.3 真正的难点：softmax(MUFU) 卡在两个 matmul 之间

即便 prefill 是"Tensor-bound"，也有个隐藏的第二瓶颈：**数据流是 `QKᵀ(Tensor) → softmax(MUFU/exp) → PV(Tensor)`，三者在关键路径上串行**。softmax 的 exp 走 **MUFU/SFU pipe**（吞吐远低于 Tensor）。于是：

- 每 head 的 exp 次数 ~ O(N²)，而 MUFU 峰值只有 Tensor 的零头。
- **naive/未重叠实现里，Tensor Core 在做 softmax 的那段时间是空转的** → 实测 Tensor 利用率被 softmax 拉低。
- 这不是"compute vs memory"，而是 **pipe 间序列化**——A1 清单里的一种"具体位置"。

### 1.4 FA-2 → FA-3：改的是这个 bound

- **FA-2**：相对 FA-1 减少了非 matmul FLOPs、把并行度从 batch·head 扩到序列维、优化 warp 间分工减少 shared 往返。但它**没有把 softmax 和 GEMM 重叠**，也没吃透 Hopper 的异步器件 → 在 **H100 上 Tensor 利用率仅 ~35%**（tensor 在等 softmax）。
- **FA-3（Hopper 专门优化）**：正是 §02 §2.6 那套——
  - **TMA** 异步搬 Q/K/V tile，**wgmma** 异步 Tensor Core；
  - **warp 专化**：生产者 warp 管 TMA，消费者 warpgroup 管 MMA+softmax；
  - **ping-pong 调度**：让 block A 的 **softmax(MUFU)** 与 block B 的 **GEMM(Tensor)** 重叠——**把 MUFU 藏进 Tensor 的影子里**；
  - （FP8 路径 + incoherent processing 处理精度）。
  - 结果：**回到贴近 Tensor-pipe bound，H100 FP16 达 ~75% 峰值（~740 TFLOP/s），FP8 ~1.2 PFLOP/s。**

> 🔑 **FA-3 的本质**：瓶颈从"Tensor 与 MUFU 序列化"搬回"纯 Tensor-bound"。手段不是减少计算，而是**用异步流水把便宜 pipe(MUFU) 的延迟塞进贵 pipe(Tensor) 的空隙**。这是 A1 结尾"把瓶颈搬到最贵的 pipe 并贴着它的峰值"的教科书案例。

### 1.5 占用率画像：低占用，靠异步流水而非 occupancy

FA kernel 的 tile 大（如 Br=Bc=128，d=128）：
- **shared**：Q/K/V tile 各 128×128×2B≈32KB，加 S/P、双缓冲 K/V → 轻松上百 KB → **shared-bound，1~2 block/SM**。
- **寄存器**：online softmax 的 O 累加器 + running max/sum → 高寄存器占用。
- → **刻意低占用率**，延迟靠 §2.6 的 warp 专化 + 深流水藏，不靠堆 warp。正是 §02 §2.3/2.5 的活体标本。

### 1.6 Nsight 定位与一句话结论

| 场景 | 主导信号 | bound 位置 |
| --- | --- | --- |
| Prefill · FA-2 | Tensor pipe util 中等(~35%)、stall 含 MUFU/`Short Scoreboard` | Tensor↔MUFU **序列化** |
| Prefill · FA-3 | Tensor pipe util 高(~75%)、Memory 不满 | **Tensor pipe** |
| Decode | `DRAM Throughput`≈峰值、Tensor util 低 | **HBM（KV cache 带宽）** |

> **一句话**：FlashAttention 把 attention 从"HBM-bound（naive 落 N×N）"救到"Tensor-bound"，FA-3 再把残余的 MUFU 序列化藏掉；但 decode 阶段绕不开 KV cache 的 HBM 带宽。

---

## 2. MoE Group GEMM（Grouped GEMM）

### 2.1 算子结构：一批"变长、小 M、不均衡"的 GEMM

MoE：每个 token 经路由送到 top-k 个专家，每个专家是一个 FFN（两个 GEMM）。路由后，**每个专家分到的 token 数 M_i 不等**（负载不均）。Group GEMM = 在**一次 kernel 启动**里，算许多个**独立、N/K 相同但 M_i 不同**的 GEMM（每专家一个）。

### 2.2 AI 分析：per-expert AI ≈ 每专家 token 数 M_i

单个专家的 GEMM：`[M_i × K] · [K × N] = [M_i × N]`，权重 `[K×N]` 从 HBM 读一次、被 M_i 个 token 复用。
- 权重字节主导（M_i 小时），**复用度 = M_i** → **AI ≈ M_i**（量级）。
- **M_i > ~300（bf16 Tensor 平衡点）→ Tensor-bound；M_i 小 → 读权重的 HBM-bound。**

**Mixtral 量级手算**（d=4096，FFN 中间 14336，top-2/8 专家，bf16）：
- **Decode**（1 token/step，batch B=32）：每专家 token ≈ B·2/8 = **8**。
  - up-proj：M=8, K=4096, N=14336。权重字节≈4096·14336·2≈**117 MB/专家**；FLOPs≈2·8·4096·14336≈9.4e8。
  - **AI ≈ 9.4e8 / 1.17e8 ≈ 8 ≪ 300 → 深度权重-带宽 bound。**
- **Prefill**（seq=4096，B=1）：每专家 token ≈ 4096·2/8 = **1024**。
  - **AI ≈ 1024 ≫ 300 → Tensor-bound**（和普通大 GEMM 一样）。

> 🔑 同一个 MoE 层：**decode 卡在权重 HBM 带宽，prefill 卡在 Tensor pipe。** 这就是"MoE 推理 decode 是 memory-bound"的微架构根因——你为极少 token 反复把整批激活专家的权重从 HBM 拉进来。

### 2.3 tile 量化 & wave 量化：变长小 M 的两种浪费

- **tile 量化（tile quantization）**：tile 固定 BM=128，某专家 M_i=8 → 仍起一个 128 行的 tile，只用 8 行 → **算力/搬运按 128 行摊，浪费**。M_i=130 → 2 个 tile，第二个只用 2 行。**变长小 M 让 tile 边界浪费极严重。**
- **wave 量化（wave quantization）**：各专家 tile 总数未必是 SM 数的整数倍 → 最后一波只占部分 SM → 尾部半空，实测占用率 ≪ 理论。
- 负载不均（有的专家 token 多、有的少）进一步加剧 SM 间不平衡。

### 2.4 调度：persistent kernel + stream-K/grouped 调度把不均摊平

朴素 Group GEMM 把每个专家静态分给一段 grid → 专家大小悬殊时 **SM 负载严重不均、尾部空转**。现代实现（CUTLASS Grouped GEMM、Triton grouped GEMM，用于 vLLM/SGLang 的 fused MoE）用：
- **persistent kernel**：常驻一批 CTA，配一个**全局 tile 调度器**动态领取 tile（grouped / stream-K 思路），把"变长不均的一堆问题"**摊成均衡的 tile 流**，喂满所有 SM。
- Hopper 上再叠 TMA + wgmma + warp 专化。

### 2.5 Nsight 定位与一句话结论

| 场景 | 主导信号 | bound 位置 |
| --- | --- | --- |
| Decode（小 M_i） | `DRAM Throughput`≈峰值、Tensor util 低 | **HBM（权重带宽）** |
| Prefill（大 M_i） | Tensor pipe util 高 | **Tensor pipe**（但注意 tile 量化侵蚀有效 FLOP） |
| 不均/小 grid | `Achieved`≪`Theoretical` occupancy、波数少、部分 SM 空 | **调度/尾部（wave 量化）** |

> **一句话**：MoE Group GEMM 的 bound 强依赖 M_i——decode 权重带宽 bound、prefill Tensor bound；变长小 M 额外带来 tile/wave 量化浪费，必须靠 persistent + 动态 tile 调度救回 SM 利用率。

---

## 3. 横向对照：把 A1 主线钉牢

| | Prefill（M 大 / 长序列） | Decode（M 小 / 单 token） |
| --- | --- | --- |
| **FlashAttention** | Tensor-bound（残余 MUFU 序列化，FA-3 藏掉） | HBM-bound（KV cache 带宽） |
| **MoE Group GEMM** | Tensor-bound（+tile 量化侵蚀） | HBM-bound（专家权重带宽） |

**共同本质（= A1 的核心）**：
1. **复用度决定 AI，AI 决定 bound**。FA 的复用 ~ N（序列长），MoE 的复用 ~ M_i（每专家 token 数）。复用大 → 抬 AI → Tensor-bound；复用小 → AI 塌 → HBM-bound。
2. **decode/生成阶段几乎注定 memory-bound**（KV cache 或权重带宽），因为"每步 token 太少、复用太低"——这是 LLM 推理系统的第一性约束。
3. **优化=搬瓶颈**：融合（FA）、批处理/加大 M、persistent 调度，本质都是**抬 AI 或喂满 pipe，把瓶颈推向最贵的 Tensor pipe 并贴着它的峰值**。

---

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **FA-3 是 Hopper 专属**：强依赖 TMA / wgmma / warp 专化 / ping-pong（§02 §2.6）。Blackwell 上进一步吃 **FP4 + TMEM**，Tensor 平衡点更高 → 更逼融合与更大 batch 才 compute-bound。
- **MoE**：Blackwell 的 FP4 + 更大 HBM（B300 288GB）缓解权重带宽压力；DeepSeek 等的**分组/共享专家 + 更细粒度专家**改变 M_i 分布，直接改 bound。生产实现（vLLM/SGLang/TensorRT-LLM）普遍用 Triton/CUTLASS grouped GEMM + persistent 调度。
- **AMD MI300X**：192GB HBM3 + Infinity Cache 对 MoE 权重带宽友好；用 rocprof/Omniperf 做同样的 SOL 下钻。

## 5. 和其它模块的挂钩

- 异步流水 / warp 专化 / ping-pong → **§02 §2.6**、**§05 同步**
- AI / SOL / pipe 定位方法 → **附录 A1**
- KV cache、权重带宽是系统级约束 → **模块 04 §05 与算法侧接口**
- Tensor Core / wgmma / TMEM → **模块 03**
- grouped GEMM 的图层/kernel 实现与调度 → **模块 04 kernel 层 + 映射问题**

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| online softmax | | 边扫 K/V 边维护 running max/sum 重缩放，免落 N×N | 01/04 |
| prefill / decode | | 长序列并行 vs 单 token 自回归；bound 完全不同 | 01/04 |
| KV cache | | 缓存历史 K/V；decode 阶段读它，HBM-bound 之源 | 01/04 |
| Group/Grouped GEMM | | 一次算多个 N/K 相同、M 不同的独立 GEMM（MoE 用） | 01/04 |
| tile 量化 | tile quantization | 固定 tile 对不整除的 M 浪费边界行 | 01 |
| wave 量化 | wave quantization | tile 总数非 SM 整数倍，尾部半空 | 01 |
| persistent kernel | | 常驻 CTA + 全局 tile 调度器，摊平不均、喂满 SM | 01/04 |
| stream-K | | 沿 K 维切分 + 全局归约的均衡调度策略 | 04 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）FA-3 ping-pong 的精确时序：两个 warpgroup 怎么在 mbarrier 上交接、寄存器怎么分？
- （待填）MoE decode 权重带宽 bound 下，batch 到多大才翻转成 Tensor-bound？和 expert 并行/张量并行怎么交互？
- （待填）tile 量化 vs wave 量化在 Nsight 里如何各自量化其损失（有效 FLOP 效率 vs 占用率缺口）？
- （待填）stream-K 在变长 Group GEMM 上的归约开销 vs 负载均衡收益的权衡点？

---

*最后更新：2026-07-05（第一版）*
