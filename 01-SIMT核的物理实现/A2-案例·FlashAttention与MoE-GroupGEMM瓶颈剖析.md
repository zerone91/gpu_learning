# 附录 A2 · 案例：FlashAttention 与 MoE Group GEMM 的瓶颈剖析

> **所属模块**：模块 01 · SIMT 核的物理实现（附录 A1 方法论的两个实战案例）
> **状态**：✅ 已按写作规范重写（第三版。知识截至 2026-08，性能数字为公开资料量级）
> **本节主旨**：附录 A1 给了方法——先手算算术强度、再钉到具体部件；这一节把方法用在两个真实算子上。案例一是 **FlashAttention**：融合把它从 HBM 受限救成 Tensor Core 受限，FA-3 再把残余的超越函数瓶颈藏进 Tensor Core 的影子里。案例二是 **MoE 的 Group GEMM**：每个专家分到的 token 数决定它的算术强度，于是 decode 阶段几乎注定卡在权重带宽上，我们会推导出一个值得背下来的翻转阈值——**每个专家攒够约 300 个 token，才喂得饱 Tensor Core**。两个案例共同印证同一条主线：**复用度决定算术强度，算术强度决定卡在哪；而 prefill 和 decode 是同一个算子的两种命运**。

---

## 0. 这一节要回答的问题

- FlashAttention 的瓶颈到底在哪？为什么 prefill 和 decode 的答案完全不同？
- softmax 里的 exp 为什么能拖慢 Tensor Core？FA-3 是怎么把它藏掉的（精确到时间线）？
- MoE 的 Group GEMM 为什么在 decode 阶段几乎总是权重带宽受限？
- "每专家约 300 个 token"这个翻转阈值是怎么推出来的？为什么换成 FP8 它几乎不变？
- 变长小矩阵会带来哪两种"量化浪费"？persistent kernel 怎么救？

---

## 1. 基础词汇：先把本节要用的词讲清楚

**attention 的形状**。attention 的计算是 `O = softmax(QKᵀ/√d)·V`：拿查询矩阵 Q 和键矩阵 K 做一次矩阵乘得到 N×N 的分数矩阵 S，对每行做 softmax 归一化，再和值矩阵 V 乘一次得到输出。N 是序列长度（几千到几十万），d 是每个注意力头的维度（常见 64 或 128）。要记住的比例关系：**两次矩阵乘的计算量是 O(N²·d)，但输入输出 Q/K/V/O 只有 O(N·d)——中间那个 N×N 的 S 才是大头**。

**prefill 与 decode**。大模型推理的两个阶段。prefill 是处理输入提示词：几千个 token 一起进来，矩阵是"胖"的；decode 是逐个生成输出：每步只有 1 个新 token，矩阵是"一行"的。**同一个模型、同一套算子，这两个阶段的矩阵形状差三四个数量级**——本节两个案例都会算出：这个形状差异直接翻转瓶颈的位置。

**KV cache**。decode 阶段每生成一个新 token，它都要对**全部历史 token** 的 K 和 V 做 attention。历史的 K/V 不必重算，缓存在显存里，这份缓存就是 KV cache。它随序列变长线性膨胀，decode 每步都要把它完整读一遍——这个"每步读一遍"就是 decode 阶段 attention 的带宽账本。

**online softmax（在线 softmax）**。softmax 需要全行的最大值和总和才能归一化，看起来必须先算完整行。online softmax 打破这个依赖：边分块扫描边维护"到目前为止的最大值和总和"（running max / running sum），来了新块就更新这两个量、并对已累计的结果做一次重缩放（correction）。数学结果完全精确，代价是每块多一点修正计算。**它是 FlashAttention 能"不落 N×N 矩阵"的数学前提**（B1 精读里有逐行代码）。

**MoE 与 Group GEMM**。MoE（Mixture of Experts，专家混合）把 FFN 层复制成 E 份"专家"，每个 token 由路由器送去其中 top-k 个。路由之后，每个专家 i 分到 M_i 个 token，要算一个 `[M_i×K]·[K×N]` 的矩阵乘——**E 个矩阵乘，K、N 都相同，只有 M_i 各不相同且不均衡**。把这一批变长矩阵乘塞进一次 kernel 启动里算完，就叫 Group GEMM（分组矩阵乘）。

---

## 2. 案例一：FlashAttention

### 2.1 它解决的原始问题：别让 N×N 落显存

先算一笔 naive 实现的账，看看问题多大。N=4096、d=128、bf16（每数 2 字节），单个注意力头：

1. 分数矩阵 S 的大小 = 4096 × 4096 × 2 字节 = **32 MB**——比整个 SM 的 shared memory 大一百多倍，只能写回 HBM。
2. naive 流程要让它在 HBM 上走三趟：QKᵀ 算完写出 32 MB，softmax 读入 32 MB 再写出 32 MB，最后和 V 乘时再读入 32 MB——**单是中间结果就搬了 128 MB**。
3. 对比真正的输入输出：Q/K/V/O 加起来 4 × 4096 × 128 × 2 ≈ 4 MB。**97% 的搬运量花在一个"本可以不存在"的中间矩阵上**。

FlashAttention 的全部思想就是消灭这三趟：把 K/V 切成小块，逐块扫描，每块的分数只活在 shared memory 和寄存器里，用 online softmax 保证边扫边归一化数学不变。N×N 从头到尾不落 HBM。

### 2.2 prefill 的算术强度：一道除法算出 Tensor-bound

用 A1 的方法给 FlashAttention 的 prefill 算 AI（每注意力头）：

1. 计算量：两次矩阵乘（QKᵀ 和 PV）各约 2·N²·d，共 **4·N²·d** FLOP。
2. 搬运量：既然 N×N 不落 HBM，剩下的就只有读 Q/K/V、写 O，共 4 × N × d × 2 字节 = **8·N·d** byte。
3. AI = 4N²d ÷ 8Nd = **N/2**。
4. 代入 N=4096：AI ≈ **2048 FLOP/byte**，远超 H100 FP16 Tensor Core 的平衡点 300。

结论：**长序列 attention 经 FlashAttention 融合后是 Tensor-pipe bound**。注意这个 AI 里 d 消掉了、只剩 N——序列越长复用越足，这就是"复用度决定 AI"在 attention 上的具体形态：每个 K/V 块被 N 个 query 行复用。

### 2.3 decode 的算术强度：同一个算子的另一种命运

decode 阶段每步只有 1 个 query token，对着长度为 N 的 KV cache 算：

1. 计算量：约 2 次矩阵-向量乘，共约 **4·N·d** FLOP。
2. 搬运量：KV cache 整个读一遍，2 × N × d × 2 字节 = **4·N·d** byte。
3. AI ≈ **1 FLOP/byte**——连 FP32 平衡点 20 都远远不到。

**深度 HBM-bound，上限就是读 KV cache 的带宽**。这和 A1 里"GEMM 对 GEMV"是同一个故事：prefill 是 GEMM 型（Tensor-bound），decode 是 GEMV 型（带宽-bound），优化手段从此分家——decode 侧的战场是 FlashDecoding（把 KV 按块切给多个 SM 并行读）、GQA/MLA（直接砍 KV cache 体积，04 模块第 5 节的账）。

### 2.4 隐藏的第二瓶颈：softmax 卡在两个矩阵乘中间

§2.2 说 prefill 是 Tensor-bound，但有个隐藏问题。每个 K/V 块的处理链是三段：

```
S = Q·K_jᵀ      （Tensor Core 干）
softmax(S)       （exp 走 MUFU，行归约走普通 ALU）
O += P·V_j       （Tensor Core 干）
```

三段在**关键路径上串行**：softmax 没算完，下一段矩阵乘的输入 P 就不存在。而 exp 的量不小——每头 O(N²) 次——走的却是吞吐只有 Tensor Core 零头的 MUFU 管线。后果：**Tensor Core 每处理完一块就要停下来等 softmax**。这不是"计算对访存"的老二分法，而是 A1 清单里"管线间序列化"那一行。实测证据：FA-2（没做重叠）在 H100 上的 Tensor Core 利用率只有约 **35%**——三分之二的时间，最贵的硬件在等一个便宜单元干活。

### 2.5 FA-3 的解法：把 MUFU 藏进 Tensor Core 的影子里

FA-3 是为 Hopper 重写的调度（数学没变），用两层重叠消掉序列化。先交代三个角色：Hopper 上 4 个 warp 组成一个 warpgroup（128 线程，异步矩阵指令 wgmma 的操作单位）；FA-3 的一个线程块里有 **1 个生产者 warpgroup**（只发 TMA 搬运指令，用 `setmaxnreg` 让出自己的寄存器）和 **2 个消费者 warpgroup**（拿到大寄存器堆放输出累加器，轮流干矩阵乘和 softmax）。

**第一层：warpgroup 内部的两级流水**。靠 wgmma 的"发了不必等"性质，把下一块的矩阵乘提前发出去，让它和当前块的 softmax 同时进行：

```
发出 S_0 = Q·K_0ᵀ；等它完成
对每个块 j：
    发出 S_{j+1} = Q·K_{j+1}ᵀ    ← Tensor Core 开始在后台算下一块
    算 softmax(S_j)              ← MUFU 干活，与上一行重叠 ✅
    发出 O += P_j·V_j
    等 S_{j+1} 完成
```

softmax 的耗时被塞进了下一块矩阵乘的执行时间里。

**第二层：两个消费者 warpgroup 的 ping-pong**。单个 warpgroup 内的重叠还不够严丝合缝，FA-3 再让两个消费者错开半个相位——用命名屏障（带编号的 `bar.sync`，第 5 节讲过）强制"A 在做矩阵乘时 B 必须在做 softmax"，反过来亦然。时间线：

| 时刻 → | t0 | t1 | t2 | t3 |
| --- | --- | --- | --- | --- |
| warpgroup A | 矩阵乘(块0) | softmax(块0) | 矩阵乘(块1) | softmax(块1) |
| warpgroup B | — | 矩阵乘(块0) | softmax(块0) | 矩阵乘(块1) |
| **Tensor Core 在给谁干活** | A | B | A | B ← 从不空闲 |
| **MUFU 在给谁干活** | — | A | B | A ← 藏进影子 |

两个 warpgroup 处理各自独立的数据块，但共享同一套 Tensor Core——ping-pong 保证这套最贵的硬件**永远被其中一个占着**。底下还垫着生产者的 TMA 在持续搬下一块 K/V，于是三种引擎（DMA 搬运、Tensor Core、MUFU）同时都在忙。同步骨架各司其职：TMA 完成靠 mbarrier 通知消费者，ping-pong 交接靠命名屏障，wgmma 完成靠 `wgmma.wait_group`（这套器件的逐行代码在 B2 精读）。

结果：H100 上 FP16 达到约 **75% 峰值（约 740 TFLOP/s）**，比 FA-2 的 35% 翻了一倍多。**手段不是减少任何计算，而是把便宜管线（MUFU）的工作塞进贵管线（Tensor Core）的空隙**——A1 结尾"把瓶颈搬到最贵的管线并贴着峰值跑"的教科书案例。

### 2.6 占用率画像：又一个"低占用是故意的"

FA kernel 的 tile 很大（典型 128×128、d=128），算一下驻留账：Q、K、V 各一个 tile 就是 3 × 128 × 128 × 2 ≈ 96 KB，加上 K/V 双缓冲，shared memory 用量轻松过百 KB——**每个 SM 只坐得下 1 到 2 个线程块**。寄存器同样紧张（输出累加器加 running max/sum 全在寄存器里）。所以 FA 的占用率极低，延迟全靠上面那套 warp 分工加深流水来藏，不靠堆 warp——它和 A1 例 2 的分块 GEMM 一样，是第 2 节"占用率是手段不是目标"的活体标本。

### 2.7 Nsight 读数对照

| 场景 | 主导信号 | 结论 |
| --- | --- | --- |
| prefill，FA-2 | Tensor 利用率中等（约 35%），stall 里有 MUFU 相关的 Short Scoreboard | Tensor 与 MUFU **序列化** |
| prefill，FA-3 | Tensor 利用率高（约 75%），Memory SOL 不满 | 贴近 **Tensor pipe** 上限 |
| decode | DRAM 吞吐贴峰值，Tensor 利用率很低 | **HBM（KV cache 带宽）** |

---

## 3. 案例二：MoE Group GEMM

### 3.1 每个专家的算术强度就约等于它分到的 token 数

单个专家的矩阵乘是 `[M_i×K]·[K×N]`：K×N 的权重从 HBM 读**一次**，被 M_i 个 token 复用。M_i 小的时候，搬运量里权重占绝对大头，于是：

- 计算量 ≈ 2·M_i·K·N，权重搬运量 ≈ K·N·2 字节（bf16）。
- AI ≈ 2·M_i·K·N ÷ 2·K·N = **M_i**——干干净净，**每专家的算术强度在数值上就约等于它分到的 token 数**。

对照平衡点立刻得出分界：M_i 超过约 300（H100 bf16 Tensor 平衡点）就是 Tensor-bound，不到就是读权重的 HBM-bound。

**用 Mixtral 的真实尺寸手算一遍**（隐藏维 4096，FFN 中间维 14336，8 个专家选 2，bf16）：

- **decode，批大小 32**：每步 32 个 token、每个去 2 个专家、摊到 8 个专家 → 每专家 M_i ≈ 32×2÷8 = **8**。up 投影的权重 4096×14336×2 ≈ 117 MB，计算量 2×8×4096×14336 ≈ 9.4×10⁸ FLOP，AI ≈ 9.4×10⁸ ÷ 1.17×10⁸ ≈ **8**。8 ≪ 300 → **深度权重带宽受限**：GPU 在为区区 8 个 token 把 117 MB 的权重从 HBM 完整拉一遍。
- **prefill，一条 4096 token 的序列**：每专家 M_i ≈ 4096×2÷8 = **1024** ≫ 300 → **Tensor-bound**，跟普通大 GEMM 无异。

同一个 MoE 层，decode 卡权重带宽、prefill 卡 Tensor 管线——这就是"MoE 推理 decode 是 memory-bound"这句行话的微架构根因。

### 3.2 翻转阈值 M*：一个值得背下来的数

把上面的分界算精确些。完整的搬运量除了权重还有输入激活（M·K）和输出（M·N），但 M 小时权重项主导，可以简化为 AI ≈ 2M ÷ b_w（b_w 是权重每个数的字节数）。令 AI 等于机器平衡点 β，解出翻转点：

**M\* ≈ β × b_w ÷ 2**

代入 H100 的两种精度，逐步算：

1. **bf16**：β ≈ 990 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOP/byte，b_w = 2 → M\* ≈ 295 × 2 ÷ 2 = **约 295**。
2. **FP8**：β ≈ 1979 ÷ 3.35 ≈ 590，b_w = 1 → M\* ≈ 590 × 1 ÷ 2 = **约 295**。

两种精度算出同一个数——这不是巧合：**降精度同时把算力抬一倍、把权重字节砍一半，两个效果在 token 阈值上正好抵消**。所以这个结论跨精度成立，值得背下来：**每个专家攒够约 256–300 个 token，Tensor Core 才吃得饱**。

换算成需要多大的并发批：decode 时每专家 M_i = B×top_k÷E（B 是并发序列数），反解 B\* = M\*×E÷top_k：

| 模型配置 | decode 时每专家 token 数 | 翻转所需并发批 B\* |
| --- | --- | --- |
| Mixtral（8 专家选 2） | B ÷ 4 | **约 1200** |
| DeepSeek 型细粒度 MoE（256 专家选 8） | B ÷ 32 | **约 9400** |

真实服务的并发批通常是几十到几百——离 1200 和 9400 都很远，所以 **MoE 的 decode 在实践中几乎总是权重带宽受限**；专家切得越细，每个专家越"饿"，阈值越高。

这笔账还有一个运营推论，解释了"为什么推理系统拼命攒批"：在带宽受限区，算 M 个 token 的耗时 ≈ 读一遍权重的耗时，**几乎与 M 无关**——于是每 token 的成本正比于 1/M，往专家里多塞 token 在越过 M\* 之前**近乎免费**。这就是 continuous batching 和专家并行（把更多 token 汇聚到同一份专家权重上）的第一动机。

### 3.3 两种"量化浪费"：tile 量化与 wave 量化

变长小 M 还带来两笔额外的浪费，名字里都有"量化"（quantization，这里指"只能取整数份"的浪费，与数值精度的量化无关）：

**tile 量化**。kernel 的 tile 尺寸是编译期定死的，比如输出按 128 行一个 tile 切。某专家 M_i = 8：照样要起一个 128 行的 tile，其中 120 行算的全是填充零——**搬运和计算都按 128 行付费，有效产出只有 8 行，效率 6%**。再看 M_i = 130：要起 2 个 tile，第二个只用 2 行——刚跨过边界一点点，浪费接近一整个 tile。这和模块 02 第 5 节脉动阵列的"形状税"是同一个数学，只是这里的"阵列尺寸"换成了软件选的 tile 尺寸。

**wave 量化**。全部专家的 tile 总数未必是"SM 能同时坐下的块数"的整数倍，最后一波只有零星几个 tile 在跑、其余 SM 空转——第 7 节的尾效应，在变长 Group GEMM 上因为块数难以凑整而更常发。

**加上负载不均**（路由可能把 token 集中到少数热门专家），三个问题叠在一起：有的 SM 分到大专家忙到最后，有的早早算完干等。

### 3.4 解法：persistent kernel 把不均摊平

朴素做法是把每个专家静态切给一段 grid，专家大小悬殊时上面三个问题全部爆发。现代实现（CUTLASS Grouped GEMM、vLLM/SGLang 的 fused MoE kernel）换成 **persistent kernel（常驻 kernel）**：只启动"刚好坐满机器"数量的线程块，每个块干完一个 tile 就去全局计数器上 `atomicAdd` 领下一个（B3 精读里有这段代码），把 E 个变长问题拆成一条均匀的 tile 流。

**一个对照例子**。三个专家的 tile 数是 12、2、2，机器能同时跑 8 个块。静态划分按专家切：分到专家 1 的块要连算多轮，分到专家 2、3 的块很快闲下来——尾部只有一半机器在干活。动态领活：16 个 tile 排成一个队列，8 个常驻块谁闲谁领，最后一波也只差 2 个 tile 的零头。**专家之间的不均衡被 tile 粒度的动态分配磨平了**。Hopper 上这套调度还会再叠上 TMA、wgmma、warp 分工——单 kernel 的技术全家桶都用上。

### 3.5 Nsight 读数对照

| 场景 | 主导信号 | 结论 |
| --- | --- | --- |
| decode（每专家 token 少） | DRAM 吞吐贴峰值，Tensor 利用率低 | **HBM（专家权重带宽）** |
| prefill（每专家 token 多） | Tensor 利用率高 | **Tensor pipe**（注意 tile 量化在侵蚀有效 FLOP） |
| 负载不均 / grid 太小 | 实测占用率 ≪ 理论，波数是零头 | **调度与尾部（wave 量化）** |

---

## 4. 横向对照：两个案例钉死一条主线

| | prefill（M 大 / 序列长） | decode（M 小 / 单 token） |
| --- | --- | --- |
| **FlashAttention** | Tensor-bound（残余 MUFU 序列化由 FA-3 藏掉） | HBM-bound（KV cache 带宽） |
| **MoE Group GEMM** | Tensor-bound（tile 量化在边缘收税） | HBM-bound（专家权重带宽） |

三条共同结论，每条都能从上面的手算里直接读出来：

1. **复用度决定 AI，AI 决定卡在哪**。FlashAttention 的复用度是序列长 N（每个 K/V 块被 N 行 query 复用），MoE 的复用度是每专家 token 数 M_i（每份权重被 M_i 个 token 复用）。复用大则 AI 高、贴 Tensor 的墙；复用小则 AI 塌、贴 HBM 的墙。
2. **decode 阶段几乎注定 memory-bound**——每步 token 太少，无论复用的对象是 KV cache 还是专家权重，都摊不开。这是大模型推理系统的第一性约束，GQA、MLA、投机解码、攒批，全是围着它转的（04 模块第 5 节）。
3. **优化等于搬瓶颈**：融合抬 AI（FA）、攒批加大 M（MoE）、动态调度喂满 SM（persistent），终点都是把瓶颈推到最贵的 Tensor 管线上并贴着它的峰值跑。

---

## 5. 开发者视角

- **先分阶段再 profile**：prefill 和 decode 的瓶颈不同，混在一起 profile 会互相稀释信号。用 `torch.profiler` 分别抓两个阶段的热点 kernel，再按 A1 的三级流程各钻各的。
- **FA 版本就是硬件版本**：调 `flash_attn` 库时版本号对应硬件代际（FA-2 通吃、FA-3 要 Hopper、FA-4 要 Blackwell），装错版本不报错、只是默默跑慢一代的调度。
- **MoE 部署的第一个数**：先算你的每专家 M_i（并发批 × top_k ÷ 专家数），对照 300 的阈值就知道自己在哪个 regime，再决定优化方向是攒批/专家并行（带宽侧）还是调 kernel（计算侧）。

## 6. 最新架构落点（时效锚点 · 知识截至 2026-08）

- **FA-3 是 Hopper 专属**（TMA、wgmma、warp 分工、ping-pong 缺一不可）；Blackwell 上的 **FA-4 论文已于 2026-03 发表**，换到 tcgen05、TMEM 与 2-CTA 配对，B200 上 BF16 达 1613 TFLOPS、约 71% 利用率（04 模块附录 A1 有完整技术拆解）。Tensor 平衡点更高，"必须融合、必须攒批"的压力更大。
- **🔥 decode 的带宽墙催生了硬件层的回应：Rubin CPX。** §4 那张表的结论是"decode 阶段几乎注定 memory-bound，这是推理系统的第一性约束"。NVIDIA 2026 年的答案不是让 decode 变快，而是**把 prefill 拆出去用另一种芯片**：Rubin CPX 单 die、**128 GB GDDR7 而非 HBM**、30 PFLOPS NVFP4，只干 prefill；decode 留给配 HBM4 的 Rubin 本体。**为什么这样分是对的，用本节的账就能说明**——prefill 的 AI ≈ N/2（§2.2 算过，N=4096 时约 2048），根本用不上 HBM 的带宽，给它配 HBM 是浪费；decode 的 AI ≈ 1，除了带宽什么都不需要。**同一个模型的两个阶段，对硬件的需求正交到可以用两种硅片分别满足——这是本节全部手算的最终归宿。**
- **对 §3 那个翻转阈值 M\* 的影响**：新硬件的平衡点更高（附录 A1 §7 有跨代际的平衡点表），而 M\* ≈ 平衡点 × 权重字节数 ÷ 2。**FP4 时代重算一遍**：Blackwell 的 NVFP4 平衡点约 1875、权重 0.5 字节/个 → M\* ≈ 1875 × 0.5 ÷ 2 ≈ **470**。比 H100 时代的 295 高了六成。**注意"降精度不改变阈值"这个漂亮的抵消关系在这里失效了**——因为 Blackwell 的算力涨幅超过了位宽降幅（15 PF 相对 H100 FP8 的 2 PF 涨了七倍多，而位宽只减半）。**结论要更新：M\* 跨精度近似不变这条规律只在"算力涨幅恰好等于位宽降幅"时成立，新代际打破了这个前提，MoE decode 的攒批压力比过去更大。**
- **MoE 侧**：Blackwell 的 FP4 和更大的 HBM（B300 到 288 GB、Rubin 到 288 GB HBM4）缓解权重带宽压力；DeepSeek 式细粒度专家加共享专家改变 M_i 的分布，等于直接改瓶颈位置。生产实现普遍是 Triton/CUTLASS 的 grouped GEMM 加 persistent 调度。
- **AMD**：MI300X 的 192 GB HBM3 加 Infinity Cache 对 MoE 权重带宽友好；当前代际 MI455X 为 432 GB HBM4、约 23 TB/s，容量优势在 MoE 上尤其明显（专家权重能多驻留一批）。同样的 SOL 下钻用 rocprof/Omniperf 做。

## 7. 术语卡

### prefill 与 decode

**定义**：大模型推理的两个阶段。prefill 并行处理整段输入提示词（一次几千 token），decode 逐个生成输出 token（一次 1 个）。同一套算子在两个阶段的矩阵形状差三四个数量级。

**为什么存在**：作为一对词存在，是因为它们的瓶颈类型系统性地不同——prefill 的复用度足、通常计算受限，decode 的复用度塌、几乎注定带宽受限。不区分阶段谈"这个 kernel 快不快"是没有意义的。

**语境例句**："这个优化 prefill 提了 40%，decode 一点没动——废话，decode 卡的是 KV cache 带宽，你优化的是 Tensor 利用率。"——意思是：两个阶段瓶颈不同，优化收益不迁移。

### KV cache

**定义**：decode 阶段缓存在显存里的全部历史 token 的 K、V 矩阵。每生成一个新 token 都要把它完整读一遍，体积随上下文长度线性增长。

**为什么存在**：不缓存就要每步重算全部历史的 K/V，计算量平方级爆炸；缓存之后计算省了，代价是"每步读一遍"变成 decode 的带宽账本、且显存占用限制并发数——它是推理系统一切"省 KV"设计（GQA、MLA、paged attention、量化 KV）的靶子。

**语境例句**："上下文拉到 128K 之后瓶颈全在 KV cache 上，算力再多也白搭。"——意思是：decode 每步读的 KV 体积太大，HBM 带宽封顶。

### Grouped GEMM（分组矩阵乘）

**定义**：在一次 kernel 启动里算完一批独立的矩阵乘，它们的 K、N 相同、只有 M 各不相同（每个 MoE 专家一个）。区别于 batched GEMM（所有矩阵形状完全相同）。

**为什么存在**：MoE 路由天然产出"一堆变长小矩阵乘"，逐个启动 kernel 的开销和尾效应都不可接受；Grouped GEMM 把它们合并成一次启动，再配合动态 tile 调度摊平专家间的不均衡。

**语境例句**："fused MoE 那个 kernel 就是个 persistent 的 grouped GEMM，专家不均衡它也能吃。"——意思是：常驻块动态领 tile，负载不均在 tile 粒度被磨平。

### tile 量化（tile quantization）

**定义**：矩阵尺寸不是 tile 尺寸整数倍时，边缘 tile 里填充部分照常消耗计算和搬运的浪费。M=8 撞上 128 行的 tile，有效率只有 6%。

**为什么存在**：tile 尺寸必须编译期定死（寄存器和 shared memory 的分配依赖它），运行期的真实 M 却千变万化——静态决策撞上动态形状，就在边界上收税。它是模块 02"形状税"的 GPU 软件版。

**语境例句**："M=130 比 M=128 慢了快一倍，tile 量化，第二个 tile 就用两行。"——意思是：刚跨过 tile 边界，为 2 行数据付了 128 行的钱。

### persistent kernel（常驻 kernel）

**定义**：只启动恰好坐满 GPU 的线程块数，每个块循环地从全局队列领取工作项（tile），直到全部干完——而不是"一个块干一份活就退休"。

**为什么存在**：静态划分在负载不均时必然有人忙死有人闲死；常驻块 + 动态领活把不均衡拆成 tile 粒度磨平，同时消掉海量小块的启动开销和尾效应。代价是 kernel 内要自建调度逻辑（原子计数器领活）。

**语境例句**："专家大小差 50 倍没关系，persistent 调度下 SM 利用率照样打满。"——意思是：动态领活让快手多领、慢工少领，机器不空转。

### 翻转阈值 M\*（每专家喂饱 Tensor Core 的 token 数）

**定义**：MoE 专家的矩阵乘从带宽受限翻转为计算受限所需的每专家 token 数，M\* ≈ 机器平衡点 × 权重字节数 ÷ 2。H100 上无论 bf16 还是 FP8 都约等于 **300**。

**为什么存在**：它把"MoE decode 为什么慢、攒批为什么有效"变成一个可以口算的判断：拿并发批算出 M_i，和 300 一比，regime 立判。跨精度不变这个性质让它特别好记——降精度抬算力和砍字节两个效应正好抵消。

**语境例句**："256 个专家选 8，你要 batch 九千多才够翻转——别指望了，按带宽受限做设计。"——意思是：细粒度 MoE 的 decode 在现实并发下永远在 M\* 左侧。

## 8. 我的困惑 / 待深挖

- FA-3 的 ping-pong 相位在不同 head dim、因果掩码（每块工作量不等）下怎么调整？
- 专家并行（EP）把 token 汇聚到本卡后，实际的 M_i 分布怎么变？与 TP/DP 的联合影响？
- tile 量化和 wave 量化的损失在 Nsight 里如何分别量化（有效 FLOP 效率对占用率缺口）？
- stream-K（沿 K 维切分再全局归约的均衡策略）在变长 Group GEMM 上，归约开销和均衡收益的平衡点在哪？

---

*最后更新：2026-08-31（第四版：§6 更新到 2026-08——FA-4 论文数字定稿、补入 Rubin CPX 作为"decode 带宽墙"的硬件级回应，并按 NVFP4 重算了翻转阈值 M*（约 470），指出"跨精度不变"这条规律在新代际失效。第三版：按写作规范重写）*
