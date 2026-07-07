# kernel 层（tiling / 手写 vs 自动调优）

> **所属模块**：模块 04 · 软件优化栈
> **状态**：🟨 学习中（第一版，知识截至 2026-01）
> **一句话主旨**：③层的全部工作可以压缩成一句话——**"计算"是固定的数学，"调度（schedule）"是它在硬件上的展开方式**（怎么切块、谁搬谁算、流水多深、数据摆哪级）。模块 01/02 学的所有技术（tiling、双缓冲、swizzle、warp 专化、向量化）**都是调度空间里的维度**。"手写 vs 自动调优"之争，本质是"**这个调度空间由人搜、模板搜、还是编译器搜**"——一条从预制菜到手工锻打的光谱，按"形状标准度 × 性能要求 × 人力"选位置。

---

## 0. 这一节要回答的问题

- [x] 从软件栈视角，一个 kernel 的"合同"是什么？
- [x] "计算 vs 调度分离"这个思想为什么是③层的地基？
- [x] 一个 GEMM 的调度参数空间具体长什么样、有多大？
- [x] 手写→模板→DSL→全自动 这条光谱上各工具的真实取舍？
- [x] autotune 到底在搜什么、什么时候赢什么时候输？
- [x] 算法侧的你，什么时候该亲自写 Triton、什么时候别写？

---

## 1. 核心概念

### 1.1 kernel 的"合同"

从栈的视角，一个 kernel 是③层交给④/⑤层的**最小执行合同**：
- **输入/输出在 HBM**（进出边界必落地——这就是②层融合想消灭的东西）；
- 附带一个 **launch 配置**（grid×block，§01-02 的占用率由此起算）；
- 内部对硬件的一切使用（shared/寄存器/Tensor Core/流水）**外界不可见**——kernel 是栈里的"黑盒颗粒度"。

### 1.2 地基思想：计算与调度分离（compute/schedule split）

Halide（图像处理 DSL）留给整个领域的核心遗产，TVM/Triton/CUTLASS 全部建立其上：

> **计算（compute）**：`C[i,j] = Σ_k A[i,k]·B[k,j]` ——数学，唯一且不变。
> **调度（schedule）**：这个求和**以什么顺序、什么粒度、放什么存储、几路流水**在硬件上展开——**同一个计算有天文数字种调度，性能差几十倍**。

模块 01/02 学的全部 kernel 技术，现在统一归位为**调度空间的维度**：

| 调度维度 | 对应硬件知识 |
| --- | --- |
| tile 尺寸 BLOCK_M/N/K | §02-04 tiling 不等式（装得下×喂得饱） |
| 线程/warp 映射 | §01-01/02（谁算哪块） |
| 数据放哪级（寄存器/shared） | §01-04 复用放大器、§02-03 数据流选择 |
| 流水深度 num_stages | §01-02 §2.6 异步流水 |
| num_warps / warp 专化 | §01-02 占用率、A2 FA-3 |
| swizzle / 向量化宽度 | §01-04 |
| split-K / stream-K | A2 §2.4（K 维切分换并行度） |

> 🔑 **一个具体的 GEMM 调度空间**：BLOCK_M/N/K ∈ {64,128,256}³ × num_warps ∈ {4,8} × num_stages ∈ {2..5} × split-K ∈ {1,2,4} ≈ **几百到几千个合法配置**，最优点随 (形状, dtype, 架构) 漂移——**这就是"调优"作为一个工程问题存在的原因**：空间太大，人脑穷举不动，但也没大到不能暴力搜。

### 1.3 光谱：谁来搜这个调度空间

```
预制菜 ◄──────────────────────────────────────────► 手工锻打
cuBLAS/cuDNN ─── CUTLASS/CuTe ─── Triton ─── Inductor codegen ─── 手写SASS
(库函数,调度   (调度=C++模板参数, (调度=DSL+   (全自动生成,      (超越编译器
 已被NV搜好)    人挑模板+实例化)   autotune搜)   零人力)           的最后%)
```

| | 性能上限 | 人力 | 灵活性（融合/怪形状） | 适用 |
| --- | --- | --- | --- | --- |
| **cuBLAS/cuDNN/FA库** | 标准形状下≈最优（NV 替你搜完了） | 零 | 差（固定算子） | 标准稠密 GEMM/conv/attention——**别自己写** |
| **CUTLASS/CuTe** | 逼近手写 | 高（C++ 模板深水区） | 中（EVT epilogue 可组合） | 库没覆盖的高性能 GEMM 变体；造库的人用 |
| **Triton** | 常达库的 80–100%（融合场景反超，因为库融不了） | 中（Python 级） | **强**——自定义融合的主战场 | 融合链、怪形状、研究迭代 |
| **Inductor 自动生成** | memory-bound 算子接近最优；GEMM 靠调库/模板 | 零 | 自动 | torch.compile 的默认输出 |
| **手写 SASS** | 极限 | 极高 | 无 | 库作者的最后 1–5%（§01-06） |

> 🔑 **判读**：光谱位置由"**你的算子离标准件有多远**"决定。标准 GEMM → 左端（库已最优，自己写是浪费）；**自定义融合链 / 新 attention 变体 / 怪形状** → Triton（库根本没有这个算子，80% 的性能 × 无限灵活 = 赢）；要造库 → CUTLASS。**Triton 的真正价值不是"比 cuBLAS 快"，而是"让不存在的融合 kernel 存在"。**

### 1.4 autotune：暴力但有效

- **在搜什么**：§1.2 那个配置空间。`@triton.autotune` 给一组候选 config，**每个都实际编译+跑一遍 benchmark**，按 (形状桶, dtype, 架构) 缓存最优。
- **代价**：首跑慢（每 config 编译+测各几十 ms~s）；形状桶变多时缓存膨胀 → 动态 shape 是它的天敌（§02-05 bucket 化在这里再现）。
- **赢/输**：怪形状/融合算子上**赢**（没人为你的形状预搜过）；标准大 GEMM 上**打平或输**给 cuBLAS（NV 用超算级资源离线搜过 + 手调 SASS）。
- 更聪明的搜法：cost model 引导（TVM Ansor 一系）、启发式剪枝（Inductor 的 max-autotune 分级）——但"实测为王"仍是主流，因为 §01 的硬件行为（占用率悬崖、bank 冲突）对解析模型太不连续。

## 2. 开发者视角：算法侧的"写不写"决策树

```
这个算子是标准件吗（稠密GEMM/conv/标准attention）？
 ├─ 是 → 用库/torch.compile，到此为止（自己写=负收益）
 └─ 否 ↓
    是 memory-bound 融合链吗（elementwise/norm/激活的组合）？
     ├─ 是 → torch.compile 先试（Inductor 很擅长）；不满意→ Triton（半天级，收益大）
     └─ 是带矩阵乘的自定义结构（新attention/MoE路由/怪形状GEMM）？
         ├─ 是 → Triton + tl.dot + autotune（主战场；验证吃到 Tensor Core：§01-06 dump asm）
         └─ 要压到极限/造轮子给别人用 → CUTLASS/CuTe（或等 FlexAttention 类模板覆盖）
```

配套检查单（全是前面模块的落地）：tile 对齐 16/128（§02-05）→ `tl.dot` 有没有下沉成 wgmma（§03-01）→ num_stages/num_warps autotune（§01-02）→ ncu 看 SOL 定位（A1）。

## 3. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **Triton**：Hopper 支持成熟（TMA/wgmma/warp 专化可达）；作为 Inductor 后端成为事实标准；AMD/Intel 后端持续跟进。
- **CUTLASS 3.x/CuTe**：Hopper/Blackwell 的官方姿势（wgmma/tcgen05 + TMA + EVT）；CuTe layout 代数统一了 §01-04 的 swizzle/fragment 描述。
- **FlexAttention**（PyTorch）：attention 的"调度参数化"产品——用户给 score_mod/mask，编译器生成融合 kernel——**"模板化打穿"的新形态**（把 FA 式人肉走私变成受限自动化）。
- **ThunderKittens / Mojo / cuTile 等新 DSL**：都在赌"比 Triton 更贴硬件抽象（tile 为一等公民）"；未定局。
- **昇腾**：Ascend C（③层）+ CANN 图层，结构对齐但生态自成一体。

## 4. 和其它模块的挂钩

- 调度维度 ↔ 硬件知识对照表（§1.2）：本节把模块 01/02 的技术**全部归位**
- kernel 合同的"进出必落 HBM" ↔ 04-02 融合想消灭的边界
- autotune 的天敌=动态 shape ↔ §02-05 bucket、04-02 graph break
- 光谱"谁来搜" ↔ 00 光谱"谁排时刻表"的③层重演（人/模板/编译器 ≈ 静态程度递增）
- 下一节 04-04：把"调度"上升为"映射"，收束全库

## 5. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| 调度 | schedule | 同一计算在硬件上的展开方式；与"计算"分离是③层地基 | 04 |
| 调度空间 | schedule/config space | tile×warps×stages×splitK…几百到几千合法点 | 04 |
| autotune | | 实测搜调度空间+按形状缓存；怪形状赢、标准件输给库 | 04 |
| split-K / stream-K | | K 维切分换并行度的调度维度（小 M 救星） | 04 |
| CUTLASS / CuTe | | 调度=C++模板参数的积木库；Hopper+官方姿势 | 04 |
| tl.dot | Triton | 触发 Tensor Core 下沉的关键原语 | 04 |
| FlexAttention | | attention 调度参数化：受限自动化的跨层打穿 | 04 |
| max-autotune | Inductor | torch.compile 的深搜模式（模板+autotune） | 04 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 6. 我的困惑 / 待深挖

- （待填）Triton 编译器内部：从 tl.dot 到 wgmma 的 lowering 决策链？什么时候静默退化？
- （待填）cost model 引导搜索（Ansor 系）在 GPU 上为何始终没干过实测？不连续性的定量刻画？
- （待填）FlexAttention 的表达边界：哪些 attention 变体表达不了？
- （待填）Blackwell tcgen05 对 Triton/CUTLASS 编程模型的重塑程度？

---

*最后更新：2026-07-06（第一版）*
