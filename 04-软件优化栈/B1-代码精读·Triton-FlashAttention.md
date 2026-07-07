# B1 · 代码精读：Triton 版 FlashAttention-2 前向

> **所属系列**：模块 04 · 代码精读 B 系列（B1 Triton FA2 → B2 Hopper FA-3/CUTLASS → B3 DeepSeek 栈）
> **状态**：🟨 学习中（第一版，知识截至 2026-01。代码为 Triton 官方 fused-attention 教程的简化重写，保留全部关键结构，省略 backward 与部分边界处理）
> **一句话主旨**：一个 ~60 行的 Triton kernel，把全库概念全部串起来——**每段代码都回答三问：它是映射四维（04-04）里的哪一维决策？编译后在硬件上变成什么（SASS/部件）？改错了会付什么代价？** 读完你应该能对任何 Triton kernel 做同样的"三问精读"。

---

## 0. 精读方法（B 系列的固定范式）

每段代码配三行注解：
- **【映射】** 这是 tiling/ordering/placement/binding 哪一维的决策（04-04）？
- **【硬件】** 编译后落到什么（哪条 SASS 指令 / 哪个部件 / 哪级存储）？
- **【反事实】** 如果改错/不这么写，会发生什么（哪个 profiler 指标会告状）？

## 1. 先摆全景：这个 kernel 的映射方案

计算：`O = softmax(QKᵀ·scale)·V`，形状 `[Z=batch, H=heads, N_CTX, HEAD_DIM]`，causal。
映射方案（先看清骨架再读代码）：

```
④binding   : 每个 program(=CTA/block) 负责一个 (batch,head) 里的一条 Q 行块 [BLOCK_M × HEAD_DIM]
①tiling    : Q 切 BLOCK_M 行；K/V 沿序列切 BLOCK_N 列，循环流过
②ordering  : 外层沿 K/V 块顺序扫（causal 时只扫下三角）；online softmax 边扫边归一
③placement : Q tile 常驻(整个循环复用)、O/m/l 累加器驻寄存器(fp32)=OS 数据流；
             K/V tile 流经 shared(编译器自动 + num_stages 流水)；S=QKᵀ 永不落 HBM ← FA 的灵魂
```

> 对照 A2 §1.1：S 不落 HBM → AI 从 O(1) 抬到 O(N) → Tensor-bound。**下面每段代码都是在实现这四行映射。**

## 2. 逐段精读

### 2.1 Autotune 装饰器：调度空间的入口

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=4),
        # ... 实际库里有十几个候选
    ],
    key=['N_CTX', 'HEAD_DIM'],   # 形状变了才重新搜
)
@triton.jit
def _attn_fwd(...):
```

- **【映射】** 这就是 04-03 的**调度空间**本体：①tiling（BLOCK_M/N）× 占用相关（num_warps）× 流水深度（num_stages）的候选集。`key` 定义"形状桶"——同桶复用搜索结果。
- **【硬件】** `num_warps=4` → 这个 CTA 有 128 线程（占用率的分子，§01-02）；`num_stages=3` → 编译器生成 **3 级 cp.async/TMA 软件流水**（§01-02 §2.6 的 N 级多缓冲，N=3）。
- **【反事实】** 只留一个 config → 换个形状/换张卡就掉到次优点；`key` 少写 `HEAD_DIM` → d=64 和 d=128 共用一个"最优"config，必有一个吃亏。**动态 shape 会让形状桶爆炸（04-03 的 autotune 天敌）。**

### 2.2 program_id：④binding 的全部内容

```python
    start_m = tl.program_id(0)      # 我负责第几条 Q 行块
    off_hz  = tl.program_id(1)      # 我负责哪个 (batch*head)
```

- **【映射】** ④binding：grid 是二维 `(cdiv(N_CTX, BLOCK_M), Z*H)`——**FA2 对 FA1 的关键改进就在这一行**：FA1 只按 (batch,head) 分（grid=Z*H），小 batch 长序列时 CTA 数 < SM 数、机器半空；FA2 把**序列维也切进 grid**，CTA 数 ×(N/BLOCK_M)，SM 全喂满（A1(04) 演化表第二行的代码实体）。
- **【硬件】** 每个 program → 一个 CTA → 被 GPU 的 block 分发器扔给某个 SM（§01-02）。
- **【反事实】** batch=1、N=8k、只按 head 分 → 32 个 CTA 对 132 个 SM → **75% 的机器在看戏**（`ncu` 里 waves=0.24、`Achieved Occupancy` 惨白——A1 的尾部/wave 量化）。

### 2.3 block_ptr：把"访存模式"声明给编译器

```python
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_hz * stride_qh,
        shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )   # K/V 各建一个，offsets 初始指向第 0 个 K/V 块
```

- **【映射】** ①tiling 的边界声明 + ③placement 的入口："我要按 [BLOCK_M×HEAD_DIM] 的块访问这个张量"。
- **【硬件】** 这是 Triton 通往 **TMA/cp.async 的门票**：块状访问声明让编译器能生成整块异步拷贝（Hopper 上可下沉为 TMA 描述符，§01-04 §2.4-4）、自动处理边界（OOB 填充）和 **swizzle**（§01-04 §1.3+，编译器代管——你从没写过 swizzle，但它就在生成的代码里）。`order=(1,0)` 声明内维连续 → coalescing 友好（§01-04）。
- **【反事实】** 用裸指针 `Q + offs_m[:,None]*stride + offs_k[None,:]` 也能工作（老写法），但编译器难以证明块状性 → 可能退化为逐元素 load、失去 TMA 路径；stride 传错 → 访存不合并，`ncu` 的 `Sectors/Request` 从 4 飙到 32（§01-04 那张表）。

### 2.4 累加器初始化：③placement 的灵魂三行

```python
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # running 分母
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)       # O 的累加器
    q = tl.load(Q_block_ptr)                              # Q tile 装进来，整个循环不再动
```

- **【映射】** ③placement 的核心决策全在这四行：**acc/m/l 驻留寄存器 = output-stationary**（02-03——psum 原地累加 K 次不动）；**q 装载一次、循环内复用 = Q 的 weight-stationary**（一份数据被所有 K/V 块反复用）。**S 呢？没有 S 的变量——它以 `qk` 的形式每轮生灭于寄存器，这就是"S 永不落 HBM"的代码形态。**
- **【硬件】** acc 是 `BLOCK_M×HEAD_DIM` 个 **fp32 寄存器**——128×128 时每 CTA 1.6 万个 fp32，寄存器压力的大头（§01-02 例 2 的"刻意低占用"就这么来的；Blackwell 的 TMEM 正是给这块卸压，§03-01）。
- **【反事实】** acc 用 bf16 → 长序列累加精度烂掉（02-03 OS 的"精度友好"反面教材）；m/l 不用 fp32 → exp 域下溢。**BLOCK_M 开太大 → 寄存器爆 → spill 到 local memory**，`-Xptxas -v` 报 spill、性能崩（§01-02 §2.2）。

### 2.5 主循环：②ordering + online softmax（FA 的数学心脏）

```python
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX   # causal: 只扫下三角
    for start_n in range(lo, hi, BLOCK_N):
        # ---- 第一个 GEMM ----
        k  = tl.load(K_block_ptr)                    # [HEAD_DIM, BLOCK_N]
        qk = tl.dot(q, k)                            # [BLOCK_M, BLOCK_N] = 一块 S
```

- **【映射】** ②ordering：沿 K/V 块流式扫描；**causal 的处理是"少扫"而不是"扫了再扔"**——`hi` 直接截断到对角线，上三角的块**根本不进循环**（02-05：把浪费挡在 tile 粒度外）。
- **【硬件】** `tl.dot` → Triton 下沉为 **mma（Ampere）/ wgmma（Hopper）**，SASS 里是 `HMMA`（§01-06 的验证点：dump asm 搜它）；`tl.load(K_block_ptr)` 在 num_stages=3 下被编译器改写成**超前 2 块的异步预取**——你写的是同步语义，编译器排成了 §01-02 §2.6 的流水。
- **【反事实】** BLOCK 不是 16 的倍数 → `tl.dot` 无法下沉到 MMA，退化为 FFMA 循环 → **Tensor pipe util = 0，慢一个数量级**（A1 的"白买"判据）；causal 用"扫全部+mask"代替截断 → 白算一半的块（浪费 2×）。

```python
        # ---- causal 对角块的掩码：谓词化，不是分支 ----
        if IS_CAUSAL and start_n + BLOCK_N > start_m * BLOCK_M:
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(mask, qk, float('-inf'))
```

- **【映射】** 只有**跨对角线的边界块**才做元素级 mask（②ordering 的收尾细节）。
- **【硬件】** `tl.where` → **谓词化指令**（`@P` 前缀，§01-03 §1.3）——32 条 lane 全执行、按谓词写回，**零分支发散**。注意外层 `if` 是 `constexpr`/块级条件（编译期/块粒度），不产生 warp 内分歧。
- **【反事实】** 若写成逐元素 `if` 的真分支 → warp 内下三角/上三角 lane 分家 → `Warp Execution Efficiency` 跳水（§01-03 的体温计）。**这一段是"发散控制"教科书：块级截断 + 元素级谓词，两层各司其职。**

```python
        # ---- online softmax：边扫边归一 ----
        m_ij  = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)      # 新的 running max
        qk    = qk * sm_scale - m_ij[:, None]
        p     = tl.math.exp2(qk * 1.44269504)                  # exp(x) = exp2(x·log2e)
        alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)        # 旧块的修正因子
        acc   = acc * alpha[:, None]                           # ← correction：重缩放旧累加
        l_i   = l_i * alpha + tl.sum(p, 1)
        m_i   = m_ij
```

- **【映射】** 这就是**让"归约分界"可融合的数学改写**（04-02：归约本是融合的天然分界；online 化让它能分块）——FA 之所以是"反向映射"（04-04：给定 S 不落 HBM 的目标，改写数学来配合）的全部秘密就这 7 行。每块只维护 `m_i`（running max）和 `l_i`（running 分母），来了新块就用 `alpha` 把**旧的 acc 整体重缩放**。
- **【硬件】** `exp2` → **MUFU pipe**（A2 §1.3：吞吐只有 FMA 零头的那条管子）。用 `exp2` 而非 `exp` 是因为硬件 MUFU 原生是 exp2，`exp` 要多一次乘法。**这里就是 FA-3 ping-pong 要藏、FA-4 用多项式绕开的那个瓶颈的代码位置**（A1(04)）。`acc*alpha` 是每块一次的 `BLOCK_M×HEAD_DIM` 次 FMA——**FA-4 的 lazy rescale 跳的就是这一行**（max 没怎么变时 alpha≈1，乘了白乘）。
- **【反事实】** 不减 max 直接 exp → fp 上溢，数值 NaN（这不是优化是正确性）；把 softmax 拆成独立 kernel → S 落 HBM，AI 塌回 memory-bound（A2 naive 行）。

```python
        # ---- 第二个 GEMM ----
        v    = tl.load(V_block_ptr)                    # [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p.to(v.dtype), v)                # O += P·V
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
```

- **【映射】** 第二个 GEMM 就地消费 p——**p（即 P=softmax(S) 的块）同样只活在寄存器**，从生到死没碰过 HBM。
- **【硬件】** `p.to(v.dtype)`：p 是 fp32，**降回 bf16 再喂 MMA**（Tensor Core 输入低精度、累加高精度——§03-01 的精度阶梯与 OS 累加）。`tl.advance` 让编译器静态知道步进模式 → 流水预取地址可提前算。
- **【反事实】** p 保持 fp32 进 dot → 走不了 bf16 Tensor pipe（吞吐减半或更多）；忘了 advance 写手动指针算术 → 编译器难分析,流水质量下降。

### 2.6 收尾：归一化 + 写回

```python
    acc = acc / l_i[:, None]                 # 最终归一化（分母到齐了）
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty))
```

- **【映射】** ②ordering 的终点：扫完所有 K/V 块，running 分母 `l_i` 才是真分母，一次性归一。写回是本 kernel **唯一一次 O 尺寸的 HBM 写**——对照 naive 的"S 写 + S 读 + P 写 + P 读 + O 写"五趟。
- **【硬件】** `tl.store` 块状写回 → 合并写事务（§01-04）；`.to(bf16)` 在写回时降精度。
- **【反事实】** 每轮循环内除 l_i → 数学等价但每块多一次除法（除法也走 MUFU！），FA2 论文明确把它挪出循环（A1(04) 演化表"减非 matmul FLOPs"的代码实体）。

## 3. 一张总账：60 行代码 ↔ 全库概念

| 代码位置 | 概念 | 出处 |
| --- | --- | --- |
| autotune configs | 调度空间、形状桶 | 04-03 |
| grid 二维化 | FA2 的序列维并行、wave 量化 | A1(04)/A1 |
| block_ptr | TMA 门票、swizzle 代管、coalescing | §01-04 |
| acc/m/l 驻寄存器 | output-stationary、寄存器压力、TMEM 伏笔 | 02-03/§01-02/§03-01 |
| q 常驻 | Q 的 weight-stationary | 02-03 |
| 没有 S 变量 | S 不落 HBM = FA 灵魂 = 反向映射 | A2/04-04 |
| causal 截断 vs tl.where | 块级少扫 + 元素级谓词化，零发散 | §01-03/02-05 |
| tl.dot | →MMA/wgmma/HMMA；16 倍数硬约束 | §03-01/§01-06 |
| num_stages | cp.async/TMA 软件流水 | §01-02 §2.6 |
| exp2/MUFU | FA3 要藏、FA4 要绕的瓶颈 | A2 §1.3/A1(04) |
| acc*alpha | FA4 lazy rescale 跳过的对象 | A1(04) |
| 除法挪出循环 | 减非 matmul FLOPs | A1(04) FA2 行 |

## 4. 验收：跑起来该看什么（profiler 三证）

1. `triton.compile(...).asm['ptx'/'sass']` 搜 **HMMA/wgmma**（Tensor Core 吃到没，§01-06 第一招）与 `cp.async/LDGSTS`（流水生成没）。
2. `ncu`：**SOL** 应为 Compute 高 Memory 不满（Tensor-bound，A1）；`Achieved Occupancy` **不高是正常的**（寄存器重、靠流水藏延迟——§01-02 §2.3 的活例）；stall 主项应是 `Wait`/MUFU 相关而非 `Long Scoreboard`（若是后者 → 流水没盖住访存，调 num_stages）。
3. 对照 A2 §1.6 的表：本 kernel 相当于"FA-2 行"——Tensor util 中等、被 softmax(MUFU) 序列化拖住。**下一篇 B2 讲 FA-3 怎么在 CUTLASS 层把这个瓶颈藏掉。**

## 5. 我的困惑 / 待深挖

- （待填）Triton 在 Hopper 上对这个 kernel 实际生成 TMA 还是 cp.async？何时选择 warp 专化路径？
- （待填）BLOCK_M=128 时的精确寄存器账（acc 16K + q/p/临时）vs 每 SM 64K 上限 → 理论占用率手算验证
- （待填）backward 的映射方案（重计算 S vs 存 logsumexp）——另一场 placement 权衡

---

*最后更新：2026-07-06（第一版）*
