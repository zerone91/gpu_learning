# PTX vs SASS：ISA 到底描述什么

> **所属模块**：模块 01 · SIMT 核的物理实现（收官节）
> **状态**：🟨 学习中（第一版：本质/费曼 + 开发者视角 + 最新架构，知识截至 2026-01）
> **一句话主旨**：**PTX 是"承诺"，SASS 是"现实"**——PTX 是一个稳定的**虚拟 ISA 合同**（面向所有过去与未来的 GPU），SASS 是某一代硅片的**真实机器码**（每代可以推倒重来）。这层分离是 NVIDIA 生态护城河的技术根基：**代码一次编译、跨代运行（前向兼容），而硬件团队每代可自由改微架构**。你调性能时看到的一切真相（寄存器、spill、控制位、有没有用上 Tensor Core）都只在 SASS 层——PTX 只是中转站。

---

## 0. 这一节要回答的问题

- [x] PTX 和 SASS 分别是什么？边界画在哪？
- [x] 从 CUDA C 到硅片的完整编译链条：谁负责哪段？JIT 和 fatbin 怎么回事？
- [x] 为什么 NVIDIA 敢每代推倒重来微架构？（虚拟 ISA 的生态学）
- [x] 前面五节散落的 SASS 知识（控制位、LDGSTS、HMMA、BSSY）怎么挂进这个框架？
- [x] 看 SASS 能回答哪几类问题？最小实用工具箱是什么？
- [x] Triton/编译器们各自"落"在这条链的哪一站？

---

## 1. 核心概念

### 1.1 费曼起点：合同与工厂

- **PTX（Parallel Thread eXecution）**：一份**虚拟 ISA 合同**。它描述一台理想化的 SIMT 机器（无限虚拟寄存器、通用指令、明确的内存空间与同步语义）。**它承诺的是语义，不承诺实现**。文本可读、文档公开、**跨代稳定且只增不删**。
- **SASS（Streaming ASSembler，真实机器码）**：某一代硅片（SM70/80/90/100…）的**原生指令**。寄存器是物理的（255 个上限，§02）、指令带控制位（§02 §1.1+）、编码不公开文档、**每代可不兼容地重来**。

> 🔑 **这层分离解决的核心问题是"时间"**：你 2020 年编出的程序要能跑在 2026 年的芯片上。把"稳定的合同"（PTX）与"善变的工厂"（SASS）解耦——软件对合同编程，工厂只要能履行合同，内部随便改。**CPU 世界里 x86 用"硬件解码器"在运行时做这个翻译（付硅片面积）；NVIDIA 用"驱动里的 JIT 编译器"做（付启动时间）——同一个问题，两种付法。**

### 1.2 完整编译链条：两段式 + 两条到达路径

```
CUDA C / Triton / ...
      │  nvcc 前端（clang）/ Triton 编译器
      ▼
    PTX  ←———— 稳定边界（合同）
      │  ptxas —— 真正的"后端编译器"：寄存器分配、指令调度、
      │           控制位生成（§02 的 stall/yield/barrier 掩码全在这步）
      ▼
    SASS（cubin，绑定具体 SM 版本）
```

**到达用户 GPU 的两条路**（fat binary 里两种货都可以带）：
1. **AOT（离线）**：编译时用 `-gencode arch=…,code=sm_90` 直接放入 SM90 的 cubin——启动零开销，但只覆盖列出的代际。
2. **JIT（运行时）**：fatbin 里带 PTX；遇到没预编译的（更新的）GPU，**驱动里的 ptxas 现场把 PTX 编成本机 SASS**（结果缓存在 `~/.nv/ComputeCache`）。**前向兼容全靠这条路**——这就是"老 CUDA 程序能跑在新卡上"的机制。

> ⚠️ 实用坑：只带 PTX 不带对应 cubin → 新卡第一次启动 JIT 可能秒级卡顿；只带 cubin 不带 PTX → 未来的卡直接跑不了。库的发布策略（帯哪些 arch + PTX）就是在这两头权衡。

### 1.3 前五节的 SASS 散点，全部挂进来

这条链解释了之前所有"SASS 层"知识的归属——**它们都是 ptxas 的产物**：

| 前面见过的 | 是什么 | 出自哪节 |
| --- | --- | --- |
| 控制位（stall/yield/屏障掩码） | ptxas 做完指令调度后**静态写进每条指令**的依赖信息——"软件辅助记分牌"的软件半边 | §02 §1.1+ |
| 寄存器数 / spill | ptxas 寄存器分配的结果（PTX 里是无限虚拟寄存器！） | §02 §2.2 |
| `LDGSTS` | `cp.async` 的 SASS 形态 | §04 |
| `HMMA/QMMA` | Tensor Core 指令的 SASS 形态（PTX 层是 `mma`/`wgmma`） | A1/A2 |
| `BSSY/BSYNC` | ITS 的收敛屏障 | §03 |
| `LDG.E.128` | 向量化访存 | §04 |
| `BAR.SYNC / MEMBAR` | 同步/栅栏家族 | §05 |

> 🔑 **看清一件事**：性能的**真相全在 ptxas 之后**。PTX 里没有寄存器压力（虚拟寄存器无限）、没有控制位、没有真实指令选择——**所以"读 PTX 调性能"基本是错觉，要读就读 SASS**。PTX 的价值在语义边界（写内联汇编、理解内存模型），不在性能诊断。

### 1.4 谁站在链条的哪一站

| 工具/系统 | 进入点 | 出口 | 备注 |
| --- | --- | --- | --- |
| nvcc | CUDA C | PTX + cubin | 标准路径 |
| **Triton** | Python DSL | **PTX**（经 LLVM NVPTX）→ 交给 ptxas | 所以 Triton 性能上限也**受 ptxas 支配** |
| CUTLASS | C++ 模板 | 同 nvcc | 靠内联 PTX（`wgmma` 等）精准控制 |
| cuBLAS/cuDNN | 预编译 | 各代 cubin | 库内常含**手写/手调 SASS**（超越 ptxas 的最后 %） |
| 驱动 JIT | PTX | 本机 SASS | 前向兼容的执行者 |

---

## 2. 关键机制 / 为什么这样设计

- **ptxas 是无名英雄也是最终裁判**：寄存器分配决定 occupancy（§02 的三方谈判实际在这里裁决）、指令调度决定 ILP 与控制位质量、指令选择决定用不用上 LDGSTS/HMMA。**同一份 PTX，不同版本 ptxas 编出的性能可差两位数百分比**——升级 CUDA toolkit 有时白得性能，根源在此。
- **为什么 SASS 不公开文档**：不公开 = 不构成合同 = 每代可自由重构微架构（Volta 改 ITS、Hopper 加 warpgroup、Blackwell 换 tcgen05，SASS 都大改）。生态付出的代价：第三方无法官方地做 SASS 级工具（社区逆向出了 maxas/CuAssembler 等）。
- **和 AMD 的镜像对照**（收 §05 的线）：AMD 反着走——**GCN/RDNA/CDNA 的 ISA 公开文档**，编译到**具体架构的机器码**（无稳定虚拟层，HIP 靠源码级可移植 + 多目标编译）。好处是透明可调（`s_waitcnt` 明晃晃，§05 §2.3-4）、坏处是二进制前向兼容弱。**NV 卖"时间兼容"，AMD 卖"透明"。**

## 3. 开发者视角：最小实用工具箱

| 想知道 | 命令/工具 | 看什么 |
| --- | --- | --- |
| 生成了什么 SASS | `cuobjdump -sass a.out` / `nvdisasm`（cubin） | 全部真实指令 |
| 寄存器/spill/shared | `-Xptxas -v`（§02 起反复用） | "Used N registers, spill…" |
| Tensor Core 用上没 | SASS 里搜 `HMMA/QMMA/BMMA` | 没有 = 白写（A1 的判据落地） |
| 异步拷贝用上没 | 搜 `LDGSTS` / TMA 相关 | cp.async 是否生效 |
| 向量化访存 | 搜 `LDG.E.128` | 128-bit 是否生效 |
| 热点行对应哪些指令 | **Nsight Compute 的 Source/SASS 对照视图**（`-lineinfo` 编译） | 最实用的入口：stall 直接标在 SASS 行上 |
| PTX 中间态 | `nvcc -ptx` / `cuobjdump -ptx` | 语义检查、写内联 PTX 时 |

**心法**：日常调优 **95% 用 Nsight 的 SASS 对照视图 + `-Xptxas -v`** 就够；手读整段 SASS 只在"怀疑编译器没做对"（该向量化没向量化、该 HMMA 没 HMMA、spill 来路不明）时出手。

### 站位阶梯（本节视角重排）

| 站位 | 与 ISA 的关系 |
| --- | --- |
| PyTorch | 完全无感（库和 torch.compile 兜底） |
| Triton | 产出 PTX；`triton.compile(...).asm["ptx"/"cubin"]` 可直接 dump 两层产物——**排查 Triton 性能问题时看它有没有生成 wgmma/LDGSTS 是第一招** |
| CUDA C | `-Xptxas -v` + Nsight SASS 视图常驻 |
| 库作者/极限党 | 内联 PTX（`asm volatile`）、乃至社区 SASS 汇编器 |

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **PTX 只增不删的演化**：`mma`（Volta）→ `wgmma`（Hopper, warpgroup 异步）→ **`tcgen05.mma`（Blackwell, TMEM 语义）**——每代 Tensor Core 大改，PTX 加新指令族而不动旧的；`mbarrier`/TMA（`cp.async.bulk.tensor`）同理。**读 PTX ISA 文档的新增章节 = 追新架构特性最可靠的一手渠道。**
- **SASS 每代重编码**：SM90→SM100 指令编码/调度规则再变；cubin 不跨代。
- **AMD**：ISA 文档持续公开（CDNA3/4）；工具 `rocobjdump`/`llvm-objdump`。
- **社区 SASS 工具**：maxas（Maxwell 时代传奇）、CuAssembler 等——存在本身就是"最后几个百分点在 SASS 层"的证据。

## 5. 和其它模块的挂钩

- 控制位/依赖屏障 = ptxas 的调度产物 → **§02 §1.1+**（软件辅助记分牌的"软件"就是 ptxas）
- 寄存器分配裁决 occupancy 谈判 → **§02 §1.3/2.2**
- `BSSY/BSYNC`、`*_sync` → **§03**；`LDGSTS/LDG.128` → **§04**；`BAR/MEMBAR/mbarrier` → **§05**
- 检查 HMMA/wgmma 生成 = 模块 03（Tensor Core）的编程接口验证手段
- Triton→PTX→ptxas 链条 → **模块 04 kernel 层**（编译器栈的最底段）
- NV 虚拟 ISA vs AMD 公开 ISA vs NPU（无用户可见 ISA，编译器直出微码）→ **模块 04 §01 栈全景、模块 05**

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| PTX | 虚拟 ISA | 稳定合同：理想 SIMT 机的语义，跨代只增不删 | 01 |
| SASS | 真实机器码 | 某代硅片的原生指令，每代可重来、不公开文档 | 01 |
| ptxas | PTX→SASS 编译器 | 寄存器分配+指令调度+控制位生成的最终裁判 | 01 |
| fatbin | fat binary | 同时打包多代 cubin + PTX 的容器 | 01 |
| JIT 编译 | driver JIT | 驱动运行时把 PTX 编成本机 SASS；前向兼容的机制 | 01 |
| cubin | | 绑定具体 SM 版本的 SASS 二进制 | 01 |
| `-lineinfo` | | 让 Nsight 能做源码↔SASS 对照的编译开关 | 01 |
| 内联 PTX | `asm volatile` | 在 CUDA C 里直接写 PTX；CUTLASS 控制 wgmma 的手段 | 01/03 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）ptxas 的指令调度器怎么权衡"控制位 stall 最小化"与"寄存器压力"？有无可控旋钮（`-O` 级别对 SASS 的实际影响）？
- （待填）JIT 缓存的失效条件（驱动升级/toolkit 变更）？生产环境预热策略？
- （待填）Triton 何时绕过 NVPTX 直接生成更优 PTX 模式？其 `wgmma` 生成质量 vs CUTLASS 差距多少？
- （待填）社区 SASS 汇编器在 Hopper/Blackwell 上还可行吗（编码逆向的现状）？

---

*最后更新：2026-07-06（第一版）*
