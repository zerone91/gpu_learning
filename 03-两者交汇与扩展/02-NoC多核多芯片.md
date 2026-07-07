# 片上 NoC、多核、多芯片扩展

> **所属模块**：模块 03 · 两者交汇与扩展（按计划：够黑话量即可，不深挖）
> **状态**：🟨 学习中（第一版，知识截至 2026-01）
> **一句话主旨**：单核（SM/阵列）之上还有四层"连接"：**片上互连（NoC/crossbar）→ die 间（chiplet/NV-HBI）→ 节点内（NVLink/NVSwitch，scale-up）→ 节点间（InfiniBand/以太，scale-out）**。每往外一层，带宽掉一个量级、延迟涨一个量级——**§04 的"复用放大器"金字塔在芯片外的延伸**。听方案时把词挂到"它在第几层"，就不会晕。

---

## 0. 这一节要回答的问题

- [x] 一颗芯片内部，几十上百个核（SM/NPU core）怎么连？
- [x] chiplet / die-to-die 是什么？Blackwell 双 die 怎么"合成一颗 GPU"？
- [x] scale-up vs scale-out 的准确分界在哪？NVLink/NVSwitch/IB 各在哪层？
- [x] 每层的带宽/延迟量级差多少？对软件（集合通信/并行策略）意味着什么？

---

## 1. 四层连接的地图（本节主体）

```
L0 片上：SM ↔ L2 ↔ SM          （crossbar/NoC，~10 TB/s 级，~百 ns）
L1 die间：die ↔ die             （NV-HBI 10TB/s / UCIe，芯片内透明）
L2 节点内：GPU ↔ GPU (≤72颗)    （NVLink5 1.8TB/s/卡 + NVSwitch = scale-up）
L3 节点间：机架 ↔ 机架          （InfiniBand/RoCE ~百GB/s/卡 = scale-out）
```

### L0 片上互连：SM 们怎么共享 L2

- GPU 内部：上百个 SM 与分区化的 L2 之间是**大型 crossbar/环网**——§01-04 说"L2 是全芯片一致性汇合点"，物理上就是所有 SM 的访存都汇到这张网上。GPC/TPC 是 SM 的分组层级（黑话：一个 GPC 含若干 TPC，一个 TPC 含 2 个 SM）。
- Hopper 的 **DSM/cluster**（§01-04）：SM↔SM 直连通道，绕过 L2 的一条"近道"。
- NPU/数据流芯片（Tenstorrent/Cerebras/Graphcore）：核间是**显式 2D mesh NoC**——**核多且同构时，mesh 是标配**；软件（或编译器）要意识到"邻居近、对角远"。GPU 把这层藏在 L2 后面，NPU 把它暴露给编译器——又一次"动态隐藏 vs 静态暴露"（§02 主线）。

### L1 die-to-die（chiplet）：光刻极限逼出来的层

- **动因**：单 die 逼近光刻极限（~800mm²），只能拆多 die 封装（chiplet）。
- **Blackwell**：两个 die 用 **NV-HBI（10 TB/s）**缝合，**软件视角完全是一颗 GPU**（一个 CUDA device、统一 L2 语义）——"缝合处带宽够高，就能假装不存在"。
- **AMD MI300**：更激进的 chiplet——8 个 XCD（计算 die）+ IOD，Infinity Fabric 互连；**苹果 UltraFusion、Intel EMIB/Foveros** 同类。黑话：**UCIe** 是 die 间互连的行业标准协议。
- 关键判断：**die 间带宽是否"够装成一颗芯片"**决定软件模型——够（NV-HBI）则透明；不够则暴露 NUMA（早期双芯卡如 K80 是两个 device）。

### L2 scale-up：节点内把多卡"缝成一台大 GPU"

- **NVLink**：GPU 间点对点高速链路（NVLink5 每卡 1.8 TB/s 聚合——**比 PCIe 高一个数量级**，这就是它存在的理由）。
- **NVSwitch**：把 NVLink 组成全交换拓扑——**NVL72**：一机架 72 颗 Blackwell 全互连，任意两卡全带宽，构成一个"超级 GPU 域"（NVLink domain）。
- **scale-up 的定义**：域内卡间带宽高到**张量并行（TP）这种细粒度、通信密集的并行也放得下**。
- 对标：AMD **Infinity Fabric**、各家 NPU 的专有互连（TPU 的 **ICI**，昇腾的 HCCS）。**UALink** 是 2024 起对标 NVLink 的开放标准联盟。

### L3 scale-out：跨节点

- **InfiniBand / RoCE 以太网**：~百 GB/s 级每卡（如 400-800Gb NIC），延迟 μs 级。
- **scale-out 的定义**：带宽/延迟只放得下**粗粒度并行**——数据并行（DP）、流水并行（PP）、专家并行（EP）的 all-to-all。
- 黑话：**GPUDirect RDMA**（NIC 直读显存）、**SHARP**（交换机内做归约）、**rail-optimized** 拓扑。
- TPU 独特路线：**OCS 光交换**——用光路开关重构 pod 拓扑（数千芯片），静态可重构、无分组交换——又是"静态 vs 动态"哲学在网络层的重演。

## 2. 一张速查表：并行策略贴哪层

| 并行策略 | 通信模式 | 通信量/频率 | 放哪层 |
| --- | --- | --- | --- |
| TP（张量并行） | 每层 all-reduce/all-gather | 极大、每算子 | **L2 scale-up 域内**（NVLink） |
| EP（专家并行） | all-to-all（token 路由） | 大、每层 | L2 为主，跨节点则痛（A2 MoE 的系统面） |
| PP（流水并行） | 点对点激活传递 | 中、每 micro-batch | L2/L3 皆可 |
| DP/FSDP | 梯度/参数 all-reduce | 大但每步一次 | **L3 scale-out** |

> 🔑 **本质**：**并行策略的设计 = 把通信模式匹配到互连层级**——通信最密的（TP）贴最快的层（NVLink 域），最疏的（DP）扔最远的层（IB）。听方案里"TP=8 域内、DP 跨机、EP 受 all-to-all 限制"这类话，翻译过来全是这张表。（深入是分布式训练/推理的领域，超出本库硬件主线，点到为止。）

## 3. 开发者视角（速览）

- **观测**：`nvidia-smi topo -m`（看卡间连接矩阵）、NCCL 日志（选了什么算法/环路）、`nsys` 的 NCCL/通信轨道。
- **拨盘**：并行策略划分（TP/PP/DP/EP 的度）、NCCL 环境变量、进程亲和；框架层 DeepSpeed/Megatron/vLLM 的并行配置。
- **红旗**：TP 跨出 NVLink 域（性能崩）、EP all-to-all 跨节点占大头、PCIe 路径混入（没走 GPUDirect）。

## 4. 最新架构落点（时效锚点 · 知识截至 2026-01）

- **NVIDIA**：GB200 NVL72（72 卡 NVLink5 域）是当前 scale-up 极致；路线图 NVL 域继续扩大（Rubin 代，未出货）。
- **AMD**：MI300/350 走 chiplet + Infinity Fabric；UALink 联盟推进中。
- **TPU**：ICI + OCS 光交换 pod（v5p 8960 芯片级），静态可重构。
- **昇腾**：910C 双 die、HCCS 互连、CloudMatrix 384 超节点（2025，以多芯拼大域对标 NVL72）。
- **趋势一句话**：**scale-up 域越做越大**（72→几百卡），因为 LLM 推理的 TP/EP 通信压不进 scale-out——"多大的 NVLink 域"成了各家系统竞争的主战场。

## 5. 和其它模块的挂钩

- L2 是片上汇合点 → **§01-04/05**；DSM/cluster → **§01-04**
- mesh NoC 静态暴露 vs GPU 动态隐藏 → **§02 主线**的网络层重演；OCS 同理
- EP all-to-all ↔ A2 MoE 的系统面
- 并行策略沿栈下传 → **模块 04 §05**；各家互连参数 → **模块 05**

## 6. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| NoC | Network-on-Chip | 片上核间网（NPU 多为显式 2D mesh，GPU 藏在 L2 后） | 03 |
| GPC/TPC | | SM 的分组层级（GPC⊃TPC⊃2×SM） | 03 |
| chiplet / UCIe | die-to-die | 多 die 封装 / 其行业互连标准 | 03 |
| NV-HBI | | Blackwell 双 die 缝合的 10TB/s 互连，软件透明 | 03 |
| NVLink / NVSwitch | | 卡间高速链路 / 其全交换组网（NVL72 域） | 03 |
| scale-up vs scale-out | | 域内高带宽缝大卡 vs 跨节点粗粒度扩展 | 03 |
| ICI / OCS | TPU | TPU 芯间互连 / 光交换可重构 pod 拓扑 | 03/05 |
| Infinity Fabric / HCCS / UALink | | AMD / 昇腾 / 开放联盟的对标互连 | 03/05 |
| GPUDirect RDMA / SHARP | | NIC 直读显存 / 交换机内归约 | 03 |
| TP/PP/DP/EP | 并行四件套 | 通信密度递减，依次贴 scale-up→scale-out | 03/04 |

> ✅ 待同步登记到 [术语表](../05-收敛-术语表与真实芯片/术语表.md)

## 7. 我的困惑 / 待深挖

- （待填）NVSwitch 内部拓扑与 SHARP 归约的实现？
- （待填）NVL72 域内 TP 的实测扩展曲线？到多少卡通信开始吃掉收益？
- （待填）OCS 光交换的重构延迟与故障切换？
- （待填）UALink/UEC 生态的实际落地进度？

---

*最后更新：2026-07-06（第一版）*
