# B3 · 代码精读：DeepSeek 栈（DeepGEMM FP8 两级累加 + DeepEP 通信骨架）

> **所属系列**：模块 04 · 代码精读 B 系列收官（B1 Triton FA2 → B2 FA-3 Hopper → **B3**）
> **状态**：🟨 学习中（第一版，知识截至 2026-01。注释性伪代码，忠实于 DeepGEMM/DeepEP 开源实现的结构与真实机制名；细节以 deepseek-ai/DeepGEMM、deepseek-ai/DeepEP 仓库为准）
> **一句话主旨**：两个 kernel、两种"新公民"——**DeepGEMM** 示范"当精度低到 FP8，连**累加本身**都成了要显式管理的资源"（细粒度缩放 + 两级累加 + persistent 调度）；**DeepEP** 示范"当瓶颈是网络，**通信也写成 kernel**"（IBGDA 把 RDMA 发起权交给 GPU、hook 把收包藏进计算）。B2 的 TMA+mbarrier 骨架在这两处原样复用——**同一套硬件积木，搭出算子和通信两种房子。**

---

## 1. DeepGEMM：FP8 grouped GEMM 的三个招牌

### 1.1 先看问题：FP8 的两笔账

- **精度账**：e4m3 只有 3 位尾数,动态范围窄 → 必须**细粒度缩放**:激活 A 按 **[1×128]**(每 token 每 128 通道)一个 fp32 scale,权重 B 按 **[128×128]** 块一个 scale——比 per-tensor 缩放细两个数量级(Transformer Engine 思路的加强版,§03-01 精度阶梯的落地)。
- **累加账**（关键硬件事实）:Hopper 的 FP8 wgmma **累加器精度有限**(尾数位不足的中间格式),K 很深时部分和会"吃"掉小数——**光靠 Tensor Core 自己累加,长 K 必然掉精度**。

### 1.2 主循环:两级累加(promotion)——本篇最重要的代码

```cpp
// 寄存器: acc_final[fp32] = 最终累加器(OS,常驻)
//         acc_wgmma      = wgmma 的临时累加器(每个 k 段清零重来)
for (int k_blk = 0; k_blk < K / BLOCK_K; ++k_blk) {
    full[s].wait(phase);                        // B2 同款: TMA + mbarrier 流水
    // ---- 第一级: Tensor Core 在低精度累加一小段 ----
    warpgroup_arrive();
    gemm(tiled_mma, sA[s], sB[s], acc_wgmma);   // FP8 wgmma, 只累加 BLOCK_K=128 深
    warpgroup_commit_batch(); warpgroup_wait<0>();
    // ---- 第二级: CUDA core 把这段"晋升"到 fp32 ----
    float scale = sA_scale[s] * sB_scale[s];    // 两个细粒度 scale 相乘
    #pragma unroll
    for (int i = 0; i < FRAG; ++i)
        acc_final[i] += scale * acc_wgmma[i];   // FFMA(FP32 pipe), 逐 fragment 晋升
    acc_wgmma = 0;                              // 临时累加器清零, 迎接下一段
    empty[s].arrive();
}
```

- **【映射】** ③placement 的新维度:**累加器分裂成两级**——短命的低精度级(Tensor Core 内)+ 常驻的 fp32 级(寄存器 OS)。scale 的乘法被**推迟到晋升时刻**、且每 128-K 段只做一次——细粒度缩放的开销被 BLOCK_K 摊薄。
- **【硬件】** 晋升跑在 **FFMA(FP32 pipe)** 上——Tensor Core 与 CUDA core **分工同框**:贵的 pipe 做矩阵,闲的 pipe 做修正(和 FA-4 软件 exp 同一个"用闲 pipe 打杂"思想,A1(04))。
- **【反事实】** 不做两级、让 wgmma 一路累加到底 → 长 K(如 7168)下精度损失可测,模型质量掉;晋升太频(每条 wgmma 一次)→ FFMA 开销盖过 Tensor;太疏(整个 K 一次)→ 精度回不来。**BLOCK_K=128 是精度×开销的平衡点,也正好等于 scale 粒度——三个 128 对齐不是巧合,是协同设计。**

### 1.3 persistent 调度器:grouped GEMM 的"工头"(A2 §2.4 的代码形态)

```cpp
// grid = 恰好铺满机器(#SM 个 CTA), 每个 CTA 是常驻工人
struct Scheduler { int next_tile;  /* global memory 里的原子计数器 */ };
while (true) {
    int tile_id = atomicAdd(&sched.next_tile, 1);        // 领活
    if (tile_id >= total_tiles) break;
    auto [expert, m_blk, n_blk] = decode(tile_id);        // 查这块属于哪个专家
    // m-grouped 连续布局: 各专家的 M 拼接在一起, tile 直接按全局行号切
    run_gemm_tile(expert, m_blk, n_blk);                  // 上面 1.2 的主循环
}
```

- **【映射】** ④binding 从"静态 grid"变"**动态领活**":变长的 M_i(各专家 token 数不等)被摊成均匀的 tile 流,SM 永不空等(A2 tile/wave 量化的解药)。
- **【硬件】** 常驻 CTA + global 原子计数器;decode 阶段还有 **masked 布局**(按 CPU 不可知的实际 token 数在 GPU 侧 mask)——为 CUDA Graph 固定形状服务(04-01 ⑤层)。
- **【反事实】** 静态平分 grid → 大专家的 SM 还在算、小专家的早收工(负载不均,`Achieved Occupancy` 尾部塌);另一个妙细节:DeepGEMM 允许 **BLOCK_N=112 这类"非 2 幂"tile**——牺牲一点对齐美感,换 tile 数与 SM 数更整除(**治 wave 量化**,A1)——"最优 tile 不一定是 2 的幂"的实证。

### 1.4 JIT:信息衰减的反向操作(04-01)

DeepGEMM 不用模板预编译,**运行时拿到 (M,N,K,专家数) 才现场编译**:所有形状成为编译期常量 → 循环全展开、边界判断消失、除法变移位。核心 GEMM 代码 ~300 行。**代价**:首次编译延迟(靠缓存摊);**收益**:每个形状都是"专属定制 kernel"——把 04-03 autotune 的"形状桶"推到极限(桶=单个形状)。

---

## 2. DeepEP:通信写成 kernel

### 2.1 low-latency dispatch 骨架(decode 用,零 SM 占用)

```cpp
// 发送侧(每个 token 一小组线程):
void dispatch_send(token, topk_experts) {
  for (e in topk_experts) {
    int dst_rank = expert_to_rank(e);
    // 打包: token 的 FP8 数据 + scale + 路由元信息 → 对称缓冲区
    pack(rdma_send_buf[dst_rank][slot], token_fp8, scales, meta);
    fence.release.sys;                        // §01-05: 保证打包的写对 NIC 可见
    // IBGDA: GPU 直接写 NIC 门铃, 发起 RDMA WRITE —— 无 CPU, 无 NCCL
    ibgda_put_nbi(dst_rank, remote_addr, local_addr, bytes);
  }
}
// 接收侧: 不占 SM —— 返回一个 hook, 计算随后"钩"一下
auto [recv_tensor, hook] = ll_dispatch(...);   // 立即返回!
other_compute();                               // 与网络传输完全重叠(SBO 的原料)
hook();                                        // 轮询完成标志, 就位才继续
```

- **【映射】** 这是"藏延迟"爬到系统层的代码形态(A1(04) 那张五层表的最上层):**传输期间 SM 一秒都不占**,重叠由调用方(SGLang 的 SBO 调度)编排。
- **【硬件】** 三个硬机制:**对称内存**(NVSHMEM 式,各 rank 同一虚拟布局,远端地址=本地地址+rank 偏移,免地址交换);**IBGDA**(GPU 写 NIC 门铃寄存器发起 RDMA——§01-05 的"同步对象是引擎"再进一步:**连发起权都给了 GPU**);**`fence.release.sys`**(§01-05 §2.3-2 的 sys-scope 可见性,真实用武之地——数据要对芯片外的 NIC 可见,必须冲刷到系统一致点)。
- **【反事实】** 少了 release fence → NIC 可能读到打包一半的数据(弱内存模型的经典翻车,§01-05);用 NCCL 走这条路 → 要起通信 kernel 占 SM + CPU 协调,μs 级延迟变多倍。

### 2.2 normal kernel(prefill 用,占 SM 换吞吐)

```cpp
// 每个 CTA 按"目标"分工: warp 0..7 → NVLink 发往本节点 8 卡, warp 8+ → RDMA 跨节点
// 跨节点路径: 先 RDMA 到远端同号卡, 再由它经 NVLink 转发给最终目标
//   —— 03-02 四层地图的实战: 让大流量走最宽的那层(NVLink), RDMA 只跳一跳
```

- **【映射】** warp 专化的第三种用法(计算→搬运→**通信**):warp 按目的地分工,和 B2 的生产者/消费者同构。
- **【硬件】** 名场面 `ld.global.nc.L1::no_allocate`:读别人写来的缓冲时**绕过 L1 且不占 cache line**(反正只读一次,§01-04 复用判据的反向应用)——技术上未定义行为,实测更快,**③层压榨到 ISA 边缘**(§01-06)。

## 3. B 系列收官:一张"全栈代码地图"

| 层(藏什么延迟) | 代码证据 | 出处 |
| --- | --- | --- |
| 指令级(算术) | FA4 软件 exp / DeepGEMM FFMA 晋升(闲 pipe 打杂) | A1(04)/B3 §1.2 |
| warp 级(访存) | B1 的 autotune num_warps | B1 §2.1 |
| kernel 内(搬运) | B1 num_stages ↔ B2 环形缓冲+mbarrier+相位票 | B1/B2 |
| kernel 内(pipe 抢占) | B2 ping-pong 命名屏障 | B2 §5 |
| kernel 间(launch) | DeepGEMM persistent 调度器 + CUDA Graph | B3 §1.3 |
| **系统级(网络)** | **DeepEP hook + IBGDA(SM 零占用)** | B3 §2.1 |

> 🔑 **B 系列总结一句话**:三篇读下来,"藏延迟五层重现"不再是口号——**每一层都有一段真实代码作证**。而且三个 kernel 用的是**同一套积木**(TMA/mbarrier/warp 专化/OS 累加/persistent),差别只在把积木搭向哪个瓶颈:FA 搭向 MUFU,DeepGEMM 搭向精度与 wave 量化,DeepEP 搭向网络。**认识积木(模块 01-03),看懂图纸(04-04 四维),就能读懂任何新 kernel——这是 B 系列想交付的终极能力。**

## 4. 验收

- **DeepGEMM repo**:搜 `promote / tma_copy / Scheduler / BLOCK_N=112 相关注释`;README 明说两级累加与 JIT 设计。
- **DeepEP repo**:搜 `ibgda / low_latency / hook / nvl_forward`;文档明说 no-SM-occupation 与 NVLink 转发。
- **ncu 侧**:DeepGEMM 应见 Tensor pipe 高 + FFMA 有规律脉冲(晋升);DeepEP low-latency 传输期 GPU 计算指标**应无扰动**(它不在 SM 上)——用 nsys 看 NIC/网络轨道。

## 5. 我的困惑 / 待深挖

- （待填）FP8 wgmma 累加器的精确格式(几位尾数)?两级累加的误差上界推导?
- （待填）IBGDA 门铃写的延迟 vs CPU proxy 的实测差?多 QP 并发的扩展性?
- （待填）BLOCK_N=112 类非对齐 tile 在 Blackwell(tcgen05)上还成立吗?
- （待填）B 系列可能的 B4:昇腾 AscendC 版算子精读(手动挡 placement 的代码形态)?

---

*最后更新：2026-07-06（第一版 · B 系列收官）*
