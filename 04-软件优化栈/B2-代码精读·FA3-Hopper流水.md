# B2 · 代码精读：FA-3 的 Hopper warp 专化流水（CUTLASS/CuTe 层）

> **所属系列**：模块 04 · 代码精读 B 系列（B1 Triton FA2 → **B2** → B3 DeepSeek 栈）
> **状态**：🟨 学习中（第一版，知识截至 2026-01。代码为忠实于 CUTLASS 3.x / FA-3 结构的**注释性伪代码**——保留全部关键机制与真实指令名，简化模板噪音与边界处理；真实实现见 flash-attention repo hopper 目录与 CUTLASS FMHA 示例）
> **一句话主旨**：B1 里 Triton 用 `num_stages=3` 一个参数代管的东西，这里全部**显式摊开**——TMA 描述符、mbarrier 的 expect_tx/phase、wgmma 异步组、setmaxnreg、命名屏障 ping-pong。读完两件事就通了：**① Triton 那个参数背后到底是什么；② "同步是异步的账单"（§01-05）在真实代码里长什么样——每一个异步器件旁边必然站着它的同步原语。**

---

## 0. 全景：映射方案与"人员编制"

一个 CTA（block）= **384 线程 = 3 个 warpgroup**，各司其职（A2 §1.7 三角色的代码化）：

```
WG0（生产者）: 只发 TMA。setmaxnreg.dec 让出寄存器（瘦身到 ~24 reg/线程）
WG1（消费者A）: wgmma + softmax。setmaxnreg.inc 增肥（~240 reg/线程，装 O 累加器）
WG2（消费者B）: 同 WG1，与 A 错半拍 ping-pong

shared memory 布局:
  sQ            : Q tile（装一次，常驻）
  sK[0..S-1]    : K tile 环形缓冲（S = 流水级数，如 3）
  sV[0..S-1]    : V tile 环形缓冲
  full[0..S-1]  : mbarrier ×S —— "第 s 格装满了"（生产者→消费者）
  empty[0..S-1] : mbarrier ×S —— "第 s 格用完了"（消费者→生产者）
```

**同步器件清单**（每个异步能力配一个同步原语——§01-05 §1.4 的活体展览）：
| 异步的事 | 配套同步 |
| --- | --- |
| TMA 搬运（DMA 引擎） | `full[s]` mbarrier + **expect_tx 字节计数** |
| 缓冲格回收 | `empty[s]` mbarrier |
| wgmma（Tensor Core 异步） | `wgmma.commit_group / wait_group` |
| 两个消费者错拍 | **命名屏障** `bar.sync 8/9, 256` |

## 1. Host 侧：TMA 描述符（§01-04 §2.4-4 的代码形态）

```cpp
// CuTe 写法：把"怎么搬"编译成一个 128B 的描述符对象，kernel 里一个线程即可触发
auto tma_load_K = make_tma_copy(
    SM90_TMA_LOAD{},                     // 拷贝原子：TMA 加载
    gK,                                  // global 张量（含 shape/stride）
    sK_layout,                           // shared 目标布局 ← 内含 Swizzle<3,4,3>！
    select<1,2>(TileShape{}),            // 每次搬的 box：BLOCK_N × HEAD_DIM
    _1{});                               // 无多播（cluster>1 时可改多播省边缘流量）
```

- **【映射】** ②ordering 的搬运粒度 + ③placement 的目的地，**在 host 侧就编译定型**——TMA 把"32 线程算地址"变成"一个描述符"（§01-04）。
- **【硬件】** 生成 `CUtensorMap`（128B 常量对象，经 `__grid_constant__` 传入）；`sK_layout` 里的 **swizzle atom**（如 `GMMA::Layout_K_SW128_Atom<bf16>`）同时约束 TMA 写入模式与 wgmma 读取模式——**B1 里"编译器代管的 swizzle"，就是这一行的显式版**（三方协同：TMA 写 / ldmatrix·wgmma 读，§01-04 §2.4-2）。
- **【反事实】** swizzle 选错或 shared 基址非 128B 对齐 → 描述符创建失败或 bank 冲突复现；行 stride 非 16B 对齐 → 直接 launch 报错（TMA 的硬约束）。

## 2. Kernel 开场：角色分派与寄存器"劫富济贫"

```cpp
__global__ void fa3_fwd(__grid_constant__ const CUtensorMap tma_K, ...) {
  int wg = threadIdx.x / 128;                    // 我是哪个 warpgroup
  if (wg == 0) {                                 // ---- 生产者 ----
    cutlass::arch::warpgroup_reg_dealloc<24>();  // PTX: setmaxnreg.dec.sync.aligned.u32 24
    producer_loop(...);
  } else {                                       // ---- 消费者 A/B ----
    cutlass::arch::warpgroup_reg_alloc<240>();   // PTX: setmaxnreg.inc...240
    consumer_loop(wg, ...);
  }
}
```

- **【映射】** ④binding 的进化：B1 里所有线程同构；这里**按 warpgroup 分工种**——SIMT 的"锁步大军"被组织成流水线工厂（00 光谱右移的代码实体）。
- **【硬件】** `setmaxnreg`（§02 §2.6b）：**寄存器堆是按 warpgroup 动态再分配的**——生产者只发 TMA 不算数，24 个寄存器够了；省下的配额给消费者装 fp32 累加器（B1 §2.4 那 1.6 万个寄存器的来源）。CTA 总寄存器不变，**内部劫富济贫**。
- **【反事实】** 不做 reg 再分配 → 消费者装不下大累加器 → 只能缩 tile 或 spill；生产者白占几千寄存器 → 占用率/驻留白白吃亏。

## 3. 生产者循环：TMA + expect_tx（"同步对象是引擎+字节数"）

```cpp
void producer_loop(...) {
  if (elect_one_sync()) {                        // PTX: elect.sync —— 128 线程里选 1 个代表
    for (int k = 0; k < n_blocks; ++k) {
      int s = k % S;                             // 环形缓冲格号
      // 等"第 s 格已被消费者用完"（首轮 empty 初始为已到达）
      empty[s].wait(phase_empty[s]);             // mbarrier.try_wait + 相位票
      // 关键一行：告诉屏障"这次要等 KV_BYTES 字节到齐"，然后发射 TMA
      full[s].arrive_and_expect_tx(KV_BYTES);    // PTX: mbarrier.arrive.expect_tx
      tma_load(tma_K, sK[s], coord_k(k), full[s]);  // PTX: cp.async.bulk.tensor.2d
      tma_load(tma_V, sV[s], coord_v(k), full[s]);  //      （完成时由硬件向 full[s] 报字节）
    }
  }
}
```

- **【映射】** ②ordering 的搬运侧：永远超前消费者 S-1 块（B1 的 num_stages，这里是手写的环形缓冲）。
- **【硬件】** 三个 §01-05 概念同框：**`elect.sync`**（TMA 只需一个线程发,选代表）；**`expect_tx`**（同步对象从"线程"扩展到"**引擎+字节数**"——TMA 引擎搬完自动向 mbarrier 报账,凑齐 KV_BYTES 才翻相位）；**相位票 `phase[s]`**（环形复用同一屏障,每绕一圈软件翻一位——§01-05 §2.3-1 的 ABA 解法,在代码里就是一个 `phase ^= 1`）。
- **【反事实】** expect_tx 字节数写错 → 屏障永不翻相位（少报）或提前翻（多报,消费者读到半成品数据,**静默算错**）；忘等 empty → 覆盖消费者还在用的格子,同样静默算错——**这类 bug ncu 查不出来,只能 racecheck/对拍**,这就是 Triton 代管的价值。

## 4. 消费者循环：wgmma 异步组 + softmax + 流水消费

```cpp
void consumer_loop(int wg, ...) {
  for (int k = 0; k < n_blocks; ++k) {
    int s = k % S;
    full[s].wait(phase_full[s]);                 // 等第 s 格的 KV 字节到齐
    // ---- GEMM0: S_blk = Q·K^T ----
    warpgroup_arrive();                          // PTX: wgmma.fence —— 保护寄存器操作数
    gemm(tiled_mma0, sQ, sK[s], tSrS);           // PTX: wgmma.mma_async.sync... ×N 条
    warpgroup_commit_batch();                    // PTX: wgmma.commit_group
    warpgroup_wait<0>();                         // PTX: wgmma.wait_group 0 —— 等这批出结果
    // ---- softmax（SFU/MUFU + FFMA）----
    online_softmax_rescale(tSrS, m, l, acc);     // B1 §2.5 那 7 行，此处在寄存器 fragment 上做
    // ---- GEMM1: O += P·V ----
    convert_and_layout(tSrS -> tSrP /*bf16*/);   // fp32→bf16 + 摆成 wgmma 的 A-fragment
    warpgroup_arrive();
    gemm(tiled_mma1, tSrP, sV[s], acc);          // 第二组 wgmma
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    empty[s].arrive();                           // 报告"第 s 格我用完了" → 生产者可重灌
  }
  epilogue(acc, l);                              // 归一化 + TMA store 写回
}
```

- **【映射】** 与 B1 的循环逐行对应——**数学一个字没变**（online softmax 原样），变的全是"等谁、谁算、何时放行"的④binding/②ordering 细节。
- **【硬件】** wgmma 的**异步三件套**（§01-05 §1.3 表里的"wgmma 组"）：`fence`（寄存器操作数保护）→ 发一批 `mma_async` → `commit_group` → `wait_group`。**wgmma 直接以 shared memory 为 B 操作数**（sK/sV 不经寄存器,§03-01 演化表 Hopper 行）——对照 B1:Triton 里 K/V 还要过寄存器 fragment。
- **【反事实】** 漏 `wgmma.fence` → 异步 MMA 还在读寄存器时 softmax 已改写它,静默错;`wait_group<0>` 换成 `<1>`（允许一批悬空）是更深的重叠,但寄存器要多养一批 fragment——**又是那道"重叠深度 vs 寄存器"的账**（§01-02 §2.6c）。

## 5. Ping-pong：两个消费者错拍（命名屏障的用法）

```cpp
// 消费者 A/B 相同代码，靠两个命名屏障强制错开半拍：
// 屏障 8 = "轮到谁用 Tensor Core（GEMM 段）"，屏障 9 = "轮到谁用 SFU（softmax 段）"
if (wg == 2) named_barrier_wait(8);        // B 先让 A 起跑（错拍的初始相位差）
...
named_barrier_wait(8);                     // PTX: bar.sync 8, 256（两个 WG 共 256 线程）
gemm(...); commit; wait;                   // ← 我占 Tensor Core 的时段
named_barrier_arrive(8);                   // 交出 Tensor Core → 对方的 GEMM 可以开始
softmax_rescale(...);                      // ← 同时对方在做 GEMM，我在用 SFU
named_barrier_arrive(9); named_barrier_wait(9);  // softmax 段的交接
```

- **【映射】** ④binding 的最后一块拼图：**A2 §1.7 那张 "Tensor: A B A B / SFU: — A B A" 时序图,物理上就是这几个 `bar.sync 8/9` 在指挥**。
- **【硬件】** 命名屏障（§01-05 §1.3:每 CTA 16 个,各带参与线程数 256）——比 `__syncthreads` 细,只同步两个消费者 WG,生产者不受影响。**Tensor Core 是子分区共享资源,两个 WG 靠屏障轮流独占它,SFU 段与对方的 GEMM 段天然重叠**——MUFU 序列化(B1 验收第 3 条的遗留瓶颈)就此藏进 Tensor 的影子。
- **【反事实】** 去掉 ping-pong（两 WG 自由跑）→ 两者的 GEMM 段随机撞车、softmax 段也撞车 → Tensor pipe 出现空洞,实测 util 明显掉——**"多加两次同步反而更快"**:因为它买到的是**资源时分复用的秩序**（§01-05 "少同步"原则的著名反例,秩序>次数）。

## 6. 总账：B1 vs B2（Triton 代管 ↔ 显式摊开）

| B1 里的一个参数/一行 | B2 里的显式形态 | 概念出处 |
| --- | --- | --- |
| `num_stages=3` | sK/sV[3] 环形缓冲 + full/empty mbarrier×3 + 相位票 | §01-02 §2.6 |
| `make_block_ptr` | host 侧 `make_tma_copy` + CUtensorMap + swizzle atom | §01-04 §2.4 |
| `tl.dot` | `wgmma.fence/mma_async/commit/wait` 四件套,B 操作数直读 shared | §03-01 |
| 线程同构 | 3 warpgroup 分工 + `setmaxnreg` 劫富济贫 + `elect.sync` | §02 §2.6b |
| （无对应,Triton 难表达） | **ping-pong 命名屏障**——SOTA 与"良好"的差距所在 | A2 §1.7 |

> 🔑 **读后感一句话**:Triton 让你用 1 个参数买到 80 分的流水;最后 20 分（ping-pong、寄存器再分配、wait_group<1> 级重叠）需要显式操纵同步原语——**而每一个显式原语都在 §01-05 的全家桶清单里**。硬件知识在这一层不再是"背景",是"操作对象"。

## 7. 验收（真代码上）

- flash-attention repo `hopper/` 目录:搜 `pipeline_tma_async / setmaxnreg / NamedBarrier / cutlass::arch`,以上每个机制都能对号入座。
- `cuobjdump -sass`:应看到 `UTMALDG`（TMA load）、`HGMMA`（wgmma 的 SASS）、`BAR.SYNC.DEFER` 系、`ELECT`。
- `ncu`:对照 A2 §1.6——Tensor util 应从 FA2 型的 ~35% 升到 ~70%+;stall 里 mbarrier/命名屏障等待应短而规律（流水节拍）,若某一项长 → 对应哪级缓冲饿了/堵了,回 §2.6c 的三方预算。

## 8. 我的困惑 / 待深挖

- （待填）wait_group<1> 级深重叠在 FA-3 real code 里用于哪段？寄存器代价实测？
- （待填）causal 时两个消费者 WG 的负载不均（对角块 vs 满块）怎么平衡？
- （待填）cluster>1 + TMA 多播在 FA-3 上省多少 L2 流量？
- （待填）B3 预告:DeepGEMM 用同样的 TMA+mbarrier 骨架跑 FP8 grouped GEMM,persistent 调度器怎么接进来？

---

*最后更新：2026-07-06（第一版）*
