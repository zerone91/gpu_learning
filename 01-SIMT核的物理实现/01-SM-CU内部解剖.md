# SM/CU 内部解剖

> **所属模块**：模块 01 · SIMT 核的物理实现
> **状态**：⬜ 待学习（建议的开刀起点）
> **一句话主旨**：〈待填〉

---

## 0. 这一节要回答的问题

- [ ] 一个 SM（NVIDIA）/ CU（AMD）内部到底有哪些部件，各干什么？
  - warp scheduler（发射单元）
  - 执行单元：ALU / FMA lane
  - 寄存器堆（register file）
  - LSU（load/store unit，即你说的 "LSM"）
  - shared memory / L1（同一块 SRAM 的两种用法）
  - Tensor Core 坐在这张图的哪个位置
- [ ] 拿一个熟悉的算子映射上去，warp、锁步、执行单元、访存这几个词怎么第一次挂到同一张图上？

## 1. 核心概念

<!-- 一张 SM/CU 解剖图 + 各部件职责。这是模块 01 的地基 -->

## 2. 关键机制 / 为什么这样设计

## 3. 和其它模块的挂钩

- Tensor Core 的位置 → 模块 03 展开
- shared memory / register 作为访存层次的一环 → 本模块 04
- register/占用与 occupancy 挂钩 → 本模块 02

## 4. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| SM | Streaming Multiprocessor |  | 01 |
| CU | Compute Unit（AMD 对应） |  | 01 |
| LSU | Load/Store Unit |  | 01 |
| 寄存器堆 | register file |  | 01 |

## 5. 我的困惑 / 待深挖

---

*最后更新：待填*
