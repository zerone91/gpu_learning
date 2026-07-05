# Tensor Core = 塞进 SIMT 核的小脉动引擎

> **所属模块**：模块 03 · 两者交汇与扩展
> **状态**：⬜ 待学习
> **一句话主旨**：〈待填〉

---

## 0. 这一节要回答的问题

- [ ] Tensor Core 在 SM 里到底是个什么部件、和 ALU/FMA lane 并列吗？
- [ ] 它内部是不是一个小脉动阵列？和模块 02 的大阵列差别在规模还是机理？
- [ ] 一条 Tensor Core 指令（如 MMA / wgmma）一次算多大的矩阵块？
- [ ] warp 怎么协作喂一个 Tensor Core？为什么"两套词"（warp / 阵列）会同时出现？
- [ ] 它怎么和 shared memory、async copy、mbarrier 组成流水？

## 1. 核心概念

## 2. 关键机制 / 为什么这样设计

<!-- 本模块题眼：把模块 01 和 02 缝死的那一针 -->

## 3. 和其它模块的挂钩

- SIMT 侧：SM 内部、warp、同步 → 模块 01
- 脉动侧：PE 阵列、数据流、tiling → 模块 02
- 编程接口 MMA/wgmma 落到 kernel 层 → 模块 04 第 03 节
- 真实实现对照 Hopper Tensor Core → 模块 05

## 4. 一句话黑话卡

| 术语 | 英文 / 别名 | 一句话解释 | 挂在哪个模块 |
| --- | --- | --- | --- |
| Tensor Core |  |  | 03 |
| MMA | matrix multiply-accumulate |  | 03 |
| wgmma | warpgroup MMA（Hopper） |  | 03 |

## 5. 我的困惑 / 待深挖

---

*最后更新：待填*
