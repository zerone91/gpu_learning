# 实验记录

用实际编译或运行来核实正文里的说法。每个实验一个脚本，结果记在这里，正文引用时指向本目录。

## Triton 的 `num_stages` 与搬运指令的选择

- 脚本：[`triton_num_stages.py`](./triton_num_stages.py)
- 日期与版本：2026-09-29，Triton 3.7.0
- 方法：只编译不运行。走 Triton 真实的 JIT 路径（`warmup`），只把目标架构替换成 sm_80（Ampere）、sm_90（Hopper）、sm_100（Blackwell），然后在生成的中间表示和 PTX 里数缓冲份数与各类指令。本机显卡是 GTX 1060，不影响结果。
- 负载：同一个 GEMM，fp16，块大小 128 × 128 × 64，4 个 warp。四种写法：逐指针 `tl.load`、`tl.make_block_ptr`、kernel 内建的 `tl.make_tensor_descriptor`、主机侧建好传入的 `TensorDescriptor`。

### 结果一：`num_stages=N` 开几份缓冲

| `num_stages` | sm_80 缓冲份数 | sm_90 缓冲份数 | sm_100 缓冲份数 | 主循环里的 `wait_group` | 换算成"允许几轮在途" |
| --- | --- | --- | --- | --- | --- |
| 1 | 不排流水 | 不排流水 | 不排流水 | 无 | — |
| 2 | 1 | 2 | 2 | 0 | 0 |
| 3 | 2 | 3 | 3 | 2 | 1 |
| 4 | 3 | 4 | 4 | 4 | 2 |
| 5 | 4 | 5 | 5 | 6 | 3 |

- Ampere 上缓冲是 N−1 份，Hopper 和 Blackwell 上是 N 份。
- 每轮迭代给 A 块和 B 块各提交一组，所以 `wait_group` 的数字是轮数的两倍。按轮数算，wait 深度是 N−2。
- 序幕预发 N−1 轮。
- Ampere 少一份缓冲的原因：主循环先把到货的块从 shared 读进寄存器再计算，数据进寄存器后那份缓冲就让出来了。Hopper 的 `wgmma` 直接从 shared 取数，计算期间缓冲不能让。

### 结果二：哪种写法生成 TMA（`num_stages=3`）

| 写法 | sm_80 | sm_90 | sm_100 |
| --- | --- | --- | --- |
| 逐指针 `tl.load` | 48 条 `cp.async` | 48 条 `cp.async` | 64 条 `cp.async` |
| `tl.make_block_ptr` | 48 条 `cp.async` | 48 条 `cp.async` | 64 条 `cp.async` |
| `tl.make_tensor_descriptor`（kernel 内） | 48 条 `cp.async` | **6 条 TMA 载入** | **8 条 TMA 载入** |
| `TensorDescriptor`（主机侧） | 128 条普通读取，不排流水 | **6 条 TMA 载入** | **8 条 TMA 载入** |

- `tl.make_block_ptr` 在三种架构上都和逐指针写法生成相同的指令，不会生成 TMA。编译时还会提示这个接口已弃用。
- 只有张量描述符会生成 TMA，而且只在 sm_90 及以后。
- 主机侧传入的描述符面向 sm_80 编译时，退回逐元素的普通读取，并且不排流水。需要兼容 Ampere 的代码要留意这一点。
- 矩阵指令：sm_80 是 `mma.sync`，sm_90 是 `wgmma`，sm_100 是 `tcgen05.mma`（PTX 里确认有 8 条）。

### 局限

- 只测了一个版本、一种负载。Triton 的流水调度在不同版本之间改动较多，换版本后应重跑。
- 只看了编译产物，没有运行，不涉及性能数字。
- 早期版本的 Triton 是否曾经由块指针生成 TMA，没有验证。

### 用到这份结果的章节

- 模块 14 第 2 节 §6.5、第 5 节 §8
- 模块 40 第 4 节 §7.3、第 9 节 §4.2、附录 B1 §3.3
