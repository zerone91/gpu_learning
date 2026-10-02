"""实验：Triton 的 num_stages 开几份缓冲；三种指针写法各生成哪种搬运指令。

只编译不运行，不需要对应的显卡：走真实的 JIT 路径（warmup），只把目标架构换掉。
JIT 会按参数缓存编译结果，所以每个（架构, 写法, num_stages）组合必须用独立进程跑。

用法（仓库根目录）：
  python3 _资源/实验/triton_num_stages.py <架构: 80|90|100> <写法: ptr|blockptr|desc|desc_inkernel> <num_stages> [--dump 目录]
  python3 _资源/实验/triton_num_stages.py --all        # 依次起子进程跑完全部组合

2026-09-29 在 Triton 3.7.0 上的结果见同目录 README.md。
"""
import re, subprocess, sys
import torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import MockTensor
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mm_ptr(a_ptr, b_ptr, c_ptr, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sa1
        b_ptrs += BK * sb0
    c_ptrs = c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    tl.store(c_ptrs, acc.to(tl.float16))


@triton.jit
def mm_blockptr(a_ptr, b_ptr, c_ptr, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(a_ptr, (M, K), (sa0, sa1), (pid_m * BM, 0), (BM, BK), (1, 0))
    b_bp = tl.make_block_ptr(b_ptr, (K, N), (sb0, sb1), (0, pid_n * BN), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))
    c_bp = tl.make_block_ptr(c_ptr, (M, N), (sc0, sc1), (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(c_bp, acc.to(tl.float16), boundary_check=(0, 1))


@triton.jit
def mm_desc(a_desc, b_desc, c_desc, M, N, K,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = a_desc.load([pid_m * BM, k * BK])
        b = b_desc.load([k * BK, pid_n * BN])
        acc = tl.dot(a, b, acc)
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))


@triton.jit
def mm_desc_inkernel(a_ptr, b_ptr, c_ptr, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[sa0, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[sb0, 1], block_shape=[BK, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = a_desc.load([pid_m * BM, k * BK])
        b = b_desc.load([k * BK, pid_n * BN])
        acc = tl.dot(a, b, acc)
    c_ptrs = c_ptr + (pid_m * BM + tl.arange(0, BM))[:, None] * sc0 + (pid_n * BN + tl.arange(0, BN))[None, :] * sc1
    tl.store(c_ptrs, acc.to(tl.float16))


BM, BN, BK = 128, 128, 64
M = N = K = 4096


def report(tag, k):
    ir, ptx = k.asm["ttgir"], k.asm["ptx"]
    n = lambda p, s: len(re.findall(p, s))
    bufs = sorted(set(re.findall(r"ttg\.local_alloc[^\n]*?memdesc<([0-9x]+)x(?:f|bf|i)\d+", ir)))
    print(f"{tag}: 缓冲={bufs} "
          f"wait_group={sorted(set(re.findall(r'cp\.async\.wait_group\s+(\d+)', ptx)))} "
          f"cp.async={n(r'cp\.async\.c[ag]\.shared', ptx)} "
          f"TMA载入={n(r'cp\.async\.bulk\.tensor\.\dd\.shared', ptx)} "
          f"ld.global={n(r'ld\.global', ptx)} wgmma={n('wgmma.mma_async', ptx)} mma.sync={n(r'mma\.sync', ptx)}")


def main(cc, name, stages, dump=None):
    triton.runtime.driver.active.get_current_target = lambda: GPUTarget("cuda", cc, 32)
    A = MockTensor(torch.float16, [M, K]); B = MockTensor(torch.float16, [K, N]); C = MockTensor(torch.float16, [M, N])
    kw = dict(grid=(32, 32), num_warps=4, num_stages=stages, BM=BM, BN=BN, BK=BK)
    ptr_args = (A, B, C, M, N, K, K, 1, N, 1, N, 1)
    if name == "desc":
        mk = lambda T, bs: TensorDescriptor(T, T.shape, T.stride(), bs)
        k = mm_desc.warmup(mk(A, [BM, BK]), mk(B, [BK, BN]), mk(C, [BM, BN]), M, N, K, **kw)
    else:
        k = {"ptr": mm_ptr, "blockptr": mm_blockptr, "desc_inkernel": mm_desc_inkernel}[name].warmup(*ptr_args, **kw)
    report(f"{name:14s} sm_{cc} num_stages={stages}", k)
    if dump:
        for ext in ("ttgir", "ptx"):
            open(f"{dump}/{name}_{cc}_{stages}.{ext}", "w").write(k.asm[ext])


if __name__ == "__main__":
    if sys.argv[1] == "--all":
        for cc in (80, 90, 100):
            for name in ("ptr", "blockptr", "desc_inkernel", "desc"):
                for st in (1, 2, 3, 4, 5):
                    r = subprocess.run([sys.executable, __file__, str(cc), name, str(st)], capture_output=True, text=True)
                    print((r.stdout.strip().splitlines() or [f"{name} sm_{cc} num_stages={st}: 失败 " + r.stderr.strip().splitlines()[-1][:160]])[-1], flush=True)
    else:
        dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
        main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), dump)
