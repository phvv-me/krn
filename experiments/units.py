"""CUDA C++ and OpenCL C translation units of the sizes the compile experiments need.

The small units are what a generated kernel of the sketch looks like. `synthetic` scales one device-function
pattern up to the size of cutok's device code (about 4.5K lines in 60 files) so compile time reads against volume.
"""

GID_DEFINE = "#define GID ((ulong)__builtin_amdgcn_workgroup_id_x() * 256ul + __builtin_amdgcn_workitem_id_x())"

SAXPY_CUDA = """
extern "C" __global__ void saxpy(float a, const float* x, float* y, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

DFA_CUDA = """
struct Dfa { const unsigned short* table; const unsigned char* accept; unsigned short start; };
struct Segments { const unsigned char* text; const unsigned* offsets; unsigned* out; unsigned count; };

__device__ unsigned short step(const Dfa& d, unsigned short state, unsigned char byte) {
    return d.table[(unsigned)state * 256u + (unsigned)byte];
}

extern "C" __global__ void run_dfa(Dfa dfa, Segments segs) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 < segs.count) {
        unsigned short state = dfa.start;
        for (unsigned p = segs.offsets[i]; p < segs.offsets[i + 1]; ++p)
            state = step(dfa, state, segs.text[p]);
        segs.out[i] = (unsigned)dfa.accept[state];
    }
}
"""

REDUX_CUDA = """
extern "C" __global__ void reduce(const unsigned* x, unsigned* out) {
    unsigned v = x[blockIdx.x * blockDim.x + threadIdx.x], r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    if ((threadIdx.x & 31) == 0) out[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = r;
}
"""

FP16_HEADER_CUDA = """
#include <cuda_fp16.h>
extern "C" __global__ void scale(const __half* x, __half* y, float a, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(a * __half2float(x[i]));
}
"""

COOP_HEADER_CUDA = """
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
extern "C" __global__ void block_sum(const float* x, float* out) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    float v = cg::reduce(warp, x[blockIdx.x * blockDim.x + threadIdx.x], cg::plus<float>());
    if (warp.thread_rank() == 0) atomicAdd(out, v);
}
"""

SAXPY_OPENCL = """
#define GID ((ulong)__builtin_amdgcn_workgroup_id_x() * 256ul + __builtin_amdgcn_workitem_id_x())
__kernel void saxpy(float a, __global const float* x, __global float* y, ulong n) {
    ulong i = GID;
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

DFA_OPENCL = """
#define GID ((ulong)__builtin_amdgcn_workgroup_id_x() * 256ul + __builtin_amdgcn_workitem_id_x())
typedef struct { __global const ushort* table; __global const uchar* accept; ushort start; } Dfa;
typedef struct { __global const uchar* text; __global const uint* offsets; __global uint* out; uint count; } Segments;

ushort step(Dfa d, ushort state, uchar byte) { return d.table[(uint)state * 256u + (uint)byte]; }

__kernel void run_dfa(Dfa dfa, Segments segs) {
    ulong i = GID;
    if (i + 1 < segs.count) {
        ushort state = dfa.start;
        for (uint p = segs.offsets[i]; p < segs.offsets[i + 1]; ++p)
            state = step(dfa, state, segs.text[p]);
        segs.out[i] = (uint)dfa.accept[state];
    }
}
"""


def synthetic(functions: int, *, dialect: str) -> tuple[str, int]:
    """One kernel that calls `functions` device functions of about 14 lines of byte and integer code each.

    Returns the source and its line count. The pattern follows cutok's device code: a loop over bytes, a table probe,
    a few branches, 32 and 64-bit arithmetic.
    """
    cuda = dialect == "cuda"
    device = "__device__ " if cuda else ""
    u32, u8 = ("unsigned", "unsigned char") if cuda else ("uint", "uchar")
    glob = "" if cuda else "__global "
    parts: list[str] = []
    for n in range(functions):
        parts.append(f"""
{device}{u32} f{n}(const {glob}{u8}* text, {u32} start, {u32} end, const {glob}{u32}* table, {u32} mask) {{
    {u32} h = start * {2 * n + 1}u;
    for ({u32} p = start; p < end; ++p) {{
        {u32} b = ({u32})text[p];
        {u32} slot = (h ^ (b << {n % 7})) & mask;
        while (true) {{
            {u32} payload = table[slot * 2u + 1u];
            if (payload == 0u) break;
            if (table[slot * 2u] == (h + b)) {{ h = payload; break; }}
            slot = (slot + 1u) & mask;
        }}
        h = h * 0x9E3779B1u + b;
    }}
    return h;
}}""")
    calls = "\n".join(f"    acc ^= f{n}(text, start, end, table, mask);" for n in range(functions))
    if cuda:
        kernel = f"""
extern "C" __global__ void unit(const unsigned char* text, const unsigned* offsets, const unsigned* table,
                                unsigned mask, unsigned* out, unsigned count) {{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned start = offsets[i], end = offsets[i + 1], acc = 0u;
{calls}
    out[i] = acc;
}}"""
    else:
        kernel = f"""
{GID_DEFINE}
__kernel void unit(__global const uchar* text, __global const uint* offsets, __global const uint* table,
                   uint mask, __global uint* out, uint count) {{
    uint i = (uint)GID;
    if (i >= count) return;
    uint start = offsets[i], end = offsets[i + 1], acc = 0u;
{calls}
    out[i] = acc;
}}"""
    source = "\n".join(parts) + kernel
    return source, source.count("\n") + 1


def library(functions: int) -> str:
    """The synthetic device functions as an external library: `extern "C"` so a separate unit can call them by name."""
    source, _ = synthetic(functions, dialect="cuda")
    body = source[: source.index('extern "C" __global__')]
    return body.replace("__device__ unsigned f", 'extern "C" __device__ unsigned f')


def caller(used: int) -> str:
    """A kernel that calls the first `used` library functions through prototypes only."""
    prototypes = "\n".join(
        f'extern "C" __device__ unsigned f{n}(const unsigned char*, unsigned, unsigned, const unsigned*, unsigned);'
        for n in range(used)
    )
    calls = "\n".join(f"    acc ^= f{n}(text, start, end, table, mask);" for n in range(used))
    return f"""{prototypes}
extern "C" __global__ void unit(const unsigned char* text, const unsigned* offsets, const unsigned* table,
                                unsigned mask, unsigned* out, unsigned count) {{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned start = offsets[i], end = offsets[i + 1], acc = 0u;
{calls}
    out[i] = acc;
}}"""
