"""Which inline PTX assembles on which target, validated by ptxas through NVRTC (no GPU needed).

Checks the claims the lowering registry rests on: redux.sync needs sm_80, block-scaled mma.sync needs sm_120a, and the
fp4 conversion instructions are architecture-specific. Run: python ptx_asm.py <nvidia-cu13 dir> <out.json>
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.paths import RESULTS
from krn.runtime.nvrtc import Nvidia, NvidiaError

log = logging.getLogger("ptx_asm")

ARCHS = ("sm_75", "sm_80", "sm_90a", "sm_100a", "sm_120", "sm_120a", "sm_121a")


@dataclass(frozen=True)
class Probe:
    name: str
    source: str
    note: str


REDUX = Probe(
    "redux.sync.add.u32",
    """extern "C" __global__ void k(const unsigned* x, unsigned* out) {
    unsigned v = x[threadIdx.x], r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    out[threadIdx.x] = r;
}""",
    "the NVIDIA lowering of warp.sum for integers",
)

NVFP4_MMA = Probe(
    "mma.sync m16n8k64 kind::mxf4nvf4 block_scale ue4m3 (NVFP4)",
    """extern "C" __global__ void k(float* out, const unsigned* a, const unsigned* b, unsigned sa, unsigned sb) {
    float d0, d1, d2, d3, c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sa), "r"(sb));
    out[threadIdx.x * 4 + 0] = d0; out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2; out[threadIdx.x * 4 + 3] = d3;
}""",
    "the NVIDIA lowering of QTile[NVFP4].mma",
)

MXFP8_MMA = Probe(
    "mma.sync m16n8k32 kind::mxf8f6f4 block_scale e4m3 ue8m0 (MXFP8)",
    """extern "C" __global__ void k(float* out, const unsigned* a, const unsigned* b, unsigned sa, unsigned sb) {
    float d0, d1, d2, d3, c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sa), "r"(sb));
    out[threadIdx.x * 4 + 0] = d0; out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2; out[threadIdx.x * 4 + 3] = d3;
}""",
    "the NVIDIA lowering of QTile[MXFP8].mma",
)

MMA_F16 = Probe(
    "mma.sync m16n8k16 f16 (unscaled baseline)",
    """extern "C" __global__ void k(float* out, const unsigned* a, const unsigned* b) {
    float d0, d1, d2, d3, c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    out[threadIdx.x * 4 + 0] = d0; out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2; out[threadIdx.x * 4 + 3] = d3;
}""",
    "the unscaled tensor-core path every tile kernel falls back to",
)

CVT_FP4 = Probe(
    "cvt.rn.satfinite.e2m1x2.f32 (fp4 encode)",
    """extern "C" __global__ void k(const float* x, unsigned char* out) {
    unsigned short packed;
    asm volatile("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; cvt.u16.u8 %0, b; }"
                 : "=h"(packed) : "f"(x[2 * threadIdx.x]), "f"(x[2 * threadIdx.x + 1]));
    out[threadIdx.x] = (unsigned char)packed;
}""",
    "the NVIDIA lowering of quantize to an E2M1 block",
)

CVT_FP8 = Probe(
    "cvt.rn.satfinite.e4m3x2.f32 (fp8 encode)",
    """extern "C" __global__ void k(const float* x, unsigned short* out) {
    unsigned short packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed)
                 : "f"(x[2 * threadIdx.x]), "f"(x[2 * threadIdx.x + 1]));
    out[threadIdx.x] = packed;
}""",
    "quantize to E4M3 (OCP)",
)

PROBES = (REDUX, MMA_F16, NVFP4_MMA, MXFP8_MMA, CVT_FP8, CVT_FP4)


def first_error(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip() and "failed with status" not in line]
    return lines[0][:140] if lines else "failed"


def main(lib_dir: Path, out: Path) -> None:
    nv = Nvidia(lib_dir)
    log.info("NVRTC %s.%s (ptxas through cubin generation)", *nv.nvrtc_version)
    table: dict[str, dict[str, str]] = {}
    for probe in PROBES:
        table[probe.name] = {}
        for arch in ARCHS:
            try:
                nv.compile(probe.source, arch, "cubin")
                table[probe.name][arch] = "ok"
            except NvidiaError as error:
                table[probe.name][arch] = "error: " + first_error(str(error))
        log.info("%s", probe.name)
        for arch, verdict in table[probe.name].items():
            log.info("    %-8s %s", arch, verdict)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]), (Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "ptx_asm.json"))
