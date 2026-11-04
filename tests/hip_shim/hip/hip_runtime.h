// A stand-in for <hip/hip_runtime.h> so that HIP C++ compiles for an AMD GPU with clang alone, with no ROCm install:
// `clang -x hip --offload-arch=gfx942 --offload-device-only -nogpulib -nogpuinc -I <this folder> -c unit.hip`.
// It carries the names the generated code uses, over clang's AMDGPU builtins. The result is a code object that is not
// linked (the math library is ROCm's), and it has never run: Sirius is the first machine that can run it.
#pragma once

#define __global__ __attribute__((amdgpu_kernel, global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))

extern "C" __device__ int printf(const char*, ...);

struct _Dim3 { unsigned x, y, z; };
#define AS4 __attribute__((address_space(4)))

__device__ __forceinline__ unsigned _packet16(int i) { return ((AS4 const unsigned short*)__builtin_amdgcn_dispatch_ptr())[i]; }
__device__ __forceinline__ unsigned _packet32(int i) { return ((AS4 const unsigned*)__builtin_amdgcn_dispatch_ptr())[i]; }
__device__ __forceinline__ _Dim3 _threadIdx() {
  return {__builtin_amdgcn_workitem_id_x(), __builtin_amdgcn_workitem_id_y(), __builtin_amdgcn_workitem_id_z()};
}
__device__ __forceinline__ _Dim3 _blockIdx() {
  return {__builtin_amdgcn_workgroup_id_x(), __builtin_amdgcn_workgroup_id_y(), __builtin_amdgcn_workgroup_id_z()};
}
__device__ __forceinline__ _Dim3 _blockDim() { return {_packet16(2), _packet16(3), _packet16(4)}; }
__device__ __forceinline__ _Dim3 _gridDim() {
  return {_packet32(3) / _packet16(2), _packet32(4) / _packet16(3), _packet32(5) / _packet16(4)};
}
#define threadIdx _threadIdx()
#define blockIdx _blockIdx()
#define blockDim _blockDim()
#define gridDim _gridDim()

__device__ __forceinline__ void __syncthreads() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
__device__ __forceinline__ int __syncthreads_count(int p) { __syncthreads(); return p; }
__device__ __forceinline__ int __syncthreads_and(int p) { __syncthreads(); return p; }
__device__ __forceinline__ int __syncthreads_or(int p) { __syncthreads(); return p; }
__device__ __forceinline__ void __threadfence_block() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup"); }
__device__ __forceinline__ void __threadfence() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent"); }
__device__ __forceinline__ void __threadfence_system() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, ""); }

__device__ __forceinline__ unsigned _lane() { return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)); }
__device__ __forceinline__ unsigned long long __ballot(int p) { return __builtin_amdgcn_ballot_w64(p != 0); }
__device__ __forceinline__ int __all(int p) { return __ballot(!p) == 0; }
__device__ __forceinline__ int __any(int p) { return __ballot(p) != 0; }
template <class T> __device__ __forceinline__ T _permute(T v, unsigned lane) {
  static_assert(sizeof(T) == 4, "the shim shuffles 32-bit values");
  int bits = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(T, __builtin_amdgcn_ds_bpermute(lane << 2, bits));
}
template <class T> __device__ __forceinline__ T __shfl(T v, int lane, int width = 64) {
  return _permute(v, (_lane() / width) * width + lane % width);
}
template <class T> __device__ __forceinline__ T __shfl_xor(T v, int mask, int width = 64) {
  unsigned source = _lane() ^ mask;
  return _permute(v, source / width == _lane() / width ? source : _lane());
}
template <class T> __device__ __forceinline__ T __shfl_up(T v, int delta, int width = 64) {
  return _permute(v, _lane() % width >= (unsigned)delta ? _lane() - delta : _lane());
}
template <class T> __device__ __forceinline__ T __shfl_down(T v, int delta, int width = 64) {
  return _permute(v, _lane() % width + delta < (unsigned)width ? _lane() + delta : _lane());
}
// ROCm's __match_any_sync and __match_all_sync (amd_warp_sync_functions.h), written over the permute for 32-bit values.
template <class T> __device__ __forceinline__ unsigned long long __match_any_sync(unsigned long long mask, T v) {
  unsigned long long same = 0;
  for (unsigned lane = 0; lane < 64; ++lane) same |= (unsigned long long)(_permute(v, lane) == v) << lane;
  return same & mask;
}
template <class T> __device__ __forceinline__ unsigned long long __match_all_sync(unsigned long long mask, T v, int* pred) {
  *pred = __match_any_sync(mask, v) == mask;
  return *pred ? mask : 0;
}

__device__ __forceinline__ int __popc(unsigned x) { return __builtin_popcount(x); }
__device__ __forceinline__ int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
__device__ __forceinline__ int __clz(int x) { return x ? __builtin_clz(x) : 32; }
__device__ __forceinline__ int __clzll(long long x) { return x ? __builtin_clzll(x) : 64; }
__device__ __forceinline__ int __ffs(int x) { return __builtin_ffs(x); }
__device__ __forceinline__ int __ffsll(long long x) { return __builtin_ffsll(x); }
__device__ __forceinline__ unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
__device__ __forceinline__ unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }

__device__ __forceinline__ unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
__device__ __forceinline__ float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
__device__ __forceinline__ long long __double_as_longlong(double x) { return __builtin_bit_cast(long long, x); }
__device__ __forceinline__ double __longlong_as_double(long long x) { return __builtin_bit_cast(double, x); }

#define _ATOMIC(name, builtin) \
  template <class T> __device__ __forceinline__ T name(T* p, T v) { return builtin(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
_ATOMIC(atomicAdd, __hip_atomic_fetch_add)
_ATOMIC(atomicSub, __hip_atomic_fetch_sub)
_ATOMIC(atomicAnd, __hip_atomic_fetch_and)
_ATOMIC(atomicOr, __hip_atomic_fetch_or)
_ATOMIC(atomicXor, __hip_atomic_fetch_xor)
_ATOMIC(atomicMax, __hip_atomic_fetch_max)
_ATOMIC(atomicMin, __hip_atomic_fetch_min)
template <class T> __device__ __forceinline__ T atomicExch(T* p, T v) {
  return __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
template <class T> __device__ __forceinline__ T atomicCAS(T* p, T expected, T v) {
  __hip_atomic_compare_exchange_strong(p, &expected, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return expected;
}
__device__ __forceinline__ unsigned atomicInc(unsigned* p, unsigned limit) {
  return __builtin_amdgcn_atomic_inc32(p, limit, __ATOMIC_RELAXED, "agent");
}
__device__ __forceinline__ unsigned atomicDec(unsigned* p, unsigned limit) {
  return __builtin_amdgcn_atomic_dec32(p, limit, __ATOMIC_RELAXED, "agent");
}

__device__ __forceinline__ float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
#define expf __builtin_expf
#define exp2f __builtin_exp2f
#define exp10f __builtin_exp10f
#define expm1f __builtin_expm1f
#define logf __builtin_logf
#define log2f __builtin_log2f
#define log10f __builtin_log10f
#define log1pf __builtin_log1pf
#define sqrtf __builtin_sqrtf
#define cbrtf __builtin_cbrtf
#define sinf __builtin_sinf
#define cosf __builtin_cosf
#define tanhf __builtin_tanhf
#define erff __builtin_erff
#define fabsf __builtin_fabsf
#define floorf __builtin_floorf
#define ceilf __builtin_ceilf
#define rintf __builtin_rintf
#define truncf __builtin_truncf
#define powf __builtin_powf
#define fmaf __builtin_fmaf
