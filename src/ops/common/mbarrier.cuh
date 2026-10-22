#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace infernix::ops {

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
}

// A consumer warp's release of a pipeline stage it read with generic-proxy loads (ld.shared,
// ldmatrix) back to a TMA producer, whose next cp.async.bulk.tensor write into the stage is an
// async-proxy access. PTX orders generic-proxy accesses before later async-proxy accesses of the same
// shared memory only through fence.proxy.async, so every lane fences its reads, the warp converges,
// and lane 0 arrives on the stage's empty barrier. Without the fence a refill can land in a stage a
// lane is still reading (docs/maintainer/qwen3_8-flash-next-design.md, "TMA tile stage release").
__device__ __forceinline__ void cta_mbarrier_release_stage(std::uint64_t* empty, int lane) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) { cta_mbarrier_arrive(empty); }
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

} // namespace infernix::ops
