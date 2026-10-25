#pragma once

// Device-side building blocks and their compile-time composition.
//
// A block is one step of a layer's work written against a virtual CTA index, so one kernel can run
// several blocks side by side (and a persistent kernel can later chain them). A block type B has:
//   static constexpr int kThreads;      the threads its CTA body uses (threads 0 .. kThreads - 1)
//   int ctas() const;                   the CTAs it needs (host)
//   __device__ void run(int cta) const; its work for virtual CTA `cta` in [0, ctas())
// A body synchronizes with cta_sync<kThreads>() instead of __syncthreads(), so it may run in a CTA
// wider than itself, and waits for its launch's dependencies itself (pdl::wait_for_dependencies or
// pdl::enter*). The standalone kernels of a body run it on blockIdx.x, so they keep their bits.

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_runtime.h>

#include <algorithm>

namespace infernix::ops::blocks {

// The barrier of a block's threads (warps 0 .. Threads / 32 - 1): barrier 1, so a composed kernel's
// other blocks never take part.
template <int Threads>
__device__ __forceinline__ void cta_sync() {
    static_assert(Threads % 32 == 0);
    asm volatile("bar.sync 1, %0;" ::"n"(Threads) : "memory");
}

// Threads past a block's kThreads only wait for the launch's dependencies (pdl's consumer protocol:
// every thread waits before it exits).
template <class Block>
__device__ __forceinline__ void run_block(const Block& block, int cta) {
    if (static_cast<int>(threadIdx.x) < Block::kThreads) {
        block.run(cta);
    } else {
        pdl::wait_for_dependencies();
    }
}

// Blocks run side by side: the first First.ctas() CTAs run First, the next ones Rest.
template <class First, class... Rest>
struct Chain {
    static constexpr int kThreads = std::max({First::kThreads, Rest::kThreads...});
    First first;
    Chain<Rest...> rest;
    int first_ctas;

    Chain(const First& f, const Rest&... r) : first(f), rest(r...), first_ctas(f.ctas()) {}
    [[nodiscard]] int ctas() const { return first_ctas + rest.ctas(); }

    __device__ __forceinline__ void run(int cta) const {
        if (cta < first_ctas) {
            run_block(first, cta);
        } else {
            rest.run(cta - first_ctas);
        }
    }
};

template <class Last>
struct Chain<Last> {
    static constexpr int kThreads = Last::kThreads;
    Last first;
    int first_ctas;

    explicit Chain(const Last& last) : first(last), first_ctas(last.ctas()) {}
    [[nodiscard]] int ctas() const { return first_ctas; }

    __device__ __forceinline__ void run(int cta) const { run_block(first, cta); }
};

template <class Composition>
__global__ void __launch_bounds__(Composition::kThreads) composed_kernel(const Composition composition) {
    composition.run(static_cast<int>(blockIdx.x));
}

// One launch of independent blocks, as a programmatic dependent of the stream's previous kernel.
template <class... Blocks>
void launch_composed(cudaStream_t stream, const Blocks&... blocks) {
    const Chain<Blocks...> chain(blocks...);
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(chain.ctas())), dim3(Chain<Blocks...>::kThreads), 0, stream},
                                    composed_kernel<Chain<Blocks...>>, chain));
}

} // namespace infernix::ops::blocks
