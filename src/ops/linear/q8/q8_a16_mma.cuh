#pragma once

// Q8G32 RowSplit x BF16 Tensor Core GEMM.
//
// out[M,N] = W[M,K] * x[K,N], where W stores one signed int8 code per element
// and one FP16 scale per 32 K elements. Raw codes and eight quantization groups' scales are
// staged with cp.async before dequantization into a swizzled BF16
// shared tile; x uses a two-stage cp.async pipeline. Tensor Cores execute
// m16n8k16 BF16 MMA with FP32 accumulation.

#include "ops/common/mma.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_shared.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

union alignas(16) Q8Bf16x8Bits {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(Q8Bf16x8Bits) == 16);

__device__ __forceinline__ int q8_g32_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Schedule>
struct Q8MmaIdentityRows {
    static constexpr int kOutputRowsPerCta = Schedule::kBlockRows;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return row0 + local_row;
    }
};
template <class Schedule, class Epilogue>
inline constexpr int q8_mma_shared_bytes = [] {
    constexpr int extra = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return static_cast<int>(Epilogue::template kSharedBytes<Schedule>);
        else
            return 0;
    }();
    return Schedule::kSharedBytes > extra ? Schedule::kSharedBytes : extra;
}();

template <class Cfg, bool Full, bool FullK, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q8_a16_mma_kernel(
    Q8LinearOperands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_begin, int token_count) {
    const auto* __restrict__ x      = operands.x;
    const auto* __restrict__ codes  = operands.codes;
    const auto* __restrict__ scales = operands.scales;
    const int m = operands.rows, k = Cfg::kStaticK > 0 ? Cfg::kStaticK : operands.k,
              n                     = token_begin + token_count;
    const int padded_k              = Cfg::kStaticK > 0 ? Cfg::kStaticK : operands.padded_k;
    constexpr int BM                = Cfg::kBlockRows;
    constexpr int BN                = Cfg::kBlockTokens;
    constexpr int BK                = Cfg::kBlockK;
    constexpr int WM                = Cfg::kWarpRows;
    constexpr int WN                = Cfg::kWarpTokens;
    constexpr int MT                = Cfg::kMmaRows;
    constexpr int NT                = Cfg::kMmaTokens;
    constexpr int KSUB              = Cfg::kMmaKSteps;
    constexpr int kOutputRowsPerCta = RowPolicy::kOutputRowsPerCta;

    struct OperandStorage {
        alignas(16) __nv_bfloat16 weights[Cfg::kWeightStages][BM * BK];
        alignas(16) __nv_bfloat16 activations[Cfg::kActivationStages][BN * BK];
        alignas(16) std::uint8_t codes[Cfg::kWeightStages][BM * BK];
        alignas(16) std::uint8_t scales[Cfg::kWeightStages][BM * Cfg::kScaleCacheBytes];
    };

    static_assert(sizeof(OperandStorage) == Cfg::kSharedBytes);
    auto* scratch = q8_shared_storage<q8_mma_shared_bytes<Cfg, Epilogue>>();
    auto& shared  = *reinterpret_cast<OperandStorage*>(scratch);
    auto& As      = shared.weights;
    auto& Bs      = shared.activations;
    auto& Cr      = shared.codes;
    auto& Sr      = shared.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::kWarpGridTokens;
    const int wn   = warp % Cfg::kWarpGridTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Group adjacent M-tiles across N-stripes so large-M weights stay hot in L2 while keeping
    // x reuse across the M-group (exact bijection over all gridDim.x * gridDim.y CTAs).
    int bx = static_cast<int>(blockIdx.x);
    int by = static_cast<int>(blockIdx.y);
    if constexpr (Cfg::kThreads <= 512) {
        const int gx = static_cast<int>(gridDim.x);
        const int gy = static_cast<int>(gridDim.y);
        if (gy >= 4 && gx >= 8) {
            constexpr int kGroupM      = 8;
            const int linear           = by * gx + bx;
            const int blocks_per_group = kGroupM * gy;
            const int group            = linear / blocks_per_group;
            const int first_m          = group * kGroupM;
            const int group_m          = min(kGroupM, gx - first_m);
            const int in_group         = linear - group * blocks_per_group;
            bx                         = first_m + (in_group % group_m);
            by                         = in_group / group_m;
        }
    }
    const int m0 = bx * kOutputRowsPerCta;
    const int n0 = token_begin + by * BN;
    const int kg = padded_k / 32;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::kThreads) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = k0 + k8 * 8;
            const int nn = n0 + nl;
            auto* dst    = &Bs[stage][nl * BK + q8_g32_swz64(nl, k8 * 8)];
            if constexpr (Full) {
                cp_async<16, Cfg::kActivationCache>(dst,
                                                    &x[static_cast<std::int64_t>(nn) * k + kk]);
            } else {
                const int valid =
                    nn < n && (FullK || kk < k) ? (FullK ? 16 : min(8, k - kk) * 2) : 0;
                infernix::ops::cp_async_zfill<16, Cfg::kPredicatedCache>(
                    dst,
                    &x[static_cast<std::int64_t>(nn < n ? nn : 0) * k + (FullK || kk < k ? kk : 0)],
                    valid);
            }
        }
    };

    constexpr int GROUPS            = BK / 32;
    constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
    static_assert(BK % 32 == 0 && (8 % GROUPS) == 0,
                  "a 16-code chunk must lie inside one Q8G32 group and the scale cache "
                  "must hold whole tiles");

    auto stage_w_codes = [&](int wbuf, int kt) {
        const int g0 = kt * GROUPS;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += Cfg::kThreads) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow  = row_policy.weight_row(m0, row, m);
            auto* dst       = &Cr[wbuf][row * BK + chunk * 16];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cfg::kWeightCache>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const bool valid_row  = (grow < m);
                const std::int64_t gi = static_cast<std::int64_t>(valid_row ? grow : 0) * kg + g0;
                infernix::ops::cp_async_zfill<16, Cfg::kPredicatedCache>(
                    dst, &codes[gi * 32 + chunk * 16], valid_row ? 16 : 0);
            }
        }
    };

    auto stage_w_scales = [&](int sbuf, int kt) {
        const int g0 = kt * GROUPS;
        for (int row = tid; row < BM; row += Cfg::kThreads) {
            const int grow = row_policy.weight_row(m0, row, m);
            auto* dst      = &Sr[sbuf][row * Cfg::kScaleCacheBytes];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cfg::kWeightCache>(dst, &scales[gi * 2]);
            } else if (kg % 8 == 0) {
                const bool valid_row   = grow < m;
                const int valid_scales = valid_row && g0 < kg ? min(8, kg - g0) : 0;
                const std::int64_t gi =
                    static_cast<std::int64_t>(valid_row ? grow : 0) * kg + g0;
                cp_async_zfill<16, Cfg::kPredicatedCache>(dst, &scales[gi * 2], valid_scales * 2);
            } else {
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    if (grow < m && g0 + pair * 2 < kg)
                        cp_async<4>(dst + pair * 4,
                                    scales + (static_cast<std::int64_t>(grow) * kg + g0) * 2 +
                                        pair * 4);
                    else
                        *reinterpret_cast<unsigned*>(dst + pair * 4) = 0;
                }
            }
        }
    };

    // Uses the exact same (row, chunk) mapping as stage_w_codes (16 codes per item, all within one
    // Q8G32 scale group), so each thread reads only the Cr bytes it staged itself via cp.async<16>.
    auto dequant_w = [&](int wbuf, int sbuf, int kt) {
        constexpr int kChunks16     = BK / 16;
        const int scale_tile_offset = (kt % SCALE_CACHE_TILES) * GROUPS * 2;
        constexpr float kBias       = 8388736.0f;
        const auto i8_to_f          = [](unsigned u, unsigned sel) {
            return __uint_as_float(__byte_perm(u, 0x4B000000U, sel)) - kBias;
        };
        for (int item = tid; item < BM * kChunks16; item += Cfg::kThreads) {
            const int row   = item / kChunks16;
            const int chunk = item - row * kChunks16;
            const int col   = chunk * 16;
            const int gg    = col >> 5;
            const float scale =
                (FullK || kt * BK + col < k)
                    ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                          &Sr[sbuf][row * Cfg::kScaleCacheBytes + scale_tile_offset + gg * 2])))
                    : 0.0f;
            const float scale_hi =
                (FullK || kt * BK + col + 8 < k) ? scale : 0.0f;
            const auto decode8 = [&](unsigned w0, unsigned w1, float s, int col_off) {
                const unsigned u0 = w0 ^ 0x80808080U;
                const unsigned u1 = w1 ^ 0x80808080U;
                Q8Bf16x8Bits out;
                out.pair[0] = __floats2bfloat162_rn(i8_to_f(u0, 0x7650U) * s,
                                                    i8_to_f(u0, 0x7651U) * s);
                out.pair[1] = __floats2bfloat162_rn(i8_to_f(u0, 0x7652U) * s,
                                                    i8_to_f(u0, 0x7653U) * s);
                out.pair[2] = __floats2bfloat162_rn(i8_to_f(u1, 0x7650U) * s,
                                                    i8_to_f(u1, 0x7651U) * s);
                out.pair[3] = __floats2bfloat162_rn(i8_to_f(u1, 0x7652U) * s,
                                                    i8_to_f(u1, 0x7653U) * s);
                store_vec(&As[wbuf][row * BK + q8_g32_swz64(row, col + col_off)], out.raw);
            };
            if constexpr (Cfg::kThreads <= 512) {
                const uint4 packed = *reinterpret_cast<const uint4*>(&Cr[wbuf][row * BK + col]);
                decode8(packed.x, packed.y, scale, 0);
                decode8(packed.z, packed.w, scale_hi, 8);
            } else {
                const uint2 lo = *reinterpret_cast<const uint2*>(&Cr[wbuf][row * BK + col]);
                decode8(lo.x, lo.y, scale, 0);
                const uint2 hi = *reinterpret_cast<const uint2*>(&Cr[wbuf][row * BK + col + 8]);
                decode8(hi.x, hi.y, scale_hi, 8);
            }
        }
    };

    auto run_mma = [&](int wbuf, int xbuf, int ks_begin, int ks_end) {
        unsigned af[2][MT][4];
        unsigned bf[2][NT][2];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = wm * WM + mi * 16 + a_rowoff;
                const int ac = ks * 16 + a_coloff;
                ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                            smem_addr(&As[wbuf][ar * BK + q8_g32_swz64(ar, ac)]));
            }
            if constexpr ((NT % 2) == 0) {
                const int b4_rowoff = ((lane >> 4) << 3) + b_rin;
#pragma unroll
                for (int ni = 0; ni < NT; ni += 2) {
                    const int br = wn * WN + ni * 8 + b4_rowoff;
                    const int bc = ks * 16 + b_koff;
                    ldmatrix_x4(bf[slot][ni][0], bf[slot][ni][1], bf[slot][ni + 1][0],
                                bf[slot][ni + 1][1],
                                smem_addr(&Bs[xbuf][br * BK + q8_g32_swz64(br, bc)]));
                }
            } else {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int br = wn * WN + ni * 8 + b_rin;
                    const int bc = ks * 16 + b_koff;
                    ldmatrix_x2(bf[slot][ni][0], bf[slot][ni][1],
                                smem_addr(&Bs[xbuf][br * BK + q8_g32_swz64(br, bc)]));
                }
            }
        };

        if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong)
            load_fragments(0, ks_begin);
#pragma unroll
        for (int ks = ks_begin; ks < ks_end; ++ks) {
            const int slot =
                Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong ? ((ks - ks_begin) & 1) : 0;
            if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::Serial) {
                load_fragments(0, ks);
            }
            if (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong && ks + 1 < ks_end) {
                load_fragments(slot ^ 1, ks + 1);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                             bf[slot][ni][0], bf[slot][ni][1]);
                }
            }
        }
    };

    const int nkt = padded_k / BK;
    if constexpr (Cfg::kWeightStages == 2) {
        // Double-buffered weights & scales: 1 __syncthreads() per K-tile. Each thread stages and
        // dequantizes its own 16-byte Cr chunk (overlapping dequant of kt+1 between the two MMA
        // halves of kt); scales for the next scale-cache block are prefetched with stage_x one
        // K-tile early so they are CTA-visible before dequant_w reads them.
        static_assert(KSUB >= 2 && (KSUB % 2) == 0);
        stage_w_codes(0, 0);
        stage_w_scales(0, 0);
        infernix::ops::cp_commit();
        infernix::ops::cp_wait<0>();
        __syncthreads();

        stage_x(0, 0);
        infernix::ops::cp_commit();
        if (1 < nkt) { stage_w_codes(1, 1); }
        infernix::ops::cp_commit();
        dequant_w(0, 0, 0);
        infernix::ops::cp_wait<1>();
        __syncthreads();

#pragma unroll Cfg::kKLoopUnroll
        for (int kt = 0; kt < nkt; ++kt) {
            const int cur   = kt & 1;
            const int nxt   = cur ^ 1;
            const int next  = kt + 1;
            const int next2 = kt + 2;

            if (next < nkt) { stage_x(nxt, next); }
            if (next2 < nkt && (next2 % SCALE_CACHE_TILES) == 0) {
                stage_w_scales((next2 / SCALE_CACHE_TILES) & 1, next2);
            }
            infernix::ops::cp_commit();

            run_mma(cur, cur, 0, KSUB / 2);

            infernix::ops::cp_wait<1>();
            if (next < nkt) { dequant_w(nxt, (next / SCALE_CACHE_TILES) & 1, next); }

            run_mma(cur, cur, KSUB / 2, KSUB);

            if (next2 < nkt) { stage_w_codes(cur, next2); }
            infernix::ops::cp_commit();

            infernix::ops::cp_wait<1>();
            __syncthreads();
        }
        infernix::ops::cp_wait<0>();
    } else {
        stage_x(0, 0);
        stage_w_codes(0, 0);
        stage_w_scales(0, 0);
        infernix::ops::cp_commit();

#pragma unroll Cfg::kKLoopUnroll
        for (int kt = 0; kt < nkt; ++kt) {
            const int stage = kt % Cfg::kStages;
            infernix::ops::cp_wait<0>();
            __syncthreads();

            dequant_w(0, 0, kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < nkt) {
                if constexpr (Cfg::kActivationStages == Cfg::kStages) {
                    stage_x(next % Cfg::kStages, next);
                }
                stage_w_codes(0, next);
                if ((next % SCALE_CACHE_TILES) == 0) { stage_w_scales(0, next); }
                infernix::ops::cp_commit();
            }

            run_mma(0, Cfg::kActivationStages == 1 ? 0 : stage, 0, KSUB);

            if constexpr (Cfg::kActivationStages == 1) {
                if (next < nkt) {
                    __syncthreads();
                    stage_x(0, next);
                    infernix::ops::cp_commit();
                }
            }
        }
    }

    const auto output_tile = linear_output_tile<kOutputRowsPerCta>(output, m0);
    if constexpr (requires {
                      epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m,
                                                               n);
                  }) {
        epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m, n);
    } else if constexpr (requires { output_tile.store_vector(0, 0, uint4{}); } &&
                         (Cfg::kThreads <= 512) && (kOutputRowsPerCta % 8 == 0) &&
                         (BN * (kOutputRowsPerCta + 8) * static_cast<int>(sizeof(__nv_bfloat16)) <=
                          q8_mma_shared_bytes<Cfg, Epilogue>)) {
        // Transpose BF16 outputs into [col][row] SMEM so global stores issue coalesced 16B uint4s
        // instead of strided 2B STG.U16 writes.
        constexpr int kOutStr = kOutputRowsPerCta + 8;
        __syncthreads();
        auto* out_s = reinterpret_cast<__nv_bfloat16*>(scratch);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int lr0 = wm * WM + mi * 16 + gid;
            const int r0  = m0 + lr0;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int lc0  = wn * WN + ni * 8 + 2 * lid;
                const int c0   = n0 + lc0;
                const float* a = acc[mi][ni];
                if (Full || (r0 < m && c0 < n))
                    out_s[lc0 * kOutStr + lr0] =
                        __float2bfloat16_rn(epilogue.apply(r0, c0, a[0]));
                if (Full || (r0 < m && c0 + 1 < n))
                    out_s[(lc0 + 1) * kOutStr + lr0] =
                        __float2bfloat16_rn(epilogue.apply(r0, c0 + 1, a[1]));
                if (Full || (r0 + 8 < m && c0 < n))
                    out_s[lc0 * kOutStr + lr0 + 8] =
                        __float2bfloat16_rn(epilogue.apply(r0 + 8, c0, a[2]));
                if (Full || (r0 + 8 < m && c0 + 1 < n))
                    out_s[(lc0 + 1) * kOutStr + lr0 + 8] =
                        __float2bfloat16_rn(epilogue.apply(r0 + 8, c0 + 1, a[3]));
            }
        }
        __syncthreads();
        constexpr int kPacksPerCol = kOutputRowsPerCta / 8;
        constexpr int kPacks       = BN * kPacksPerCol;
        for (int pack = tid; pack < kPacks; pack += Cfg::kThreads) {
            const int lc = pack / kPacksPerCol;
            const int lr = (pack - lc * kPacksPerCol) * 8;
            const int c  = n0 + lc;
            const int r  = m0 + lr;
            if (Full || (c < n && r < m)) {
                if (Full || ((m & 7) == 0 && r + 8 <= m)) {
                    output_tile.store_vector(r, c, load_vec<uint4>(&out_s[lc * kOutStr + lr]));
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (r + i < m) {
                            output_tile.store(r + i, c,
                                              __bfloat162float(out_s[lc * kOutStr + lr + i]));
                        }
                    }
                }
            }
        }
    } else {
        const auto store = [&](int row, int token, float value) {
            if (Full || (row < m && token < n))
                output_tile.store(row, token, epilogue.apply(row, token, value));
        };
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = m0 + wm * WM + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0   = n0 + wn * WN + ni * 8 + 2 * lid;
                const float* a = acc[mi][ni];
                store(r0, c0, a[0]);
                store(r0, c0 + 1, a[1]);
                store(r0 + 8, c0, a[2]);
                store(r0 + 8, c0 + 1, a[3]);
            }
        }
    }
}
} // namespace infernix::ops::detail
