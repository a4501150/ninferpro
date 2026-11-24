// Exact top-k selection over per-column FP32 scores (exact_topk_select.h): the radix select
// moved out of the QSA select stage so both pooled-key indexers share one facility. The
// arithmetic and the launch topology are unchanged from the code QSA qualified.

#include "ops/common/exact_topk_select.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::ops::detail {
namespace {

constexpr unsigned kFullMask = 0xFFFFFFFFU;

struct SelectShared {
    using Scan = cub::BlockScan<int, kTopkSelectThreads>;
    typename Scan::TempStorage scan;
    unsigned radix[256];
    int pick_digit, pick_above, pick_count; // radix_threshold's digit
    int bin, bin_above, bin_count;          // the score bin holding the top-th score
    unsigned candidates;                    // keys appended to this CTA's candidate buffer
    unsigned long long and_key, or_key;     // over this CTA's share of the bin's keys
    int last;                               // this CTA arrived last at its column
};

// The monotone bin of a finite score: bits [30:20] of the ordered bits for non-negative scores
// (their top bit is set), bits [31:21] over the whole float order otherwise.
template <int BinShift>
__device__ __forceinline__ int score_bin(float score) {
    return static_cast<int>((ordered_score_bits(score) >> BinShift) & (kTopkScoreBins - 1));
}

// First block of slice s of S over n blocks; multiples of 4 so float4 loads stay aligned.
__host__ __device__ __forceinline__ int slice_begin(int n, int s, int slices) {
    if (s >= slices) { return n; }
    return static_cast<int>((static_cast<std::int64_t>(n) * s / slices) & ~std::int64_t{3});
}

// Walks scores [lo, hi) of a column (lo a multiple of 4) in chunks of 128, each warp one chunk at
// a time with kWalkBatch chunks' loads in flight. Lane l sees scores base..base+3, base = lo +
// 128 chunk + 4 l; `valid` masks those < hi. Calls are warp-uniform.
constexpr int kWalkBatch = 4; // 128-score chunks each warp keeps in flight

template <int Warps, class Visit>
__device__ __forceinline__ void walk_scores(const float* column, int lo, int hi, Visit&& visit) {
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int chunks = (hi - lo + 127) / 128;
    for (int first = warp; first < chunks; first += Warps * kWalkBatch) {
        float4 v[kWalkBatch];
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int base = lo + (first + u * Warps) * 128 + 4 * lane;
            v[u]           = base < hi ? *reinterpret_cast<const float4*>(column + base)
                                       : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int chunk = first + u * Warps;
            if (chunk >= chunks) { break; }
            const int base    = lo + chunk * 128 + 4 * lane;
            const int left    = hi - base;
            const unsigned ok = left >= 4 ? 0xFU : left > 0 ? (1U << left) - 1 : 0U;
            visit(chunk, base, v[u], ok);
        }
    }
}

// Lanes 8m..8m+7 hold four flags each of scores 32m..32m+31 of a chunk; lane 8m returns the word.
__device__ __forceinline__ unsigned chunk_word(unsigned nibble, int lane) {
    unsigned word = nibble << (4 * (lane & 7));
    word |= __shfl_xor_sync(kFullMask, word, 1);
    word |= __shfl_xor_sync(kFullMask, word, 2);
    word |= __shfl_xor_sync(kFullMask, word, 4);
    return word;
}

template <int BinShift>
__device__ void score_histogram(const float* column, int lo, int hi, unsigned* hist) {
    walk_scores<kTopkSelectThreads / 32>(column, lo, hi, [&](int, int, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) { atomicAdd(hist + score_bin<BinShift>(s[k]), 1U); }
        }
    });
}

// The bin holding the top-th highest score: sh.bin, the count above it and its own count.
__device__ void find_score_bin(const unsigned* hist, int top, SelectShared& sh) {
    const int tid = static_cast<int>(threadIdx.x);
    int count[4], local = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        count[k] = static_cast<int>(hist[kTopkScoreBins - 1 - 4 * tid - k]);
        local += count[k];
    }
    int above = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, above);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (above < top && top <= above + count[k]) {
            sh.bin       = kTopkScoreBins - 1 - 4 * tid - k;
            sh.bin_above = above;
            sh.bin_count = count[k];
        }
        above += count[k];
    }
    __syncthreads();
}

// One pass over scores [lo, hi): scores above the bin set their bit in `bitmap` (word w holds
// scores lo + 32 w ..), the bin's scores append their order keys to `candidates` (up to
// kTopkCandidateCapacity; `count` counts them all) and fold into sh.and_key / sh.or_key.
template <int BinShift>
__device__ void filter_scores(const float* column, int lo, int hi, int bin, unsigned* bitmap,
                              std::uint64_t* candidates, unsigned* count, SelectShared& sh) {
    const int lane             = static_cast<int>(threadIdx.x) % 32;
    unsigned long long and_key = ~0ULL, or_key = 0;
    walk_scores<kTopkSelectThreads / 32>(
        column, lo, hi, [&](int chunk, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
            unsigned above = 0, inside = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((ok >> k) & 1U) {
                    const int b = score_bin<BinShift>(s[k]);
                    above |= static_cast<unsigned>(b > bin) << k;
                    inside |= static_cast<unsigned>(b == bin) << k;
                }
            }
            const unsigned word = chunk_word(above, lane);
            if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] = word; }
            const int mine = __popc(inside);
            if (__any_sync(kFullMask, mine != 0)) {
                int inclusive = mine;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int other = __shfl_up_sync(kFullMask, inclusive, o);
                    if (lane >= o) { inclusive += other; }
                }
                unsigned start = 0;
                if (lane == 31) { start = atomicAdd(count, static_cast<unsigned>(inclusive)); }
                start    = __shfl_sync(kFullMask, start, 31);
                int slot = static_cast<int>(start) + inclusive - mine;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if ((inside >> k) & 1U) {
                        const std::uint64_t key = score_id_order_key(s[k], base + k);
                        if (slot < kTopkCandidateCapacity) { candidates[slot] = key; }
                        ++slot;
                        and_key &= key;
                        or_key |= key;
                    }
                }
            }
        });
    const unsigned and_hi = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key >> 32));
    const unsigned and_lo = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key));
    const unsigned or_hi  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key >> 32));
    const unsigned or_lo  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key));
    if (lane == 0) {
        atomicAnd(&sh.and_key, (static_cast<unsigned long long>(and_hi) << 32) | and_lo);
        atomicOr(&sh.or_key, (static_cast<unsigned long long>(or_hi) << 32) | or_lo);
    }
}

// Keys of the candidate buffer.
struct CandidateKeys {
    const std::uint64_t* keys;
    int count;

    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        for (int i = static_cast<int>(threadIdx.x); i < count; i += kTopkSelectThreads) {
            f(keys[i]);
        }
    }
};

// Keys of the bin's scores, read from the column (when the bin exceeds the candidate buffer).
template <int BinShift>
struct BinKeys {
    const float* column;
    int n;
    int bin;

    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        walk_scores<kTopkSelectThreads / 32>(
            column, 0, n, [&](int, int base, float4 v, unsigned ok) {
                const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (((ok >> k) & 1U) && score_bin<BinShift>(s[k]) == bin) {
                        f(score_id_order_key(s[k], base + k));
                    }
                }
            });
    }
};

// Warp 0: the digit holding the `remaining`-th highest key among sh.radix's 256 bins.
__device__ __forceinline__ void choose_digit(SelectShared& sh, int remaining) {
    const int lane = static_cast<int>(threadIdx.x);
    int count[8], local = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        count[k] = static_cast<int>(sh.radix[255 - 8 * lane - k]);
        local += count[k];
    }
    int inclusive = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kFullMask, inclusive, o);
        if (lane >= o) { inclusive += other; }
    }
    int above = inclusive - local;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (above < remaining && remaining <= above + count[k]) {
            sh.pick_digit = 255 - 8 * lane - k;
            sh.pick_above = above;
            sh.pick_count = count[k];
        }
        above += count[k];
    }
}

// The `needed`-th highest of a set of distinct 64-bit keys, as a threshold T: exactly `needed` keys
// of the set are >= T. Radix select, 8 bits a pass from the highest bit in which the keys differ
// (and_keys / or_keys over the set), stopping once a digit's keys are all taken. Block-wide.
template <class Keys>
__device__ std::uint64_t radix_threshold(const Keys& keys, int needed, std::uint64_t and_keys,
                                         std::uint64_t or_keys, SelectShared& sh) {
    const std::uint64_t differ = and_keys ^ or_keys;
    if (differ == 0) { return or_keys; } // a single key
    const int top      = 63 - __clzll(static_cast<long long>(differ));
    std::uint64_t mask = top == 63 ? 0 : ~((std::uint64_t{2} << top) - 1); // bits every key shares
    std::uint64_t prefix = or_keys & mask;
    int remaining        = needed;
    int high             = top;
    for (;;) {
        const int low             = high >= 7 ? high - 7 : 0;
        const unsigned digit_mask = (2U << (high - low)) - 1;
        for (int i = static_cast<int>(threadIdx.x); i < 256; i += kTopkSelectThreads) {
            sh.radix[i] = 0;
        }
        __syncthreads();
        keys.for_each([&](std::uint64_t key) {
            if ((key & mask) == prefix) {
                atomicAdd(&sh.radix[static_cast<unsigned>(key >> low) & digit_mask], 1U);
            }
        });
        __syncthreads();
        if (threadIdx.x < 32) { choose_digit(sh, remaining); }
        __syncthreads();
        remaining -= sh.pick_above;
        const bool done = sh.pick_count == remaining || low == 0;
        prefix |= static_cast<std::uint64_t>(sh.pick_digit) << low;
        mask |= static_cast<std::uint64_t>(digit_mask) << low;
        __syncthreads(); // every thread has read sh.pick_* before the next pass or caller reuses sh
        if (done) { return prefix; }
        high = low - 1;
    }
}

// Writes id_base + (bit positions of words[0..nwords)) in ascending order to out. Block-wide.
__device__ void compact_words(const unsigned* words, int nwords, int id_base, std::int32_t* out,
                              SelectShared& sh) {
    const int per   = (nwords + kTopkSelectThreads - 1) / kTopkSelectThreads;
    const int begin = min(nwords, static_cast<int>(threadIdx.x) * per);
    const int end   = min(nwords, begin + per);
    int local       = 0;
    for (int w = begin; w < end; ++w) { local += __popc(words[w]); }
    int offset = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, offset);
    for (int w = begin; w < end; ++w) {
        unsigned bits = words[w];
        while (bits != 0) {
            out[offset++] = id_base + 32 * w + (__ffs(static_cast<int>(bits)) - 1);
            bits &= bits - 1;
        }
    }
}

// Grid (slices, columns of the group). Each CTA adds the histogram of its slice
// of the column into the column's global histogram (zeroed by the launch); the last CTA to
// arrive finds the bin, filters the column (bitmap above the bin, candidates in it), selects the
// candidates by radix and writes the ids in ascending order.
template <int BinShift>
__global__ void __launch_bounds__(kTopkSelectThreads)
    topk_select_kernel(const std::int32_t* __restrict__ positions, int ratio, int top,
                       int group_begin, const float* __restrict__ scores, int score_stride,
                       unsigned* __restrict__ histograms, unsigned* __restrict__ arrivals,
                       std::int32_t* __restrict__ selected, std::int32_t* __restrict__ counts) {
    pdl::enter();
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    __shared__ SelectShared sh;
    const int c = static_cast<int>(blockIdx.y), t = group_begin + c;
    const int slice = static_cast<int>(blockIdx.x), slices = static_cast<int>(gridDim.x);
    const int n = (positions[t] + 1) / ratio;
    if (n <= top) {
        if (slice == 0 && threadIdx.x == 0) { counts[t] = -1; }
        return;
    }
    const float* column = scores + static_cast<std::size_t>(c) * score_stride;
    auto* hist = reinterpret_cast<unsigned*>(dynamic_smem); // [kTopkScoreBins], then candidates
    for (int i = static_cast<int>(threadIdx.x); i < kTopkScoreBins; i += kTopkSelectThreads) {
        hist[i] = 0;
    }
    __syncthreads();
    score_histogram<BinShift>(column, slice_begin(n, slice, slices),
                              slice_begin(n, slice + 1, slices), hist);
    __syncthreads();
    unsigned* column_hist = histograms + static_cast<std::size_t>(c) * kTopkScoreBins;
    for (int i = static_cast<int>(threadIdx.x); i < kTopkScoreBins; i += kTopkSelectThreads) {
        if (hist[i] != 0) { atomicAdd(column_hist + i, hist[i]); }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        sh.last = atomicAdd(arrivals + c, 1U) + 1 == static_cast<unsigned>(slices);
    }
    __syncthreads();
    if (sh.last == 0) { return; }
    __threadfence();

    for (int i = static_cast<int>(threadIdx.x); i < kTopkScoreBins; i += kTopkSelectThreads) {
        hist[i] = __ldcg(column_hist + i);
    }
    if (threadIdx.x == 0) {
        sh.candidates = 0;
        sh.and_key    = ~0ULL;
        sh.or_key     = 0;
    }
    __syncthreads();
    find_score_bin(hist, top, sh);
    const int bin = sh.bin, needed = top - sh.bin_above, in_bin = sh.bin_count;
    auto* candidates = reinterpret_cast<std::uint64_t*>(dynamic_smem); // the histogram is read
    auto* bitmap =
        reinterpret_cast<unsigned*>(dynamic_smem + sizeof(std::uint64_t) * kTopkCandidateCapacity);
    filter_scores<BinShift>(column, 0, n, bin, bitmap, candidates, &sh.candidates, sh);
    __syncthreads();
    const bool buffered         = in_bin <= kTopkCandidateCapacity;
    const std::uint64_t and_key = sh.and_key, or_key = sh.or_key;
    const std::uint64_t threshold =
        buffered ? radix_threshold(CandidateKeys{candidates, in_bin}, needed, and_key, or_key, sh)
                 : radix_threshold(BinKeys<BinShift>{column, n, bin}, needed, and_key, or_key, sh);
    if (buffered) {
        for (int i = static_cast<int>(threadIdx.x); i < in_bin; i += kTopkSelectThreads) {
            const std::uint64_t key = candidates[i];
            if (key >= threshold) {
                const int id = id_from_order_key(key);
                atomicOr(bitmap + id / 32, 1U << (id % 32));
            }
        }
    } else {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        walk_scores<kTopkSelectThreads / 32>(
            column, 0, n, [&](int chunk, int base, float4 v, unsigned ok) {
                const float s[4] = {v.x, v.y, v.z, v.w};
                unsigned chosen  = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (((ok >> k) & 1U) && score_bin<BinShift>(s[k]) == bin &&
                        score_id_order_key(s[k], base + k) >= threshold) {
                        chosen |= 1U << k;
                    }
                }
                const unsigned word = chunk_word(chosen, lane);
                if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] |= word; }
            });
    }
    __syncthreads();
    compact_words(bitmap, (n + 31) / 32, 0, selected + static_cast<std::size_t>(t) * top, sh);
    if (threadIdx.x == 0) { counts[t] = top; }
}

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int multiprocessors() {
    static const int count = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

// Opts a kernel into `bytes` of dynamic shared memory beyond the default 48 KiB, once per size.
template <class Kernel>
void reserve_shared(Kernel kernel, std::size_t bytes, std::atomic<std::size_t>& reserved) {
    if (bytes <= 48 * 1024 || bytes <= reserved.load(std::memory_order_acquire)) { return; }
    int device = 0, limit = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (bytes > static_cast<std::size_t>(limit)) {
        throw std::runtime_error(
            "exact_topk_select: max bounds need more shared memory than a CTA has");
    }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(bytes)));
    std::size_t seen = reserved.load(std::memory_order_relaxed);
    while (seen < bytes &&
           !reserved.compare_exchange_weak(seen, bytes, std::memory_order_acq_rel)) {}
}

std::atomic<std::size_t> g_topk_reserved{0};

} // namespace

std::size_t topk_select_state_bytes(std::int32_t columns) {
    if (columns <= 0) {
        throw std::invalid_argument("exact_topk_select: columns must be positive");
    }
    // kTopkScoreBins histogram words, then one arrival counter, per column.
    return align256(sizeof(std::uint32_t) * static_cast<std::size_t>(columns) *
                    (kTopkScoreBins + 1));
}

void exact_topk_select(const std::int32_t* positions, int ratio, int top, int group_begin,
                       int columns, const float* scores, int stride, std::uint32_t* state,
                       std::int32_t* selected, std::int32_t* counts, int max_bounds,
                       bool non_negative_scores, cudaStream_t stream) {
    if (positions == nullptr || scores == nullptr || state == nullptr || selected == nullptr ||
        counts == nullptr) {
        throw std::invalid_argument("exact_topk_select: missing buffers");
    }
    if (columns <= 0 || ratio <= 0 || top <= 0 || stride < max_bounds + 1 || max_bounds < 0) {
        throw std::invalid_argument("exact_topk_select: invalid geometry");
    }
    const std::size_t state_size = topk_select_state_bytes(columns);
    CUDA_CHECK(cudaMemsetAsync(state, 0, state_size, stream));

    // Dynamic shared memory: candidates (aliasing the histogram) and the column bitmap.
    const std::size_t smem =
        sizeof(std::uint64_t) * kTopkCandidateCapacity +
        sizeof(unsigned) * 4 * static_cast<std::size_t>(ceil_div(max_bounds + 1, 128));
    reserve_shared(topk_select_kernel<20>, smem, g_topk_reserved);
    reserve_shared(topk_select_kernel<21>, smem, g_topk_reserved);

    const int slices  = std::clamp(std::min(ceil_div(max_bounds, kTopkSliceBlocks),
                                            ceil_div(std::int64_t{4} * multiprocessors(), columns)),
                                   1, kTopkMaxSlices);
    const auto launch = [&]<int BinShift>() {
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(static_cast<unsigned>(slices), static_cast<unsigned>(columns)),
             dim3(kTopkSelectThreads), smem, stream},
            topk_select_kernel<BinShift>, positions, ratio, top, group_begin,
            static_cast<const float*>(scores), stride, state,
            state + static_cast<std::size_t>(columns) * kTopkScoreBins, selected, counts));
    };
    if (non_negative_scores) {
        launch.template operator()<20>();
    } else {
        launch.template operator()<21>();
    }
}

} // namespace infernix::ops::detail
