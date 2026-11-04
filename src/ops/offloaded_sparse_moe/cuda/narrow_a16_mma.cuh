// The tensor-core narrow route of NVFP4 W4A16 experts (design §17.3's arithmetic; adapted from
// upstream Infernix 19c47f14), included by moe_layer.cu after the A16 section: the narrow jobs
// (at most moe::kMaxColumns columns) of the bank's calls of more than kA16RowsMaxColumns columns;
// a call's wider jobs take the wide route, and its narrower calls stay on the dp4a rows kernels
// (which measured faster in their down pass at decode and verification widths, and slower at
// prefill shapes, on the RTX PRO 6000; moe_layer.cu's commit message carries the boundary table).
// Every row sum S = sum_b e4m3_scaled(s_b) P_b is the same exact int64 as the dp4a kernels' and the
// CPU engine's, so the outputs are theirs bit for bit wherever an expert runs.
//
// - The call's x is encoded once (a16_nmma_prepare_kernel) into the workspace's x limb rows
//   (a16_limbs.cuh) with its column exponents, and the jobs' gate/up CTA counts are cleared.
// - Per 16-element block, rows g and g + 8 of a 16-row group form one m16n8k32 A fragment with K the
//   block twice, A = [8 c | c] (c the doubled E2M1 codes, |8 c| <= 96 a signed byte): one record
//   word per thread gives both rows (its low and high nibbles). Against B = [H1 | H0] (s8) the MMA
//   gives c . (X >> 11), against B = [L1 | L0] (u8) c . (X & 2047), so P_b = 2048 (c . (X >> 11)) +
//   c . (X & 2047) exactly in int32, for up to eight columns at once (the n of the MMA), as the wide
//   route forms it (wide_expert_a16.cuh). Thread (g, t) then adds P times the row's scale into its
//   four int64 sums: rows g and g + 8, columns 2 t and 2 t + 1.
// - Gate/up: a warp owns one row group (eight intermediates' gate and up rows) and streams its units
//   through a private cp.async ring; a gate row's thread pairs with its up row's (lane + 4) for
//   SwiGLU and writes BF16 h. A job of at most kNmma16SharedX columns reads x from a shared-memory
//   copy, wider ones from L2. The job's last CTA to finish (job_ctas) rewrites every entry's h in
//   place as its limbs, with h_emax, once for the down CTAs.
// - Down: a warp owns one row group, its units copied before the PDL wait; the job's h limbs then
//   come to shared memory.
//
// Upstream measured the schedule (RTX 5090, cold L2, moe_experts): T = 1 0.87x, T = 3 0.76x, T = 8
// 0.66x of the dp4a kernels' time, output hashes equal at every point; a sweep kept 4 warps and 4
// stages of 8 units. Its helpers (a16_limbs.cuh) and ggml::tile and pdl come from the including
// moe_layer.cu.

constexpr int kNmma16Warps      = 4; // row groups of a gate/up CTA and of a down CTA
constexpr int kNmma16Threads    = 32 * kNmma16Warps;
constexpr int kNmma16StageUnits = 8; // units per stage of a gate/up warp's ring (1,152 bytes)
constexpr int kNmma16Depth      = 4; // stages in a gate/up warp's ring
constexpr int kNmma16SharedX    = 2; // columns whose x a gate/up CTA copies to shared memory
// A limb row's stride in shared memory: 16 bytes past the row, so two columns' words t fall in different banks.
template <int RowBytes>
inline constexpr int kNmma16RowStride = RowBytes + 16;

constexpr int kNmma16XRowBytes  = moe::wide::kA16XRowBytes;
constexpr int kNmma16HRowBytes = moe::wide::kA16HRowBytes;
constexpr int kNmma16StageBytes = kNmma16StageUnits * static_cast<int>(moe::kUnitBytes);
constexpr int kNmma16Stages     = moe::kGateUpBlocks / kNmma16StageUnits;
constexpr int kNmma16GateUpCtas = moe::kGateUpRowGroups / kNmma16Warps; // per job
constexpr int kNmma16DownCtas   = moe::kDownRowGroups / kNmma16Warps;   // per job
constexpr int kNmma16DownBytes  = moe::kDownBlocks * static_cast<int>(moe::kUnitBytes); // a row group's
static_assert(moe::kGateUpBlocks % kNmma16StageUnits == 0 && kNmma16Stages >= kNmma16Depth, "whole stages, a full ring");
static_assert(moe::kGateUpRowGroups % kNmma16Warps == 0 && moe::kDownRowGroups % kNmma16Warps == 0,
              "every CTA owns whole row groups");
static_assert(kNmma16StageBytes % 16 == 0 && kNmma16DownBytes % 16 == 0 && kNmma16XRowBytes % 16 == 0 &&
                  kNmma16HRowBytes % 16 == 0,
              "16-byte copies");
static_assert(moe::kMaxColumns <= 8, "one MMA's n covers a job's columns");

// These kernels take the narrow jobs of an NVFP4 W4A16 bank's calls of more than kA16RowsMaxColumns
// columns (launch_pass reaches them after the GGUF and block-FP8 banks' own kernels; moe_experts
// prepares their x limbs and job counts once per call, before their passes).

// D = A B on int8 tensor cores with signed A and unsigned B (ggml::tile::mma_k32 is signed times signed).
__device__ __forceinline__ void mma_k32_su(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

__device__ __forceinline__ std::uint32_t nmma16_lds32(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

// One unit's contribution to thread (g, t)'s sums: its A fragment from the unit's word 32 t + 4 g (rows g and
// g + 8, elements 4 t .. 4 t + 3), B from column g's limb words t of the unit's block (`limbs`: that column's
// limb row at the block's tile, offset by 16 (b % 4) + 4 t), the rows' scales through `scaled`.
template <bool kGlobal>
__device__ __forceinline__ void nmma16_unit(const std::uint8_t* unit, const std::uint8_t* limbs, const std::int32_t* scaled,
                                            int g, int t, long long (&s)[4]) {
    const std::uint32_t word = nmma16_lds32(unit + 32 * t + 4 * g);
    const std::uint32_t c_lo = canon::e2m1_x2_quad(word & 0x0F0F0F0FU);        // row g
    const std::uint32_t c_hi = canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU); // row g + 8
    const int a0 = static_cast<int>((c_lo << 3) & 0xF8F8F8F8U); // 8 c: each byte's own bits
    const int a1 = static_cast<int>((c_hi << 3) & 0xF8F8F8F8U);
    std::uint32_t h1, l1, small;
    if constexpr (kGlobal) {
        h1    = __ldg(reinterpret_cast<const unsigned*>(limbs));
        l1    = __ldg(reinterpret_cast<const unsigned*>(limbs + moe::wide::kA16KTile));
        small = __ldg(reinterpret_cast<const unsigned*>(limbs + 2 * moe::wide::kA16KTile));
    } else {
        h1    = nmma16_lds32(limbs);
        l1    = nmma16_lds32(limbs + moe::wide::kA16KTile);
        small = nmma16_lds32(limbs + 2 * moe::wide::kA16KTile);
    }
    int hi[4], lo[4];
    ggml::tile::mma_k32(hi, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(h1),
                  static_cast<int>(small & 0x07070707U));
    mma_k32_su(lo, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(l1),
               static_cast<int>((small >> 4) & 0x07070707U));
    const int w_lo = scaled[unit[128 + g]], w_hi = scaled[unit[136 + g]];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // P = 2048 hi + lo in wrapping arithmetic: its true value fits int32.
        const int p = static_cast<int>((static_cast<unsigned>(hi[i]) << 11) + static_cast<unsigned>(lo[i]));
        moe::wide::fold(s[i], p, i < 2 ? w_lo : w_hi);
    }
}

// The limb words of block b for thread t within a limb row.
__device__ __forceinline__ int nmma16_block_offset(int b, int t) {
    return (b / 4) * moe::wide::kA16LimbTileBytes + 16 * (b % 4) + 4 * t;
}

__device__ __forceinline__ void nmma16_fill_scaled(std::int32_t* scaled, int threads) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += threads) {
        scaled[i] = canon::e4m3_scaled(static_cast<unsigned>(i));
    }
}

// Every token's x limbs and exponent (CTA t < columns, one block per thread), and the jobs' gate/up
// CTA counts cleared (CTA `columns`).
__global__ void __launch_bounds__(moe::kHidden / 16)
    a16_nmma_prepare_kernel(const std::uint16_t* __restrict__ x, int columns, std::uint8_t* __restrict__ limbs,
                            std::int32_t* __restrict__ emax, std::uint32_t* __restrict__ job_ctas, int max_jobs) {
    constexpr int kEncodeThreads = moe::kHidden / 16;
    static_assert(kEncodeThreads % 32 == 0);
    __shared__ unsigned keys[kEncodeThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    if (t == columns) {
        for (int j = b; j < max_jobs; j += kEncodeThreads) { job_ctas[j] = 0U; }
        return;
    }
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * moe::kHidden + 16 * b);
    std::uint16_t v[16];
    moe::wide::unpack16(source[0], source[1], v);
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, moe::wide::a16_key(v));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kEncodeThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = moe::wide::a16_emax(all);
    moe::wide::store_a16_block(limbs + static_cast<std::size_t>(t) * kNmma16XRowBytes, b, v, e);
    if (b == 0) { emax[t] = e; }
}

// Entry `entry`'s h (BF16, written by the job's gate/up CTAs) rewritten in place as its limbs, and its exponent:
// one warp, holding the row's blocks in registers (lane l blocks l, l + 32, ...) before it writes. Reads through L2.
__device__ __forceinline__ void nmma16_encode_h(std::uint8_t* __restrict__ h_rows, std::int32_t* __restrict__ h_emax,
                                                int entry, int lane) {
    constexpr int kBlocks = moe::kHBlocks;
    constexpr int kPerLane = (kBlocks + 31) / 32;
    std::uint8_t* row = h_rows + static_cast<std::size_t>(entry) * kNmma16HRowBytes;
    std::uint16_t v[kPerLane][16];
    unsigned key = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) {
            const auto* p = reinterpret_cast<const uint4*>(row + 32 * b);
            moe::wide::unpack16(__ldcg(p), __ldcg(p + 1), v[i]);
            key = max(key, moe::wide::a16_key(v[i]));
        }
    }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    const int e = moe::wide::a16_emax(key);
    __syncwarp(); // every lane holds its blocks before any limb overwrites the row
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) { moe::wide::store_a16_block(row, b, v[i], e); }
    }
    if (lane == 0) { h_emax[entry] = e; }
}

struct Nmma16Args {
    const std::uint8_t* x_limbs; // [T][kXRowBytes]
    const std::int32_t* x_emax;  // [T]
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base;
    std::uint8_t* h_rows; // [entries][kHRowBytes]: BF16 h, then the job's last gate/up CTA's limbs
    std::int32_t* h_emax; // [entries]
    std::uint32_t* job_ctas; // [max_jobs]: gate/up CTAs finished per job
    bf16* outputs;           // [entries][hidden]
    PassPhase phase;
};

struct Nmma16GateUpShared {
    std::int32_t scaled[256];
    alignas(16) std::uint8_t x[kNmma16SharedX][kNmma16RowStride<kNmma16XRowBytes>];
    alignas(16) std::uint8_t ring[kNmma16Warps][kNmma16Depth][kNmma16StageBytes];
};

__global__ void __launch_bounds__(kNmma16Threads) gate_up_nmma16_kernel(Nmma16Args a) {
    __shared__ Nmma16GateUpShared sm;
    __shared__ int last_cta;
    // The down pass (a programmatic dependent in captured graphs) may launch and start its weight copies now: it
    // reads h only after its PDL wait.
    pdl::trigger_dependents();
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (a.cpu_flags != nullptr && a.cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(a.phase, a.source, expert, a.job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = a.source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return; }
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * kNmma16Warps + warp;
    const std::uint8_t* units = record + static_cast<std::size_t>(group) * moe::kGateUpBlocks * moe::kUnitBytes;
    const auto token          = [&](int c) { return dispatch.entries[first + c] / a.top_k; };
    const bool shared_x       = count <= kNmma16SharedX;
    if (shared_x) {
        constexpr int kChunks = kNmma16XRowBytes / 16;
        for (int i = static_cast<int>(threadIdx.x); i < count * kChunks; i += kNmma16Threads) {
            const int c = i / kChunks, k = i % kChunks;
            ggml::tile::cp_async16(sm.x[c] + 16 * k, a.x_limbs + static_cast<std::size_t>(token(c)) * kNmma16XRowBytes + 16 * k,
                                   16);
        }
    }
    ggml::tile::cp_async_commit();
    auto& ring = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * kNmma16StageBytes;
        for (int i = lane; i < kNmma16StageBytes / 16; i += 32) {
            ggml::tile::cp_async16(ring[stage % kNmma16Depth] + 16 * i, src + 16 * i, 16);
        }
    };
#pragma unroll
    for (int k = 0; k < kNmma16Depth - 1; ++k) {
        issue(k);
        ggml::tile::cp_async_commit();
    }
    nmma16_fill_scaled(sm.scaled, kNmma16Threads);
    ggml::tile::cp_async_wait<kNmma16Depth - 1>(); // the x group
    __syncthreads();                               // x and the scale table
    // Column g's limb row (a thread past the job's columns reads its last one; its sums are not used).
    const int column = min(g, count - 1);
    long long s[4]  = {0, 0, 0, 0};
    const auto k_loop = [&](auto global_x, const std::uint8_t* limbs) {
        constexpr bool kGlobal = decltype(global_x)::value;
        for (int st = 0; st < kNmma16Stages; ++st) {
            ggml::tile::cp_async_wait<kNmma16Depth - 2>();
            __syncwarp();
            if (st + kNmma16Depth - 1 < kNmma16Stages) { issue(st + kNmma16Depth - 1); }
            ggml::tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kNmma16Depth];
#pragma unroll
            for (int u = 0; u < kNmma16StageUnits; ++u) {
                const int b = st * kNmma16StageUnits + u;
                nmma16_unit<kGlobal>(stage + u * moe::kUnitBytes, limbs + nmma16_block_offset(b, t), sm.scaled, g, t, s);
            }
            __syncwarp(); // every lane is done with this stage before a later issue refills it
        }
    };
    if (shared_x) {
        k_loop(std::false_type{}, sm.x[column]);
    } else {
        k_loop(std::true_type{}, a.x_limbs + static_cast<std::size_t>(token(column)) * kNmma16XRowBytes);
    }
    // Rows g and g + 8 are gate rows for even g (row 2 i is intermediate i's gate, 2 i + 1 its up): the up row's
    // outputs come from lane + 4.
    const bool gate = (g & 1) == 0;
    std::uint16_t y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = 2 * t + (i & 1);
        y[i] = canon::a16_row_output(s[i], a.x_emax[token(min(c, count - 1))], gate ? scales.alpha_gate : scales.alpha_up);
    }
    std::uint16_t up[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
    if (gate) {
        auto* h = reinterpret_cast<std::uint16_t*>(a.h_rows);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = 2 * t + (i & 1);
            const int k = group * 8 + g / 2 + (i < 2 ? 0 : 4); // rows g, g + 8: intermediates g / 2, g / 2 + 4
            if (c < count) {
                h[static_cast<std::size_t>(first + c) * moe::wide::kA16HRowElements + k] = canon::swiglu_bf16(y[i], up[i]);
            }
        }
    }
    // The job's last CTA encodes its h. The barrier orders every thread's h stores before thread 0's acq_rel count
    // (a cumulative release), and the last CTA's acquire reaches its other threads through the barrier.
    __syncthreads();
    if (threadIdx.x == 0) {
        std::uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(done) : "l"(a.job_ctas + job) : "memory");
        last_cta = done == kNmma16GateUpCtas - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last_cta == 0) { return; } // (a job belongs to one pass; the call's prepare kernel clears the counts)
    for (int c = warp; c < count; c += kNmma16Warps) { nmma16_encode_h(a.h_rows, a.h_emax, first + c, lane); }
}

// A down CTA's shared memory: the warps' row groups, then the job's h limb rows (as many as its columns, sized at
// launch).
struct Nmma16DownShared {
    std::int32_t scaled[256];
    int emax[moe::kMaxColumns];
    alignas(16) std::uint8_t units[kNmma16Warps][kNmma16DownBytes];
    alignas(16) std::uint8_t h[1][kNmma16RowStride<kNmma16HRowBytes>]; // [columns] at launch
};

std::size_t nmma16_down_shared_bytes(int columns) {
    return offsetof(Nmma16DownShared, h) + static_cast<std::size_t>(columns) * kNmma16RowStride<kNmma16HRowBytes>;
}

__global__ void __launch_bounds__(kNmma16Threads) down_nmma16_kernel(Nmma16Args a) {
    extern __shared__ __align__(16) unsigned char nmma16_shared[];
    auto& sm = *reinterpret_cast<Nmma16DownShared*>(nmma16_shared);
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    // Only h_rows and h_emax are read after the wait (the job's last gate/up CTA encoded its h); the weights do not
    // depend on it, so their copies start first.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    moe::ExpertScales scales{};
    if (job < *dispatch.job_count && (a.cpu_flags == nullptr || a.cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(a.phase, a.source, expert, a.job_records, job);
    }
    if (record != nullptr) {
        scales = a.source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * kNmma16Warps + warp;
    if (record != nullptr) {
        const std::uint8_t* src = record + moe::kGateUpBytes + static_cast<std::size_t>(group) * kNmma16DownBytes;
        for (int i = lane; i < kNmma16DownBytes / 16; i += 32) { ggml::tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    }
    ggml::tile::cp_async_commit();
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    constexpr int kChunks = kNmma16HRowBytes / 16;
    for (int i = static_cast<int>(threadIdx.x); i < count * kChunks; i += kNmma16Threads) {
        const int c = i / kChunks, k = i % kChunks;
        ggml::tile::cp_async16(sm.h[c] + 16 * k, a.h_rows + static_cast<std::size_t>(first + c) * kNmma16HRowBytes + 16 * k, 16);
    }
    ggml::tile::cp_async_commit();
    if (static_cast<int>(threadIdx.x) < count) { sm.emax[threadIdx.x] = __ldcg(a.h_emax + first + threadIdx.x); }
    nmma16_fill_scaled(sm.scaled, kNmma16Threads);
    ggml::tile::cp_async_wait<0>();
    __syncthreads(); // the h limbs and exponents, the scale table and every lane's weight copies
    const std::uint8_t* limbs = sm.h[min(g, count - 1)];
    const std::uint8_t* units = sm.units[warp];
    long long s[4] = {0, 0, 0, 0};
#pragma unroll 8
    for (int b = 0; b < moe::kDownBlocks; ++b) {
        nmma16_unit<false>(units + b * moe::kUnitBytes, limbs + nmma16_block_offset(b, t), sm.scaled, g, t, s);
    }
    auto* out = reinterpret_cast<std::uint16_t*>(a.outputs);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = 2 * t + (i & 1);
        if (c < count) {
            const int row = group * 16 + g + (i < 2 ? 0 : 8);
            out[static_cast<std::size_t>(dispatch.entries[first + c]) * moe::kHidden + row] =
                canon::a16_row_output(s[i], sm.emax[c], scales.alpha_down);
        }
    }
}

// Encodes the call's x and clears its jobs' gate/up CTA counts (once per call, before its passes).
void launch_prepare_a16_nmma(const Tensor& x, int max_jobs, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    const int columns = static_cast<int>(x.ne[1]);
    a16_nmma_prepare_kernel<<<static_cast<unsigned>(columns + 1), moe::kHidden / 16, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), columns, layout.x_limbs, layout.x_emax, layout.job_ctas, max_jobs);
    check_launch("A16 tensor-core prepare");
}

// One pass of the tensor-core narrow kernels: a gate/up CTA per kNmma16Warps row groups and a down CTA per
// kNmma16Warps output row groups of each job. `columns` is the call's T; a job has at most
// min(columns, kMaxColumns) columns, which sizes the down CTAs' h rows.
void launch_expert_pass_a16_nmma(int columns, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                 const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                 PassPhase phase) {
    const Nmma16Args args{layout.x_limbs, layout.x_emax,  dispatch, source, top_k, job_records, flags, base,
                          layout.h_rows,  layout.h_emax, layout.job_ctas, outputs, phase};
    gate_up_nmma16_kernel<<<dim3(kNmma16GateUpCtas, jobs), kNmma16Threads, 0, stream>>>(args);
    check_launch("A16 tensor-core gate/up");
    const std::size_t down_bytes = nmma16_down_shared_bytes(std::min(columns, moe::kMaxColumns));
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer({dim3(kNmma16DownCtas, jobs), dim3(kNmma16Threads), down_bytes, stream},
                                    down_nmma16_kernel, args));
    check_launch("A16 tensor-core down");
}
