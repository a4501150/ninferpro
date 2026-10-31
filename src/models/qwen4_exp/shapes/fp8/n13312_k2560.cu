// Qwen3.8-Flash-Next QSA query/gate/key/value projection (the BF16 indexer runs apart), row-scaled FP8 with
// BF16 activations (W8A16, recipe C). Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090; the
// GEMV and SIMT routes through T = 8 both reduce a column as 16 values per lane in one accumulator
// chain, so a column has the same bits at every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_decode_routes.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<13312, 2560>;
using Gemv     = Fp8N13312K2560Gemv;
using Simt     = Fp8N13312K2560Simt;
using TmaT64R256  = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 96) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 192) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 384) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 1536) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 97.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission (2026-10-15 sweep, RTX 5090): slower than the exact route at T=32, 80
// and 96 (1.06-1.08), faster from 112 (0.65-0.89).
constexpr std::int32_t kPairMinTokens = 97;

// A16ColumnInvariant: the tensor-core tiles at every T. Each accumulates an output as one FP32 chain
// of k16 MMA steps in K order and applies the row scale once, so a column's bits do not depend on the
// tile, the call's width or the column's place in it; at or below 16 columns, where the A16 route
// takes the CUDA-core or sliced-K reductions, the 32-column tile runs with its unused columns masked.
void launch_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 16) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT16R64>(x, weight, out, stream);
    if (x.ne[1] <= 16) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    launch_a16(x, weight, out, stream);
}
} // namespace

const Fp8LinearShape kFp8N13312K2560{13312,   2560, launch_a16, nullptr, uses_a8, nullptr,
                                     launch_invariant, launch_fp8_a8_pair<Geometry>, kPairMinTokens,
                                     launch_fp8_a8_pair_rows<Geometry>};
} // namespace infernix::ops::detail
