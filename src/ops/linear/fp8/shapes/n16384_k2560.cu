// Qwen3.8-Flash-Next GDN query/key/value/z projection, row-scaled FP8 with BF16 activations (W8A16, recipe C).
// Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090. The GEMV at T = 1 and the SIMT tile
// through T = 8 both reduce a column as 16 values per lane in one accumulator chain, so a column has
// the same bits at every width (column invariance, design §11.3); the tensor-core tiles beyond.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/fp8/fp8_decode_routes.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 2560>;
using Gemv     = Fp8N16384K2560Gemv;
using Simt     = Fp8N16384K2560Simt;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 4096) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 33.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission (2026-10-15 sweep, RTX 5090): faster than the exact route at every
// swept width from 48 (0.70-0.93), slower at T=32 (1.02).
constexpr std::int32_t kPairMinTokens = 33;

// A16ColumnInvariant: the tensor-core tiles at every T. Each accumulates an output as one FP32 chain
// of k16 MMA steps in K order and applies the row scale once, so a column's bits do not depend on the
// tile, the call's width or the column's place in it; narrow calls (where the A16 route takes the
// CUDA-core or sliced-K reductions) run the 16- or 32-column tile with their unused columns masked.
void launch_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 16) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT16R64>(x, weight, out, stream);
    if (x.ne[1] <= 32) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    launch_a16(x, weight, out, stream);
}
} // namespace

const Fp8LinearShape kFp8N16384K2560{16384,   2560, launch_a16, nullptr, uses_a8, nullptr,
                                     launch_invariant, launch_fp8_a8_pair<Geometry>, kPairMinTokens,
                                     launch_fp8_a8_pair_rows<Geometry>};
} // namespace infernix::ops::detail
