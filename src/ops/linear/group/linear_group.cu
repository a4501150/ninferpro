#include "ops/linear/group/linear_group.h"

#include "ops/common/blocks.cuh"
#include "ops/linear/bf16/bf16_a16_gemv.cuh"
#include "ops/linear/bf16/bf16_decode_routes.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/linear/bf16/bf16_skinny.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_decode_routes.cuh"
#include "ops/linear/fp8/fp8_operands.h"

namespace infernix::ops::detail {
namespace {

bool is(const Weight& w, QType qtype, std::int32_t n, std::int32_t k) {
    return w.qtype == qtype && w.n == n && w.k == k;
}

// The FP8 GEMV block of a T = 1 projection on `Schedule` (its shape's own T = 1 route).
template <class Schedule>
auto fp8_gemv(const Tensor& x, const Weight& w, const LinearBf16RowParts& parts) {
    using Instance          = Fp8ScheduleInstance<Schedule, 2560>;
    const Fp8A16Operands p  = fp8_a16_operands(x, w);
    validate_fp8_operands<Instance>(p);
    if (p.rows % Instance::kBlockRows || p.k % (32 * Instance::kValuesPerLane))
        throw std::invalid_argument("FP8 GEMV requires complete row/K tiles");
    return Fp8A16GemvBlock<Instance, LinearBf16RowParts, LinearIdentityEpilogue, Fp8IdentityRows>{
        p.x, p.codes, p.scales, parts, {}, {}, p.rows, p.k};
}

// Qwen3.8-Flash-Next's GDN input: q/k/v/z [16384, 2560] (row-FP8) beside a/b [96, 2560] (BF16).
bool gdn_input(const Tensor& x, const Weight& qkvz, const LinearBf16RowParts& qkvz_parts, const Weight& ab,
               const LinearBf16RowParts& ab_parts, cudaStream_t stream) {
    if (!is(qkvz, QType::FP8_E4M3FN_ROW_BF16, 16384, 2560) || !is(ab, QType::BF16, 96, 2560)) return false;
    using Bf16             = Bf16N96K2560Gemv;
    const Bf16A16Operands q = bf16_a16_operands(x, ab);
    validate_bf16_operands<Bf16>(q);
    blocks::launch_composed(stream, fp8_gemv<Fp8N16384K2560Gemv>(x, qkvz, qkvz_parts),
                            Bf16A16GemvBlock<Bf16, LinearBf16RowParts, LinearIdentityEpilogue>{
                                q.x, q.weight, ab_parts, {}, q.rows, q.k});
    return true;
}

// Qwen3.8-Flash-Next's attention input: q/gate/k/v [13312, 2560] (row-FP8) beside the QSA index
// projection [640, 2560] (BF16, the skinny GEMV of the runtime-shape route).
bool attention_input(const Tensor& x, const Weight& qkvg, const LinearBf16RowParts& qkvg_parts,
                     const Weight& index, const LinearBf16RowParts& index_parts, cudaStream_t stream) {
    if (!is(qkvg, QType::FP8_E4M3FN_ROW_BF16, 13312, 2560) || !is(index, QType::BF16, 640, 2560)) return false;
    using Skinny = Bf16SkinnyGemvBlock<1, kBf16SkinnyWarps, kBf16SkinnyRowsPerWarp, 0, LinearBf16RowParts>;
    const Bf16A16Operands q = bf16_a16_operands(x, index);
    const auto aligned      = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
    if (!aligned(q.x) || !aligned(q.weight) || q.k % 8) return false; // the runtime-shape route's vector path
    blocks::launch_composed(stream, fp8_gemv<Fp8N13312K2560Gemv>(x, qkvg, qkvg_parts),
                            Skinny{q.x, q.weight, index_parts, q.rows, q.k});
    return true;
}

} // namespace

bool linear_group_composed(const Tensor& x, const Weight& first, const LinearBf16RowParts& first_parts,
                           const Weight& second, const LinearBf16RowParts& second_parts, LinearPolicy policy,
                           cudaStream_t stream) {
    if (policy != LinearPolicy::A16Only || x.ne[1] != 1) return false;
    return gdn_input(x, first, first_parts, second, second_parts, stream) ||
           attention_input(x, first, first_parts, second, second_parts, stream);
}

} // namespace infernix::ops::detail
