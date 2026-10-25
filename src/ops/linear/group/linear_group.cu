#include "ops/linear/group/linear_group.h"

#include "ops/common/blocks.cuh"
#include "ops/linear/bf16/bf16_a16_gemv.cuh"
#include "ops/linear/bf16/bf16_a16_simt.cuh"
#include "ops/linear/bf16/bf16_decode_routes.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/linear/bf16/bf16_skinny.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
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

// The FP8 SIMT block of a T = 2 .. 8 projection on `Schedule` (its shape's own route: capacity 8).
template <class Schedule>
auto fp8_simt(const Tensor& x, const Weight& w, const LinearBf16RowParts& parts) {
    using Instance         = Fp8ScheduleInstance<Schedule, 2560, 8, false>;
    const Fp8A16Operands p = fp8_a16_operands(x, w);
    validate_fp8_operands<Instance>(p);
    if (p.rows % Instance::kBlockRows || p.k % (32 * Instance::kValuesPerLane) || p.tokens > 8)
        throw std::invalid_argument("FP8 SIMT requires complete row/K tiles and matching token capacity");
    return Fp8A16SimtBlock<Instance, LinearBf16RowParts, LinearIdentityEpilogue, Fp8IdentityRows>{p, parts, {}, {}};
}

// Qwen3.8-Flash-Next's GDN input: q/k/v/z [16384, 2560] (row-FP8) beside a/b [96, 2560] (BF16).
bool gdn_input(const Tensor& x, const Weight& qkvz, const LinearBf16RowParts& qkvz_parts, const Weight& ab,
               const LinearBf16RowParts& ab_parts, cudaStream_t stream) {
    if (!is(qkvz, QType::FP8_E4M3FN_ROW_BF16, 16384, 2560) || !is(ab, QType::BF16, 96, 2560)) return false;
    const Bf16A16Operands q = bf16_a16_operands(x, ab);
    if (x.ne[1] == 1) {
        validate_bf16_operands<Bf16N96K2560Gemv>(q);
        blocks::launch_composed(stream, fp8_gemv<Fp8N16384K2560Gemv>(x, qkvz, qkvz_parts),
                                Bf16A16GemvBlock<Bf16N96K2560Gemv, LinearBf16RowParts, LinearIdentityEpilogue>{
                                    q.x, q.weight, ab_parts, {}, q.rows, q.k});
        return true;
    }
    using Simt = Bf16N96K2560Simt;
    validate_bf16_operands<Simt>(q);
    if (q.rows % Simt::kBlockRows) return false;
    blocks::launch_composed(stream, fp8_simt<Fp8N16384K2560Simt>(x, qkvz, qkvz_parts),
                            Bf16A16SimtBlock<Simt, LinearBf16RowParts, LinearIdentityEpilogue>{
                                q.x, q.weight, ab_parts, {}, q.rows, q.k, q.tokens});
    return true;
}

// Qwen3.8-Flash-Next's attention input: q/gate/k/v [13312, 2560] (row-FP8) beside the QSA index
// projection [640, 2560] (BF16, the skinny GEMV of the runtime-shape route).
bool attention_input(const Tensor& x, const Weight& qkvg, const LinearBf16RowParts& qkvg_parts,
                     const Weight& index, const LinearBf16RowParts& index_parts, cudaStream_t stream) {
    if (!is(qkvg, QType::FP8_E4M3FN_ROW_BF16, 13312, 2560) || !is(index, QType::BF16, 640, 2560)) return false;
    const Bf16A16Operands q = bf16_a16_operands(x, index);
    const auto aligned      = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
    if (!aligned(q.x) || !aligned(q.weight) || q.k % 8) return false; // the runtime-shape route's vector path
    const auto launch = [&]<int Tokens>() {
        using Skinny = Bf16SkinnyGemvBlock<Tokens, kBf16SkinnyWarps, kBf16SkinnyRowsPerWarp, 0, LinearBf16RowParts>;
        const Skinny skinny{q.x, q.weight, index_parts, q.rows, q.k};
        if constexpr (Tokens == 1) {
            blocks::launch_composed(stream, fp8_gemv<Fp8N13312K2560Gemv>(x, qkvg, qkvg_parts), skinny);
        } else {
            blocks::launch_composed(stream, fp8_simt<Fp8N13312K2560Simt>(x, qkvg, qkvg_parts), skinny);
        }
    };
    switch (x.ne[1]) {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
    case 7: launch.template operator()<7>(); break;
    default: launch.template operator()<8>(); break;
    }
    return true;
}

} // namespace

bool linear_group_composed(const Tensor& x, const Weight& first, const LinearBf16RowParts& first_parts,
                           const Weight& second, const LinearBf16RowParts& second_parts, LinearPolicy policy,
                           cudaStream_t stream) {
    if (policy != LinearPolicy::A16Only || x.ne[1] > 8) return false;
    return gdn_input(x, first, first_parts, second, second_parts, stream) ||
           attention_input(x, first, first_parts, second, second_parts, stream);
}

} // namespace infernix::ops::detail
