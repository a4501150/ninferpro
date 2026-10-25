#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_shapes.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Bf16Launch (*select)(std::int32_t);
    Bf16Launch (*invariant)(std::int32_t) = nullptr; // A16ColumnInvariant, or null where none is registered
    Bf16Launch (*index_mma)(std::int32_t) = nullptr; // LinearRoutes::Bf16IndexMma, or null
};

constexpr std::array kShapes{
    ShapeEntry{14336, 5120, select_bf16_n14336_k5120},
    ShapeEntry{5120, 6144, select_bf16_n5120_k6144},
    ShapeEntry{48, 2560, select_bf16_n48_k2560},
    ShapeEntry{96, 2560, select_bf16_n96_k2560, select_bf16_n96_k2560_invariant},
    ShapeEntry{640, 2560, select_bf16_n640_k2560, select_bf16_n640_k2560_invariant, select_bf16_n640_k2560_index_mma},
    ShapeEntry{1664, 2560, select_bf16_n1664_k2560},
    ShapeEntry{324, 10240, select_bf16_n324_k10240},
    ShapeEntry{320, 10240, select_bf16_n320_k10240},
    ShapeEntry{10240, 320, select_bf16_n10240_k320},
    ShapeEntry{12800, 2560, select_bf16_n12800_k2560},
    ShapeEntry{248320, 2560, select_bf16_n248320_k2560},
    ShapeEntry{13952, 2560, select_bf16_n13952_k2560},
    ShapeEntry{2560, 6144, select_bf16_n2560_k6144},
    ShapeEntry{2560, 2560, select_bf16_n2560_k2560},
    ShapeEntry{16384, 2560, select_bf16_n16384_k2560},
    ShapeEntry{1280, 2560, select_bf16_n1280_k2560},
    ShapeEntry{2560, 640, select_bf16_n2560_k640},
    ShapeEntry{1152, 1536, select_bf16_n1152_k1536},
    ShapeEntry{3456, 1152, select_bf16_n3456_k1152},
    ShapeEntry{1152, 1152, select_bf16_n1152_k1152},
    ShapeEntry{4304, 1152, select_bf16_n4304_k1152},
    ShapeEntry{1152, 4304, select_bf16_n1152_k4304},
    ShapeEntry{4608, 4608, select_bf16_n4608_k4608},
    ShapeEntry{2560, 4608, select_bf16_n2560_k4608},
    ShapeEntry{256, 5120, select_bf16_n256_k5120},
    ShapeEntry{5120, 4608, select_bf16_n5120_k4608},
};
} // namespace

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearRoutes routes) {
    if (t <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k) continue;
        if (has_route(routes, LinearRoutes::Bf16IndexMma) && entry.index_mma != nullptr) return entry.index_mma(t);
        return entry.select(t);
    }
    // Shapes without a specialised kernel (for example a full-precision vocab head) fall back to the
    // general runtime-shape GEMM.
    return select_bf16_general_launch(t);
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy,
                              LinearRoutes routes) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("bf16 linear: unsupported policy");
    if (column_invariant(policy)) {
        if (t <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
        for (const auto& entry : kShapes) {
            if (entry.n == n && entry.k == k && entry.invariant != nullptr) return entry.invariant(t);
        }
        throw std::invalid_argument("bf16 linear: no column-invariant route for this problem");
    }
    return select_bf16_a16_launch(n, k, t, routes);
}

bool bf16_column_invariant_supported(std::int32_t n, std::int32_t k) noexcept {
    return std::any_of(kShapes.begin(), kShapes.end(),
                       [&](const ShapeEntry& entry) { return entry.n == n && entry.k == k && entry.invariant != nullptr; });
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   LinearRoutes routes, cudaStream_t stream) {
    select_bf16_launch(weight.n, weight.k, x.ne[1], policy, routes)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
