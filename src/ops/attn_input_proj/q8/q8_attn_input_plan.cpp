#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"

#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kH2048Kv512Routes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, Q8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, Q8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, Q8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kH2048Kv1024Routes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, Q8AttnInputScheduleId::MmaR32C64},
    {193, 288, Q8AttnInputScheduleId::MmaR64C96},
    {289, 320, Q8AttnInputScheduleId::MmaR64C64},
    {321, 384, Q8AttnInputScheduleId::MmaR64C128},
    {385, 448, Q8AttnInputScheduleId::MmaR128C64},
    {449, 560, Q8AttnInputScheduleId::MmaR128C80},
    {561, kAnyCols, Q8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 6> kWideHiddenRoutes{{
    {1, 48, Q8AttnInputScheduleId::WideHSmallT},
    {49, 63, Q8AttnInputScheduleId::WideHMmaR16C64K128},
    {64, 96, Q8AttnInputScheduleId::WideHMmaR32C32K128},
    {97, 128, Q8AttnInputScheduleId::WideHMmaR32C64K128},
    {129, 192, Q8AttnInputScheduleId::WideHMmaR32C64},
    {193, kAnyCols, Q8AttnInputScheduleId::WideHMmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kH2048Kv512Routes),
              "Q8 H2048 KV512 attention input routes must be exact and closed");
static_assert(catalog_is_closed(kH2048Kv1024Routes),
              "Q8 H2048 KV1024 attention input routes must be exact and closed");
static_assert(catalog_is_closed(kWideHiddenRoutes),
              "Q8 wide-H attention input routes must be exact and closed");

bool is_h2048_kv1024_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool is_wide_hidden_shape(const Q8AttnInputProblem& problem) noexcept {
    // Qwen3.8-27B's drafter (H 5120) and Qwen3.8-Flash-Next's (H 2560).
    return (problem.input_rows == 5120 || problem.input_rows == 2560) && problem.query_rows == 4096 &&
           problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == problem.input_rows;
}

bool supported_shape(const Q8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    const bool target_shape = problem.input_rows == 2048 && problem.padded_k == 2048 && target_qkgv;
    return target_shape || is_h2048_kv1024_shape(problem) || is_wide_hidden_shape(problem);
}

} // namespace

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.q8.decode.r8.direct.k2048";
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.q8.splitk.mma.r16.direct";
    case Q8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.q8.simt.r8.c4";
    case Q8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.q8.mma.r32.c64";
    case Q8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.q8.mma.r32.c128";
    case Q8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.q8.mma.r64.c64";
    case Q8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.q8.mma.r64.c96";
    case Q8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.q8.mma.r64.c128";
    case Q8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.q8.mma.r128.c64";
    case Q8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.q8.mma.r128.c80";
    case Q8AttnInputScheduleId::WideHSmallT:
        return "attn_input_proj.q8.wide_h.small_t";
    case Q8AttnInputScheduleId::WideHMmaR16C64K128:
        return "attn_input_proj.q8.wide_h.mma.r16.c64.k128";
    case Q8AttnInputScheduleId::WideHMmaR32C32K128:
        return "attn_input_proj.q8.wide_h.mma.r32.c32.k128";
    case Q8AttnInputScheduleId::WideHMmaR32C64K128:
        return "attn_input_proj.q8.wide_h.mma.r32.c64.k128";
    case Q8AttnInputScheduleId::WideHMmaR32C64:
        return "attn_input_proj.q8.wide_h.mma.r32.c64";
    case Q8AttnInputScheduleId::WideHMmaR64C128:
        return "attn_input_proj.q8.wide_h.mma.r64.c128";
    }
    return "attn_input_proj.q8.unknown";
}

bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem) {
    if (!q8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q8 attention input: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("Q8 attention input: admitted problem has no covering route");
    };
    if (is_wide_hidden_shape(problem)) { return resolve_from(kWideHiddenRoutes); }
    if (is_h2048_kv1024_shape(problem)) { return resolve_from(kH2048Kv1024Routes); }
    return resolve_from(kH2048Kv512Routes);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 9216 || problem.kv_rows != 512 ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
    case Q8AttnInputScheduleId::MmaR64C64:
    case Q8AttnInputScheduleId::MmaR64C96:
    case Q8AttnInputScheduleId::MmaR128C64:
    case Q8AttnInputScheduleId::MmaR128C80:
    case Q8AttnInputScheduleId::WideHSmallT:
    case Q8AttnInputScheduleId::WideHMmaR32C64:
    case Q8AttnInputScheduleId::WideHMmaR64C128:
    case Q8AttnInputScheduleId::WideHMmaR16C64K128:
    case Q8AttnInputScheduleId::WideHMmaR32C32K128:
    case Q8AttnInputScheduleId::WideHMmaR32C64K128:
        throw std::logic_error("Q8 attention input: three-output schedule in four-output plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, gate, k, v,
                               stream);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    const bool companion           = is_h2048_kv1024_shape(problem);
    const bool wide_h              = is_wide_hidden_shape(problem);
    if ((!companion && !wide_h) || resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact three-output problem");
    }
    if (wide_h) {
        switch (plan.schedule) {
        case Q8AttnInputScheduleId::WideHSmallT:
            q8_attn_input_wide_h_small_t_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::WideHMmaR16C64K128:
            q8_attn_input_wide_h_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::WideHMmaR32C32K128:
            q8_attn_input_wide_h_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::WideHMmaR32C64K128:
            q8_attn_input_wide_h_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::WideHMmaR32C64:
            q8_attn_input_wide_h_mma_r32_c64_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::WideHMmaR64C128:
            q8_attn_input_wide_h_mma_r64_c128_launch(x, weight, q, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: non-wide-H schedule in wide-H plan");
        }
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
        q8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C64:
        q8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C96:
        q8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C64:
        q8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C80:
        q8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::WideHSmallT:
    case Q8AttnInputScheduleId::WideHMmaR32C64:
    case Q8AttnInputScheduleId::WideHMmaR64C128:
    case Q8AttnInputScheduleId::WideHMmaR16C64K128:
    case Q8AttnInputScheduleId::WideHMmaR32C32K128:
    case Q8AttnInputScheduleId::WideHMmaR32C64K128:
        throw std::logic_error("Q8 attention input: wide-H schedule in H2048 KV1024 plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace infernix::ops::detail
