#include "infernix/ops/silu_mul.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr PointwiseCriterion silu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> silu_mul_oracle(const std::vector<float>& gate, const std::vector<float>& up,
                                    double limit = std::numeric_limits<double>::infinity()) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = std::min<double>(gate[i], limit);
        const double u = std::clamp<double>(up[i], -limit, limit);
        expected[i]    = (g / (1.0 + std::exp(-g))) * static_cast<double>(u);
    }
    return expected;
}

int run_contiguous_case(const char* label, std::int32_t rows, std::int32_t columns,
                        std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += verify_exact("silu_mul up unchanged",
                             from_device<std::uint16_t>(device_up.data(), count), up_bits);
    failures += device_gate.verify_guards("silu_mul gate");
    failures += device_up.verify_guards("silu_mul up");
    failures += device_out.verify_guards("silu_mul out");
    return failures;
}

int run_strided_gate_up_case() {
    constexpr std::int32_t rows    = 17408;
    constexpr std::int32_t columns = 17;
    const std::size_t count        = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 301u, -12.0f, 12.0f);
    fill_uniform(up, 302u, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<float> gate_up(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, gate_up.data() + target);
        std::copy_n(up.data() + source, rows, gate_up.data() + target + rows);
    }

    const auto expected     = silu_mul_oracle(gate, up);
    const auto gate_up_bits = encode_bf16(gate_up);
    GuardedDeviceBuffer device_gate_up(gate_up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate_up.copy_from_host(gate_up_bits.data(), device_gate_up.bytes());

    Tensor gate_up_tensor(device_gate_up.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = gate_up_tensor.slice(0, 0, rows);
    Tensor up_tensor   = gate_up_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_pointwise("silu_mul strided gate/up", from_device_bf16(device_out.data(), count),
                         expected, silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul strided inputs unchanged",
                             from_device<std::uint16_t>(device_gate_up.data(), gate_up_bits.size()),
                             gate_up_bits);
    failures += device_gate_up.verify_guards("silu_mul strided inputs");
    failures += device_out.verify_guards("silu_mul strided out");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-30.0f, -8.0f, -1.0f, 0.0f, 1.0f, 8.0f, 30.0f};
    std::vector<float> up{2.0f, -3.0f, 0.5f, -4.0f, 5.0f, -6.0f, 7.0f};
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(expected.size())});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("silu_mul edge values",
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul edge gate");
    failures += device_up.verify_guards("silu_mul edge up");
    failures += device_out.verify_guards("silu_mul edge out");
    return failures;
}

// silu_mul_clamped: h = SiLU(min(g, L)) * clamp(u, -L, L), FP32 math from the BF16 inputs, rounded
// once. The oracle applies min/clamp in FP64 to the represented values; the criterion is silu_mul's
// (the clamped arithmetic is silu's plus two fmin/fmax, so its error profile is unchanged).
int run_clamped_case(const char* label, std::int32_t rows, std::int32_t columns, float limit,
                     std::uint32_t seed, bool strided = false) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    if (limit <= 12.0f) {
        // Force values at and beyond both clamp boundaries.
        for (std::size_t i = 0; i < count; i += 3) {
            gate[i] = limit + (i % 2 == 0 ? 1.5f : -3.0f);
        }
        for (std::size_t i = 1; i < count; i += 3) {
            up[i] = (i % 2 == 0 ? 1.0f : -1.0f) * limit * 1.25f;
        }
    }
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up, limit);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    if (strided) {
        // Interleave gate and up rows in one buffer and slice, as silu_mul's strided case does.
        GuardedDeviceBuffer device_gate_up(2 * count * sizeof(std::uint16_t));
        std::vector<std::uint16_t> gate_up(2 * count);
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::size_t source = static_cast<std::size_t>(column) * rows;
            const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
            std::copy_n(gate_bits.data() + source, rows, gate_up.data() + target);
            std::copy_n(up_bits.data() + source, rows, gate_up.data() + target + rows);
        }
        device_gate_up.copy_from_host(gate_up.data(), device_gate_up.bytes());
        Tensor gate_up_tensor(device_gate_up.data(), DType::BF16, {2 * rows, columns});
        Tensor gate_slice = gate_up_tensor.slice(0, 0, rows);
        Tensor up_slice   = gate_up_tensor.slice(0, rows, rows);
        ops::silu_mul_clamped(gate_slice, up_slice, limit, out_tensor, nullptr);
        cuda_synchronize();
        int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                        silu_mul_bf16_criterion());
        failures += verify_exact("silu_mul_clamped strided inputs unchanged",
                                 from_device<std::uint16_t>(device_gate_up.data(), gate_up.size()),
                                 gate_up);
        failures += device_gate_up.verify_guards("silu_mul_clamped strided inputs");
        failures += device_out.verify_guards("silu_mul_clamped strided out");
        return failures;
    }
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    ops::silu_mul_clamped(gate_tensor, up_tensor, limit, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul_clamped gate");
    failures += device_up.verify_guards("silu_mul_clamped up");
    failures += device_out.verify_guards("silu_mul_clamped out");
    return failures;
}

// With a limit above every input magnitude the clamped op is silu_mul bit for bit.
int run_clamped_matches_silu_mul() {
    constexpr std::int32_t rows = 17408, columns = 3;
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 401u, -12.0f, 12.0f);
    fill_uniform(up, 402u, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_plain(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_clamped(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor plain_tensor(device_plain.data(), DType::BF16, {rows, columns});
    Tensor clamped_tensor(device_clamped.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, plain_tensor, nullptr);
    ops::silu_mul_clamped(gate_tensor, up_tensor, 1e30f, clamped_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_exact("silu_mul_clamped with an inert limit equals silu_mul",
                                from_device<std::uint16_t>(device_clamped.data(), count),
                                from_device<std::uint16_t>(device_plain.data(), count));
    return failures;
}

int run_clamped_refusals() {
    std::vector<float> gate(16, 1.0f), up(16, 1.0f);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(gate_bits.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {16});
    Tensor up_tensor(device_up.data(), DType::BF16, {16});
    Tensor out_tensor(device_out.data(), DType::BF16, {16});
    int failures = 0;
    for (const float limit : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()}) {
        try {
            ops::silu_mul_clamped(gate_tensor, up_tensor, limit, out_tensor, nullptr);
            std::cerr << "silu_mul_clamped: limit " << limit << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_contiguous_case("silu_mul [17408,1]", 17408, 1, 101u);
    failures += run_contiguous_case("silu_mul [17408,48]", 17408, 48, 102u);
    failures += run_strided_gate_up_case();
    failures += run_edge_case();
    failures += run_clamped_case("silu_mul_clamped [17408,1] limit 10", 17408, 1, 10.0f, 111u);
    failures += run_clamped_case("silu_mul_clamped [17408,48] limit 10", 17408, 48, 10.0f, 112u);
    failures += run_clamped_case("silu_mul_clamped [6144,16] limit 2.5", 6144, 16, 2.5f, 113u);
    failures += run_clamped_case("silu_mul_clamped strided gate/up", 17408, 17, 10.0f, 114u, true);
    failures += run_clamped_matches_silu_mul();
    failures += run_clamped_refusals();
    std::cout << (failures ? "FAIL" : "OK") << " silu_mul\n";
    return failures ? 1 : 0;
}
