#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace infernix::ops {

/**
 * Elementwise SwiGLU activation:
 *
 *   ideal[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates `ideal` in FP64 from the represented inputs. The BF16
 * output is promoted and compared directly with that result; output storage rounding belongs to
 * the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op writes all of out and uses no workspace or persistent state.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

/**
 * Op: silu_mul_clamped — clamped SwiGLU activation:
 *
 *   ideal[i] = (min(gate[i], limit) / (1 + exp(-min(gate[i], limit)))) * clamp(up[i], -limit,
 * limit)
 *
 * The gate clamps above only; the up value clamps on both sides. `gate`, `up`, and `out` are
 * same-shaped BF16 tensors. out is contiguous; gate and up may use arbitrary valid Tensor strides.
 * out must not overlap either input (the two read-only inputs may overlap one another). limit must
 * be finite and positive; with a limit at or above every |gate|/|up| magnitude the result is
 * bit-identical to silu_mul's. The oracle evaluates `ideal` in FP64 from the represented inputs,
 * with min/clamp applied in FP64 to the represented values. The BF16 output is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. Private kernel arithmetic is implementation-defined. The Op writes all
 * of out and uses no workspace or persistent state.
 */
void silu_mul_clamped(const Tensor& gate, const Tensor& up, float limit, Tensor& out,
                      cudaStream_t stream);

} // namespace infernix::ops
