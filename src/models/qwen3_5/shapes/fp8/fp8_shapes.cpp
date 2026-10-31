// The Qwen3.5 family's tuned FP8 linear shapes: one row per shape file in this directory. The
// table's external linkage (fp8_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/fp8/fp8_shapes.h"

namespace infernix::ops::detail {

const Fp8LinearShape* const qwen3_5_fp8_shapes[] = {
    &kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
    &kFp8N5120K6144,  &kFp8N5120K17408,  &kFp8N248320K5120,
};
const std::size_t qwen3_5_fp8_shapes_count =
    sizeof(qwen3_5_fp8_shapes) / sizeof(qwen3_5_fp8_shapes[0]);

} // namespace infernix::ops::detail
