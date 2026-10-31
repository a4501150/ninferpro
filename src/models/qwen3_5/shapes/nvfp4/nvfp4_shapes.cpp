// The Qwen3.5 family's tuned NVFP4 linear shapes: one row per shape file in this directory. The
// table's external linkage (nvfp4_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/nvfp4/nvfp4_shapes.h"

namespace infernix::ops::detail {

const Nvfp4LinearShape* const qwen3_5_nvfp4_shapes[] = {
    &kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
    &kNvfp4N5120K6144,  &kNvfp4N5120K17408,
};
const std::size_t qwen3_5_nvfp4_shapes_count =
    sizeof(qwen3_5_nvfp4_shapes) / sizeof(qwen3_5_nvfp4_shapes[0]);

} // namespace infernix::ops::detail
