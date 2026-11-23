#pragma once

#include <cstddef>

namespace infernix::ops::detail {

// One table of tuned linear shapes of a format. `count` is held by address so that a registry of
// sets is constant-initialized: the table owners define their counts in other translation units.
template <class Row>
struct LinearShapeSet {
    const Row* rows;
    const std::size_t* count;
};

} // namespace infernix::ops::detail
