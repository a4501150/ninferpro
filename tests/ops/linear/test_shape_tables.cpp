// Disjointness of the linear dispatch's registered shape tables
// (src/models/<family>/shapes/<dtype>/<dtype>_shapes.cpp, registered by src/models/shapes.cpp).
//
// The dispatch scans the registered tables in order and the first matching (n, k) row wins, so a
// key tuned in two tables would silently shadow one table's entry. This asserts the invariant on
// the linked registries themselves: across every table of a dtype, no (n, k) key repeats.
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q6/q6_shapes.h"
#include "ops/linear/q8/q8_shapes.h"

#include <cstdio>
#include <map>
#include <utility>

namespace {

using infernix::ops::detail::LinearShapeSet;
using Key = std::pair<std::int32_t, std::int32_t>;

template <class Row>
const Row& row_entry(const Row& row) {
    return row;
}

template <class Row>
const Row& row_entry(const Row* row) {
    return *row;
}

template <class Row>
int unique_keys(const char* dtype, const LinearShapeSet<Row>* sets, std::size_t count) {
    std::map<Key, std::size_t> owner;
    int failures = 0;
    for (std::size_t s = 0; s < count; ++s) {
        for (std::size_t i = 0; i < *sets[s].count; ++i) {
            const auto& entry = row_entry(sets[s].rows[i]);
            const Key key{entry.n, entry.k};
            const auto [it, inserted] = owner.emplace(key, s);
            if (inserted) continue;
            std::printf("FAIL %s: (n,k) = (%d,%d) is in table %zu and table %zu; the dispatch "
                        "would run only table %zu's entry\n",
                        dtype, key.first, key.second, it->second, s, it->second);
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    namespace d  = infernix::ops::detail;
    int failures = 0;
    failures += unique_keys("bf16", d::bf16_linear_shape_sets, d::bf16_linear_shape_set_count);
    failures += unique_keys("fp8", d::fp8_linear_shape_sets, d::fp8_linear_shape_set_count);
    failures += unique_keys("nvfp4", d::nvfp4_linear_shape_sets, d::nvfp4_linear_shape_set_count);
    failures += unique_keys("q4", d::q4_linear_shape_sets, d::q4_linear_shape_set_count);
    failures += unique_keys("q5", d::q5_linear_shape_sets, d::q5_linear_shape_set_count);
    failures += unique_keys("q6", d::q6_linear_shape_sets, d::q6_linear_shape_set_count);
    failures += unique_keys("q8", d::q8_linear_shape_sets, d::q8_linear_shape_set_count);
    if (failures == 0) std::puts("OK linear shape tables disjoint");
    return failures == 0 ? 0 : 1;
}
