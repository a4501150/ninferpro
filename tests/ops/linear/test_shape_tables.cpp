// Cross-family disjointness of the linear dispatch's family shape tables
// (src/models/<family>/shapes/<dtype>/<dtype>_shapes.cpp).
//
// The dispatch scans the family tables in order and the first matching (n, k) row wins, so a
// key tuned in two families would silently shadow one family's entry (the former single closed
// table made an overlap impossible; the removed registration seam aborted on any duplicate key
// at static init). This asserts the invariant on the linked tables themselves: wherever a dtype
// has entries in both families their (n, k) sets are disjoint, and no table repeats a key.
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q6/q6_shapes.h"
#include "ops/linear/q8/q8_shapes.h"

#include <cstdio>
#include <set>
#include <utility>
#include <vector>

namespace {

using infernix::ops::detail::qwen3_5_bf16_shapes;
using infernix::ops::detail::qwen3_5_bf16_shapes_count;
using infernix::ops::detail::qwen3_5_fp8_shapes;
using infernix::ops::detail::qwen3_5_fp8_shapes_count;
using infernix::ops::detail::qwen3_5_nvfp4_shapes;
using infernix::ops::detail::qwen3_5_nvfp4_shapes_count;
using infernix::ops::detail::qwen3_5_q4_shapes;
using infernix::ops::detail::qwen3_5_q4_shapes_count;
using infernix::ops::detail::qwen3_5_q5_shapes;
using infernix::ops::detail::qwen3_5_q5_shapes_count;
using infernix::ops::detail::qwen3_5_q6_shapes;
using infernix::ops::detail::qwen3_5_q6_shapes_count;
using infernix::ops::detail::qwen3_5_q8_shapes;
using infernix::ops::detail::qwen3_5_q8_shapes_count;
using infernix::ops::detail::qwen4_exp_bf16_shapes;
using infernix::ops::detail::qwen4_exp_bf16_shapes_count;
using infernix::ops::detail::qwen4_exp_fp8_shapes;
using infernix::ops::detail::qwen4_exp_fp8_shapes_count;
using infernix::ops::detail::qwen4_exp_q8_shapes;
using infernix::ops::detail::qwen4_exp_q8_shapes_count;

using Key = std::pair<std::int32_t, std::int32_t>;

template <class Entry>
std::vector<Key> entry_keys(const Entry* shapes, std::size_t count) {
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) keys.emplace_back(shapes[i].n, shapes[i].k);
    return keys;
}

template <class Entry>
std::vector<Key> entry_keys(const Entry* const* shapes, std::size_t count) {
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) keys.emplace_back(shapes[i]->n, shapes[i]->k);
    return keys;
}

int unique(const char* table, const std::vector<Key>& keys) {
    int failures = 0;
    std::set<Key> seen;
    for (const Key& key : keys) {
        if (seen.insert(key).second) continue;
        std::printf("FAIL %s: duplicate (n,k) = (%d,%d)\n", table, key.first, key.second);
        ++failures;
    }
    return failures;
}

int disjoint(const char* dtype, const std::vector<Key>& qwen3_5, const std::vector<Key>& qwen4_exp) {
    const std::set<Key> other(qwen4_exp.begin(), qwen4_exp.end());
    int failures = 0;
    for (const Key& key : qwen3_5) {
        if (other.count(key) == 0) continue;
        std::printf("FAIL %s: (n,k) = (%d,%d) is tuned in both qwen3_5 and qwen4_exp; "
                    "the dispatch would run only qwen3_5's entry\n",
                    dtype, key.first, key.second);
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;

    const auto bf16_qwen3_5 = entry_keys(qwen3_5_bf16_shapes, qwen3_5_bf16_shapes_count);
    const auto bf16_qwen4_exp = entry_keys(qwen4_exp_bf16_shapes, qwen4_exp_bf16_shapes_count);
    failures += unique("qwen3_5 bf16", bf16_qwen3_5);
    failures += unique("qwen4_exp bf16", bf16_qwen4_exp);
    failures += disjoint("bf16", bf16_qwen3_5, bf16_qwen4_exp);

    const auto fp8_qwen3_5 = entry_keys(qwen3_5_fp8_shapes, qwen3_5_fp8_shapes_count);
    const auto fp8_qwen4_exp = entry_keys(qwen4_exp_fp8_shapes, qwen4_exp_fp8_shapes_count);
    failures += unique("qwen3_5 fp8", fp8_qwen3_5);
    failures += unique("qwen4_exp fp8", fp8_qwen4_exp);
    failures += disjoint("fp8", fp8_qwen3_5, fp8_qwen4_exp);

    const auto q8_qwen3_5 = entry_keys(qwen3_5_q8_shapes, qwen3_5_q8_shapes_count);
    const auto q8_qwen4_exp = entry_keys(qwen4_exp_q8_shapes, qwen4_exp_q8_shapes_count);
    failures += unique("qwen3_5 q8", q8_qwen3_5);
    failures += unique("qwen4_exp q8", q8_qwen4_exp);
    failures += disjoint("q8", q8_qwen3_5, q8_qwen4_exp);

    failures += unique("qwen3_5 q4", entry_keys(qwen3_5_q4_shapes, qwen3_5_q4_shapes_count));
    failures += unique("qwen3_5 q5", entry_keys(qwen3_5_q5_shapes, qwen3_5_q5_shapes_count));
    failures += unique("qwen3_5 q6", entry_keys(qwen3_5_q6_shapes, qwen3_5_q6_shapes_count));
    failures += unique("qwen3_5 nvfp4", entry_keys(qwen3_5_nvfp4_shapes, qwen3_5_nvfp4_shapes_count));

    if (failures == 0) std::puts("OK linear family shape tables disjoint");
    return failures == 0 ? 0 : 1;
}
