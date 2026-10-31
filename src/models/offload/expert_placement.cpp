#include "models/offload/expert_placement.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>


namespace infernix::models::offload {
namespace {

// ---- the --experts spec ----

// One `tier=value` term: the tier name, and whether the value is a count, a percent (kept as
// permille) or `all`.
struct SpecTerm {
    std::string tier;
    bool all = false;
    std::optional<std::uint64_t> count;
    std::optional<std::uint64_t> permille; // percent * 10
};

[[nodiscard]] std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t");
    if (first == std::string::npos) { return {}; }
    const auto last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1);
}

// "N" (a count), "N%" (a percent of at most 100) or "all"; throws with `what` on anything else.
[[nodiscard]] SpecTerm parse_term(const std::string& raw, const std::string& what) {
    SpecTerm term;
    term.tier = what;
    const std::string value = trim(raw);
    if (value == "all") {
        term.all = true;
        return term;
    }
    const bool percent = !value.empty() && value.back() == '%';
    const std::string digits = percent ? value.substr(0, value.size() - 1) : value;
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
                                      [](unsigned char c) { return std::isdigit(c) != 0; })) {
        throw std::invalid_argument("--experts: " + what + " expects a count, a percent or 'all', not '" +
                                    value + "'");
    }
    if (digits.size() > 18) { throw std::invalid_argument("--experts: " + what + " is out of range"); }
    const std::uint64_t n = std::stoull(digits);
    if (percent) {
        if (n > 100) { throw std::invalid_argument("--experts: " + what + " must be at most 100%"); }
        term.permille = n * 10;
    } else {
        term.count = n;
    }
    return term;
}

} // namespace

ExpertsSpec parse_experts_spec(const std::string& text) {
    ExpertsSpec spec;
    std::vector<SpecTerm> terms;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t comma = text.find(',', at);
        const std::string part = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? text.size() : comma + 1;
        const std::size_t equals = part.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument("--experts: every tier is `tier=value`; '" + trim(part) + "' has no '='");
        }
        const std::string tier = trim(part.substr(0, equals));
        if (tier != "vram" && tier != "ram" && tier != "ssd") {
            throw std::invalid_argument("--experts: the tiers are vram, ram and ssd, not '" + tier + "'");
        }
        if (std::any_of(terms.begin(), terms.end(), [&](const SpecTerm& t) { return t.tier == tier; })) {
            throw std::invalid_argument("--experts: " + tier + " appears twice");
        }
        terms.push_back(parse_term(part.substr(equals + 1), tier));
    }
    if (terms.empty()) { throw std::invalid_argument("--experts: the spec is empty"); }
    const auto find = [&](const char* tier) -> const SpecTerm* {
        const auto it = std::find_if(terms.begin(), terms.end(),
                                     [&](const SpecTerm& t) { return t.tier == tier; });
        return it == terms.end() ? nullptr : &*it;
    };
    const SpecTerm* vram = find("vram");
    const SpecTerm* ram = find("ram");
    const SpecTerm* ssd = find("ssd");
    if (vram == nullptr) { throw std::invalid_argument("--experts: the vram tier is required"); }
    // vram=100% with no other tier reads as vram=all.
    const bool vram_all = vram->all || (vram->permille.has_value() && *vram->permille == 1000 &&
                                        ram == nullptr && ssd == nullptr);
    if (vram_all) {
        if (ram != nullptr || ssd != nullptr) {
            throw std::invalid_argument("--experts: vram=all excludes the ram and ssd tiers");
        }
        spec.all = true;
        return spec;
    }
    const bool shares = vram->permille.has_value() || (ram != nullptr && ram->permille.has_value()) ||
                        (ssd != nullptr && ssd->permille.has_value());
    const bool counts = vram->count.has_value() || (ram != nullptr && ram->count.has_value()) ||
                        (ssd != nullptr && ssd->count.has_value());
    if (shares && counts) {
        throw std::invalid_argument("--experts: counts and percentages cannot be mixed; write them all one way");
    }
    if ((vram->count.has_value() && *vram->count == 0) ||
        (vram->permille.has_value() && *vram->permille == 0)) {
        throw std::invalid_argument("--experts: vram is 0; no expert would be in VRAM. Use vram=all for "
                                    "every expert in VRAM, or give the other tiers a share");
    }
    if (shares) {
        spec.vram_permille = vram->permille;
        if (ram != nullptr) { spec.ram_permille = ram->permille; }
        if (ssd != nullptr) { spec.ssd_permille = ssd->permille; }
        const std::uint64_t named = spec.vram_permille.value_or(0) + spec.ram_permille.value_or(0) +
                                    spec.ssd_permille.value_or(0);
        if (spec.ssd_permille.has_value()) {
            if (named != 1000) {
                throw std::invalid_argument("--experts: the shares must add up to 100%, not " +
                                            std::to_string(named / 10.0) + "%");
            }
        } else if (named > 1000) {
            throw std::invalid_argument("--experts: vram and ram add up to more than 100%");
        }
    } else {
        spec.vram_count = vram->count;
        if (ram != nullptr) { spec.ram_count = ram->count; }
        if (ssd != nullptr) { spec.ssd_count = ssd->count; }
    }
    return spec;
}

std::pair<std::uint64_t, std::uint64_t> experts_spec_counts(const ExpertsSpec& spec, std::uint64_t total) {
    if (spec.all) { return {total, 0}; }
    if (spec.vram_count.has_value()) {
        const std::uint64_t vram = *spec.vram_count;
        const std::uint64_t ram = spec.ram_count.value_or(0);
        const std::uint64_t ssd = spec.ssd_count.value_or(0);
        if (vram + ram + ssd > total) {
            throw std::invalid_argument("--experts: " + std::to_string(vram + ram + ssd) +
                                        " experts named of " + std::to_string(total));
        }
        return {vram, ram};
    }
    // Shares: largest remainder, ties to vram then ram, so the counts add up exactly to `total`.
    const std::uint64_t vram_pm = spec.vram_permille.value_or(0);
    const std::uint64_t ram_pm = spec.ram_permille.value_or(0);
    const std::uint64_t ssd_pm = spec.ssd_permille.has_value()
                                     ? *spec.ssd_permille
                                     : 1000 - std::min<std::uint64_t>(vram_pm + ram_pm, 1000);
    const auto share = [&](std::uint64_t permille) { return static_cast<double>(total) * permille / 1000.0; };
    std::uint64_t vram = static_cast<std::uint64_t>(share(vram_pm));
    std::uint64_t ram = static_cast<std::uint64_t>(share(ram_pm));
    std::uint64_t ssd = static_cast<std::uint64_t>(share(ssd_pm));
    std::uint64_t left = total - (vram + ram + ssd);
    while (left != 0) {
        // The largest fractional part gains the expert: vram first, then ram, then ssd.
        const double fv = share(vram_pm) - static_cast<double>(vram);
        const double fr = share(ram_pm) - static_cast<double>(ram);
        const double fs = share(ssd_pm) - static_cast<double>(ssd);
        if (fv >= fr && fv >= fs) {
            ++vram;
        } else if (fr >= fs) {
            ++ram;
        } else {
            ++ssd;
        }
        --left;
    }
    return {vram, ram};
}

namespace {

// The layer-interleaved order of tier_fill_order: expert e of every layer before expert e + 1, so
// no layer is left wholly on the SSD.
[[nodiscard]] std::vector<std::uint32_t> spread_order(std::uint32_t layers, std::uint32_t experts) {
    const std::uint32_t keys = layers * experts;
    std::vector<std::uint32_t> order;
    order.reserve(keys);
    for (std::uint32_t i = 0; i < keys; ++i) { order.push_back((i % layers) * experts + i / layers); }
    return order;
}

void append_spread_tail(std::vector<std::uint32_t>& order, std::uint32_t layers, std::uint32_t experts,
                        std::vector<std::uint8_t>& listed) {
    for (const std::uint32_t key : spread_order(layers, experts)) {
        if (listed[key] == 0) {
            order.push_back(key);
            listed[key] = 1;
        }
    }
}

} // namespace

ExpertPlacementSpec placement_from_counts(std::uint32_t vram_count, std::uint32_t ram_count, std::uint32_t layers,
                                           std::uint32_t experts, std::span<const std::uint32_t> counts,
                                           std::span<const std::uint32_t> ranked) {
    const std::uint32_t keys = layers * experts;
    ExpertPlacementSpec spec;
    spec.from_saved_state = !ranked.empty() || !counts.empty();
    std::vector<std::uint32_t> order;
    std::vector<std::uint8_t> listed(keys, 0);
    // The ranking: the saved order first, then any key the session used (most used first), then
    // the rest spread across the layers.
    for (const std::uint32_t key : ranked) {
        if (key < keys && listed[key] == 0) {
            order.push_back(key);
            listed[key] = 1;
        }
    }
    if (counts.size() == keys) {
        std::vector<std::uint32_t> used;
        for (std::uint32_t key = 0; key < keys; ++key) {
            if (listed[key] == 0 && counts[key] != 0) { used.push_back(key); }
        }
        const auto spread = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
        std::sort(used.begin(), used.end(), [&](std::uint32_t a, std::uint32_t b) {
            const std::uint32_t ca = counts[a], cb = counts[b];
            return ca != cb ? ca > cb : spread(a) < spread(b);
        });
        for (const std::uint32_t key : used) {
            order.push_back(key);
            listed[key] = 1;
        }
    }
    append_spread_tail(order, layers, experts, listed);

    // The top vram_count of the order pin in VRAM, the next ram_count hold RAM slots.
    spec.order = std::move(order);
    const std::size_t in_vram = vram_count != 0 ? std::min<std::size_t>(vram_count, spec.order.size()) : 0;
    spec.vram.assign(spec.order.begin(), spec.order.begin() + static_cast<std::ptrdiff_t>(in_vram));
    if (ram_count != 0) {
        const std::size_t in_ram = std::min<std::size_t>(ram_count, spec.order.size() - in_vram);
        spec.ram.assign(spec.order.begin() + static_cast<std::ptrdiff_t>(in_vram),
                        spec.order.begin() + static_cast<std::ptrdiff_t>(in_vram + in_ram));
    }
    return spec;
}

ExpertPlacementSpec parse_expert_placement_file(const std::filesystem::path& path, std::uint32_t layers,
                                                std::uint32_t experts) {
    const std::uint32_t keys = layers * experts;
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("expert placement: cannot open " + path.string()); }
    ExpertPlacementSpec spec;
    std::vector<std::uint32_t> ssd; // rows explicitly ssd (ranked above the unlisted tail)
    std::vector<std::uint8_t> listed(keys, 0);
    std::string line;
    std::size_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        const auto hash = line.find('#');
        if (hash != std::string::npos) { line.erase(hash); }
        const auto blank = line.find_first_not_of(" \t\r");
        if (blank == std::string::npos) { continue; }
        if (line.compare(0, 19, "layer,expert,tier,") == 0 || line.compare(0, 17, "layer,expert,tier") == 0) {
            continue; // the export's header
        }
        std::istringstream row(line);
        std::string layer_text, expert_text, tier, extra;
        if (!std::getline(row, layer_text, ',') || !std::getline(row, expert_text, ',') ||
            !std::getline(row, tier, ',')) {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " is not `layer,expert,tier`");
        }
        // A fourth column (the export's `uses`) is accepted and ignored: the export writes it so a
        // loaded file reproduces the ranking it recorded.
        (void)std::getline(row, extra, ',');
        if (std::getline(row, extra, ',')) {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " has more than four columns");
        }
        char* end  = nullptr;
        const long layer  = std::strtol(layer_text.c_str(), &end, 10);
        if (end == layer_text.c_str() || *end != '\0' || layer < 0 || layer >= static_cast<long>(layers)) {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " names layer " + layer_text + " outside [0, " + std::to_string(layers) + ")");
        }
        const long expert = std::strtol(expert_text.c_str(), &end, 10);
        if (end == expert_text.c_str() || *end != '\0' || expert < 0 || expert >= static_cast<long>(experts)) {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " names expert " + expert_text + " outside [0, " + std::to_string(experts) + ")");
        }
        while (!tier.empty() && (tier.back() == ' ' || tier.back() == '\t' || tier.back() == '\r')) { tier.pop_back(); }
        const std::uint32_t key = static_cast<std::uint32_t>(layer) * experts + static_cast<std::uint32_t>(expert);
        if (listed[key] != 0) {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " names layer " + layer_text + " expert " + expert_text + " twice");
        }
        listed[key] = 1;
        if (tier == "vram") { spec.vram.push_back(key); }
        else if (tier == "ram") { spec.ram.push_back(key); }
        else if (tier == "ssd") { ssd.push_back(key); }
        else {
            throw std::runtime_error("expert placement: " + path.string() + " line " + std::to_string(number) +
                                     " names tier " + tier + "; expected vram, ram or ssd");
        }
    }
    if (spec.vram.empty() && spec.ram.empty() && ssd.empty()) {
        throw std::runtime_error("expert placement: " + path.string() + " assigns no expert");
    }
    // The fill order: the vram set, then the ram set, then the explicitly-ssd rows, then every
    // unlisted key spread across the layers.
    spec.order.reserve(keys);
    std::vector<std::uint8_t> ordered(keys, 0);
    const auto take = [&](std::uint32_t key) {
        if (ordered[key] == 0) {
            spec.order.push_back(key);
            ordered[key] = 1;
        }
    };
    for (const std::uint32_t key : spec.vram) { take(key); }
    for (const std::uint32_t key : spec.ram) { take(key); }
    for (const std::uint32_t key : ssd) { take(key); }
    for (std::uint32_t key = 0; key < keys; ++key) {
        if (ordered[key] == 0 && listed[key] != 0) { take(key); }
    }
    append_spread_tail(spec.order, layers, experts, ordered);
    return spec;
}

bool write_expert_placement_file(const std::filesystem::path& path, std::uint32_t layers, std::uint32_t experts,
                                 std::span<const std::int32_t> frame_table, std::span<const std::uint32_t> ram_keys,
                                 std::span<const std::uint32_t> counts, std::string* error) {
    const std::uint32_t keys = layers * experts;
    if (frame_table.size() != keys) {
        if (error != nullptr) { *error = "the frame table has " + std::to_string(frame_table.size()) + " entries"; }
        return false;
    }
    // A host copy in the RAM tier ranks the key ram unless it is also in VRAM (the frame wins).
    std::vector<std::uint8_t> host_copy(keys, 0);
    for (const std::uint32_t key : ram_keys) {
        if (key < keys) { host_copy[key] = 1; }
    }
    // Best first: the session's use counts, so the file reproduces the ranking; a key without a
    // count ranks by its spread position after them.
    std::vector<std::uint32_t> order;
    order.reserve(keys);
    const auto spread = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
    if (counts.size() == keys) {
        for (std::uint32_t key = 0; key < keys; ++key) { order.push_back(key); }
        std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            const std::uint32_t ca = counts[a], cb = counts[b];
            return ca != cb ? ca > cb : spread(a) < spread(b);
        });
    } else {
        order = spread_order(layers, experts);
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        if (error != nullptr) { *error = "cannot open " + path.string() + " for writing"; }
        return false;
    }
    out << "# infernix expert placement v1\n";
    out << "# tier is vram, ram or ssd; rows are ordered by use count, best first\n";
    out << "layer,expert,tier,uses\n";
    for (const std::uint32_t key : order) {
        const char* tier = frame_table[key] >= 0 ? "vram" : host_copy[key] != 0 ? "ram" : "ssd";
        out << (key / experts) << ',' << (key % experts) << ',' << tier;
        if (counts.size() == keys) { out << ',' << counts[key]; }
        out << '\n';
    }
    if (!out.good()) {
        if (error != nullptr) { *error = "failed writing " + path.string(); }
        return false;
    }
    return true;
}

std::vector<std::uint32_t> placement_refill_batch(std::span<const std::uint32_t> order, std::size_t& at,
                                                  std::uint32_t batch, std::span<const std::int32_t> frame_table) {
    std::vector<std::uint32_t> want;
    while (at < order.size() && want.size() < batch) {
        const std::uint32_t key = order[at++];
        if (key < frame_table.size() && frame_table[key] < 0) { want.push_back(key); }
    }
    return want;
}

} // namespace infernix::models::offload
