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
