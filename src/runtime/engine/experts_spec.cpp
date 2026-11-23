#include "runtime/engine/experts_spec.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::runtime {
namespace {

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
    term.tier               = what;
    const std::string value = trim(raw);
    if (value == "all") {
        term.all = true;
        return term;
    }
    const bool percent       = !value.empty() && value.back() == '%';
    const std::string digits = percent ? value.substr(0, value.size() - 1) : value;
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; })) {
        throw std::invalid_argument("--experts: " + what +
                                    " expects a count, a percent or 'all', not '" + value + "'");
    }
    if (digits.size() > 18) {
        throw std::invalid_argument("--experts: " + what + " is out of range");
    }
    const std::uint64_t n = std::stoull(digits);
    if (percent) {
        if (n > 100) {
            throw std::invalid_argument("--experts: " + what + " must be at most 100%");
        }
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
        const std::string part =
            text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at                       = comma == std::string::npos ? text.size() : comma + 1;
        const std::size_t equals = part.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument("--experts: every tier is `tier=value`; '" + trim(part) +
                                        "' has no '='");
        }
        const std::string tier = trim(part.substr(0, equals));
        if (tier != "vram" && tier != "ram" && tier != "ssd") {
            throw std::invalid_argument("--experts: the tiers are vram, ram and ssd, not '" + tier +
                                        "'");
        }
        if (std::any_of(terms.begin(), terms.end(),
                        [&](const SpecTerm& t) { return t.tier == tier; })) {
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
    const SpecTerm* ram  = find("ram");
    const SpecTerm* ssd  = find("ssd");
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
    const bool shares = vram->permille.has_value() ||
                        (ram != nullptr && ram->permille.has_value()) ||
                        (ssd != nullptr && ssd->permille.has_value());
    const bool counts = vram->count.has_value() || (ram != nullptr && ram->count.has_value()) ||
                        (ssd != nullptr && ssd->count.has_value());
    if (shares && counts) {
        throw std::invalid_argument(
            "--experts: counts and percentages cannot be mixed; write them all one way");
    }
    if ((vram->count.has_value() && *vram->count == 0) ||
        (vram->permille.has_value() && *vram->permille == 0)) {
        throw std::invalid_argument(
            "--experts: vram is 0; no expert would be in VRAM. Use vram=all for "
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

std::pair<std::uint64_t, std::uint64_t> experts_spec_counts(const ExpertsSpec& spec,
                                                            std::uint64_t total) {
    if (spec.all) { return {total, 0}; }
    if (spec.vram_count.has_value()) {
        const std::uint64_t vram = *spec.vram_count;
        const std::uint64_t ram  = spec.ram_count.value_or(0);
        const std::uint64_t ssd  = spec.ssd_count.value_or(0);
        if (vram + ram + ssd > total) {
            throw std::invalid_argument("--experts: " + std::to_string(vram + ram + ssd) +
                                        " experts named of " + std::to_string(total));
        }
        return {vram, ram};
    }
    // Shares: largest remainder, ties to vram then ram, so the counts add up exactly to `total`.
    const std::uint64_t vram_pm = spec.vram_permille.value_or(0);
    const std::uint64_t ram_pm  = spec.ram_permille.value_or(0);
    const std::uint64_t ssd_pm  = spec.ssd_permille.has_value()
                                      ? *spec.ssd_permille
                                      : 1000 - std::min<std::uint64_t>(vram_pm + ram_pm, 1000);
    const auto share            = [&](std::uint64_t permille) {
        return static_cast<double>(total) * permille / 1000.0;
    };
    std::uint64_t vram = static_cast<std::uint64_t>(share(vram_pm));
    std::uint64_t ram  = static_cast<std::uint64_t>(share(ram_pm));
    std::uint64_t ssd  = static_cast<std::uint64_t>(share(ssd_pm));
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

} // namespace infernix::runtime
