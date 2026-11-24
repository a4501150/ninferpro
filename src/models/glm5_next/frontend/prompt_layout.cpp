#include "models/glm5_next/frontend/prompt_layout.h"

#include <array>
#include <stdexcept>

namespace infernix::models::glm5_next::frontend {
namespace {

// GLM frames a turn with a role marker only; a turn runs to the next marker.
struct RoleMarker {
    std::string_view text;
    ChatRole role;
};
constexpr std::array kRoleMarkers{
    RoleMarker{"<|system|>", ChatRole::System},
    RoleMarker{"<|user|>", ChatRole::User},
    RoleMarker{"<|assistant|>", ChatRole::Assistant},
    RoleMarker{"<|observation|>", ChatRole::Tool},
};
constexpr std::string_view kThinkOpen = "<think>";

bool template_bytes(const text::TemplateOutput& output, std::size_t begin, std::size_t end) {
    return !text::overlaps(output.literal_spans, begin, end);
}

std::size_t find_marker(const text::TemplateOutput& output, std::string_view marker,
                        std::size_t begin) {
    for (auto pos = output.text.find(marker, begin); pos != std::string::npos;
         pos      = output.text.find(marker, pos + marker.size())) {
        if (template_bytes(output, pos, pos + marker.size())) return pos;
    }
    return std::string::npos;
}

// The earliest template-emitted role marker at or after begin.
std::optional<std::pair<std::size_t, const RoleMarker*>>
next_role(const text::TemplateOutput& output, std::size_t begin) {
    std::optional<std::pair<std::size_t, const RoleMarker*>> result;
    for (const auto& marker : kRoleMarkers) {
        const auto pos = find_marker(output, marker.text, begin);
        if (pos != std::string::npos && (!result || pos < result->first)) result = {pos, &marker};
    }
    return result;
}

} // namespace

std::optional<text::ByteSpan> unique_output_region(const text::TemplateOutput& output,
                                                   std::uint32_t tag) {
    std::optional<text::ByteSpan> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag) continue;
        if (result) return std::nullopt;
        result = text::ByteSpan{region.begin, region.end};
    }
    return result;
}

std::optional<std::size_t> source_boundary(const text::TemplateOutput& output, std::uint32_t tag,
                                           std::size_t offset) {
    std::optional<std::size_t> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag || !region.source_offset) continue;
        const auto begin = *region.source_offset;
        if (offset < begin || offset - begin > region.end - region.begin) continue;
        const auto mapped = region.begin + offset - begin;
        if (result && *result != mapped) return std::nullopt;
        result = mapped;
    }
    for (const auto& mapping : output.boundary_mappings) {
        if (mapping.tag != tag || offset < mapping.source_begin || offset > mapping.source_end)
            continue;
        if (result && *result != mapping.offset) return std::nullopt;
        result = mapping.offset;
    }
    return result;
}

PromptLayout inspect_prompt_layout(const text::TemplateOutput& output,
                                   std::span<const Modality> media) {
    if (!media.empty()) throw std::invalid_argument("this model accepts text input only");
    PromptLayout result;
    const std::string_view source = output.text;
    for (auto role = next_role(output, 0); role;) {
        const auto [pos, marker] = *role;
        const auto content_begin = pos + marker->text.size();
        const auto next          = next_role(output, content_begin);
        const bool closed        = next.has_value();
        const auto end           = closed ? next->first : source.size();
        result.messages.push_back({marker->role, pos, content_begin, end, end, closed});
        if (marker->role == ChatRole::Assistant) {
            result.execution_boundaries.push_back(content_begin);
            const auto body = source.substr(content_begin, end - content_begin);
            const bool opens =
                body.starts_with(kThinkOpen) &&
                template_bytes(output, content_begin, content_begin + kThinkOpen.size());
            auto close = std::string::npos;
            if (opens) {
                result.execution_boundaries.push_back(content_begin + kThinkOpen.size());
                close = find_marker(output, kCanonicalReasoningCloseSerialization,
                                    content_begin + kThinkOpen.size());
                if (close != std::string::npos && close < end) {
                    result.execution_boundaries.push_back(
                        close + kCanonicalReasoningCloseSerialization.size());
                }
            }
            // Only the final assistant prefix controls the initial output channel.
            if (!closed) result.starts_in_reasoning = opens && close == std::string::npos;
        }
        role = next;
    }
    return result;
}

} // namespace infernix::models::glm5_next::frontend
