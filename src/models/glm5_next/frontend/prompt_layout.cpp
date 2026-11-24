#include "models/glm5_next/frontend/prompt_layout.h"

#include <algorithm>
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

// The template's media wrappers; the processor expands the middle token.
struct MediaWrapper {
    std::string_view open, token, close;
};

constexpr MediaWrapper kImageWrapper{"<|begin_of_image|>", "<|image|>", "<|end_of_image|>"};
constexpr MediaWrapper kVideoWrapper{"<|begin_of_video|>", "<|video|>", "<|end_of_video|>"};

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

    // Each media input contributes one complete wrapper in input order. The next token of each
    // kind is searched again only once it is consumed, so the scan stays linear in the prompt.
    std::size_t image = find_marker(output, kImageWrapper.token, 0);
    std::size_t video = find_marker(output, kVideoWrapper.token, 0);
    for (;;) {
        const auto at = std::min(image, video);
        if (at == std::string::npos) break;
        const auto modality         = at == image ? Modality::Image : Modality::Video;
        const MediaWrapper& wrapper = modality == Modality::Image ? kImageWrapper : kVideoWrapper;
        const auto end              = at + wrapper.token.size();
        if (at < wrapper.open.size() ||
            source.substr(at - wrapper.open.size(), wrapper.open.size()) != wrapper.open ||
            source.substr(end, wrapper.close.size()) != wrapper.close ||
            !template_bytes(output, at - wrapper.open.size(), end + wrapper.close.size())) {
            throw std::invalid_argument("chat template media token requires its complete wrapper");
        }
        const auto index = result.media_placeholders.size();
        if (index >= media.size() || media[index] != modality) {
            throw std::invalid_argument(
                "chat template media placeholders must match input count, type and order");
        }
        result.media_placeholders.push_back({{at, end}, modality, index});
        (modality == Modality::Image ? image : video) = find_marker(output, wrapper.token, end);
    }
    if (result.media_placeholders.size() != media.size()) {
        throw std::invalid_argument("chat template omitted an input media placeholder");
    }
    return result;
}

} // namespace infernix::models::glm5_next::frontend
