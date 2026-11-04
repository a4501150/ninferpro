#pragma once

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace infernix::models::qwen_common::frontend {

// Immutable declarations and parameter codecs shared by the grammar builder and output decoder.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        InferredTypes,
        DeclaredTypes,
    };

    enum class Encoding : std::uint8_t { NonStrict, RawString, Json };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::InferredTypes;
        TypeSet types;
        Encoding encoding = Encoding::NonStrict;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        std::string schema_json;
        std::size_t declaration_index = 0;
        bool strict                   = false;
    };

    std::vector<Tool> tools;
    bool constrained = false;
    bool required    = false;
    bool parallel    = true;
    // The client knows tools the request does not declare (PromptOptions::open_tool_names): a
    // well-formed call to an undeclared name is the client's to judge, so the parser keeps it a call.
    bool open_names  = false;
};

// Listed declarations come first, then deferred ones (declared but absent from the prompt's tool
// section); declaration_index counts across both.
[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons,
                                std::span<const std::string> deferred_tool_jsons = {},
                                bool open_names = false);

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice);

} // namespace infernix::models::qwen_common::frontend
