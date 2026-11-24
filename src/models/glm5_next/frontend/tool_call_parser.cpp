#include "models/glm5_next/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace infernix::models::glm5_next::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = common::frontend::ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
    // Tolerant mode repaired its opener or kept an undeclared name the strict parser rejects.
    bool repaired = false;
};

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

// GLM writes a call only as `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
// </tool_call>`; no other spelling opens a call.
bool is_prefix_of_marker(std::string_view prefix) { return kToolOpen.starts_with(prefix); }

bool matches_marker(std::string_view text) { return text.starts_with(kToolOpen); }

std::size_t find_first_tool_marker(std::string_view text) { return text.find(kToolOpen); }

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

// Model bytes need not be valid UTF-8 (a recovered call can end inside a sequence): invalid
// sequences encode as U+FFFD instead of throwing.
std::string encode_json_string(std::string_view value) {
    return Json(std::string(value)).dump(-1, ' ', false, Json::error_handler_t::replace);
}

// A model writing an array or object argument often puts a real line break (or tab) inside one of
// its strings, a multi-line edit for example, which strict JSON refuses (Strata #4630). The value
// with the control bytes inside its string literals escaped is the one the model meant; nullopt
// when there were none.
std::optional<std::string> escape_string_controls(std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 16);
    bool in_string = false, escaped = false, changed = false;
    for (const char c : value) {
        const auto byte = static_cast<unsigned char>(c);
        if (!in_string) {
            in_string = c == '"';
        } else if (escaped) {
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            in_string = false;
        } else if (byte < 0x20) {
            changed = true;
            switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                out += "\\u00";
                out.push_back(kHex[byte >> 4U]);
                out.push_back(kHex[byte & 15U]);
            }
            continue;
        }
        out.push_back(c);
    }
    if (!changed) { return std::nullopt; }
    return out;
}

// Strict JSON first; an array or object refused only for control bytes inside its strings is read
// with them escaped.
std::optional<std::string> parse_json_value(std::string_view value, JsonValueKind& kind) {
    if (classify_json_value(value, kind)) { return std::string(value); }
    if (value.empty() || (value.front() != '[' && value.front() != '{')) { return std::nullopt; }
    auto escaped = escape_string_controls(value);
    if (escaped && classify_json_value(*escaped, kind)) { return escaped; }
    return std::nullopt;
}

// The deepest array / object nesting a generated argument value keeps as JSON. The server builds
// JSON trees from a call's arguments (Anthropic `input`) and serializes them recursively, so a
// degenerate value nested thousands deep could exhaust a server thread's stack; real arguments
// stay far below it (candidates LE-14).
constexpr int kMaxArgumentNesting = 256;

// The deepest array / object nesting of JSON text, outside its strings.
int json_nesting(std::string_view text) {
    int depth = 0, deepest = 0;
    bool quoted = false, escaped = false;
    for (const char c : text) {
        if (quoted) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                quoted = false;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == '[' || c == '{') {
            deepest = std::max(deepest, ++depth);
        } else if (c == ']' || c == '}') {
            --depth;
        }
    }
    return deepest;
}

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    // The template writes a string value verbatim and any other value as JSON.
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(encoded_value)};
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (auto json = parse_json_value(value, kind)) {
        if (json_nesting(*json) > kMaxArgumentNesting) {
            return {.disposition = ParameterNormalization::SchemaMismatch,
                    .json_value  = encode_json_string(encoded_value)};
        }
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::move(*json)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(encoded_value)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    // JSON nested past kMaxArgumentNesting stays text: a JSON string, never a tree too deep to
    // serialize.
    const auto json_or_text = [&](std::string json) -> NormalizedParameter {
        if (json_nesting(json) > kMaxArgumentNesting) {
            return {.disposition = ParameterNormalization::SchemaMismatch,
                    .json_value  = encode_json_string(value)};
        }
        return {.json_value = std::move(json)};
    };
    if (Json::accept(value.begin(), value.end())) { return json_or_text(std::string(value)); }
    JsonValueKind kind;
    if (auto json = parse_json_value(value, kind)) { return json_or_text(std::move(*json)); }
    return {.json_value = encode_json_string(value)};
}

constexpr std::string_view kKeyOpen    = "<arg_key>";
constexpr std::string_view kKeyClose   = "</arg_key>";
constexpr std::string_view kValueOpen  = "<arg_value>";
constexpr std::string_view kValueClose = "</arg_value>";

// Free output's calls. Format whitespace may separate the call's pieces; a value runs to its first
// closer, as the grammar's raw string does.
class GlmToolRegionParser {
public:
    GlmToolRegionParser(std::string_view text, std::size_t max_name_length,
                        const Contract& contract, bool tolerant)
        : text_(text), max_name_length_(max_name_length), contract_(contract),
          tolerant_(tolerant) {}

    [[nodiscard]] std::uint32_t duplicate_parameters_repaired() const noexcept {
        return duplicate_parameters_repaired_;
    }

    FallbackReason parse(std::vector<RawToolCall>& calls) {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (!starts_with_at(text_, pos, kToolOpen)) {
                // Tolerant: a trailing suffix after one or more complete calls is discarded
                // rather than failing the whole output.
                if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }
            RawToolCall call;
            const FallbackReason terminal = finish_call(calls, call, parse_tool_call(pos, call));
            if (terminal != FallbackReason::None) { return terminal; }
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    // Applies tolerant recovery to a call's result and reports whether parse() should terminate.
    // In tolerant mode, a malformed suffix after one or more complete calls is discarded, and a
    // single truncated final call whose name and at least one argument are complete is retained.
    FallbackReason finish_call(std::vector<RawToolCall>& calls, RawToolCall& call,
                               FallbackReason failure) {
        if (failure == FallbackReason::None) {
            calls.push_back(std::move(call));
            return FallbackReason::None;
        }
        if (tolerant_ && !calls.empty()) { return FallbackReason::TruncatedTail; }
        if (tolerant_ && failure == FallbackReason::TruncatedTail && calls.empty() &&
            !call.parameters.empty()) {
            calls.push_back(std::move(call));
            return FallbackReason::TruncatedTail;
        }
        return failure;
    }

    // The failure for a region that ends where more markup was due: a truncation in tolerant
    // mode, a malformed structure otherwise.
    FallbackReason cut(std::size_t pos) const {
        std::size_t at = pos;
        skip_format_whitespace(text_, at);
        return tolerant_ && at == text_.size() ? FallbackReason::TruncatedTail
                                               : FallbackReason::MalformedStructure;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) {
        if (!consume(pos, kToolOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t key   = text_.find(kKeyOpen, pos);
        const std::size_t close = text_.find(kToolClose, pos);
        const std::size_t name_end = std::min(key, close);
        if (name_end == std::string_view::npos) {
            return tolerant_ ? FallbackReason::TruncatedTail : FallbackReason::MalformedStructure;
        }
        call.name = trim_format_whitespace(text_.substr(pos, name_end - pos));
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        // Strict mode rejects a name outside the declared tool set, unless the client keeps tool
        // definitions out of the request (Contract::open_names). Tolerant mode keeps an otherwise
        // well-formed call structured and leaves the identity judgment to the consumer.
        if (find_tool_contract(contract_, call.name) == nullptr) {
            if (!tolerant_ && !contract_.open_names) { return FallbackReason::UndeclaredTool; }
            if (!contract_.open_names) { call.repaired = true; }
        }
        pos = name_end;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, kToolClose)) { return FallbackReason::None; }
            if (pos == text_.size()) { return cut(pos); }
            const FallbackReason failure = parse_argument(pos, call);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_argument(std::size_t& pos, RawToolCall& call) {
        if (!consume(pos, kKeyOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t key_end = text_.find(kKeyClose, pos);
        if (key_end == std::string_view::npos) { return cut(text_.size()); }
        const std::string_view name = trim_format_whitespace(text_.substr(pos, key_end - pos));
        if (name.empty() || name.find_first_of("<>\r\n") != std::string_view::npos) {
            return FallbackReason::MalformedStructure;
        }
        pos = key_end + kKeyClose.size();
        skip_format_whitespace(text_, pos);
        if (!consume(pos, kValueOpen)) { return cut(pos); }
        const std::size_t value_end = text_.find(kValueClose, pos);
        if (value_end == std::string_view::npos) {
            if (!tolerant_) { return FallbackReason::MalformedStructure; }
            // Tolerant: the output budget cut the value. Keep it up to the cut and flag the tail.
            record(call, name, text_.substr(pos));
            pos = text_.size();
            return FallbackReason::TruncatedTail;
        }
        record(call, name, text_.substr(pos, value_end - pos));
        pos = value_end + kValueClose.size();
        return FallbackReason::None;
    }

    // Last occurrence wins, as it would in JSON object syntax, rather than discarding an
    // otherwise well-formed call.
    void record(RawToolCall& call, std::string_view name, std::string_view value) {
        const auto existing =
            std::find_if(call.parameters.begin(), call.parameters.end(),
                         [&](const RawParameter& candidate) { return candidate.name == name; });
        if (existing != call.parameters.end()) {
            existing->value = value;
            ++duplicate_parameters_repaired_;
        } else {
            call.parameters.push_back(RawParameter{.name = name, .value = value});
        }
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    std::uint32_t duplicate_parameters_repaired_ = 0;
    bool tolerant_ = false;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

// Canonical constrained calls are parsed one at a time. A terminal interruption can discard the
// unfinished suffix while retaining preceding calls. JSON values are delimited lexically, so a
// marker inside a quoted value never becomes an argument boundary.
class ConstrainedToolRegionParser {
public:
    enum class Status { Complete, Incomplete, Invalid };

    ConstrainedToolRegionParser(std::string_view input, const Contract& contract)
        : input_(input), contract_(contract) {}

    Status parse(std::vector<GeneratedToolCall>& calls, ToolCallParseDiagnostics& diagnostics) {
        while (position_ < input_.size()) {
            RawToolCall raw;
            if (!take(kToolOpen)) return status_;
            // The name runs to the first argument or the call's close; the grammar spells one of
            // the declared names there.
            const auto rest     = input_.substr(position_);
            const auto name_end = std::min(rest.find(kKeyOpen), rest.find(kToolClose));
            if (name_end == std::string_view::npos) return Status::Incomplete;
            raw.name         = rest.substr(0, name_end);
            const auto* tool = find_tool_contract(contract_, raw.name);
            if (!tool) return Status::Invalid;
            position_ += name_end;
            std::ptrdiff_t previous = -1;
            while (!input_.substr(position_).starts_with(kToolClose)) {
                if (kToolClose.starts_with(input_.substr(position_))) return Status::Incomplete;
                if (!take(kKeyOpen)) return status_;
                const auto name_close = input_.find(kKeyClose, position_);
                if (name_close == std::string_view::npos) return Status::Incomplete;
                const auto name       = input_.substr(position_, name_close - position_);
                const auto* parameter = find_parameter_contract(*tool, name);
                if (name.empty() || name.find_first_of("<>\r\n") != std::string_view::npos)
                    return Status::Invalid;
                if (tool->strict) {
                    if (!parameter) return Status::Invalid;
                    const auto index = parameter - tool->parameters.data();
                    if (index <= previous) return Status::Invalid;
                    previous = index;
                }
                position_ = name_close + kKeyClose.size();
                if (!take(kValueOpen)) return status_;
                const auto value_begin = position_;
                if (parameter && parameter->encoding == Contract::Encoding::Json) {
                    if (!json_value()) return status_;
                } else {
                    // A raw value cannot hold its closer, so its first closer ends it.
                    const auto close = input_.find(kValueClose, position_);
                    if (close == std::string_view::npos) return Status::Incomplete;
                    position_ = close;
                }
                const auto value = input_.substr(value_begin, position_ - value_begin);
                if (!take(kValueClose)) return status_;
                // Non-strict arguments use the last value for a repeated name, as JSON object
                // consumers do. Keep the first key position and publish each key only once.
                const auto existing =
                    tool->strict
                        ? raw.parameters.end()
                        : std::find_if(raw.parameters.begin(), raw.parameters.end(),
                                       [&](const auto& item) { return item.name == name; });
                if (existing == raw.parameters.end())
                    raw.parameters.push_back({name, value});
                else
                    existing->value = value;
            }
            if (!take(kToolClose)) return status_;
            if (!contract_.parallel && !calls.empty()) return Status::Invalid;
            std::string arguments = "{";
            bool first            = true;
            for (const auto& value : raw.parameters) {
                const auto* parameter = find_parameter_contract(*tool, value.name);
                NormalizedParameter normalized;
                if (parameter && parameter->encoding == Contract::Encoding::RawString) {
                    normalized.json_value = encode_json_string(value.value);
                } else if (parameter && parameter->encoding == Contract::Encoding::Json) {
                    if (!Json::accept(value.value)) return Status::Invalid;
                    normalized.json_value = value.value;
                } else {
                    normalized = normalize_parameter(value.value, parameter);
                }
                if (normalized.disposition == ParameterNormalization::Omitted) {
                    ++diagnostics.empty_arguments_omitted;
                    continue;
                }
                if (normalized.disposition == ParameterNormalization::SchemaMismatch)
                    ++diagnostics.schema_mismatch_arguments;
                if (!first) arguments += ',';
                first = false;
                arguments += encode_json_string(value.name) + ":" + normalized.json_value;
            }
            arguments += '}';
            calls.push_back(
                {.name = std::string(raw.name), .arguments_json = std::move(arguments)});
        }
        return Status::Complete;
    }

private:
    bool take(std::string_view literal) {
        const auto remaining = input_.substr(position_);
        if (remaining.starts_with(literal)) {
            position_ += literal.size();
            return true;
        }
        status_ = literal.starts_with(remaining) ? Status::Incomplete : Status::Invalid;
        return false;
    }

    // A JSON value ends at the value closer's '<', which only a quoted string can hold.
    bool json_value() {
        bool quoted = false, escaped = false;
        int depth = 0;
        for (; position_ < input_.size(); ++position_) {
            const char c = input_[position_];
            if (quoted) {
                if (escaped)
                    escaped = false;
                else if (c == '\\')
                    escaped = true;
                else if (c == '"')
                    quoted = false;
            } else if (c == '"')
                quoted = true;
            else if (c == '{' || c == '[') {
                // A strict value streams as the model writes it, so the bound ends the request
                // rather than turning the value into text: its call would otherwise reach a
                // client as a tree too deep to serialize.
                if (++depth > kMaxArgumentNesting) {
                    throw std::length_error("a tool call's argument nests arrays or objects deeper than " +
                                            std::to_string(kMaxArgumentNesting) + " levels");
                }
            } else if (c == '}' || c == ']') {
                if (--depth < 0) {
                    status_ = Status::Invalid;
                    return false;
                }
            } else if (c == '<' && depth == 0)
                return true;
        }
        status_ = Status::Incomplete;
        return false;
    }

    std::string_view input_;
    const Contract& contract_;
    std::size_t position_ = 0;
    Status status_        = Status::Incomplete;
};

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

// Normalizes an accepted region's calls into the structured turn.
void publish_structured_calls(ParsedToolCallOutput& out, const std::vector<RawToolCall>& raw_calls,
                              std::uint32_t duplicate_repairs, const Contract& contract) {
    out.tool_calls.reserve(raw_calls.size());
    for (const RawToolCall& raw : raw_calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
    }
    out.diagnostics.duplicate_parameters_repaired = duplicate_repairs;
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
}

} // namespace

void MarkdownCodeTracker::feed(std::string_view text) {
    for (const char byte : text) {
        if (byte == '\n') {
            end_line();
        } else {
            line_.push_back(byte);
        }
    }
}

void MarkdownCodeTracker::end_line() {
    std::string_view line(line_);
    const std::size_t first = line.find_first_not_of(" \t\r\f\v");
    line = first == std::string_view::npos ? std::string_view{} : line.substr(first);
    const auto run_of_three = [](std::string_view text, char fence) {
        return text.size() >= 3 && text[0] == fence && text[1] == fence && text[2] == fence;
    };
    if (fence_ != '\0') {
        if (run_of_three(line, fence_)) { fence_ = '\0'; }
    } else if (!line.empty() && (line[0] == '`' || line[0] == '~') && run_of_three(line, line[0]) &&
               line.substr(3).find(std::string(3, line[0])) == std::string_view::npos) {
        // A ``` or ~~~ opener whose run is not closed on the same line (that is inline code).
        fence_ = line[0];
    } else if (line.empty()) {
        ticks_ = 0; // a blank line ends the paragraph and any inline code left open in it
    } else {
        ticks_ += static_cast<std::uint32_t>(std::count(line_.begin(), line_.end(), '`'));
    }
    line_.clear();
}

bool MarkdownCodeTracker::in_code() const noexcept {
    if (fence_ != '\0') { return true; }
    // A marker at a line's start (only leading whitespace before it on the line) is never inside
    // a code span an earlier line opened: a streamed reader cannot know that such a backtick run
    // is never closed (CommonMark then makes it text), and a stray backtick in a sentence before
    // the call must not turn the call into text (Strata #4923).
    if (line_.find('`') == std::string::npos &&
        line_.find_first_not_of(" \t\r\f\v") == std::string::npos) {
        return false;
    }
    return (ticks_ + static_cast<std::uint32_t>(std::count(line_.begin(), line_.end(), '`'))) % 2 ==
           1;
}

bool contains_unquoted_tool_marker(std::string_view text) {
    // A marker inside Markdown code is a quoted example; only a marker the code state leaves is
    // one the reasoning section would refuse.
    MarkdownCodeTracker code;
    std::size_t tracked = 0;
    const auto quoted   = [&](std::size_t at) {
        code.feed(text.substr(tracked, at - tracked));
        tracked = at;
        return code.in_code();
    };
    std::size_t candidate = find_first_tool_marker(text);
    while (candidate != std::string_view::npos && quoted(candidate)) {
        const std::size_t next = find_first_tool_marker(text.substr(candidate + 1));
        candidate = next == std::string_view::npos ? next : candidate + 1 + next;
    }
    return candidate != std::string_view::npos;
}

ParsedToolCallOutput
parse_glm_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                           const common::frontend::ToolCallOutputContract& contract, bool tolerant,
                           MarkdownCodeTracker code) {
    const std::string_view source(text);
    // A marker inside Markdown code is a quoted example, never a call. Candidates
    // only move forward, so the tracker reads each byte once.
    std::size_t tracked = 0;
    const auto quoted   = [&](std::size_t at) {
        code.feed(source.substr(tracked, at - tracked));
        tracked = at;
        return code.in_code();
    };
    const auto next_marker = [&](std::size_t from) {
        const std::size_t found = find_first_tool_marker(source.substr(from));
        return found == std::string_view::npos ? std::string::npos : from + found;
    };
    // A bare marker right after a quoted opener spelling belongs to that wrapper, which is text:
    // its own markup must not open a region.
    std::size_t quoted_opener = std::string::npos;
    const auto skipped = [&](std::size_t at) {
        if (quoted(at)) {
            if (source.substr(at, kToolOpen.size()) == kToolOpen) { quoted_opener = at; }
            return true;
        }
        if (quoted_opener == std::string::npos) { return false; }
        const std::size_t between = at - quoted_opener - kToolOpen.size();
        return source.substr(quoted_opener + kToolOpen.size(), between)
                   .find_first_not_of(" \t\r\n\f\v") == std::string_view::npos;
    };
    std::size_t candidate = find_first_tool_marker(source);
    while (candidate != std::string::npos && skipped(candidate)) { candidate = next_marker(candidate + 1); }
    if (candidate == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try the first marker, then
    // each later `<tool_call>` wrapper, and accept the first region that parses; earlier markers
    // stay ordinary content. A truncated tail that still kept a complete call (tolerant mode) is a
    // recovered region, not a failure.
    std::vector<RawToolCall> raw_calls;
    std::size_t accepted                   = std::string::npos;
    FallbackReason accepted_reason         = FallbackReason::None;
    std::uint32_t duplicate_repairs        = 0;
    FallbackReason first_failure           = FallbackReason::MalformedStructure;
    bool first_failure_recorded            = false;
    while (candidate != std::string::npos) {
        std::vector<RawToolCall> calls;
        GlmToolRegionParser parser(source.substr(candidate), max_tool_name_length, contract,
                                    tolerant);
        const FallbackReason failure = parser.parse(calls);
        if (failure == FallbackReason::None ||
            (failure == FallbackReason::TruncatedTail && !calls.empty())) {
            accepted          = candidate;
            accepted_reason   = failure;
            duplicate_repairs = parser.duplicate_parameters_repaired();
            raw_calls         = std::move(calls);
            break;
        }
        if (!first_failure_recorded) {
            first_failure          = failure;
            first_failure_recorded = true;
        }
        // Retries move to the next unquoted opener.
        candidate = text.find(kToolOpen, candidate + 1);
        while (candidate != std::string::npos && quoted(candidate)) {
            candidate = text.find(kToolOpen, candidate + 1);
        }
    }
    if (accepted == std::string::npos) {
        // No region parsed. A truncated tail that kept no call carries no arguments either, so
        // the response is returned as text with the first region's reason recorded.
        out.diagnostics.fallback_reason = first_failure;
        return fallback(text, out.diagnostics);
    }
    // A recovered truncated tail keeps its reason for transparency without demoting the output.
    // Only tolerant mode produces one, or repairs a kept call.
    out.diagnostics.fallback_reason = accepted_reason;
    out.diagnostics.tolerant_recovered =
        accepted_reason == FallbackReason::TruncatedTail ||
        std::any_of(raw_calls.begin(), raw_calls.end(),
                    [](const RawToolCall& call) { return call.repaired; });

    out.content = rtrim_format_whitespace(source.substr(0, accepted));
    publish_structured_calls(out, raw_calls, duplicate_repairs, contract);
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(
    std::shared_ptr<const common::frontend::ToolCallOutputContract> contract,
    std::size_t max_tool_name_length, bool tolerant)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length),
      tolerant_(tolerant) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    fed_content_ = true;
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    std::string visible;
    // Every published byte also feeds the code tracker; the held whitespace joins it when it is
    // published, so the tracker always describes the text before the held bytes.
    const auto publish = [&](std::string_view bytes) {
        visible.append(bytes);
        code_.feed(bytes);
        published_tail_.append(bytes);
        if (published_tail_.size() > kToolOpen.size()) {
            published_tail_.erase(0, published_tail_.size() - kToolOpen.size());
        }
    };
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (!pending_tag_.empty()) {
            pending_tag_.push_back(byte);
            if (matches_marker(pending_tag_)) {
                MarkdownCodeTracker before_marker = code_;
                before_marker.feed(trailing_whitespace_);
                // A bare marker right after a quoted opener spelling belongs to that wrapper,
                // which is text: its own markup must not open a region.
                if (before_marker.in_code() || published_tail_ == kToolOpen) {
                    // A quoted example inside Markdown code: the marker is text.
                    publish(trailing_whitespace_);
                    trailing_whitespace_.clear();
                    publish(pending_tag_);
                    pending_tag_.clear();
                    continue;
                }
                region_code_ = code_;
                tool_region_ = std::move(trailing_whitespace_);
                trailing_whitespace_.clear();
                tool_region_.append(pending_tag_);
                pending_tag_.clear();
                tool_region_.append(text.substr(index + 1));
                saw_tool_marker_ = true;
                break;
            }
            if (is_prefix_of_marker(pending_tag_)) { continue; }
            // Publish the bytes before this one; the byte itself may start the next marker.
            pending_tag_.pop_back();
            publish(trailing_whitespace_);
            trailing_whitespace_.clear();
            publish(pending_tag_);
            pending_tag_.clear();
        }

        if (byte == '<') {
            pending_tag_.push_back(byte);
        } else if (is_format_whitespace(byte)) {
            trailing_whitespace_.push_back(byte);
        } else {
            publish(trailing_whitespace_);
            trailing_whitespace_.clear();
            publish(std::string_view(&byte, 1));
        }
    }
    const auto suppressed = std::min(continuation_withheld_bytes_, visible.size());
    visible.erase(0, suppressed);
    continuation_withheld_bytes_ -= suppressed;
    return visible;
}

void ToolCallOutputDecoder::initialize_continuation(std::string_view prefix) {
    (void)feed(prefix);
    if (contract_ && contract_->constrained && saw_tool_marker_) {
        std::vector<GeneratedToolCall> calls;
        ToolCallParseDiagnostics diagnostics;
        auto region = std::string_view(tool_region_);
        region.remove_prefix(region.find(kToolOpen));
        const auto status =
            ConstrainedToolRegionParser(region, *contract_).parse(calls, diagnostics);
        if (!calls.empty() || status == ConstrainedToolRegionParser::Status::Invalid)
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed or malformed calls");
    } else if (saw_tool_marker_) {
        std::vector<RawToolCall> calls;
        (void)GlmToolRegionParser(tool_region_, max_tool_name_length_, *contract_, tolerant_)
            .parse(calls);
        if (!calls.empty())
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool continuation must contain no completed calls");
    }
    continuation_withheld_bytes_ =
        trailing_whitespace_.size() + pending_tag_.size() + tool_region_.size();
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish(FinishReason reason) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    if (contract_->constrained && saw_tool_marker_) {
        Terminal result;
        result.diagnostics.marker_seen = true;
        auto region                    = std::string_view(tool_region_);
        region.remove_prefix(region.find(kToolOpen));
        const auto status = ConstrainedToolRegionParser(region, *contract_)
                                .parse(result.tool_calls, result.diagnostics);
        const bool interrupted = reason == FinishReason::OutputLimit ||
                                 reason == FinishReason::ContextCapacity ||
                                 reason == FinishReason::Cancelled;
        if (status == ConstrainedToolRegionParser::Status::Invalid ||
            (!interrupted && status != ConstrainedToolRegionParser::Status::Complete))
            throw std::logic_error("committed tool grammar and parameter decoder disagree");
        result.diagnostics.structured_call_count =
            static_cast<std::uint32_t>(result.tool_calls.size());
        return result;
    }
    if (contract_->constrained && contract_->required && reason == FinishReason::StopToken)
        throw std::logic_error("required tool grammar ended without a call");
    if (contract_->constrained && contract_->required) return {};

    ParsedToolCallOutput parsed = parse_glm_tool_call_output(tool_region_, max_tool_name_length_,
                                                              *contract_, tolerant_, region_code_);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        // The parser reports the held bytes before the accepted structured region, which are the
        // bytes after an earlier quoted marker that this decoder has not published yet.
        std::string content = std::move(parsed.content);
        trailing_whitespace_.clear();
        tool_region_.clear();
        pending_tag_.clear();
        return Terminal{.content     = std::move(content),
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }

    std::string tail = std::move(trailing_whitespace_);
    tail.append(pending_tag_);
    pending_tag_.clear();
    tail += tool_region_;
    tail.erase(0, std::min(continuation_withheld_bytes_, tail.size()));
    continuation_withheld_bytes_ = 0;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

} // namespace infernix::models::glm5_next::frontend
