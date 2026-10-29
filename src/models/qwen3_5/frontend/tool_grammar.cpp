#include "models/qwen3_5/frontend/tool_grammar.h"
#include "text/json_schema.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <string_view>

namespace infernix::models::qwen3_5::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

constexpr std::string_view kCallOpen = "<tool_call>";

// MarkdownCodeTracker as a finite automaton over answer bytes, with a matcher of the call opener
// that runs only outside code. Only the parity of a paragraph's backticks decides inline code, and
// a fence opener needs only its first three characters and the longest run of them after.
struct AnswerState {
    enum Phase : std::uint8_t { Lead, Run1, Run2, Opener, Plain, Closer };
    char fence      = '\0'; // the open fence's character
    bool paragraph  = false; // odd backticks in the paragraph's finished lines
    bool line       = false; // odd backticks in the current line
    Phase phase     = Lead;
    char run_char   = '\0'; // the ` or ~ that starts the line (Run1, Run2, Opener)
    std::uint8_t rest = 0;  // the current run of run_char after an opener's first three
    std::uint8_t matched = 0; // bytes of kCallOpen matched outside code

    [[nodiscard]] bool in_code() const noexcept { return fence != '\0' || paragraph != line; }
    auto key() const noexcept {
        return std::tuple(fence, paragraph, line, phase, run_char, rest, matched);
    }
    bool operator<(const AnswerState& other) const noexcept { return key() < other.key(); }

    // The state after `byte`; `opened` is set when the byte completes a call opener.
    AnswerState step(unsigned char byte, bool& opened) const {
        AnswerState next = *this;
        opened           = false;
        const bool outside = !in_code();
        if (byte == '\n') {
            if (fence != '\0') {
                if (phase == Closer) next.fence = '\0';
            } else if (phase == Lead) {
                next.paragraph = false;
            } else if (phase == Opener) {
                next.fence = run_char;
            } else {
                next.paragraph = paragraph != line;
            }
            next.line     = false;
            next.phase    = Lead;
            next.run_char = '\0';
            next.rest     = 0;
        } else {
            if (fence == '\0' && byte == '`') next.line = !line;
            const bool blank = byte == ' ' || byte == '\t' || byte == '\r' || byte == '\f' ||
                               byte == '\v';
            const char fence_char = fence != '\0' ? fence : static_cast<char>(byte);
            switch (phase) {
            case Lead:
                if (blank) break;
                if ((byte == '`' || byte == '~') && fence_char == static_cast<char>(byte)) {
                    next.phase    = Run1;
                    next.run_char = static_cast<char>(byte);
                } else {
                    next.phase = Plain;
                }
                break;
            case Run1:
                next.phase = byte == static_cast<unsigned char>(run_char) ? Run2 : Plain;
                break;
            case Run2:
                if (byte != static_cast<unsigned char>(run_char))
                    next.phase = Plain;
                else
                    next.phase = fence != '\0' ? Closer : Opener;
                break;
            case Opener:
                // A second run of three on the opener's line makes the line inline code.
                if (byte != static_cast<unsigned char>(run_char))
                    next.rest = 0;
                else if (++next.rest == 3)
                    next.phase = Plain;
                break;
            case Plain:
            case Closer: break;
            }
            if (next.phase == Plain || next.phase == Closer) {
                next.run_char = '\0';
                next.rest     = 0;
            }
        }
        // The opener holds no backtick, tilde or line break, so the code state at its first byte
        // holds through its last.
        next.matched = 0;
        if (outside) {
            if (byte == static_cast<unsigned char>(kCallOpen[matched])) {
                next.matched = static_cast<std::uint8_t>(matched + 1);
            } else if (byte == static_cast<unsigned char>(kCallOpen[0])) {
                next.matched = 1;
            }
            if (next.matched == kCallOpen.size()) {
                opened       = true;
                next.matched = 0;
            }
        }
        return next;
    }
};

// Free answer text that opens `calls` at the first call opener outside Markdown code; the text
// may end at any byte. The parser applies the same rule, so a quoted opener stays content.
std::int32_t answer_then_calls(xgrammar::GrammarBuilder& builder, std::int32_t calls) {
    // Bytes that move the automaton; every other byte (and every non-ASCII code point) behaves
    // like `x`, the last column.
    std::string distinct = "\n \t\r\f\v`~";
    for (char c : kCallOpen)
        if (distinct.find(c) == std::string::npos) distinct.push_back(c);
    std::string columns = distinct + 'x';
    constexpr int kOpened = -1;

    std::map<AnswerState, int> index;
    std::vector<AnswerState> states;
    std::vector<std::vector<int>> moves;
    const auto visit = [&](const AnswerState& state) {
        auto [found, inserted] = index.try_emplace(state, static_cast<int>(states.size()));
        if (inserted) states.push_back(state);
        return found->second;
    };
    visit(AnswerState{});
    for (std::size_t at = 0; at < states.size(); ++at) {
        std::vector<int> row;
        for (char c : columns) {
            bool opened     = false;
            const auto next = states[at].step(static_cast<unsigned char>(c), opened);
            row.push_back(opened ? kOpened : visit(next));
        }
        moves.push_back(std::move(row));
    }

    // Merge states with the same future (Moore refinement): every state accepts the end, so only
    // where an opener completes tells them apart. Each merged state becomes one rule.
    std::vector<int> block(states.size(), 0);
    for (std::size_t count = 1;;) {
        std::map<std::vector<int>, int> signatures;
        std::vector<int> refined(states.size());
        for (std::size_t s = 0; s < states.size(); ++s) {
            std::vector<int> signature{block[s]};
            for (int next : moves[s]) signature.push_back(next == kOpened ? kOpened : block[next]);
            refined[s] = signatures.try_emplace(signature, static_cast<int>(signatures.size()))
                             .first->second;
        }
        block = std::move(refined);
        if (signatures.size() == count) break;
        count = signatures.size();
    }
    const int blocks = 1 + *std::max_element(block.begin(), block.end());
    std::vector<int> representative(blocks, -1);
    for (std::size_t s = 0; s < states.size(); ++s)
        if (representative[block[s]] < 0) representative[block[s]] = static_cast<int>(s);
    std::vector<std::int32_t> rules(blocks);
    for (auto& rule : rules) rule = builder.AddEmptyRuleWithHint("answer");

    using Elements = std::vector<xgrammar::GrammarBuilder::CharacterClassElement>;
    Elements named;
    for (char c : distinct)
        named.push_back({static_cast<unsigned char>(c), static_cast<unsigned char>(c)});
    for (int b = 0; b < blocks; ++b) {
        const auto& row = moves[representative[b]];
        const auto target = [&](std::size_t column) {
            return row[column] == kOpened ? kOpened : block[row[column]];
        };
        // Bytes that keep the state form one starred class, so an ordinary token matches in one
        // grammar position; the rest are grouped by their destination.
        const bool other_stays = target(distinct.size()) == b;
        std::map<int, Elements> targets;
        Elements stay, moving;
        for (std::size_t column = 0; column < distinct.size(); ++column) {
            const std::int32_t c = static_cast<unsigned char>(distinct[column]);
            if (target(column) == b) {
                stay.push_back({c, c});
            } else {
                targets[target(column)].push_back({c, c});
                moving.push_back({c, c});
            }
        }
        std::vector<std::int32_t> tail{builder.AddEmptyStr()};
        for (const auto& [next, elements] : targets)
            tail.push_back(builder.AddSequence(
                {builder.AddCharacterClass(elements),
                 builder.AddRuleRef(next == kOpened ? calls : rules[next])}));
        if (!other_stays) {
            const int next = target(distinct.size());
            tail.push_back(builder.AddSequence(
                {builder.AddCharacterClass(named, true),
                 builder.AddRuleRef(next == kOpened ? calls : rules[next])}));
        }
        const auto move = builder.AddRuleWithHint("answer_move", builder.AddChoices(tail));
        std::vector<std::int32_t> body;
        if (other_stays)
            body.push_back(builder.AddCharacterClassStar(moving, true));
        else if (!stay.empty())
            body.push_back(builder.AddCharacterClassStar(stay));
        body.push_back(builder.AddRuleRef(move));
        builder.UpdateRuleBody(rules[b], builder.AddSequence(body));
    }
    return rules[block[0]];
}

xgrammar::Grammar arguments(const Contract::Tool& tool) {
    try {
        std::string source;
        if (tool.strict)
            source = text::prepare_json_schema(tool.schema_json);
        else
            source = R"({"type":"object","additionalProperties":{"type":"string"}})";
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, false, xgrammar::JSONFormat::kQwenXML, {});
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
                              ? RequestErrorKind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? RequestErrorKind::UnsatisfiableJsonSchema
                              : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer,
                           RequestErrorSource::Tools);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters",
                           RequestErrorSource::Tools);
    }
}

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body) {
    std::optional<xgrammar::Grammar> json;
    if (body) json = text::build_json_grammar(*body);
    if (json && contract.tools.empty()) return *json;
    xgrammar::GrammarBuilder builder;
    if (contract.tools.empty())
        return builder.Get(
            builder.AddRuleWithHint("root", builder.AddTagDispatch({{}, false, {"<tool_call>"}})));
    std::vector<int32_t> choices;
    std::optional<int32_t> basic_arguments;
    for (const auto& tool : contract.tools) {
        const auto body = !tool.strict && basic_arguments
                              ? *basic_arguments
                              : xgrammar::SubGrammarAdder::Apply(&builder, arguments(tool));
        if (!tool.strict) basic_arguments = body;
        choices.push_back(builder.AddSequence(
            {builder.AddByteString("\n<function=" + tool.name + ">\n"), builder.AddRuleRef(body),
             builder.AddByteString("</function>\n</tool_call>")}));
    }
    const auto call_body = builder.AddRuleWithHint("call_body", builder.AddChoices(choices));
    const auto next_call = builder.AddSequence(
        {builder.AddByteString("\n<tool_call>"), builder.AddRuleRef(call_body)});
    const auto tail  = contract.parallel ? builder.AddRepeatFromExpr("calls", next_call, 0, -1)
                                         : builder.AddEmptyStr();
    const auto first = builder.AddRuleWithHint(
        "calls", builder.AddSequence({builder.AddRuleRef(call_body), tail}));
    const auto root =
        (contract.required || body)
            ? builder.AddSequence({builder.AddByteString(std::string(kCallOpen)),
                                   builder.AddRuleRef(first)})
            : builder.AddRuleRef(answer_then_calls(builder, first));
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
}
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel}};
    if (body && !body->choices.empty())
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    if (body) identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"] = Json::parse(tool.schema_json);
            // Failed compilations are cached too, including their declaration error location.
            entry["source_index"] = tool.declaration_index;
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    // An auto turn may also end in its thinking, as on the free route: calls the model wrote there
    // can end the turn instead of looping until the output limit. A call or a JSON answer that
    // the turn owes still needs the close.
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body); }, reasoning_close, continuation,
        !contract.required && !body);
}
} // namespace infernix::models::qwen3_5::frontend
