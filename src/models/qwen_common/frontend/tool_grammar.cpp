#include "models/qwen_common/frontend/tool_grammar.h"
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

namespace infernix::models::qwen_common::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

constexpr std::string_view kCallOpen = "<tool_call>";

// The length of the longest prefix of `close` that is a suffix of `close.substr(0, matched)
// + byte`; the reasoning section tracks it to end at the close wherever the model writes it.
std::uint8_t close_advance(std::string_view close, std::uint8_t matched, unsigned char byte) {
    std::string candidate(close.substr(0, matched));
    candidate += static_cast<char>(byte);
    std::size_t best = std::min(candidate.size(), close.size());
    while (best && !candidate.ends_with(close.substr(0, best))) { --best; }
    return static_cast<std::uint8_t>(best);
}

enum class Section : std::uint8_t { Answer, Reasoning };

// MarkdownCodeTracker as a finite automaton over section bytes, with a matcher of the call opener
// that runs only outside code and, in the reasoning section, a matcher of the reasoning close that
// runs everywhere. Only the parity of a paragraph's backticks decides inline code, and a fence
// opener needs only its first three characters and the longest run of them after.
struct TextState {
    enum Phase : std::uint8_t { Lead, Run1, Run2, Opener, Plain, Closer };
    char fence      = '\0'; // the open fence's character
    bool paragraph  = false; // odd backticks in the paragraph's finished lines
    bool line       = false; // odd backticks in the current line
    Phase phase     = Lead;
    char run_char   = '\0'; // the ` or ~ that starts the line (Run1, Run2, Opener)
    std::uint8_t rest = 0;  // the current run of run_char after an opener's first three
    std::uint8_t matched = 0; // bytes of kCallOpen matched outside code
    std::uint8_t closed   = 0; // bytes of the reasoning close matched (reasoning section)
    std::uint8_t indent   = 0; // columns of leading whitespace consumed in Lead

    // A marker at a line's start (phase Lead: only leading whitespace before it on the line) is
    // never inside a code span an earlier line opened: a streamed reader cannot know that such a
    // backtick run is never closed (Strata #4923).
    [[nodiscard]] bool in_code() const noexcept {
        return fence != '\0' || (phase != Lead && paragraph != line);
    }
    auto key() const noexcept {
        return std::tuple(fence, paragraph, line, phase, run_char, rest, matched, closed, indent);
    }
    bool operator<(const TextState& other) const noexcept { return key() < other.key(); }

    // The state after `byte`. The answer section sets `opened` when the byte completes a call
    // opener outside code; the reasoning section instead refuses that byte, and sets `exited`
    // when the byte completes the reasoning close.
    template <Section S>
    TextState step(unsigned char byte, std::string_view close, bool& opened, bool& exited,
                   bool& refused) const {
        TextState next = *this;
        opened         = false;
        exited         = false;
        refused        = false;
        // A marker's code decision is made at its first byte, as the tracker does: a partial
        // opener's continuation holds it (its bytes leave the line's start, where an earlier
        // line's span must not decide, Strata #4923), while a fresh candidate start samples the
        // current state.
        const bool continues = matched != 0 && byte == static_cast<unsigned char>(kCallOpen[matched]);
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
            next.indent   = 0;
        } else {
            if (fence == '\0' && byte == '`') next.line = !line;
            const bool blank = byte == ' ' || byte == '\t' || byte == '\r' || byte == '\f' ||
                               byte == '\v';
            const char fence_char = fence != '\0' ? fence : static_cast<char>(byte);
            switch (phase) {
            case Lead:
                if (blank) {
                    next.indent = indent < 255 ? static_cast<std::uint8_t>(indent + 1) : indent;
                    break;
                }
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
        if (!in_code() || continues) {
            if (continues) {
                next.matched = static_cast<std::uint8_t>(matched + 1);
            } else if (byte == static_cast<unsigned char>(kCallOpen[0])) {
                next.matched = 1;
            }
            if (next.matched == kCallOpen.size()) {
                if constexpr (S == Section::Answer) {
                    opened = true;
                } else {
                    // A call cannot start inside thinking: the opener's last byte is refused.
                    refused = true;
                }
                next.matched = 0;
            }
        }
        if constexpr (S == Section::Reasoning) {
            next.closed = close_advance(close, closed, byte);
            if (next.closed == close.size()) {
                exited     = true;
                next.closed = 0;
            }
        }
        return next;
    }
};

// Free section text. The answer section opens `target_rule` (the calls) at the first call opener
// outside Markdown code and may end at any byte; the parser applies the same rule, so a quoted
// opener stays content. The reasoning section instead ends at its close, whose matcher runs
// everywhere like the decode layer's, and refuses the opener's last byte outside code, so a call
// cannot start inside thinking and the turn cannot end there.
template <Section S>
std::int32_t text_section(xgrammar::GrammarBuilder& builder, std::int32_t target_rule,
                          std::string_view close) {
    // Bytes that move the automaton; every other byte (and every non-ASCII code point) behaves
    // like `x`, the last column.
    std::string distinct = "\n \t\r\f\v`~";
    for (char c : kCallOpen)
        if (distinct.find(c) == std::string::npos) distinct.push_back(c);
    if constexpr (S == Section::Reasoning)
        for (char c : close)
            if (distinct.find(c) == std::string::npos) distinct.push_back(c);
    std::string columns = distinct + 'x';
    constexpr int kExit = -1, kRefused = -2;

    std::map<TextState, int> index;
    std::vector<TextState> states;
    std::vector<std::vector<int>> moves;
    const auto visit = [&](const TextState& state) {
        auto [found, inserted] = index.try_emplace(state, static_cast<int>(states.size()));
        if (inserted) states.push_back(state);
        return found->second;
    };
    visit(TextState{});
    for (std::size_t at = 0; at < states.size(); ++at) {
        std::vector<int> row;
        for (char c : columns) {
            bool opened = false, exited = false, refused = false;
            const auto next =
                states[at].step<S>(static_cast<unsigned char>(c), close, opened, exited, refused);
            if constexpr (S == Section::Answer)
                row.push_back(opened ? kExit : visit(next));
            else
                row.push_back(exited ? kExit : refused ? kRefused : visit(next));
        }
        moves.push_back(std::move(row));
    }

    // Merge states with the same future (Moore refinement): in the answer every state accepts the
    // end, so only where an opener completes tells them apart; in the reasoning, states also part
    // by which bytes they refuse and where the close completes. Each merged state becomes one
    // rule.
    std::vector<int> block(states.size(), 0);
    for (std::size_t count = 1;;) {
        std::map<std::vector<int>, int> signatures;
        std::vector<int> refined(states.size());
        for (std::size_t s = 0; s < states.size(); ++s) {
            std::vector<int> signature{block[s]};
            for (int next : moves[s]) signature.push_back(next >= 0 ? block[next] : next);
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
    for (auto& rule : rules)
        rule = builder.AddEmptyRuleWithHint(S == Section::Answer ? "answer" : "reasoning");

    using Elements = std::vector<xgrammar::GrammarBuilder::CharacterClassElement>;
    Elements named;
    for (char c : distinct)
        named.push_back({static_cast<unsigned char>(c), static_cast<unsigned char>(c)});
    for (int b = 0; b < blocks; ++b) {
        const auto& row = moves[representative[b]];
        const auto destination = [&](std::size_t column) {
            return row[column] >= 0 ? block[row[column]] : row[column];
        };
        // Bytes that keep the state form one starred class, so an ordinary token matches in one
        // grammar position; the rest are grouped by their destination. A refused byte joins no
        // class but the negated ones, so no token holding it can be produced there.
        const bool other_stays = destination(distinct.size()) == b;
        std::map<int, Elements> targets;
        Elements stay, moving;
        for (std::size_t column = 0; column < distinct.size(); ++column) {
            const std::int32_t c = static_cast<unsigned char>(distinct[column]);
            const std::int32_t to = destination(column);
            if (to == kRefused) {
                moving.push_back({c, c});
                continue;
            }
            if (to == b) {
                stay.push_back({c, c});
            } else {
                targets[to].push_back({c, c});
                moving.push_back({c, c});
            }
        }
        std::vector<std::int32_t> tail;
        if constexpr (S == Section::Answer) tail.push_back(builder.AddEmptyStr());
        for (const auto& [to, elements] : targets)
            tail.push_back(builder.AddSequence(
                {builder.AddCharacterClass(elements),
                 builder.AddRuleRef(to == kExit ? target_rule : rules[to])}));
        if (!other_stays) {
            const int to = destination(distinct.size());
            tail.push_back(builder.AddSequence(
                {builder.AddCharacterClass(named, true),
                 builder.AddRuleRef(to == kExit ? target_rule : rules[to])}));
        }
        const auto move = builder.AddRuleWithHint("section_move", builder.AddChoices(tail));
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

// The reasoning section: thinking text that ends at its close, and refuses a call opener outside
// Markdown code, so the model must close its thinking before calling and the turn cannot end
// inside it.
xgrammar::Grammar reasoning_section(std::string_view close) {
    xgrammar::GrammarBuilder builder;
    const auto exit = builder.AddRuleWithHint("reasoning_close", builder.AddEmptyStr());
    const auto root  = text_section<Section::Reasoning>(builder, exit, close);
    return xgrammar::GrammarNormalizer::Apply(builder.Get(root));
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

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body,
                        std::string_view close) {
    std::optional<xgrammar::Grammar> json;
    if (body) json = text::build_json_grammar(*body);
    if (json && contract.tools.empty())
        return close.empty() ? *json : xgrammar::Grammar::Concat({reasoning_section(close), *json});
    xgrammar::GrammarBuilder builder;
    if (contract.tools.empty()) {
        const auto dispatch = builder.Get(builder.AddRuleWithHint(
            "root", builder.AddTagDispatch({{}, false, {std::string(kCallOpen)}})));
        return close.empty() ? dispatch
                             : xgrammar::Grammar::Concat({reasoning_section(close), dispatch});
    }
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
            : builder.AddRuleRef(text_section<Section::Answer>(builder, first, {}));
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    const xgrammar::Grammar answer =
        json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
    return close.empty() ? answer : xgrammar::Grammar::Concat({reasoning_section(close), answer});
}
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel},
                  {"reasoning_close", std::string(reasoning_close)}};
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
    // The reasoning section the grammar builds refuses a call opener outside Markdown code, so
    // a call cannot start inside thinking: the model closes its thinking first, and the turn
    // cannot end inside it.
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body, reasoning_close); }, continuation);
}
} // namespace infernix::models::qwen_common::frontend
