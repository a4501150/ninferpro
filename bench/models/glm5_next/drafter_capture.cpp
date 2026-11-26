// The DFlash2 drafter-training corpus, step 2 (tools/drafter/README.md): teacher-forced features of
// every record, through the family's causal-scoring Program with the serving numerics. Each
// record's tokens (its rendered prompt and, for a generated record, the target's own answer) are
// prefilled once; every position's stream-mean residual after the drafter's target layers
// (5, 14, 24, 33, 42) and its final stream mean (the final norm's input) are appended to
// DIR/features.bin ([tokens][5 x 4096] BF16) and DIR/final.bin ([tokens][4096] BF16), the tokens
// to DIR/ids.bin (int32), and one line {"id", "offset", "n", "loss_start"} to DIR/index.jsonl
// (offset in tokens; positions >= loss_start are the target's answer, or the whole conversation of
// an off-policy record from position 1). Records already in the index are skipped.
//
//   infernix_glm5_next_drafter_capture --in RECORDS.jsonl --out-dir DIR [--max-context M]
//       [--experts SPEC] [--artifact FILE]

#include "drafter_corpus.h"

#include "core/device.h"
#include "infernix/types.h"
#include "models/glm5_next/frontend/prepared_prompt.h"
#include "models/glm5_next/instance.h"
#include "models/registry.h"
#include "runtime/engine/engine_options.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace infernix;
namespace glm = infernix::models::glm5_next;

int main(int argc, char** argv) {
    std::string in, dir, experts = "vram=55%,ram=45%";
    const char* env = std::getenv("INFERNIX_GLM_ARTIFACT");
    std::string artifact = env != nullptr ? env : "";
    std::uint32_t max_context = 16384;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i], value = argv[i + 1];
        if (arg == "--in") {
            in = value;
        } else if (arg == "--out-dir") {
            dir = value;
        } else if (arg == "--experts") {
            experts = value;
        } else if (arg == "--artifact") {
            artifact = value;
        } else if (arg == "--max-context") {
            max_context = static_cast<std::uint32_t>(std::stoul(value));
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (in.empty() || dir.empty() || artifact.empty()) {
        std::fprintf(stderr, "--in, --out-dir and the artifact (INFERNIX_GLM_ARTIFACT) are required\n");
        return 2;
    }
    try {
        std::filesystem::create_directories(dir);
        std::set<std::string> done;
        std::uint64_t offset = 0;
        {
            std::ifstream index(dir + "/index.jsonl");
            for (std::string line; std::getline(index, line);) {
                if (line.empty()) { continue; }
                const auto row = nlohmann::json::parse(line);
                done.insert(row["id"].get<std::string>());
                offset = std::max(offset, row["offset"].get<std::uint64_t>() + row["n"].get<std::uint64_t>());
            }
        }

        EngineOptions options;
        options.purpose               = EnginePurpose::CausalScoring;
        options.artifact_path         = artifact;
        options.kv_cache              = KvCacheStorage::Int8Group64;
        options.max_concurrency       = 1;
        options.max_context           = max_context;
        options.kv_capacity           = KvCapacityPolicy::automatic();
        options.experts_spec          = experts;
        options.context_cache.enabled = false;
        const runtime::FamilyOptionRules& family = models::family_option_rules(models::Architecture::Glm5Next);
        runtime::validate_engine_options(options, family);
        options = runtime::normalize_engine_options(std::move(options), family);
        DeviceContext device(0);
        auto instance = runtime::construct_glm5_next_instance(options, device);
        glm::Program& program = *instance->program;
        program.enable_feature_capture({5, 14, 24, 33, 42});

        std::ifstream records(in);
        std::ofstream features(dir + "/features.bin", std::ios::binary | std::ios::app);
        std::ofstream finals(dir + "/final.bin", std::ios::binary | std::ios::app);
        std::ofstream ids(dir + "/ids.bin", std::ios::binary | std::ios::app);
        std::ofstream index(dir + "/index.jsonl", std::ios::app);
        std::vector<std::uint16_t> f, h;
        std::uint64_t captured = 0, rows = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::string line; std::getline(records, line);) {
            if (line.empty()) { continue; }
            const auto record = nlohmann::json::parse(line);
            const std::string id = record["id"].get<std::string>();
            if (done.contains(id)) { continue; }
            const glm::PreparedPrompt rendered = instance->frontend.prepare(drafter_corpus::prompt_input(record));
            std::vector<TokenId> tokens = glm::PreparedPromptAccess::view(rendered).token_ids;
            std::uint32_t loss_start    = 1;
            if (record.contains("output_ids")) {
                loss_start = static_cast<std::uint32_t>(tokens.size());
                for (const auto& t : record["output_ids"]) { tokens.push_back(t.get<TokenId>()); }
            }
            if (tokens.size() > max_context) { tokens.resize(max_context); }
            if (tokens.size() < 16 || loss_start + 8 >= tokens.size()) { continue; }
            const auto n = static_cast<std::uint32_t>(tokens.size());
            (void)program.causal_score(instance->frontend.prepare_tokens(tokens, false), n - 1, {});
            program.take_features(f, h);
            features.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(2 * f.size()));
            finals.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(2 * h.size()));
            ids.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(4 * tokens.size()));
            // The data lands before its index line: a stopped run leaves no line without its data.
            features.flush();
            finals.flush();
            ids.flush();
            index << nlohmann::json{{"id", id}, {"offset", offset}, {"n", n}, {"loss_start", loss_start}}.dump() << '\n';
            index.flush();
            offset += n;
            captured += n;
            if (++rows % 50 == 0) {
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::printf("%llu records, %llu tokens, %.0f tok/s\n", static_cast<unsigned long long>(rows),
                            static_cast<unsigned long long>(captured), static_cast<double>(captured) / s);
                std::fflush(stdout);
            }
        }
        std::printf("done: %llu records, %llu tokens\n", static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(captured));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
