// The DFlash2 drafter-training corpus, step 1 (tools/drafter/README.md): the target answers prompt
// records through the public Engine, and each answer's own token ids are appended to the output.
// Reads INFERNIX_GLM_ARTIFACT. Plain decoding (no drafter), so the answers are the target's
// distribution at the request's sampling; C worker threads keep C requests in flight. Records whose
// id is already in --out are skipped, so a stopped run resumes.
//
//   infernix_glm5_next_drafter_generate --in PROMPTS.jsonl --out GEN.jsonl [--concurrency C]
//       [--max-tokens N] [--temperature T] [--experts SPEC] [--max-context M] [--minutes MIN]

#include "drafter_corpus.h"
#include "infernix/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_GLM_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::fprintf(stderr, "INFERNIX_GLM_ARTIFACT must be set\n");
        return 2;
    }
    std::string in, out, experts = "vram=50%,ram=50%";
    std::uint32_t concurrency = 8, max_tokens = 2048, max_context = 16384;
    float temperature         = 1.0F;
    double minutes            = 1e9;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i], value = argv[i + 1];
        if (arg == "--in") {
            in = value;
        } else if (arg == "--out") {
            out = value;
        } else if (arg == "--experts") {
            experts = value;
        } else if (arg == "--concurrency") {
            concurrency = static_cast<std::uint32_t>(std::stoul(value));
        } else if (arg == "--max-tokens") {
            max_tokens = static_cast<std::uint32_t>(std::stoul(value));
        } else if (arg == "--max-context") {
            max_context = static_cast<std::uint32_t>(std::stoul(value));
        } else if (arg == "--temperature") {
            temperature = std::stof(value);
        } else if (arg == "--minutes") {
            minutes = std::stod(value);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (in.empty() || out.empty() || concurrency < 1 || concurrency > 8) {
        std::fprintf(stderr, "--in and --out are required; --concurrency 1..8\n");
        return 2;
    }
    try {
        std::set<std::string> done;
        {
            std::ifstream existing(out);
            for (std::string line; std::getline(existing, line);) {
                if (!line.empty()) { done.insert(nlohmann::json::parse(line)["id"].get<std::string>()); }
            }
        }
        std::vector<nlohmann::json> records;
        {
            std::ifstream prompts(in);
            for (std::string line; std::getline(prompts, line);) {
                if (line.empty()) { continue; }
                auto record = nlohmann::json::parse(line);
                if (!done.contains(record["id"].get<std::string>())) { records.push_back(std::move(record)); }
            }
        }
        infernix::EngineOptions options;
        options.artifact_path        = artifact;
        options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
        options.max_context          = max_context;
        options.kv_capacity          = infernix::KvCapacityPolicy::automatic();
        options.max_concurrency      = concurrency;
        options.max_pending_requests = 2 * concurrency;
        options.experts_spec         = experts;
        infernix::Engine engine(options);
        std::printf("engine up; %zu records to answer (%zu done)\n", records.size(), done.size());
        std::fflush(stdout);

        std::ofstream sink(out, std::ios::app);
        std::mutex lock;
        std::atomic<std::size_t> next{0};
        std::atomic<std::uint64_t> tokens{0}, answered{0};
        const auto start    = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::duration<double, std::ratio<60>>(minutes);
        auto worker = [&](std::uint32_t w) {
            for (;;) {
                const std::size_t i = next++;
                if (i >= records.size() || std::chrono::steady_clock::now() > deadline) { return; }
                const nlohmann::json& record = records[i];
                try {
                    infernix::RequestOptions request;
                    request.execution.requested_output_tokens = max_tokens;
                    request.execution.sampling.temperature    = temperature;
                    if (temperature > 0.0F) {
                        request.execution.sampling.top_p = 0.95F;
                        request.execution.sampling.seed  = 0x5eed0000ULL + i;
                    }
                    auto handle = engine.submit(engine.prepare(drafter_corpus::prompt_input(record)), request);
                    const infernix::GenerationResult r = handle.wait();
                    nlohmann::json row = record;
                    row["output_ids"]  = r.generated_token_ids;
                    tokens += r.generated_token_ids.size();
                    const std::uint64_t n = ++answered;
                    const std::lock_guard guard(lock);
                    sink << row.dump() << '\n';
                    sink.flush();
                    if (n % 20 == 0) {
                        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                        std::printf("%llu answered, %llu tokens, %.1f tok/s\n", static_cast<unsigned long long>(n),
                                    static_cast<unsigned long long>(tokens.load()), static_cast<double>(tokens) / s);
                        std::fflush(stdout);
                    }
                } catch (const std::exception& e) {
                    const std::lock_guard guard(lock);
                    std::fprintf(stderr, "worker %u record %s: %s\n", w, record["id"].get<std::string>().c_str(), e.what());
                }
            }
        };
        std::vector<std::thread> threads;
        for (std::uint32_t w = 0; w < concurrency; ++w) { threads.emplace_back(worker, w); }
        for (auto& t : threads) { t.join(); }
        std::printf("done: %llu answered, %llu tokens\n", static_cast<unsigned long long>(answered.load()),
                    static_cast<unsigned long long>(tokens.load()));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
