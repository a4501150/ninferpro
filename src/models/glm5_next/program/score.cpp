// Teacher-forced causal scoring for Glm5Next (EnginePurpose::CausalScoring; infernix-perplexity):
// the prompt is prefilled on lane 0 through the generation path itself, every call keeping its
// final residuals; the stream mean, final norm and LM head then run over the saved residuals in
// tiles of kScoreTileColumns, producing each target's log-probability, the top_k distribution and
// the candidates' log-probabilities from its logits (FP32, or BF16: ProgramOptions::bf16_logits).

#include "models/glm5_next/program/program_impl.h"

#include "infernix/ops/cast.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/target_logprobs.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::glm5_next::detail {

ScoreResult ProgramImpl::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    if (!options_.causal_scoring) { throw std::logic_error("Glm5Next: the Program was not built for causal scoring"); }
    if (in_transaction() || pending_transaction_ != 0 ||
        std::any_of(lanes_.begin(), lanes_.end(), [](const Lane& lane) { return lane.phase != Phase::Free; })) {
        throw std::logic_error("Glm5Next: causal scoring needs an idle Program");
    }
    const auto& data = PreparedPromptAccess::view(prompt);
    if (data.has_media()) { throw std::invalid_argument("causal scoring accepts text tokens only"); }
    const std::size_t n = data.token_ids.size();
    if (n < 2 || n > options_.max_context) { throw std::invalid_argument("causal score token count must be in [2,max_context]"); }
    if (first_target == 0 || first_target >= n) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    const std::uint32_t top_k = options.top_k, per_position = options.candidates_per_position;
    const std::size_t scored  = n - first_target;
    if (top_k > kMaximumScoreTopTokens || per_position > kMaximumScoreTopTokens ||
        options.candidates.size() != scored * per_position) {
        throw std::invalid_argument("causal score distribution outputs have an invalid shape");
    }
    for (const TokenId candidate : options.candidates) {
        if (candidate < 0 || candidate >= token_domain_) {
            throw std::invalid_argument("causal score candidate is not a public token");
        }
    }
    const std::vector<TokenId> tokens(data.token_ids.begin(), data.token_ids.end());

    // The prompt binds and prefills as a one-token greedy request whose token is discarded.
    runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1;
    execution.allow_prefix_reuse      = false;
    RequestBasePlan base                       = plan_request(std::move(prompt), execution);
    const std::vector<SourceCandidate> sources = hybrid_sources(base, UINT32_MAX);
    if (!start_binding(base, runtime::LaneId{0}, sources.back(), nullptr).reserved) {
        throw std::runtime_error("Glm5Next: the KV cache cannot hold the scored window");
    }
    const ContextProgress bound = poll_context(runtime::CancellationFlagView{});
    if (!bound.complete || !bound.sequence) {
        release(0);
        throw std::logic_error("Glm5Next: the scored prompt did not bind");
    }
    const SequenceHandle sequence = *bound.sequence;

    ScoreResult output;
    output.logprobs.reserve(scored);
    output.top_ids.reserve(scored * top_k);
    output.top_logprobs.reserve(scored * top_k);
    output.candidate_logprobs.reserve(scored * per_position);
    std::optional<PendingBatch> pending;
    try {
        scoring_ = true;
        for (;;) {
            PrefillProgress progress = advance_prefill(sequence, nullptr);
            if (progress.complete) {
                if (progress.pending) { pending.emplace(std::move(*progress.pending)); }
                break;
            }
        }
        scoring_ = false;
        check_device_error();
        if (capture_rows_ > 0) {
            // The final norm's input of every position, beside the taps the calls wrote.
            Tensor final(capture_final_.p, DType::BF16, {dim(c_.hidden_size), static_cast<std::int32_t>(n)});
            ops::hyper_connection_stream_mean(score_residuals(0, static_cast<std::int32_t>(n)),
                                              dim(c_.hc.streams), final, device_.stream);
            captured_ = n;
        }

        // Position p's logits (p in [first_target - 1, n - 2]) predict token p + 1.
        const cudaStream_t s = device_.stream;
        constexpr auto tile  = static_cast<std::size_t>(kScoreTileColumns);
        auto* io             = static_cast<std::int32_t*>(score_io_.p);
        auto* host           = static_cast<std::int32_t*>(score_host_.data());
        // Word offsets in score_io_ / score_host_ (kScoreIoWords).
        const std::size_t targets_at = 0, logprobs_at = tile, top_ids_at = 2 * tile,
                          top_logprobs_at       = top_ids_at + kMaximumScoreTopTokens * tile,
                          candidates_at         = top_logprobs_at + kMaximumScoreTopTokens * tile,
                          candidate_logprobs_at = candidates_at + kMaximumScoreTopTokens * tile;
        for (std::size_t first = first_target - 1; first + 1 < n; first += tile) {
            const auto columns     = static_cast<std::int32_t>(std::min(tile, n - 1 - first));
            const std::size_t done = output.logprobs.size(); // scored positions before this tile
            std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(first + 1), columns, host + targets_at);
            if (per_position > 0) {
                std::copy_n(options.candidates.begin() + static_cast<std::ptrdiff_t>(done * per_position),
                            static_cast<std::size_t>(columns) * per_position, host + candidates_at);
            }
            CUDA_CHECK(cudaMemcpyAsync(io, host, 4ULL * kScoreIoWords, cudaMemcpyHostToDevice, s));
            // The head's logits: FP32, or each FP32 sum rounded once to BF16.
            const bool bf16 = options_.bf16_logits;
            Tensor logits   = bf16 ? Tensor(score_logits16_.p, DType::BF16, {vocab_, columns})
                                   : Tensor(score_logits_.p, DType::FP32, {vocab_, columns});
            if (bf16) {
                Tensor wide(score_logits_.p, DType::FP32, {vocab_, columns});
                forward_->head_logits(score_residuals(static_cast<std::int32_t>(first), columns), wide);
                ops::cast_fp32_to_bf16(wide, logits, s);
            } else {
                forward_->head_logits(score_residuals(static_cast<std::int32_t>(first), columns), logits);
            }
            Tensor targets(io + targets_at, DType::I32, {1, columns});
            Tensor logprobs(io + logprobs_at, DType::FP32, {1, columns});
            ops::target_logprobs(logits, targets, token_domain_, logprobs, s);
            if (top_k > 0) {
                Tensor ids(io + top_ids_at, DType::I32, {static_cast<std::int32_t>(top_k), columns});
                Tensor values(io + top_logprobs_at, DType::FP32, {static_cast<std::int32_t>(top_k), columns});
                ops::top_logprobs(logits, token_domain_, ids, values, s);
            }
            if (per_position > 0) {
                Tensor ids(io + candidates_at, DType::I32, {static_cast<std::int32_t>(per_position), columns});
                Tensor values(io + candidate_logprobs_at, DType::FP32, {static_cast<std::int32_t>(per_position), columns});
                ops::target_logprobs(logits, ids, token_domain_, values, s);
            }
            CUDA_CHECK(cudaMemcpyAsync(host, io, score_io_.bytes, cudaMemcpyDeviceToHost, s));
            device_.synchronize();
            const auto* floats = reinterpret_cast<const float*>(host);
            output.logprobs.insert(output.logprobs.end(), floats + logprobs_at, floats + logprobs_at + columns);
            output.top_ids.insert(output.top_ids.end(), host + top_ids_at,
                                  host + top_ids_at + static_cast<std::size_t>(columns) * top_k);
            output.top_logprobs.insert(output.top_logprobs.end(), floats + top_logprobs_at,
                                       floats + top_logprobs_at + static_cast<std::size_t>(columns) * top_k);
            output.candidate_logprobs.insert(output.candidate_logprobs.end(), floats + candidate_logprobs_at,
                                             floats + candidate_logprobs_at +
                                                 static_cast<std::size_t>(columns) * per_position);
        }
    } catch (...) {
        scoring_ = false;
        if (pending) {
            (void)discard(std::move(*pending));
        } else {
            (void)abort(sequence);
        }
        throw;
    }
    // The prompt's sampled token is not part of the score; discarding it releases the lane.
    if (pending) {
        (void)discard(std::move(*pending));
    } else {
        (void)abort(sequence);
    }
    return output;
}

void ProgramImpl::enable_feature_capture(std::vector<std::uint32_t> layers) {
    if (!options_.causal_scoring || drafter_ || layers.empty()) {
        throw std::logic_error("Glm5Next: feature capture needs a causal-scoring Program without a drafter");
    }
    for (const std::uint32_t l : layers) {
        if (l >= c_.num_hidden_layers) { throw std::invalid_argument("Glm5Next: capture layer out of range"); }
    }
    const std::size_t H = c_.hidden_size, T = options_.max_context;
    capture_rows_       = static_cast<std::int32_t>(layers.size() * H);
    capture_features_   = DeviceBuffer(2ULL * static_cast<std::size_t>(capture_rows_) * T);
    capture_final_      = DeviceBuffer(2ULL * H * T);
    forward_->set_feature_taps(std::move(layers));
}

void ProgramImpl::take_features(std::vector<std::uint16_t>& features, std::vector<std::uint16_t>& final) {
    if (capture_rows_ == 0 || captured_ == 0) { throw std::logic_error("Glm5Next: nothing captured"); }
    features.resize(static_cast<std::size_t>(capture_rows_) * captured_);
    final.resize(c_.hidden_size * captured_);
    CUDA_CHECK(cudaMemcpyAsync(features.data(), capture_features_.p, 2 * features.size(), cudaMemcpyDeviceToHost,
                               device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(final.data(), capture_final_.p, 2 * final.size(), cudaMemcpyDeviceToHost,
                               device_.stream));
    device_.synchronize();
    captured_ = 0;
}

Tensor ProgramImpl::score_residuals(std::int32_t begin, std::int32_t columns) const {
    return Tensor(static_cast<std::byte*>(score_residuals_.p) + 2ULL * width_ * static_cast<std::size_t>(begin),
                  DType::BF16, {width_, columns});
}

} // namespace infernix::models::glm5_next::detail
