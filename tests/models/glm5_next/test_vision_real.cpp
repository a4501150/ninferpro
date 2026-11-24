// The GLM-5.3-Flash vision tower on the real artifact against the binary64 reference tower
// (tools/glm5_next/vision_reference.py) on the Hugging Face processor's pixels.
// INFERNIX_GLM_ARTIFACT names the artifact and INFERNIX_GLM_VISION_REFERENCE the reference's --out
// directory, run with --no-bf16-boundaries --hf-check --one-step (the oracle, its per-stage
// form, and Transformers' BF16 tower's errors against both); without both the test skips (77).
//
// The tower amplifies rounding in its middle blocks: two BF16 implementations of it land 7-25 %
// from the oracle and from each other end to end. So the arithmetic is qualified stage by stage:
// every block and the merger, given the oracle's own BF16-rounded state, must stay within
// kFloorFactor times Transformers' BF16 error on the same input (a block's error relative to its
// update, the merger's relative to its output). End to end each image must stay within
// kChaosFactor times Transformers' error, and the pass over every image at once must give each
// image's bits alone. With vision offload (the tower in pinned memory, streamed through device
// staging layer by layer) the pass must give the resident tower's bits.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/common/vision/weight_stream.h"
#include "models/glm5_next/execution/parameters.h"
#include "models/glm5_next/execution/vision.h"
#include "models/glm5_next/load.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace infernix;
namespace glm  = infernix::models::glm5_next;
namespace exec = infernix::models::glm5_next::execution;

namespace {

constexpr double kFloorFactor = 1.25;
constexpr double kChaosFactor = 2.0;

int failures = 0;

void expect(bool ok, const std::string& message) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
}

template <typename T>
std::vector<T> read_raw(const std::string& path, std::size_t count, std::size_t offset = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path);
    file.seekg(static_cast<std::streamoff>(offset * sizeof(T)));
    std::vector<T> out(count);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T))
        throw std::runtime_error(path + " is short");
    return out;
}

std::uint16_t to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

double from_bf16(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// ||got - want|| / ||want - base|| (base empty: ||want||).
double relative(const std::vector<std::uint16_t>& got, const float* want,
                const std::uint16_t* base = nullptr) {
    double err = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = from_bf16(got[i]) - want[i],
                     s = want[i] - (base ? from_bf16(base[i]) : 0.0);
        err += d * d;
        scale += s * s;
    }
    return std::sqrt(err / scale);
}

std::vector<std::uint16_t> download(const Tensor& t) {
    std::vector<std::uint16_t> out(t.bytes() / 2);
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

struct Item {
    std::uint32_t height = 0, width = 0;
    std::size_t patches = 0, merged = 0, first_row = 0; // first_row: in embeddings.f32
    std::vector<std::uint16_t> pixels;
    std::vector<std::int32_t> positions;
    const nlohmann::json* check = nullptr; // the item's hf_check entry
};

// One pass over `items` on device memory of its own.
struct Pass {
    exec::VisionPassLayout layout;
    DeviceBuffer arena, output;
    std::vector<Tensor> outputs;
    std::unique_ptr<exec::VisionTowerPass> tower;

    Pass(DeviceContext& device, const glm::VisionConfig& config,
         const exec::VisionParameters& parameters, const std::vector<const Item*>& items,
         models::common::VisionWeightStream* stream = nullptr) {
        std::vector<std::size_t> patches;
        std::size_t merged = 0;
        for (const Item* item : items) {
            patches.push_back(item->patches);
            merged += item->merged;
        }
        layout              = exec::plan_vision_pass(config, parameters, patches);
        const std::size_t O = config.out_hidden_size;
        arena               = DeviceBuffer(layout.bytes);
        output              = DeviceBuffer(2 * O * merged);
        std::vector<exec::VisionTowerItem> pass;
        std::size_t column = 0;
        for (const Item* item : items) {
            outputs.emplace_back(
                static_cast<std::byte*>(output.p) + 2 * O * column, DType::BF16,
                std::initializer_list<std::int32_t>{static_cast<std::int32_t>(O),
                                                    static_cast<std::int32_t>(item->merged)});
            pass.push_back(
                {.patches = item->pixels, .positions = item->positions, .output = outputs.back()});
            column += item->merged;
        }
        tower = std::make_unique<exec::VisionTowerPass>(device, config, parameters, std::move(pass),
                                                        DeviceSpan{arena.p, layout.bytes}, layout,
                                                        stream);
    }
};

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_GLM_ARTIFACT");
    const char* reference = std::getenv("INFERNIX_GLM_VISION_REFERENCE");
    if (artifact == nullptr || reference == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT and INFERNIX_GLM_VISION_REFERENCE\n");
        return 77;
    }
    try {
        const std::string dir = reference;
        nlohmann::json summary;
        std::ifstream(dir + "/vision.json") >> summary;
        const auto& checks = summary.at("hf_check");
        if (summary.at("bf16_boundaries").get<bool>() ||
            checks.size() != summary.at("images").size() ||
            !checks.front().contains("hf_one_step")) {
            throw std::invalid_argument(
                "the reference must be run with --no-bf16-boundaries --hf-check --one-step");
        }

        DeviceContext device(0);
        artifact::Reader reader(artifact);
        models::LoadOptions options;
        options.stream_experts = true;
        options.vision         = true;
        auto model             = glm::materialize_model(glm::plan_load(reader, options), device);
        const exec::Parameters parameters(*model);
        const glm::VisionConfig& config     = *model->config().vision;
        const exec::VisionParameters& tower = *parameters.vision;
        const std::size_t O = config.out_hidden_size, H = config.hidden_size,
                          width = config.patch_width();
        if (summary.at("out_hidden_size").get<std::size_t>() != O) {
            throw std::invalid_argument(
                "the reference's out_hidden_size differs from the artifact's");
        }

        std::vector<Item> items;
        std::size_t row = 0;
        for (std::size_t k = 0; k < summary.at("images").size(); ++k) {
            const auto& image = summary.at("images")[k];
            Item item;
            const auto& grid = image.at("grid");
            if (grid[0].get<int>() != 1) throw std::invalid_argument("single-frame grids only");
            item.height    = grid[1].get<std::uint32_t>();
            item.width     = grid[2].get<std::uint32_t>();
            item.patches   = image.at("patch_count").get<std::size_t>();
            item.merged    = image.at("merged_tokens").get<std::size_t>();
            item.first_row = row;
            row += item.merged;
            const auto pixels = read_raw<float>(dir + "/pixels/Image-" + std::to_string(k) + ".f32",
                                                item.patches * width);
            item.pixels.resize(pixels.size());
            for (std::size_t i = 0; i < pixels.size(); ++i) item.pixels[i] = to_bf16(pixels[i]);
            item.positions =
                exec::vision_patch_positions(item.height, item.width, config.spatial_merge_size);
            item.check = &checks[k];
            items.push_back(std::move(item));
        }

        for (std::size_t k = 0; k < items.size(); ++k) {
            const Item& item = items[k];
            const std::string name =
                std::to_string(item.height) + "x" + std::to_string(item.width) + " patches";
            const std::string blocks = dir + "/blocks/Image-" + std::to_string(k);
            const std::size_t plane  = item.patches * H;
            const auto inputs =
                read_raw<std::uint16_t>(blocks + ".in.bf16", (config.depth + 1) * plane);
            const auto outputs = read_raw<float>(blocks + ".out.f32", config.depth * plane);
            const auto merged  = read_raw<float>(blocks + ".merged.f32", item.merged * O);

            // Stage by stage from the oracle's states.
            Pass pass(device, config, tower, {&item});
            pass.tower->advance(1);
            const Tensor x = pass.tower->residual(0);
            double worst   = 0;
            for (std::uint32_t b = 0; b <= config.depth; ++b) {
                CUDA_CHECK(cudaMemcpyAsync(x.data, inputs.data() + b * plane, x.bytes(),
                                           cudaMemcpyHostToDevice, device.stream));
                pass.tower->advance(b + 2);
                device.synchronize();
                const bool merger  = b == config.depth;
                const double got   = merger ? relative(download(pass.outputs[0]), merged.data())
                                            : relative(download(x), outputs.data() + b * plane,
                                                       inputs.data() + b * plane);
                const double floor = merger ? item.check->at("hf_merger").get<double>()
                                            : item.check->at("hf_one_step")[b].get<double>();
                worst              = std::max(worst, got / floor);
                expect(got <= kFloorFactor * floor,
                       name + (merger ? " merger" : " block " + std::to_string(b)) + ": " +
                           std::to_string(got) + " against Transformers' " + std::to_string(floor));
            }

            // End to end from the pixels.
            Pass whole(device, config, tower, {&item});
            whole.tower->advance(exec::VisionTowerPass::stages(config.depth));
            device.synchronize();
            const auto embeddings =
                read_raw<float>(dir + "/embeddings.f32", item.merged * O, item.first_row * O);
            const double end    = relative(download(whole.outputs[0]), embeddings.data());
            const double hf_end = item.check->at("hf_relative_l2").get<double>();
            std::printf("%s (%zu merged tokens): worst stage %.2fx Transformers' BF16 error; end "
                        "to end %.4f "
                        "(Transformers %.4f)\n",
                        name.c_str(), item.merged, worst, end, hf_end);
            expect(std::isfinite(end) && end <= kChaosFactor * hf_end, name + " end to end");
        }

        std::vector<const Item*> all;
        for (const Item& item : items) all.push_back(&item);
        Pass together(device, config, tower, all);
        CudaEventTimer resident_time(device);
        resident_time.start();
        together.tower->advance(exec::VisionTowerPass::stages(config.depth));
        const float resident_ms = resident_time.stop_ms();
        for (std::size_t k = 0; k < items.size(); ++k) {
            Pass alone(device, config, tower, {&items[k]});
            alone.tower->advance(exec::VisionTowerPass::stages(config.depth));
            device.synchronize();
            expect(download(together.outputs[k]) == download(alone.outputs[0]),
                   "image " + std::to_string(k) + " differs beside the others");
        }

        models::LoadOptions offload = options;
        offload.vision_offload      = true;
        auto pinned        = glm::materialize_model(glm::plan_load(reader, offload), device);
        const auto& assets = *pinned->overlay_vision();
        const exec::Parameters pinned_parameters(*pinned);
        DeviceBuffer staging(assets.layout.staging_bytes);
        CudaEventTimer streamed_time(device);
        streamed_time.start();
        models::common::VisionWeightStream stream(device, assets,
                                                  static_cast<std::byte*>(staging.p));
        const exec::VisionParameters window =
            exec::vision_window_weights(stream, *pinned_parameters.vision);
        Pass streamed(device, config, window, all, &stream);
        streamed.tower->advance(exec::VisionTowerPass::stages(config.depth));
        stream.finish(device.stream);
        const float streamed_ms = streamed_time.stop_ms();
        std::printf("vision offload: %zu MiB of the tower streamed through %zu MiB of staging; the "
                    "pass over every image %.1f ms (resident %.1f ms)\n",
                    stream.uploaded_bytes() >> 20, assets.layout.staging_bytes >> 20, streamed_ms,
                    resident_ms);
        for (std::size_t k = 0; k < items.size(); ++k) {
            expect(download(streamed.outputs[k]) == download(together.outputs[k]),
                   "image " + std::to_string(k) + " differs with the tower streamed");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::printf("PASS\n");
    return 0;
}
