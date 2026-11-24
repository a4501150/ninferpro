#include "models/glm5_next/frontend/processor.h"

#include "media/decode/decode.h"
#include "models/common/frontend/digest.h"
#include "models/glm5_next/frontend/media_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::models::glm5_next::frontend {
namespace {

[[noreturn]] void throw_decode_error(const media::decode::Error& error) {
    switch (error.kind()) {
    case media::decode::ErrorKind::BudgetExceeded:
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded, error.what());
    case media::decode::ErrorKind::InvalidInput:
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown media decode error kind");
}

using Clock = std::chrono::steady_clock;

constexpr int kPatch                              = 14;
constexpr int kTemporal                           = 2;
constexpr int kMerge                              = 2;
constexpr int kFactor                             = kPatch * kMerge;
constexpr int kPatchFeatures                      = 3 * kTemporal * kPatch * kPatch;
constexpr std::uint64_t kPixelsPerToken           = kTemporal * kFactor * kFactor;
constexpr std::uint64_t kMinimumRawPatchesPerItem = kMerge * kMerge;
constexpr std::string_view kImageToken            = "<|image|>";
constexpr std::string_view kVideoToken            = "<|video|>";
constexpr std::string_view kImageOpen             = "<|begin_of_image|>";
constexpr std::string_view kImageClose            = "<|end_of_image|>";
// OPENAI_CLIP_MEAN and OPENAI_CLIP_STD.
constexpr std::array<float, 3> kMean{0.48145466F, 0.4578275F, 0.40821073F};
constexpr std::array<float, 3> kStd{0.26862954F, 0.26130258F, 0.27577711F};

static_assert(kPatchFeatures == static_cast<int>(kPreparedVisionPatchFeatures));

struct Size {
    int h = 0;
    int w = 0;
};

struct Prepared {
    VisionItem item;
    std::shared_ptr<PreparedMediaPayload> payload;
};

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflow");
    }
    return a * b;
}

std::uint64_t align(std::uint64_t value) { return (value + kFactor - 1) / kFactor * kFactor; }

// smart_resize: the canvas, whole merge blocks, whose `frames` frames hold between min_tokens and
// max_tokens merged tokens.
Size smart_resize(int frames, int height, int width, std::uint64_t min_tokens,
                  std::uint64_t max_tokens) {
    if (frames <= 0 || height <= 0 || width <= 0) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, "media has an empty frame");
    }
    const std::uint64_t min_pixels     = min_tokens * kPixelsPerToken;
    const std::uint64_t max_pixels     = max_tokens * kPixelsPerToken;
    const std::uint64_t aligned_frames = std::max<std::uint64_t>(
        kTemporal,
        static_cast<std::uint64_t>(std::nearbyint(static_cast<double>(frames) / kTemporal)) *
            kTemporal);
    std::uint64_t h = align(static_cast<std::uint64_t>(height));
    std::uint64_t w = align(static_cast<std::uint64_t>(width));
    if (checked_mul(aligned_frames, checked_mul(h, w, "media area"), "media pixels") < min_pixels) {
        const double scale =
            std::sqrt(static_cast<double>(min_pixels) /
                      static_cast<double>(static_cast<std::uint64_t>(frames) * height * width));
        h = align(
            std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(height * scale))));
        w = align(std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(width * scale))));
    }
    if (checked_mul(aligned_frames, checked_mul(h, w, "media area"), "media pixels") > max_pixels) {
        if (max_pixels < aligned_frames * kFactor * kFactor) {
            throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                 "the Vision token budget holds no merge block");
        }
        // The largest content height, and its proportional width, whose canvas fits the budget.
        std::uint64_t low = 1, high = static_cast<std::uint64_t>(height);
        h = w = kFactor;
        while (low <= high) {
            const std::uint64_t content = (low + high) / 2;
            const std::uint64_t cw =
                std::max<std::uint64_t>(1, static_cast<std::uint64_t>(width) * content /
                                               static_cast<std::uint64_t>(height));
            const std::uint64_t ch = align(content), cwa = align(cw);
            if (aligned_frames * ch * cwa <= max_pixels) {
                h   = ch;
                w   = cwa;
                low = content + 1;
            } else {
                high = content - 1;
            }
        }
    }
    if (h > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        w > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded, "media canvas is too large");
    }
    return {static_cast<int>(h), static_cast<int>(w)};
}

// The content's size inside the canvas: the largest aspect-preserving fit, never upscaled once
// the source alone holds min_tokens; the rest of the canvas is zero padding.
Size content_size(int frames, int height, int width, Size canvas, std::uint64_t min_tokens) {
    double scale =
        std::min(static_cast<double>(canvas.h) / height, static_cast<double>(canvas.w) / width);
    if (static_cast<std::uint64_t>(frames) * static_cast<std::uint64_t>(height) *
            static_cast<std::uint64_t>(width) >=
        kPixelsPerToken * min_tokens) {
        scale = std::min(1.0, scale);
    }
    return {std::max(1, std::min(canvas.h, static_cast<int>(std::floor(height * scale)))),
            std::max(1, std::min(canvas.w, static_cast<int>(std::floor(width * scale))))};
}

// Torchvision's antialiased bicubic resize of uint8 images (ATen's separable uint8 kernels): the
// Keys a = -0.5 filter widened by the downscale, normalized per output, quantized to int16
// weights with the largest precision that keeps the largest weight in range, and each pass
// rounded to uint8. Width first, then height; a dimension of unchanged size is not resampled.
double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) { x = -x; }
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1; }
    if (x < 2.0) { return (((x - 5) * x + 8) * x - 4) * a; }
    return 0.0;
}

struct Resample {
    int taps           = 0;
    unsigned precision = 0;
    std::vector<int> first;
    std::vector<int> count;
    std::vector<std::int16_t> weights; // [output, taps]
};

Resample resample(int input, int output) {
    const double scale    = static_cast<double>(input) / output;
    const double support  = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    Resample r;
    r.taps = static_cast<int>(std::ceil(support)) * 2 + 1;
    r.first.resize(static_cast<std::size_t>(output));
    r.count.resize(static_cast<std::size_t>(output));
    std::vector<double> weights(static_cast<std::size_t>(output) * r.taps, 0.0);
    double largest = 0.0;
    for (int i = 0; i < output; ++i) {
        const double center = scale * (i + 0.5);
        const auto begin =
            std::max<std::int64_t>(static_cast<std::int64_t>(center - support + 0.5), 0);
        const auto size = std::clamp<std::int64_t>(
            std::min<std::int64_t>(static_cast<std::int64_t>(center + support + 0.5), input) -
                begin,
            0, r.taps);
        double* w    = weights.data() + static_cast<std::size_t>(i) * r.taps;
        double total = 0.0;
        for (std::int64_t j = 0; j < size; ++j) {
            w[j] = bicubic((static_cast<double>(j + begin) - center + 0.5) * invscale);
            total += w[j];
        }
        if (total != 0.0) {
            for (std::int64_t j = 0; j < size; ++j) {
                w[j] /= total;
                largest = std::max(largest, w[j]);
            }
        }
        r.first[static_cast<std::size_t>(i)] = static_cast<int>(begin);
        r.count[static_cast<std::size_t>(i)] = static_cast<int>(size);
    }
    for (r.precision = 0; r.precision < 22; ++r.precision) {
        if (static_cast<int>(0.5 + largest * (1 << (r.precision + 1))) >= (1 << 15)) { break; }
    }
    r.weights.resize(weights.size());
    const double one = static_cast<double>(1 << r.precision);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double v = weights[i];
        r.weights[i]   = static_cast<std::int16_t>(v < 0 ? static_cast<int>(-0.5 + v * one)
                                                         : static_cast<int>(0.5 + v * one));
    }
    return r;
}

std::uint8_t clip8(int value, unsigned precision) {
    return static_cast<std::uint8_t>(std::clamp(value >> precision, 0, 255));
}

void resize_bicubic(media::decode::Image& image, Size size, const PreparationControl& control) {
    if (image.width != size.w) {
        const Resample r = resample(image.width, size.w);
        std::vector<std::uint8_t> out(static_cast<std::size_t>(image.height) * size.w * 3);
        for (int y = 0; y < image.height; ++y) {
            if (y % 16 == 0) { check_preparation_control(control); }
            const std::uint8_t* row =
                image.rgb.data() + static_cast<std::size_t>(y) * image.width * 3;
            std::uint8_t* target = out.data() + static_cast<std::size_t>(y) * size.w * 3;
            for (int x = 0; x < size.w; ++x) {
                const std::int16_t* w = r.weights.data() + static_cast<std::size_t>(x) * r.taps;
                const std::uint8_t* source =
                    row + static_cast<std::size_t>(r.first[static_cast<std::size_t>(x)]) * 3;
                std::array<int, 3> sum;
                sum.fill(1 << (r.precision - 1));
                for (int j = 0; j < r.count[static_cast<std::size_t>(x)]; ++j, source += 3) {
                    for (int c = 0; c < 3; ++c) { sum[c] += w[j] * source[c]; }
                }
                for (int c = 0; c < 3; ++c) { target[3 * x + c] = clip8(sum[c], r.precision); }
            }
        }
        image.rgb   = std::move(out);
        image.width = size.w;
    }
    if (image.height != size.h) {
        const Resample r         = resample(image.height, size.h);
        const std::size_t stride = static_cast<std::size_t>(image.width) * 3;
        std::vector<std::uint8_t> out(static_cast<std::size_t>(size.h) * stride);
        std::vector<int> sum(stride);
        for (int y = 0; y < size.h; ++y) {
            if (y % 16 == 0) { check_preparation_control(control); }
            const std::int16_t* w = r.weights.data() + static_cast<std::size_t>(y) * r.taps;
            std::fill(sum.begin(), sum.end(), 1 << (r.precision - 1));
            for (int j = 0; j < r.count[static_cast<std::size_t>(y)]; ++j) {
                const std::uint8_t* source =
                    image.rgb.data() +
                    static_cast<std::size_t>(r.first[static_cast<std::size_t>(y)] + j) * stride;
                for (std::size_t i = 0; i < stride; ++i) { sum[i] += w[j] * source[i]; }
            }
            std::uint8_t* target = out.data() + static_cast<std::size_t>(y) * stride;
            for (std::size_t i = 0; i < stride; ++i) { target[i] = clip8(sum[i], r.precision); }
        }
        image.rgb    = std::move(out);
        image.height = size.h;
    }
}

// The frame resized to `content` and zero-padded right and bottom to `canvas`.
void fit_canvas(media::decode::Image& image, Size content, Size canvas,
                const PreparationControl& control) {
    resize_bicubic(image, content, control);
    if (content.h == canvas.h && content.w == canvas.w) { return; }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(canvas.h) * canvas.w * 3, 0);
    for (int y = 0; y < content.h; ++y) {
        std::copy_n(image.rgb.data() + static_cast<std::size_t>(y) * content.w * 3,
                    static_cast<std::size_t>(content.w) * 3,
                    out.data() + static_cast<std::size_t>(y) * canvas.w * 3);
    }
    image.rgb    = std::move(out);
    image.width  = canvas.w;
    image.height = canvas.h;
}

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// The fused rescale and normalization in FP32, (v - 255 mean) / (255 std), rounded to BF16.
const std::array<std::array<std::uint16_t, 256>, 3>& normalization_lut() {
    static const auto values = [] {
        std::array<std::array<std::uint16_t, 256>, 3> out{};
        for (std::size_t c = 0; c < 3; ++c) {
            const float mean = kMean[c] * 255.0F, std = kStd[c] * 255.0F;
            for (std::size_t v = 0; v < 256; ++v) {
                out[c][v] = to_bf16((static_cast<float>(v) - mean) / std);
            }
        }
        return out;
    }();
    return values;
}

// One patch's features in (channel, frame, row, column) order.
void append_patch(const std::array<const media::decode::Image*, kTemporal>& frames, int grid_y,
                  int grid_x, std::span<std::uint16_t> out, std::size_t& cursor) {
    if (cursor > out.size() || out.size() - cursor < kPatchFeatures) {
        throw std::logic_error("Vision patch writer exceeded its allocation");
    }
    const auto& lut            = normalization_lut();
    std::uint16_t* destination = out.data() + cursor;
    for (int channel = 0; channel < 3; ++channel) {
        const auto& table = lut[static_cast<std::size_t>(channel)];
        for (const media::decode::Image* frame : frames) {
            for (int y = 0; y < kPatch; ++y) {
                const std::uint8_t* source =
                    frame->rgb.data() +
                    (static_cast<std::size_t>(grid_y * kPatch + y) * frame->width +
                     grid_x * kPatch) *
                        3 +
                    channel;
                for (int x = 0; x < kPatch; ++x) { *destination++ = table[source[3 * x]]; }
            }
        }
    }
    cursor += kPatchFeatures;
}

// Every patch of one frame pair, the m x m blocks row-major and each block's patches row-major.
void append_frame_pair(const std::array<const media::decode::Image*, kTemporal>& frames, int gh,
                       int gw, std::span<std::uint16_t> out, std::size_t& cursor) {
    for (int by = 0; by < gh / kMerge; ++by) {
        for (int bx = 0; bx < gw / kMerge; ++bx) {
            for (int my = 0; my < kMerge; ++my) {
                for (int mx = 0; mx < kMerge; ++mx) {
                    append_patch(frames, by * kMerge + my, bx * kMerge + mx, out, cursor);
                }
            }
        }
    }
}

void add_budget(PrepareStats& stats, const VisionItem& item);
void enforce_media_item_resource_limits(const VisionItem& item);
void enforce_media_resource_limits(const PrepareStats& stats, const ProcessorOptions& options);

// Miss builders run concurrently. Claim their aggregate extent before allocating the retained
// patch payload so an invalid prompt cannot fill the live-byte account and leave another worker
// waiting for memory that this same request will never release.
class ConcurrentMediaBudget {
public:
    explicit ConcurrentMediaBudget(const ProcessorOptions& options) : options_(options) {}

    void claim(const VisionItem& item) {
        std::lock_guard lock(mutex_);
        add_budget(stats_, item);
        enforce_media_resource_limits(stats_, options_);
    }

private:
    const ProcessorOptions& options_;
    std::mutex mutex_;
    PrepareStats stats_;
};

VisionItem image_item(const media::decode::ImageInfo& image, const ProcessorOptions& options) {
    const Size canvas = smart_resize(kTemporal, image.height, image.width, options.image_min_tokens,
                                     options.image_max_tokens);
    VisionItem item;
    item.modality = PromptModality::Image;
    item.grid     = {.temporal = 1, .height = canvas.h / kPatch, .width = canvas.w / kPatch};
    item.source   = {.width = image.width, .height = image.height};
    return item;
}

Prepared prepare_image(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    Prepared out;
    out.item = image_item({.width = image.width, .height = image.height}, options);
    enforce_media_item_resource_limits(out.item);
    request_budget.claim(out.item);
    const int gh = out.item.grid.height, gw = out.item.grid.width;
    const Size canvas{gh * kPatch, gw * kPatch};
    const Size content =
        content_size(kTemporal, image.height, image.width, canvas, options.image_min_tokens);
    const std::size_t elements = static_cast<std::size_t>(gh) * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    fit_canvas(image, content, canvas, control);
    std::size_t cursor = 0;
    append_frame_pair({&image, &image}, gh, gw, out.payload->mutable_span(), cursor);
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

// Glm5NextVideoProcessor.sample_frames, without its final duplication of an odd count's last
// index (the frame pairs repeat that frame instead).
std::vector<int> sample_frames(int total, double fps, double duration, double target_fps,
                               int max_frames) {
    if (total <= 0 || !(fps > 0.0)) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, "video has no frame rate");
    }
    if (!(duration > 0.0)) { duration = std::nearbyint((total - 1) / fps) + 1; }
    const double max_seconds = std::trunc(duration);
    const auto extract       = static_cast<int>(
        std::min<double>(std::trunc(duration * target_fps), static_cast<double>(max_frames)));
    if (extract <= 0) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "video is too short to sample one frame");
    }
    const auto linspace = [extract](int start, int end) {
        std::vector<int> out(static_cast<std::size_t>(extract));
        const double step = extract > 1 ? static_cast<double>(end - start) / (extract - 1) : 0.0;
        for (int i = 0; i < extract; ++i) {
            out[static_cast<std::size_t>(i)] = static_cast<int>(i * step + start);
        }
        if (extract > 1) { out.back() = end; }
        return out;
    };
    std::vector<int> indices;
    if (total < extract) {
        indices = linspace(0, total - 1);
    } else {
        double current      = 0.0;
        const double inv    = 1.0 / target_fps;
        const double period = 1.0 / fps;
        for (int frame = 0; frame < total; ++frame) {
            if (frame * period >= current) {
                current += inv;
                indices.push_back(frame);
                if (current >= max_seconds) { break; }
            }
        }
    }
    if (static_cast<int>(indices.size()) < extract) {
        indices =
            indices.empty() ? linspace(0, total - 1) : linspace(indices.front(), indices.back());
    } else if (static_cast<int>(indices.size()) > extract) {
        indices = linspace(0, total - 1);
    }
    std::vector<int> unique;
    unique.reserve(indices.size());
    for (const int index : indices) {
        if (std::find(unique.begin(), unique.end(), index) == unique.end()) {
            unique.push_back(index);
        }
    }
    std::sort(unique.begin(), unique.end());
    return unique;
}

media::decode::FrameSampler frame_sampler(const ProcessorOptions& options) {
    return [&options](int total, double fps, double duration) {
        return sample_frames(total, fps, duration, options.video_fps, options.video_max_frames);
    };
}

// The frame pairs' first frames' times (Glm5NextProcessor: metadata.timestamps[::2]).
std::vector<double> video_timestamps(std::span<const int> indices, int groups, double fps) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(groups));
    for (int t = 0; t < groups; ++t) {
        out.push_back(indices[std::min<std::size_t>(2U * t, indices.size() - 1)] / fps);
    }
    return out;
}

VisionItem video_item(int sampled, int height, int width, std::span<const int> indices, double fps,
                      const ProcessorOptions& options) {
    if (sampled <= 0) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, "video contains no decoded frame");
    }
    const int frames = (sampled + kTemporal - 1) / kTemporal * kTemporal;
    const Size canvas =
        smart_resize(frames, height, width, options.video_min_tokens, options.video_max_tokens);
    VisionItem item;
    item.modality = PromptModality::Video;
    item.grid     = {
        .temporal = frames / kTemporal, .height = canvas.h / kPatch, .width = canvas.w / kPatch};
    item.source = {.width = width, .height = height};
    item.timestamps =
        video_timestamps(indices.first(static_cast<std::size_t>(sampled)), item.grid.temporal, fps);
    return item;
}

Prepared prepare_video(std::span<const std::uint8_t> bytes, const ProcessorOptions& options,
                       const media::decode::Policy& policy, MediaPreprocessCache& cache,
                       ConcurrentMediaBudget& request_budget, const PreparationControl& control) {
    media::decode::Video video =
        media::decode::decode_video(bytes, policy, frame_sampler(options), 1);
    Prepared out;
    out.item = video_item(static_cast<int>(video.frames.size()), video.height, video.width,
                          video.indices, video.fps, options);
    enforce_media_item_resource_limits(out.item);
    request_budget.claim(out.item);
    const int gt = out.item.grid.temporal, gh = out.item.grid.height, gw = out.item.grid.width;
    const Size canvas{gh * kPatch, gw * kPatch};
    const Size content =
        content_size(2 * gt, video.height, video.width, canvas, options.video_min_tokens);
    const std::size_t elements = static_cast<std::size_t>(gt) * gh * gw * kPatchFeatures;
    out.payload                = cache.allocate_payload(elements, control);
    for (media::decode::Image& frame : video.frames) {
        if (frame.width != video.width || frame.height != video.height) {
            throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                                 "video frames change size mid-stream");
        }
        fit_canvas(frame, content, canvas, control);
    }
    std::size_t cursor = 0;
    for (int t = 0; t < gt; ++t) {
        check_preparation_control(control);
        const std::size_t last = video.frames.size() - 1;
        append_frame_pair({&video.frames[std::min<std::size_t>(2U * t, last)],
                           &video.frames[std::min<std::size_t>(2U * t + 1, last)]},
                          gh, gw, out.payload->mutable_span(), cursor);
    }
    if (cursor != elements) { throw std::logic_error("Vision patch writer left a short payload"); }
    return out;
}

std::vector<ChatPart*> media_parts(std::vector<ChatMessage>& messages) {
    std::vector<ChatPart*> out;
    for (ChatMessage& message : messages) {
        for (ChatPart& part : message.parts) {
            if (part.kind != ChatPartKind::Text) { out.push_back(&part); }
        }
    }
    return out;
}

std::size_t validate_media_inputs(std::span<ChatPart* const> parts,
                                  const ProcessorOptions& options) {
    const std::uint64_t maximum_items_from_extents =
        std::min(options.max_raw_patches / kMinimumRawPatchesPerItem, options.max_vision_tokens);
    if (std::cmp_greater(parts.size(), maximum_items_from_extents)) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "minimum Vision grids exceed processor extent budget");
    }
    std::size_t remaining = options.max_encoded_media_bytes;
    for (const ChatPart* part : parts) {
        if (part->media.bytes.size() > remaining) {
            throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                 "request media bytes exceed processor budget");
        }
        remaining -= part->media.bytes.size();
    }
    return options.max_encoded_media_bytes - remaining;
}

VisionItem inspect_item(const ChatPart& part, const ProcessorOptions& options,
                        const media::decode::Policy& policy) {
    if (part.kind == ChatPartKind::Image) {
        return image_item(media::decode::inspect_image(part.media.bytes, policy), options);
    }
    const media::decode::VideoInfo video =
        media::decode::inspect_video(part.media.bytes, policy, frame_sampler(options), 1);
    return video_item(video.sampled_frames, video.height, video.width, video.indices, video.fps,
                      options);
}

// oversized_image = "error": refuse an image the processor would scale down to fit its budget.
void enforce_image_resize_policy(const ChatPart& part, const ProcessorOptions& options,
                                 const media::decode::Policy& policy) {
    if (part.kind != ChatPartKind::Image ||
        part.media.image_resize_policy != ImageResizePolicy::RejectOversized) {
        return;
    }
    const media::decode::ImageInfo image = media::decode::inspect_image(part.media.bytes, policy);
    const std::uint64_t canvas = kTemporal * align(static_cast<std::uint64_t>(image.height)) *
                                 align(static_cast<std::uint64_t>(image.width));
    if (canvas > options.image_max_tokens * kPixelsPerToken) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia,
                             "image exceeds the native Vision geometry and oversized_image is "
                             "set to 'error'");
    }
}

std::uint64_t frame_tokens(const VisionItem& item) {
    return static_cast<std::uint64_t>(item.grid.height / kMerge) * (item.grid.width / kMerge);
}

void append_media_token_run(std::string& out, const VisionItem& item, Modality modality,
                            std::size_t item_index, std::size_t frame_index,
                            std::vector<MediaTokenRunByteSpec>& runs) {
    const std::uint64_t count = frame_tokens(item);
    if (count > (std::numeric_limits<std::size_t>::max() - out.size()) / kImageToken.size()) {
        throw std::invalid_argument("vision placeholder expansion is too large");
    }
    const std::size_t begin = out.size();
    for (std::uint64_t i = 0; i < count; ++i) { out += kImageToken; }
    runs.push_back({.bytes       = text::ByteSpan{begin, out.size()},
                    .modality    = modality,
                    .item_index  = item_index,
                    .frame_index = frame_index});
}

// Glm5NextProcessor: an image's token repeats once per merged token; a video's becomes one
// image wrapper per frame pair followed by the pair's time.
void append_placeholder(std::string& out, const VisionItem& item, std::size_t item_index,
                        std::vector<MediaTokenRunByteSpec>& runs) {
    if (item.modality == PromptModality::Image) {
        append_media_token_run(out, item, Modality::Image, item_index, 0, runs);
        return;
    }
    if (item.timestamps.size() != static_cast<std::size_t>(item.grid.temporal)) {
        throw std::logic_error("video timestamp count does not match grid");
    }
    for (std::size_t frame = 0; frame < item.timestamps.size(); ++frame) {
        out += kImageOpen;
        append_media_token_run(out, item, Modality::Video, item_index, frame, runs);
        out += kImageClose;
        std::array<char, 64> time{};
        const int length =
            std::snprintf(time.data(), time.size(), "%.1f seconds", item.timestamps[frame]);
        if (length <= 0 || static_cast<std::size_t>(length) >= time.size()) {
            throw std::invalid_argument("video timestamp is out of range");
        }
        out.append(time.data(), static_cast<std::size_t>(length));
    }
}

struct PlaceholderExpansion {
    std::size_t source_begin = 0;
    std::size_t source_end   = 0;
    std::size_t rendered_end = 0;
};

std::size_t map_expanded_boundary(std::size_t boundary, std::size_t source_size,
                                  std::span<const PlaceholderExpansion> expansions,
                                  std::string_view kind) {
    if (boundary > source_size) {
        throw std::logic_error(std::string(kind) + " byte offset exceeds rendered chat");
    }
    const auto first_incomplete =
        std::upper_bound(expansions.begin(), expansions.end(), boundary,
                         [](std::size_t value, const PlaceholderExpansion& expansion) {
                             return value < expansion.source_end;
                         });
    if (first_incomplete != expansions.end() && first_incomplete->source_begin < boundary) {
        throw std::logic_error(std::string(kind) + " intersects a media placeholder");
    }
    if (first_incomplete == expansions.begin()) { return boundary; }
    const PlaceholderExpansion& completed = *std::prev(first_incomplete);
    return completed.rendered_end + boundary - completed.source_end;
}

RenderedChat expand_placeholders(RenderedChat rendered, const std::vector<VisionItem>& items) {
    if (!rendered.media_token_runs.empty()) {
        throw std::logic_error("rendered chat media placeholders were already expanded");
    }
    if (rendered.media_placeholders.size() != items.size()) {
        throw std::invalid_argument("chat media count does not match rendered placeholders");
    }
    std::string source = std::move(rendered.text);
    std::string expanded;
    expanded.reserve(source.size());
    std::vector<PlaceholderExpansion> expansions;
    expansions.reserve(items.size());
    std::size_t source_cursor = 0;
    for (std::size_t index = 0; index < items.size(); ++index) {
        const VisionItem& item                     = items[index];
        const MediaPlaceholderByteSpec placeholder = rendered.media_placeholders[index];
        const bool image                           = item.modality == PromptModality::Image;
        if (placeholder.item_index != index ||
            placeholder.modality != (image ? Modality::Image : Modality::Video)) {
            throw std::invalid_argument("chat media order does not match rendered placeholders");
        }
        const std::string_view needle = image ? kImageToken : kVideoToken;
        if (placeholder.bytes.begin < source_cursor || placeholder.bytes.end > source.size() ||
            placeholder.bytes.end - placeholder.bytes.begin != needle.size() ||
            source.compare(placeholder.bytes.begin, needle.size(), needle) != 0) {
            throw std::logic_error("rendered media placeholder metadata does not match its text");
        }
        expanded.append(source, source_cursor, placeholder.bytes.begin - source_cursor);
        append_placeholder(expanded, item, index, rendered.media_token_runs);
        expansions.push_back({.source_begin = placeholder.bytes.begin,
                              .source_end   = placeholder.bytes.end,
                              .rendered_end = expanded.size()});
        source_cursor = placeholder.bytes.end;
    }
    expanded.append(source, source_cursor, source.size() - source_cursor);

    const auto map_boundary = [&](std::size_t boundary, std::string_view kind) {
        return map_expanded_boundary(boundary, source.size(), expansions, kind);
    };
    for (auto& span : rendered.literal_spans) {
        span.begin = map_boundary(span.begin, "literal span");
        span.end   = map_boundary(span.end, "literal span");
    }
    if (rendered.rewrite_checkpoint) {
        rendered.rewrite_checkpoint->offset =
            map_boundary(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
        rendered.rewrite_checkpoint->recovery_offset =
            map_boundary(rendered.rewrite_checkpoint->recovery_offset, "input recovery");
    }
    for (std::size_t& boundary : rendered.rewrite_execution_boundaries) {
        boundary = map_boundary(boundary, "rewrite execution boundary");
    }
    for (std::optional<std::size_t>& boundary : rendered.message_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "message boundary"); }
    }
    for (std::optional<std::size_t>& boundary : rendered.cache_boundaries) {
        if (boundary) { *boundary = map_boundary(*boundary, "cache boundary"); }
    }
    rendered.media_placeholders.clear();
    rendered.text = std::move(expanded);
    return rendered;
}

void add_budget(PrepareStats& stats, const VisionItem& item) {
    const std::uint64_t spatial =
        checked_mul(static_cast<std::uint64_t>(item.grid.height),
                    static_cast<std::uint64_t>(item.grid.width), "vision spatial grid");
    const std::uint64_t patches =
        checked_mul(static_cast<std::uint64_t>(item.grid.temporal), spatial, "vision raw patches");
    stats.raw_patches += patches;
    stats.vision_tokens += patches / (kMerge * kMerge);
    stats.attention_pairs += checked_mul(static_cast<std::uint64_t>(item.grid.temporal),
                                         checked_mul(spatial, spatial, "vision attention pairs"),
                                         "vision attention pairs");
}

// The tower runs each frame pair as its own item.
void enforce_media_item_resource_limits(const VisionItem& item) {
    if (frame_tokens(item) > kMaximumVisionItemTokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "single media frame tokens exceed Vision execution capacity");
    }
}

void enforce_media_resource_limits(const PrepareStats& stats, const ProcessorOptions& options) {
    if (stats.raw_patches > options.max_raw_patches) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision raw patches exceed processor budget");
    }
    if (stats.vision_tokens > options.max_vision_tokens) {
        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                             "vision tokens exceed processor budget");
    }
}

// Each item's token spans and the token types, from the encoded runs.
void assign_media_tokens(ProcessedInput& output,
                         std::span<const EncodedChat::MediaTokenRun> media_runs) {
    const std::size_t length = output.input_ids.size();
    output.token_types.assign(length, 0);
    std::size_t cursor = 0;
    for (const EncodedChat::MediaTokenRun& run : media_runs) {
        if (run.tokens.begin < cursor || run.tokens.count == 0 ||
            run.tokens.count > length - run.tokens.begin) {
            throw std::logic_error("encoded media token runs are not ordered disjoint spans");
        }
        if (run.item_index >= output.vision_items.size()) {
            throw std::logic_error("encoded media token run references no Vision item");
        }
        VisionItem& item = output.vision_items[run.item_index];
        const bool image = item.modality == PromptModality::Image;
        if ((image ? Modality::Image : Modality::Video) != run.modality ||
            run.frame_index != item.token_spans.size() ||
            run.frame_index >= static_cast<std::size_t>(item.grid.temporal)) {
            throw std::logic_error("encoded media token run identity is inconsistent");
        }
        if (run.tokens.count != frame_tokens(item)) {
            throw std::invalid_argument("vision placeholder run does not match media grid");
        }
        item.token_spans.push_back(run.tokens);
        cursor = run.tokens.begin + run.tokens.count;
        std::fill(output.token_types.begin() + static_cast<std::ptrdiff_t>(run.tokens.begin),
                  output.token_types.begin() + static_cast<std::ptrdiff_t>(cursor),
                  static_cast<std::uint8_t>(run.modality));
    }
    for (const VisionItem& item : output.vision_items) {
        if (item.token_spans.size() != static_cast<std::size_t>(item.grid.temporal)) {
            throw std::invalid_argument("media grid count does not match placeholder runs");
        }
    }
}

TokenId media_token_id(const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    if (ids.size() != 1) {
        throw std::invalid_argument("Vision placeholder must encode to one tokenizer token: " +
                                    std::string(text));
    }
    return ids.front();
}

media::decode::Policy decode_policy(const ProcessorOptions& options,
                                    const PreparationControl& control) {
    return {
        .max_bytes                  = options.max_encoded_media_bytes,
        .max_decoded_pixels         = options.max_decoded_pixels,
        .max_decoded_video_pixels   = options.max_decoded_video_pixels,
        .max_video_source_frames    = options.max_video_source_frames,
        .max_video_duration_seconds = options.max_video_duration_seconds,
        .checkpoint                 = [&control] { check_preparation_control(control); },
    };
}

} // namespace

Processor::Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
                     const ProcessorOptions& options,
                     std::shared_ptr<MediaPreprocessCache> media_cache)
    : tokenizer_(tokenizer), chat_template_(chat_template), options_(options),
      media_cache_(std::move(media_cache)) {
    if (options_.max_encoded_media_bytes == 0 || options_.max_decoded_pixels == 0 ||
        options_.max_decoded_video_pixels == 0 || options_.max_raw_patches == 0 ||
        options_.max_vision_tokens == 0 || options_.image_min_tokens == 0 ||
        options_.image_max_tokens < options_.image_min_tokens || options_.video_min_tokens == 0 ||
        options_.video_max_tokens < options_.video_min_tokens || !(options_.video_fps > 0.0) ||
        options_.video_max_frames <= 0 ||
        options_.max_video_source_frames < options_.video_max_frames ||
        !(options_.max_video_duration_seconds > 0.0)) {
        throw std::invalid_argument("processor budgets must be positive");
    }
    if (!media_cache_) { throw std::invalid_argument("processor media cache must not be null"); }
    image_token_ = media_token_id(tokenizer_, kImageToken);
    (void)media_token_id(tokenizer_, kVideoToken);
}

std::size_t Processor::count_tokens(std::vector<ChatMessage> messages,
                                    ChatRenderOptions render_options,
                                    const PreparationControl& control) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    (void)validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
    const media::decode::Policy policy    = decode_policy(options_, control);
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    PrepareStats stats;
    try {
        for (const ChatPart* part : parts) {
            check_preparation_control(control);
            enforce_image_resize_policy(*part, options_, policy);
            VisionItem item = inspect_item(*part, options_, policy);
            enforce_media_item_resource_limits(item);
            add_budget(stats, item);
            enforce_media_resource_limits(stats, options_);
            items.push_back(std::move(item));
        }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    rendered = expand_placeholders(std::move(rendered), items);
    const std::size_t count =
        encode_rendered_chat(tokenizer_, rendered, std::numeric_limits<std::size_t>::max(),
                             image_token_)
            .input_ids.size();
    check_preparation_control(control, "tokenization");
    return count;
}

ProcessedInput Processor::process(std::vector<ChatMessage> messages,
                                  ChatRenderOptions render_options,
                                  const PreparationControl& control,
                                  std::size_t maximum_prompt_tokens) const {
    check_preparation_control(control);
    const std::vector<ChatPart*> parts = media_parts(messages);
    const std::size_t media_bytes      = validate_media_inputs(parts, options_);
    RenderedChat rendered = chat_template_.render(messages, std::move(render_options), control);
    const std::size_t encode_limit =
        maximum_prompt_tokens == std::numeric_limits<std::size_t>::max()
            ? maximum_prompt_tokens
            : maximum_prompt_tokens + 1U;
    PrepareStats stats;
    if (maximum_prompt_tokens != std::numeric_limits<std::size_t>::max()) {
        // The text alone already over the limit fails before any media is decoded.
        const auto started = Clock::now();
        const std::size_t text_tokens =
            tokenizer_
                .encode_with_boundaries(rendered.text, {},
                                        EncodeOptions{.max_tokens = encode_limit},
                                        rendered.literal_spans)
                .input_ids.size();
        stats.tokenize_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        check_preparation_control(control, "tokenization");
        if (text_tokens > maximum_prompt_tokens) {
            const std::size_t counted =
                tokenizer_
                    .encode_with_boundaries(
                        rendered.text, {},
                        EncodeOptions{.max_tokens = context_recount_limit(maximum_prompt_tokens)},
                        rendered.literal_spans)
                    .input_ids.size();
            throw ProcessorError(context_length_details(counted, maximum_prompt_tokens));
        }
    }
    MediaPreparationPermit request_permit = media_cache_->acquire_request(control);
    std::atomic<bool> stop_preparation{false};
    const PreparationControl worker_control{
        .deadline     = control.deadline,
        .cancellation = CancellationView(
            [&stop_preparation] { return stop_preparation.load(std::memory_order_relaxed); }),
    };
    const media::decode::Policy policy = decode_policy(options_, worker_control);
    try {
        for (const ChatPart* part : parts) { enforce_image_resize_policy(*part, options_, policy); }
    } catch (const media::decode::Error& error) { throw_decode_error(error); }
    ProcessedInput output;
    std::vector<VisionItem> items;
    items.reserve(parts.size());
    stats.media_items = parts.size();
    stats.media_bytes = media_bytes;

    std::vector<PendingMedia> pending_items;
    pending_items.reserve(parts.size());
    ConcurrentMediaBudget request_budget(options_);
    std::exception_ptr preparation_error;
    const auto media_phase_started = Clock::now();
    for (ChatPart* part : parts) {
        try {
            check_preparation_control(control);
            const auto digest = common::frontend::sha256(
                part->media.bytes, [&control] { check_preparation_control(control); });
            const ChatPartKind kind = part->kind;
            const MediaCacheKey key{
                .digest   = digest,
                .modality = kind == ChatPartKind::Image ? Modality::Image : Modality::Video,
            };
            PendingMedia pending = media_cache_->begin_prepare(
                key, worker_control,
                [this, part, kind, digest, &policy, &request_budget, &worker_control]() {
                    Prepared built =
                        kind == ChatPartKind::Image
                            ? prepare_image(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control)
                            : prepare_video(part->media.bytes, options_, policy, *media_cache_,
                                            request_budget, worker_control);
                    built.item.content_digest = digest;
                    return PreparedMedia{std::move(built.item), std::move(built.payload)};
                });
            pending_items.push_back(std::move(pending));
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            break;
        }
    }
    MediaCacheRequestStats cache_stats;
    std::size_t patch_cursor = 0;
    for (PendingMedia& pending : pending_items) {
        PreparedMedia media;
        try {
            media = media_cache_->await(pending, preparation_error ? PreparationControl{} : control,
                                        cache_stats);
        } catch (...) {
            if (!preparation_error) {
                preparation_error = std::current_exception();
                stop_preparation.store(true, std::memory_order_relaxed);
            }
            MediaCacheRequestStats discarded;
            try {
                (void)media_cache_->await(pending, {}, discarded);
            } catch (...) {}
            continue;
        }
        if (preparation_error) { continue; }
        if (!media.payload || media.payload->patch_elements % kPatchFeatures != 0) {
            preparation_error = std::make_exception_ptr(
                std::logic_error("preprocessed patch buffer is not row aligned"));
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        VisionItem item  = media.item;
        item.patch_begin = patch_cursor;
        item.patch_count = media.payload->patch_elements / kPatchFeatures;
        patch_cursor += item.patch_count;
        try {
            enforce_media_item_resource_limits(item);
            add_budget(stats, item);
            enforce_media_resource_limits(stats, options_);
        } catch (...) {
            preparation_error = std::current_exception();
            stop_preparation.store(true, std::memory_order_relaxed);
            continue;
        }
        stats.patch_bytes += media.payload->patch_elements * sizeof(std::uint16_t);
        items.push_back(std::move(item));
        output.media_payloads.push_back(std::move(media.payload));
    }
    for (ChatPart* part : parts) { std::vector<std::uint8_t>().swap(part->media.bytes); }
    if (preparation_error) {
        try {
            std::rethrow_exception(preparation_error);
        } catch (const media::decode::Error& error) { throw_decode_error(error); }
    }
    if (patch_cursor != stats.raw_patches || output.media_payloads.size() != items.size()) {
        throw std::logic_error("preprocessed patch count does not match processor budget");
    }
    const double media_preprocess_seconds =
        std::chrono::duration<double>(Clock::now() - media_phase_started).count();
    request_permit.reset();

    check_preparation_control(control);
    rendered                    = expand_placeholders(std::move(rendered), items);
    const auto tokenize_started = Clock::now();
    EncodedChat encoded = encode_rendered_chat(tokenizer_, rendered, encode_limit, image_token_);
    stats.tokenize_seconds +=
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    check_preparation_control(control, "tokenization");
    if (encoded.input_ids.size() > maximum_prompt_tokens) {
        const std::size_t counted =
            encode_rendered_chat(tokenizer_, rendered, context_recount_limit(maximum_prompt_tokens),
                                 image_token_)
                .input_ids.size();
        throw ProcessorError(context_length_details(counted, maximum_prompt_tokens));
    }
    output.starts_in_reasoning         = rendered.starts_in_reasoning;
    output.continuation_content        = rendered.continuation_content;
    output.input_ids                   = std::move(encoded.input_ids);
    output.rewrite_checkpoint          = encoded.rewrite_checkpoint;
    output.rewrite_execution_frontiers = std::move(encoded.rewrite_execution_frontiers);
    output.message_boundaries          = std::move(encoded.message_boundaries);
    output.cache_boundaries            = std::move(encoded.cache_boundaries);

    stats.media_cache_hits              = cache_stats.hits;
    stats.media_cache_misses            = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    stats.media_preprocess_seconds      = media_preprocess_seconds;
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    output.vision_items                 = std::move(items);
    assign_media_tokens(output, encoded.media_token_runs);
    check_preparation_control(control);
    output.stats = stats;
    return output;
}

} // namespace infernix::models::glm5_next::frontend
