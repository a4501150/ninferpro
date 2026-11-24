#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::media::decode {

enum class ErrorKind {
    BudgetExceeded,
    InvalidInput,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Policy {
    std::size_t max_bytes                  = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::function<void()> checkpoint;
};

struct Image {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct ImageInfo {
    int width  = 0;
    int height = 0;
};

struct Video {
    int width        = 0;
    int height       = 0;
    int total_frames = 0;
    double fps       = 0.0;
    double duration  = 0.0;
    std::vector<int> indices;
    std::vector<Image> frames;
};

struct VideoInfo {
    int width          = 0;
    int height         = 0;
    int total_frames   = 0;
    int sampled_frames = 0;
    double fps         = 0.0;
    double duration    = 0.0;
    std::vector<int> indices;
};

// The source frame indices to keep (ascending, unique) of a video of total_frames frames at fps
// lasting duration seconds (0 when the container does not say).
using FrameSampler = std::function<std::vector<int>(int total_frames, double fps, double duration)>;

// Exact display geometry without RGB conversion. Image probing decodes the first displayed frame
// so codec crop and orientation match decode_image(). Video probing walks the selected frame
// sequence so frame-count, truncation, pixel-budget, crop, and sampling semantics match
// decode_video(), but it does not materialize RGB frames.
ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy);
VideoInfo inspect_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                        double target_fps, int min_frames, int max_frames);

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy);
Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames);

// The same with the caller's frame selection; at least min_frames of its indices must decode.
VideoInfo inspect_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                        const FrameSampler& sampler, int min_frames);
Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                   const FrameSampler& sampler, int min_frames);

enum class LibraryLogSeverity {
    Error,
    Warning,
    Info,
};

// One complete FFmpeg diagnostic line. `source` names the FFmpeg component that reported it, such
// as "swscaler" or "mjpeg", and is empty when none did; `message` carries no line break.
struct LibraryLogLine {
    LibraryLogSeverity severity = LibraryLogSeverity::Info;
    std::string_view source;
    std::string_view message;
};

using LibraryLogHandler = std::function<void(const LibraryLogLine& line)>;

// FFmpeg reports through one process-wide callback whose default writes straight to stderr, past
// whatever owns it. An installed handler receives instead each line that FFmpeg's own threshold
// admits (info and more severe), on the decoding thread; an empty handler restores FFmpeg's stderr
// output. Returns the handler it replaces, so a scoped owner can restore it.
LibraryLogHandler set_library_log_handler(LibraryLogHandler handler);

} // namespace infernix::media::decode
