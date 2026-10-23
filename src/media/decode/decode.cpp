#include "media/decode/decode.h"
#include "media/decode/png_decode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::media::decode {
namespace {

std::string av_error(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

[[noreturn]] void throw_invalid_media(std::string message) {
    throw Error(ErrorKind::InvalidInput, std::move(message));
}

void validate_input(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (bytes.empty()) { throw_invalid_media("media bytes are empty"); }
    if (bytes.size() > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
    }
}

int exif_orientation(std::span<const std::uint8_t> bytes) {
    constexpr std::size_t limit = 1ULL << 20;
    bytes                       = bytes.first(std::min(limit, bytes.size()));
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) { return 1; }
    auto be16 = [&](std::size_t offset) {
        return static_cast<std::uint16_t>((bytes[offset] << 8U) | bytes[offset + 1]);
    };
    std::size_t marker = 2;
    while (marker + 4 <= bytes.size()) {
        if (bytes[marker] != 0xff) { break; }
        const std::uint8_t kind = bytes[marker + 1];
        if (kind == 0xda || kind == 0xd9) { break; }
        const std::uint16_t length = be16(marker + 2);
        if (length < 2 || marker + 2 + length > bytes.size()) { break; }
        const std::size_t payload      = marker + 4;
        const std::size_t payload_size = length - 2;
        if (kind == 0xe1 && payload_size >= 14 &&
            std::memcmp(bytes.data() + payload, "Exif\0\0", 6) == 0) {
            const std::size_t tiff = payload + 6;
            const bool little      = bytes[tiff] == 'I' && bytes[tiff + 1] == 'I';
            const bool big         = bytes[tiff] == 'M' && bytes[tiff + 1] == 'M';
            if (!little && !big) { return 1; }
            auto u16 = [&](std::size_t offset) -> std::uint16_t {
                if (offset + 2 > bytes.size()) { return 0; }
                return little ? static_cast<std::uint16_t>(bytes[offset] | bytes[offset + 1] << 8U)
                              : static_cast<std::uint16_t>(bytes[offset] << 8U | bytes[offset + 1]);
            };
            auto u32 = [&](std::size_t offset) -> std::uint32_t {
                if (offset + 4 > bytes.size()) { return 0; }
                if (little) {
                    return static_cast<std::uint32_t>(bytes[offset]) |
                           static_cast<std::uint32_t>(bytes[offset + 1]) << 8U |
                           static_cast<std::uint32_t>(bytes[offset + 2]) << 16U |
                           static_cast<std::uint32_t>(bytes[offset + 3]) << 24U;
                }
                return static_cast<std::uint32_t>(bytes[offset]) << 24U |
                       static_cast<std::uint32_t>(bytes[offset + 1]) << 16U |
                       static_cast<std::uint32_t>(bytes[offset + 2]) << 8U |
                       static_cast<std::uint32_t>(bytes[offset + 3]);
            };
            if (u16(tiff + 2) != 42) { return 1; }
            const std::size_t ifd = tiff + u32(tiff + 4);
            if (ifd + 2 > bytes.size()) { return 1; }
            const std::uint16_t count = u16(ifd);
            for (std::uint16_t i = 0; i < count; ++i) {
                const std::size_t entry = ifd + 2 + static_cast<std::size_t>(i) * 12;
                if (entry + 12 > bytes.size()) { return 1; }
                if (u16(entry) == 0x0112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
                    const int orientation = u16(entry + 8);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
            return 1;
        }
        marker += 2 + length;
    }
    return 1;
}

struct BufferCursor {
    const std::uint8_t* data = nullptr;
    std::size_t size         = 0;
    std::size_t offset       = 0;
};

int read_packet(void* opaque, std::uint8_t* output, int size) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (cursor.offset == cursor.size) { return AVERROR_EOF; }
    const std::size_t amount =
        std::min<std::size_t>(static_cast<std::size_t>(size), cursor.size - cursor.offset);
    std::memcpy(output, cursor.data + cursor.offset, amount);
    cursor.offset += amount;
    return static_cast<int>(amount);
}

class AvImageBuffer {
public:
    AvImageBuffer(int width, int height, AVPixelFormat format) {
        constexpr int alignment = 64;
        const int rc =
            av_image_alloc(data_.data(), linesize_.data(), width, height, format, alignment);
        if (rc == AVERROR(ENOMEM)) { throw std::bad_alloc(); }
        if (rc < 0) {
            throw std::runtime_error("failed to allocate media conversion buffer: " + av_error(rc));
        }
    }

    ~AvImageBuffer() { av_freep(data_.data()); }

    AvImageBuffer(const AvImageBuffer&)            = delete;
    AvImageBuffer& operator=(const AvImageBuffer&) = delete;

    [[nodiscard]] std::uint8_t* const* data() const noexcept { return data_.data(); }

    [[nodiscard]] const int* linesize() const noexcept { return linesize_.data(); }

private:
    std::array<std::uint8_t*, 4> data_{};
    std::array<int, 4> linesize_{};
};

std::int64_t seek_packet(void* opaque, std::int64_t offset, int whence) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (whence == AVSEEK_SIZE) { return static_cast<std::int64_t>(cursor.size); }
    const int base_kind = whence & ~AVSEEK_FORCE;
    std::int64_t base   = 0;
    if (base_kind == SEEK_CUR) {
        base = static_cast<std::int64_t>(cursor.offset);
    } else if (base_kind == SEEK_END) {
        base = static_cast<std::int64_t>(cursor.size);
    } else if (base_kind != SEEK_SET) {
        return AVERROR(EINVAL);
    }
    if (offset < -base || base + offset < 0 ||
        static_cast<std::uint64_t>(base + offset) > cursor.size) {
        return AVERROR(EINVAL);
    }
    cursor.offset = static_cast<std::size_t>(base + offset);
    return static_cast<std::int64_t>(cursor.offset);
}

Image rotate_image(Image image, int orientation) {
    if (orientation == 1) { return image; }
    const int width  = image.width;
    const int height = image.height;
    Image rotated;
    const bool swap = orientation >= 5;
    rotated.width   = swap ? height : width;
    rotated.height = swap ? width : height;
    rotated.rgb.resize(image.rgb.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rx = 0;
            int ry = 0;
            if (orientation == 2) {
                rx = width - 1 - x;
                ry = y;
            } else if (orientation == 3) {
                rx = width - 1 - x;
                ry = height - 1 - y;
            } else if (orientation == 4) {
                rx = x;
                ry = height - 1 - y;
            } else if (orientation == 5) {
                rx = y;
                ry = x;
            } else if (orientation == 6) {
                rx = height - 1 - y;
                ry = x;
            } else if (orientation == 7) {
                rx = height - 1 - y;
                ry = width - 1 - x;
            } else { // orientation 8
                rx = y;
                ry = width - 1 - x;
            }
            const std::size_t src = (static_cast<std::size_t>(y) * width + x) * 3;
            const std::size_t dst_index =
                (static_cast<std::size_t>(ry) * rotated.width + rx) * 3;
            std::copy_n(image.rgb.data() + src, 3, rotated.rgb.data() + dst_index);
        }
    }
    return rotated;
}

// libjpeg's exact still-image color decode: the "fancy" triangle chroma upsampling of
// jdsample.c and the fixed-point YCbCr->RGB tables of jdcolor.c. The reference image
// processors of the served models (HF/PIL) decode JPEG through libjpeg, while swscale
// upsamples subsampled chroma by nearest neighbour (SWS_POINT) and converts with its own
// tables; on an ordinary 4:2:0 JPEG that colors every sharp edge by tens of levels.
namespace jpeg_reference {

constexpr int kScaleBits = 16;
constexpr int kOneHalf   = 1 << (kScaleBits - 1);
constexpr int kCenter    = 128;

constexpr int fix(double value) {
    return static_cast<int>(value * 65536.0 + 0.5);
}

struct Tables {
    // R and B hold the rounded delta added to Y; G holds the 16.16 scaled sums
    // (Cb carries the +OneHalf rounding of the pair).
    int cr_r[256];
    int cb_b[256];
    int cb_g[256];
    int cr_g[256];

    Tables() {
        constexpr int cr_r_factor = fix(1.40200);
        constexpr int cb_b_factor = fix(1.77200);
        constexpr int cr_g_factor = -fix(0.71414);
        constexpr int cb_g_factor = -fix(0.34414);
        for (int value = 0; value < 256; ++value) {
            const int x = value - kCenter;
            cr_r[value] = (cr_r_factor * x + kOneHalf) >> kScaleBits;
            cb_b[value] = (cb_b_factor * x + kOneHalf) >> kScaleBits;
            cb_g[value] = cb_g_factor * x + kOneHalf;
            cr_g[value] = cr_g_factor * x;
        }
    }
};

const Tables& tables() {
    static const Tables instance;
    return instance;
}

std::uint8_t clamp_byte(int value) {
    return value < 0 ? 0 : (value > 255 ? 255 : static_cast<std::uint8_t>(value));
}

// h2v1_fancy_upsample: one full-resolution row from one downsampled row. libjpeg alternates
// the rounding bias between output columns instead of rounding every half up. Requires
// width > 2; narrower chroma takes the box path in rgb_jpeg.
void upsample_h2v1(const std::uint8_t* row, int width, std::uint8_t* out) {
    out[0] = row[0];
    out[1] = clamp_byte((row[0] * 3 + row[1] + 2) >> 2);
    for (int x = 1; x < width - 1; ++x) {
        out[2 * x]     = clamp_byte((row[x] * 3 + row[x - 1] + 1) >> 2);
        out[2 * x + 1] = clamp_byte((row[x] * 3 + row[x + 1] + 2) >> 2);
    }
    out[2 * width - 2] = clamp_byte((row[width - 1] * 3 + row[width - 2] + 1) >> 2);
    out[2 * width - 1] = row[width - 1];
}

// h2v2_fancy_upsample, one output row at a time: first the vertical (3,1)/4 sum of the two
// nearest chroma rows, then the horizontal (3,1)/4 pass over those sums, /16 in total.
// Requires width > 2; narrower chroma takes the box path in rgb_jpeg.
void upsample_h2v2(const int* colsum, int width, std::uint8_t* out) {
    out[0] = clamp_byte((colsum[0] * 4 + 8) >> 4);
    out[1] = clamp_byte((colsum[0] * 3 + colsum[1] + 7) >> 4);
    for (int x = 1; x < width - 1; ++x) {
        out[2 * x]     = clamp_byte((colsum[x] * 3 + colsum[x - 1] + 8) >> 4);
        out[2 * x + 1] = clamp_byte((colsum[x] * 3 + colsum[x + 1] + 7) >> 4);
    }
    out[2 * width - 2] = clamp_byte((colsum[width - 1] * 3 + colsum[width - 2] + 8) >> 4);
    out[2 * width - 1] = clamp_byte((colsum[width - 1] * 4 + 7) >> 4);
}

} // namespace jpeg_reference

class Decoder {
public:
    Decoder(std::span<const std::uint8_t> input, std::uint64_t max_decoded_pixels)
        : max_decoded_pixels_(max_decoded_pixels) {
        try {
            format_ = avformat_alloc_context();
            if (format_ == nullptr) { throw std::bad_alloc(); }
            cursor_              = BufferCursor{input.data(), input.size(), 0};
            std::uint8_t* buffer = static_cast<std::uint8_t*>(av_malloc(32'768));
            if (buffer == nullptr) { throw std::bad_alloc(); }
            io_ =
                avio_alloc_context(buffer, 32'768, 0, &cursor_, read_packet, nullptr, seek_packet);
            if (io_ == nullptr) {
                av_free(buffer);
                throw std::bad_alloc();
            }
            format_->pb = io_;
            format_->flags |= AVFMT_FLAG_CUSTOM_IO;
            AVFormatContext* raw = format_;
            int rc               = avformat_open_input(&raw, nullptr, nullptr, nullptr);
            format_              = raw;
            if (rc < 0) { throw_invalid_media("failed to open media: " + av_error(rc)); }
            if ((rc = avformat_find_stream_info(format_, nullptr)) < 0) {
                throw_invalid_media("failed to inspect media: " + av_error(rc));
            }
            stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream_index_ < 0) { throw_invalid_media("media has no decodable video stream"); }
            stream_              = format_->streams[stream_index_];
            const AVCodec* codec = avcodec_find_decoder(stream_->codecpar->codec_id);
            if (codec == nullptr) { throw_invalid_media("media codec is not supported"); }
            codec_ = avcodec_alloc_context3(codec);
            if (codec_ == nullptr) { throw std::bad_alloc(); }
            if ((rc = avcodec_parameters_to_context(codec_, stream_->codecpar)) < 0 ||
                (rc = avcodec_open2(codec_, codec, nullptr)) < 0) {
                throw_invalid_media("failed to open media codec: " + av_error(rc));
            }
            packet_ = av_packet_alloc();
            frame_  = av_frame_alloc();
            if (packet_ == nullptr || frame_ == nullptr) { throw std::bad_alloc(); }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Decoder() { cleanup(); }

    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] AVStream* stream() const noexcept { return stream_; }

    [[nodiscard]] double duration_seconds() const noexcept {
        if (stream_->duration > 0) {
            return static_cast<double>(stream_->duration) * av_q2d(stream_->time_base);
        }
        if (format_->duration > 0) { return static_cast<double>(format_->duration) / AV_TIME_BASE; }
        return 0.0;
    }

    template <typename Callback>
    void frames(Callback&& callback) {
        int index    = 0;
        auto receive = [&] {
            while (true) {
                const int rc = avcodec_receive_frame(codec_, frame_);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return true; }
                if (rc < 0) {
                    throw_invalid_media("failed to decode media frame: " + av_error(rc));
                }
                if (frame_->crop_top != 0 || frame_->crop_bottom != 0 || frame_->crop_left != 0 ||
                    frame_->crop_right != 0) {
                    const int crop = av_frame_apply_cropping(frame_, 0);
                    if (crop < 0) {
                        throw_invalid_media("failed to apply media display crop: " +
                                            av_error(crop));
                    }
                }
                const bool keep_decoding = callback(index++, frame_);
                av_frame_unref(frame_);
                if (!keep_decoding) { return false; }
            }
        };
        while (true) {
            const int rc = av_read_frame(format_, packet_);
            if (rc == AVERROR_EOF) { break; }
            if (rc < 0) { throw_invalid_media("failed while reading media: " + av_error(rc)); }
            if (packet_->stream_index == stream_index_) {
                const int send = avcodec_send_packet(codec_, packet_);
                if (send < 0 && send != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    throw_invalid_media("failed to submit media packet: " + av_error(send));
                }
                if (!receive()) {
                    av_packet_unref(packet_);
                    return;
                }
            }
            av_packet_unref(packet_);
        }
        const int flush = avcodec_send_packet(codec_, nullptr);
        if (flush < 0 && flush != AVERROR_EOF) {
            throw_invalid_media("failed to flush media decoder: " + av_error(flush));
        }
        (void)receive();
    }

    Image rgb(const AVFrame* frame, int orientation, bool composite_alpha = false) {
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) {
            throw_invalid_media("decoded media dimensions are invalid");
        }
        if (static_cast<std::uint64_t>(width) * height > max_decoded_pixels_) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        Image out;
        out.width  = width;
        out.height = height;
        const AVPixFmtDescriptor* descriptor =
            av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        const bool alpha = composite_alpha && descriptor != nullptr &&
                           (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        const AVPixelFormat destination_format = alpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
        AvImageBuffer converted(width, height, destination_format);
        sws_ = sws_getCachedContext(sws_, width, height, static_cast<AVPixelFormat>(frame->format),
                                    width, height, destination_format, SWS_POINT, nullptr, nullptr,
                                    nullptr);
        if (sws_ == nullptr) { throw std::runtime_error("failed to create media color converter"); }
        const int rows = sws_scale(sws_, frame->data, frame->linesize, 0, height, converted.data(),
                                   converted.linesize());
        if (rows != height) { throw std::runtime_error("failed to convert media frame to RGB"); }
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        out.rgb.resize(pixels * 3);
        if (!alpha) {
            const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
            for (int y = 0; y < height; ++y) {
                std::memcpy(out.rgb.data() + static_cast<std::size_t>(y) * row_bytes,
                            converted.data()[0] +
                                static_cast<std::size_t>(y) * converted.linesize()[0],
                            row_bytes);
            }
        } else {
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* source =
                    converted.data()[0] + static_cast<std::size_t>(y) * converted.linesize()[0];
                std::uint8_t* destination =
                    out.rgb.data() + static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    const int a = source[4 * x + 3];
                    for (int c = 0; c < 3; ++c) {
                        destination[3 * x + c] = static_cast<std::uint8_t>(
                            ((255 - a) * 255 + a * source[4 * x + c] + 127) / 255);
                    }
                }
            }
        }
        return rotate_image(std::move(out), orientation);
    }

    // Still images decoded by the JPEG codec take libjpeg's color path (jpeg_reference
    // above); every other layout and codec keeps the swscale converter.
    Image rgb_jpeg(const AVFrame* frame, int orientation) {
        const AVPixFmtDescriptor* descriptor =
            av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        if (descriptor == nullptr || (descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0 ||
            descriptor->nb_components == 0 || descriptor->comp[0].depth != 8 ||
            descriptor->log2_chroma_w > 1 || descriptor->log2_chroma_h > 1 ||
            (descriptor->nb_components != 1 && descriptor->nb_components != 3)) {
            return rgb(frame, orientation);
        }
        for (int component = 0; component < descriptor->nb_components; ++component) {
            if (descriptor->comp[component].plane != component ||
                descriptor->comp[component].offset != 0 ||
                descriptor->comp[component].step != 1) {
                return rgb(frame, orientation);
            }
        }
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) { throw_invalid_media("decoded media dimensions are invalid"); }
        if (static_cast<std::uint64_t>(width) * height > max_decoded_pixels_) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        Image out;
        out.width  = width;
        out.height = height;
        out.rgb.resize(static_cast<std::size_t>(width) * height * 3);
        const std::uint8_t* y_plane = frame->data[0];
        const int y_stride          = frame->linesize[0];
        if (descriptor->nb_components == 1) {
            // Grayscale: libjpeg passes Y to all three channels.
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* source = y_plane + static_cast<std::size_t>(y) * y_stride;
                std::uint8_t* destination =
                    out.rgb.data() + static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    destination[3 * x] = destination[3 * x + 1] = destination[3 * x + 2] = source[x];
                }
            }
            return rotate_image(std::move(out), orientation);
        }
        const int lw = descriptor->log2_chroma_w;
        const int lh = descriptor->log2_chroma_h;
        const int cw = AV_CEIL_RSHIFT(width, lw);
        const int ch = AV_CEIL_RSHIFT(height, lh);
        const std::uint8_t* cb_plane = frame->data[1];
        const std::uint8_t* cr_plane = frame->data[2];
        const int cb_stride          = frame->linesize[1];
        const int cr_stride          = frame->linesize[2];
        const jpeg_reference::Tables& tables = jpeg_reference::tables();
        std::vector<std::uint8_t> cb_row(static_cast<std::size_t>(width));
        std::vector<std::uint8_t> cr_row(static_cast<std::size_t>(width));
        std::vector<int> cb_sums(static_cast<std::size_t>(cw));
        std::vector<int> cr_sums(static_cast<std::size_t>(cw));
        // libjpeg falls back from fancy to box upsampling when the downsampled width is
        // at most two samples (jinit_upsampler's do_fancy guard).
        const bool narrow = cw <= 2;
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* y_row = y_plane + static_cast<std::size_t>(y) * y_stride;
            std::uint8_t* destination = out.rgb.data() + static_cast<std::size_t>(y) * width * 3;
            for (int plane = 0; plane < 2; ++plane) {
                const std::uint8_t* source =
                    (plane == 0 ? cb_plane : cr_plane) +
                    static_cast<std::size_t>(y >> lh) * (plane == 0 ? cb_stride : cr_stride);
                std::uint8_t* upsampled = plane == 0 ? cb_row.data() : cr_row.data();
                int* sums               = plane == 0 ? cb_sums.data() : cr_sums.data();
                if (lw == 0 && lh == 0) {
                    std::copy_n(source, width, upsampled);
                } else if (narrow) {
                    for (int x = 0; x < cw; ++x) {
                        upsampled[2 * x] = source[x];
                        if (2 * x + 1 < width) { upsampled[2 * x + 1] = source[x]; }
                    }
                } else if (lw == 1 && lh == 0) {
                    jpeg_reference::upsample_h2v1(source, cw, upsampled);
                } else if (lw == 0 && lh == 1) {
                    // h1v2_fancy_upsample: the vertical triangle only, with the same
                    // alternating bias the h2v1 pass uses between its two output rows.
                    const int r   = y >> 1;
                    const int far = (y & 1) == 0 ? (r > 0 ? r - 1 : 0)
                                                  : (r < ch - 1 ? r + 1 : ch - 1);
                    const std::uint8_t* far_row =
                        (plane == 0 ? cb_plane : cr_plane) +
                        static_cast<std::size_t>(far) * (plane == 0 ? cb_stride : cr_stride);
                    const int bias = (y & 1) == 0 ? 1 : 2;
                    for (int x = 0; x < cw; ++x) {
                        upsampled[x] = jpeg_reference::clamp_byte(
                            (source[x] * 3 + far_row[x] + bias) >> 2);
                    }
                } else {
                    // h2v2: vertical (3,1)/4 sums, then the horizontal pass, /16 overall.
                    const int r   = y >> 1;
                    const int far = (y & 1) == 0 ? (r > 0 ? r - 1 : 0)
                                                  : (r < ch - 1 ? r + 1 : ch - 1);
                    const std::uint8_t* far_row =
                        (plane == 0 ? cb_plane : cr_plane) +
                        static_cast<std::size_t>(far) * (plane == 0 ? cb_stride : cr_stride);
                    for (int x = 0; x < cw; ++x) {
                        sums[x] = source[x] * 3 + far_row[x];
                    }
                    jpeg_reference::upsample_h2v2(sums, cw, upsampled);
                }
            }
            for (int x = 0; x < width; ++x) {
                const int y_value = y_row[x];
                const int cb      = cb_row[static_cast<std::size_t>(x)];
                const int cr      = cr_row[static_cast<std::size_t>(x)];
                destination[3 * x] =
                    jpeg_reference::clamp_byte(y_value + tables.cr_r[cr]);
                destination[3 * x + 1] = jpeg_reference::clamp_byte(
                    y_value + ((tables.cb_g[cb] + tables.cr_g[cr]) >> jpeg_reference::kScaleBits));
                destination[3 * x + 2] =
                    jpeg_reference::clamp_byte(y_value + tables.cb_b[cb]);
            }
        }
        return rotate_image(std::move(out), orientation);
    }

    [[nodiscard]] bool stream_is_jpeg() const noexcept {
        return stream_ != nullptr && stream_->codecpar->codec_id == AV_CODEC_ID_MJPEG;
    }

    // The still-image entry: JPEG streams take the libjpeg color path, everything else the
    // swscale converter. Video keeps swscale (its frames have no libjpeg reference).
    Image still_rgb(const AVFrame* frame, int orientation) {
        return stream_is_jpeg() ? rgb_jpeg(frame, orientation) : rgb(frame, orientation);
    }

private:
    void cleanup() noexcept {
        sws_freeContext(sws_);
        sws_ = nullptr;
        av_frame_free(&frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_);
        if (format_ != nullptr) { avformat_close_input(&format_); }
        if (io_ != nullptr) { avio_context_free(&io_); }
    }

    AVFormatContext* format_ = nullptr;
    AVIOContext* io_         = nullptr;
    AVCodecContext* codec_   = nullptr;
    AVPacket* packet_        = nullptr;
    AVFrame* frame_          = nullptr;
    AVStream* stream_        = nullptr;
    SwsContext* sws_         = nullptr;
    BufferCursor cursor_;
    int stream_index_                 = -1;
    std::uint64_t max_decoded_pixels_ = 0;
};

int rotation_of(const AVStream* stream) {
    int rotation                 = 0;
    const AVDictionaryEntry* tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
    if (tag != nullptr) {
        rotation = static_cast<int>(std::nearbyint(std::strtod(tag->value, nullptr)));
    }
    const AVPacketSideData* side =
        av_packet_side_data_get(stream->codecpar->coded_side_data,
                                stream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
    if (side != nullptr && side->size >= 9 * sizeof(std::int32_t)) {
        rotation = static_cast<int>(std::nearbyint(
            -av_display_rotation_get(reinterpret_cast<const std::int32_t*>(side->data))));
    }
    rotation %= 360;
    if (rotation < 0) { rotation += 360; }
    if (rotation == 89 || rotation == 91) { rotation = 90; }
    if (rotation == 179 || rotation == 181) { rotation = 180; }
    if (rotation == 269 || rotation == 271) { rotation = 270; }
    return rotation == 90 || rotation == 180 || rotation == 270 ? rotation : 0;
}

int orientation_from_rotation(int rotation) {
    if (rotation == 90) { return 6; }
    if (rotation == 180) { return 3; }
    if (rotation == 270) { return 8; }
    return 1;
}

ImageInfo frame_info(const AVFrame* frame, int orientation, std::uint64_t max_decoded_pixels) {
    const int width  = frame->width;
    const int height = frame->height;
    if (width <= 0 || height <= 0) { throw_invalid_media("decoded media dimensions are invalid"); }
    if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) >
        max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    const bool swap = orientation >= 5;
    return ImageInfo{.width = swap ? height : width, .height = swap ? width : height};
}

double fps_of(const AVStream* stream) {
    double fps = av_q2d(stream->avg_frame_rate);
    if (!(fps > 0.0) || !std::isfinite(fps)) { fps = av_q2d(stream->r_frame_rate); }
    return fps > 0.0 && std::isfinite(fps) ? fps : 24.0;
}

int count_frames(std::span<const std::uint8_t> input, const Policy& policy) {
    Decoder decoder(input, policy.max_decoded_pixels);
    int count = 0;
    decoder.frames([&](int index, const AVFrame*) {
        if (policy.checkpoint) { policy.checkpoint(); }
        count = index + 1;
        if (count > policy.max_video_source_frames) {
            throw Error(ErrorKind::BudgetExceeded,
                        "video source frame count exceeds processor limit");
        }
        return true;
    });
    return count;
}

std::vector<int> sample_indices(int total, double source_fps, double target_fps, int min_frames,
                                int max_frames) {
    if (total <= 0 || source_fps <= 0.0 || target_fps <= 0.0 || min_frames <= 0 ||
        max_frames < min_frames) {
        throw std::invalid_argument("invalid video sampling configuration");
    }
    int count = static_cast<int>(static_cast<double>(total) / source_fps * target_fps);
    count     = std::min({std::max(count, min_frames), max_frames, total});
    std::vector<int> indices(static_cast<std::size_t>(count));
    if (count == 1) {
        indices[0] = 0;
        return indices;
    }
    for (int i = 0; i < count; ++i) {
        const double value                   = static_cast<double>(i) * (total - 1) / (count - 1);
        indices[static_cast<std::size_t>(i)] = static_cast<int>(std::nearbyint(value));
    }
    return indices;
}

struct VideoPlan {
    int total_frames   = 0;
    int minimum_frames = 0;
    double fps         = 0.0;
    double duration    = 0.0;
    int orientation    = 1;
    std::vector<int> indices;
};

VideoPlan make_video_plan(std::span<const std::uint8_t> bytes, const Policy& policy,
                          double target_fps, int min_frames, int max_frames) {
    Decoder probe(bytes, policy.max_decoded_pixels);
    VideoPlan plan;
    plan.fps          = fps_of(probe.stream());
    plan.total_frames = probe.stream()->nb_frames > 0 &&
                                probe.stream()->nb_frames <= std::numeric_limits<int>::max()
                            ? static_cast<int>(probe.stream()->nb_frames)
                            : 0;
    plan.duration     = probe.duration_seconds();
    if (plan.total_frames == 0 && plan.duration > 0.0) {
        plan.total_frames = static_cast<int>(std::nearbyint(plan.duration * plan.fps));
    }
    if (plan.duration > policy.max_video_duration_seconds) {
        throw Error(ErrorKind::BudgetExceeded, "video duration exceeds processor limit");
    }
    if (plan.total_frames == 0) { plan.total_frames = count_frames(bytes, policy); }
    if (plan.total_frames == 0) { throw_invalid_media("video contains no decoded frame"); }
    if (plan.total_frames > policy.max_video_source_frames) {
        throw Error(ErrorKind::BudgetExceeded, "video source frame count exceeds processor limit");
    }
    plan.indices = sample_indices(plan.total_frames, plan.fps, target_fps, min_frames, max_frames);
    const std::uint64_t coded_pixels =
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->width, 0)) *
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->height, 0));
    if (coded_pixels != 0 && plan.indices.size() > policy.max_decoded_video_pixels / coded_pixels) {
        throw Error(ErrorKind::BudgetExceeded,
                    "sampled source video pixels exceed processor limit");
    }
    plan.minimum_frames = std::min(min_frames, plan.total_frames);
    plan.orientation    = orientation_from_rotation(rotation_of(probe.stream()));
    return plan;
}

struct VideoScan {
    ImageInfo first_frame;
    int sampled_frames = 0;
};

template <typename Retain>
VideoScan scan_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                     const VideoPlan& plan, Retain&& retain) {
    Decoder decoder(bytes, policy.max_decoded_pixels);
    VideoScan scan;
    std::size_t wanted            = 0;
    int decoded                   = 0;
    std::uint64_t retained_pixels = 0;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        decoded = index + 1;
        if (wanted < plan.indices.size() && index == plan.indices[wanted]) {
            const ImageInfo info = frame_info(frame, plan.orientation, policy.max_decoded_pixels);
            const std::uint64_t pixels = static_cast<std::uint64_t>(frame->width) *
                                         static_cast<std::uint64_t>(frame->height);
            if (retained_pixels > policy.max_decoded_video_pixels ||
                pixels > policy.max_decoded_video_pixels - retained_pixels) {
                throw Error(ErrorKind::BudgetExceeded,
                            "sampled source video pixels exceed processor limit");
            }
            retained_pixels += pixels;
            if (wanted == 0) { scan.first_frame = info; }
            retain(decoder, frame, plan.orientation);
            ++wanted;
        }
        return wanted != plan.indices.size();
    });
    if (wanted < static_cast<std::size_t>(plan.minimum_frames)) {
        throw_invalid_media("video decoder produced fewer than the minimum sampled frames");
    }
    for (std::size_t index = wanted; index < plan.indices.size(); ++index) {
        if (plan.indices[index] < decoded) {
            throw_invalid_media("video decoder skipped a requested internal frame");
        }
    }
    scan.sampled_frames = static_cast<int>(wanted);
    return scan;
}

std::mutex library_log_mutex;
std::shared_ptr<const LibraryLogHandler> library_log_handler;

std::shared_ptr<const LibraryLogHandler> current_library_log_handler() {
    std::lock_guard lock(library_log_mutex);
    return library_log_handler;
}

std::string_view library_log_source(void* context) noexcept {
    if (context == nullptr) { return {}; }
    const AVClass* av_class = *static_cast<const AVClass* const*>(context);
    if (av_class == nullptr || av_class->item_name == nullptr) { return {}; }
    const char* name = av_class->item_name(context);
    return name == nullptr ? std::string_view{} : std::string_view(name);
}

LibraryLogSeverity library_log_severity(int level) noexcept {
    if (level <= AV_LOG_ERROR) { return LibraryLogSeverity::Error; }
    if (level <= AV_LOG_WARNING) { return LibraryLogSeverity::Warning; }
    return LibraryLogSeverity::Info;
}

// FFmpeg may build one line from several calls, so each decoding thread assembles its own; the
// line keeps the source of its first part and the most severe level of any part.
struct PendingLibraryLine {
    std::string text;
    std::string source;
    int level = AV_LOG_QUIET;
};

void forward_library_log(void* context, int level, const char* format, std::va_list args) {
    const std::shared_ptr<const LibraryLogHandler> handler = current_library_log_handler();
    if (!handler) {
        // The default was restored between FFmpeg reading its callback and calling it.
        av_log_default_callback(context, level, format, args);
        return;
    }
    // The high byte carries FFmpeg's colour hint.
    const int severity_level = level & 0xff;
    if (severity_level > av_log_get_level()) { return; }
    thread_local PendingLibraryLine pending;
    try {
        std::array<char, 1024> buffer;
        std::va_list measured;
        va_copy(measured, args);
        const int length = std::vsnprintf(buffer.data(), buffer.size(), format, measured);
        va_end(measured);
        if (length < 0) { return; }
        if (pending.text.empty()) {
            pending.source = library_log_source(context);
            pending.level  = severity_level;
        } else {
            pending.level = std::min(pending.level, severity_level);
        }
        if (static_cast<std::size_t>(length) < buffer.size()) {
            pending.text.append(buffer.data(), static_cast<std::size_t>(length));
        } else {
            const std::size_t offset = pending.text.size();
            pending.text.resize(offset + static_cast<std::size_t>(length));
            std::vsnprintf(pending.text.data() + offset, static_cast<std::size_t>(length) + 1,
                           format, args);
        }
        std::size_t start = 0;
        for (std::size_t end = pending.text.find('\n'); end != std::string::npos;
             start = end + 1, end = pending.text.find('\n', start)) {
            std::string_view message(pending.text.data() + start, end - start);
            if (message.ends_with('\r')) { message.remove_suffix(1); }
            if (message.empty()) { continue; }
            (*handler)(LibraryLogLine{.severity = library_log_severity(pending.level),
                                      .source   = pending.source,
                                      .message  = message});
        }
        pending.text.erase(0, start);
    } catch (...) {
        // An exception must not unwind through FFmpeg; the line is dropped.
        pending.text.clear();
    }
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
#ifdef INFERNIX_MEDIA_NATIVE_PNG
    if (is_png(bytes)) {
        return png_image_info(bytes, policy.max_decoded_pixels);
    }
#endif
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    ImageInfo result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result = frame_info(frame, orientation, policy.max_decoded_pixels);
            return false;
        }
        return true;
    });
    if (result.width == 0 || result.height == 0) {
        throw_invalid_media("image contains no decoded frame");
    }
    return result;
}

VideoInfo inspect_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                        double target_fps, int min_frames, int max_frames) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    VideoPlan plan       = make_video_plan(bytes, policy, target_fps, min_frames, max_frames);
    const VideoScan scan = scan_video(bytes, policy, plan, [](Decoder&, const AVFrame*, int) {});
    VideoInfo out;
    out.width          = scan.first_frame.width;
    out.height         = scan.first_frame.height;
    out.total_frames   = plan.total_frames;
    out.sampled_frames = scan.sampled_frames;
    out.fps            = plan.fps;
    out.duration =
        plan.duration > 0.0 ? plan.duration : static_cast<double>(plan.total_frames) / plan.fps;
    out.indices = std::move(plan.indices);
    return out;
}

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    // Some FFmpeg builds (the Windows vcpkg tree) ship without the png decoder; the
    // native path decodes png directly so those builds accept png images too.
#ifdef INFERNIX_MEDIA_NATIVE_PNG
    if (is_png(bytes)) {
        return decode_png(bytes, policy.max_decoded_pixels);
    }
#endif
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    Image result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result = decoder.still_rgb(frame, orientation);
            return false;
        }
        return true;
    });
    if (result.rgb.empty()) { throw_invalid_media("image contains no decoded frame"); }
    return result;
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    VideoPlan plan = make_video_plan(bytes, policy, target_fps, min_frames, max_frames);
    Video out;
    out.total_frames = plan.total_frames;
    out.fps          = plan.fps;
    out.duration =
        plan.duration > 0.0 ? plan.duration : static_cast<double>(plan.total_frames) / plan.fps;
    out.frames.reserve(plan.indices.size());
    const VideoScan scan = scan_video(
        bytes, policy, plan, [&](Decoder& decoder, const AVFrame* frame, int orientation) {
            out.frames.push_back(decoder.rgb(frame, orientation, true));
        });
    out.width   = scan.first_frame.width;
    out.height  = scan.first_frame.height;
    out.indices = std::move(plan.indices);
    return out;
}

LibraryLogHandler set_library_log_handler(LibraryLogHandler handler) {
    std::shared_ptr<const LibraryLogHandler> installed;
    if (handler) { installed = std::make_shared<const LibraryLogHandler>(std::move(handler)); }
    std::shared_ptr<const LibraryLogHandler> previous;
    {
        // FFmpeg's callback changes under the same lock, so concurrent owners cannot leave it
        // disagreeing with the installed handler.
        std::lock_guard lock(library_log_mutex);
        previous = std::exchange(library_log_handler, installed);
        av_log_set_callback(installed ? forward_library_log : av_log_default_callback);
    }
    return previous ? *previous : LibraryLogHandler{};
}

} // namespace infernix::media::decode
