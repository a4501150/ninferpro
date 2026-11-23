// RoPE benchmark over the public ops::rope contract (include/infernix/ops/rope.h): the registered
// Text D256/R64 Q/K head geometries 24/4 and 16/2 with one or three position axes, the full-head
// D128/R128 1-D Text geometry 32/8, and the Vision 2-D geometry 16/16. The benchmark includes no
// private kernels or launch policy; every measured route goes through the public entry. --control
// runs a bench-local same-payload kernel: it reads the same positions, computes one angle per
// rotary pair, and reads and writes the same rotary region of Q/K, with bench-fixed launch
// geometry (one 256-thread CTA per token); it is a traffic and sincos reference, not the
// production kernel and not its launch policy.
// Examples:
//   ./infernix_rope_bench --text --geometry d128 --axes 1 \
//       --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
//   ./infernix_rope_bench --text --geometry all --tokens 16 --control
//   ncu ... ./infernix_rope_bench --text --geometry d128 --tokens 1024 --profile
//   ./infernix_rope_bench --vision --patches 8,256,4096,49152,65536
#include "core/device.h"
#include "infernix/ops/rope.h"
#include "infernix_bench_common.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::bench;

namespace {

constexpr int kTextHeadDim    = 256;
constexpr int kTextRotaryDim  = 64;
constexpr float kTextTheta    = 1.0e7F;
constexpr int kFullHeadDim    = 128;
constexpr int kFullHeadQHeads = 32;
constexpr int kFullHeadKHeads = 8;
constexpr int kVisionHeadDim  = 72;
constexpr int kVisionHeads    = 16;
constexpr float kVisionTheta  = 10'000.0F;
// Bench-fixed control geometry, independent of any private launch policy.
constexpr int kControlBlock = 256;

std::vector<int> parse_csv(const char* value) {
    std::vector<int> values;
    const std::string text(value);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string part = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int parsed       = std::stoi(part);
        if (parsed <= 0) { throw std::invalid_argument("benchmark dimensions must be positive"); }
        values.push_back(parsed);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (values.empty()) { throw std::invalid_argument("empty benchmark dimension list"); }
    return values;
}

DeviceBuffer copy_positions(const std::vector<std::int32_t>& host) {
    DeviceBuffer device(host.size() * sizeof(std::int32_t));
    cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice);
    return device;
}

DeviceBuffer make_text_positions(int tokens, int axes) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(tokens) * axes);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            host[static_cast<std::size_t>(axis) * tokens + token] = axis * 97 + token;
        }
    }
    return copy_positions(host);
}

struct VisionGrid {
    int temporal;
    int height;
    int width;
};

VisionGrid canonical_vision_grid(int patches) {
    switch (patches) {
    case 8:
        return {2, 2, 2};
    case 256:
        return {1, 16, 16};
    case 4096:
        return {1, 64, 64};
    case 49152:
        return {3, 128, 128};
    case 65536:
        return {1, 256, 256};
    default:
        throw std::invalid_argument("Vision benchmark supports P=8,256,4096,49152,65536");
    }
}

DeviceBuffer make_vision_positions(int patches) {
    const VisionGrid grid = canonical_vision_grid(patches);
    std::vector<std::int32_t> host(static_cast<std::size_t>(patches) * 2);
    int token = 0;
    for (int temporal = 0; temporal < grid.temporal; ++temporal) {
        (void)temporal;
        for (int block_h = 0; block_h < grid.height / 2; ++block_h) {
            for (int block_w = 0; block_w < grid.width / 2; ++block_w) {
                for (int inner_h = 0; inner_h < 2; ++inner_h) {
                    for (int inner_w = 0; inner_w < 2; ++inner_w) {
                        host[token]                                     = block_h * 2 + inner_h;
                        host[static_cast<std::size_t>(patches) + token] = block_w * 2 + inner_w;
                        ++token;
                    }
                }
            }
        }
    }
    return copy_positions(host);
}

__device__ __forceinline__ void copy_bf16x8(__nv_bfloat16* data, std::int64_t index) {
    void* address = data + index;
    unsigned int x, y, z, w;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(address)
                 : "memory");
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}

// Bench-local payload control: one sincos per rotary pair from the represented positions, then the
// same rotary-region reads and writes as the Op, with the bench-fixed kControlBlock geometry. It
// is not the production kernel and does not represent its launch policy.
template <int HeadDim, int RotaryDim, int QHeads, int KHeads>
__global__ void rope_payload_control_kernel(const std::int32_t* positions, int axes, float theta,
                                            __nv_bfloat16* q, __nv_bfloat16* k, int tokens,
                                            std::int64_t q_token_stride,
                                            std::int64_t k_token_stride) {
    static_assert(RotaryDim % 8 == 0, "the control copies whole 8-element vectors");
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    constexpr int half = RotaryDim / 2;
    __shared__ volatile float cos_cache[half];
    __shared__ volatile float sin_cache[half];
    if (threadIdx.x < half) {
        const int pair = static_cast<int>(threadIdx.x);
        float position;
        if (axes == 1) {
            position = static_cast<float>(positions[token]);
        } else {
            position = static_cast<float>(positions[token * axes + pair % axes]);
        }
        const float phi = position * powf(theta, -2.0F * static_cast<float>(pair) /
                                                     static_cast<float>(RotaryDim));
        float sine      = 0.0F;
        float cosine    = 0.0F;
        sincosf(phi, &sine, &cosine);
        cos_cache[pair] = cosine;
        sin_cache[pair] = sine;
    }
    __syncthreads();
    if (threadIdx.x < half) {
        const float cosine = cos_cache[threadIdx.x];
        const float sine   = sin_cache[threadIdx.x];
        asm volatile("" : : "f"(cosine), "f"(sine) : "memory");
    }

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < QHeads + KHeads; combined_head += block_warps) {
        const bool is_q             = combined_head < QHeads;
        const int head              = is_q ? combined_head : combined_head - QHeads;
        __nv_bfloat16* data         = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        const std::int64_t base =
            static_cast<std::int64_t>(token) * stride_t + static_cast<std::int64_t>(head) * HeadDim;
        for (int vector = lane; vector < RotaryDim / 8; vector += 32) {
            copy_bf16x8(data, base + vector * 8);
        }
    }
}

template <int HeadDim, int RotaryDim, int QHeads, int KHeads>
void launch_control(const Tensor& positions, Tensor& q, Tensor& k, float theta,
                    cudaStream_t stream) {
    const int tokens = q.ne[2];
    const int axes   = positions.ne[1];
    rope_payload_control_kernel<HeadDim, RotaryDim, QHeads, KHeads>
        <<<tokens, kControlBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data), axes, theta,
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data), tokens,
            q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

template <int QHeads, int KHeads>
void run_text(DeviceExecutionView execution, int tokens, int axes, bool control,
              const char* geometry) {
    const std::size_t q_elements = static_cast<std::size_t>(kTextHeadDim) * QHeads * tokens;
    const std::size_t k_elements = static_cast<std::size_t>(kTextHeadDim) * KHeads * tokens;
    DeviceBuffer positions       = make_text_positions(tokens, axes);
    DeviceBuffer q               = make_bf16(q_elements, 103U);
    DeviceBuffer k               = make_bf16(k_elements, 105U);
    SavedBuffer initial_q(q);
    SavedBuffer initial_k(k);
    const auto restore = [&](cudaStream_t stream) {
        initial_q.restore(stream);
        initial_k.restore(stream);
    };
    Tensor tpos(positions.p, DType::I32, {tokens, axes});
    Tensor tq(q.p, DType::BF16, {kTextHeadDim, QHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kTextHeadDim, KHeads, tokens});
    const double bytes  = 2.0 * static_cast<double>((QHeads + KHeads) * kTextRotaryDim) * tokens *
                          sizeof(__nv_bfloat16);
    const Result result = bench_loop_prepared(
        restore,
        [&](cudaStream_t stream) {
            if (control) {
                launch_control<kTextHeadDim, kTextRotaryDim, QHeads, KHeads>(tpos, tq, tk,
                                                                             kTextTheta, stream);
            } else {
                ops::rope(tpos, kTextRotaryDim, kTextTheta, tq, tk, execution.on_stream(stream));
            }
        },
        bytes);
    const std::string label = std::string("rope text ") + geometry +
                              " axes=" + std::to_string(axes) + " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

template <int Heads>
void run_text_single(DeviceExecutionView execution, int tokens, int axes, bool control,
                     const char* geometry, const char* role) {
    const std::size_t elements = static_cast<std::size_t>(kTextHeadDim) * Heads * tokens;
    DeviceBuffer positions     = make_text_positions(tokens, axes);
    DeviceBuffer x             = make_bf16(elements, 101U);
    SavedBuffer initial_x(x);
    const auto restore = [&](cudaStream_t stream) { initial_x.restore(stream); };
    Tensor tpos(positions.p, DType::I32, {tokens, axes});
    Tensor tx(x.p, DType::BF16, {kTextHeadDim, Heads, tokens});
    const double bytes =
        2.0 * static_cast<double>(Heads * kTextRotaryDim) * tokens * sizeof(__nv_bfloat16);
    const Result result = bench_loop_prepared(
        restore,
        [&](cudaStream_t stream) {
            if (control) {
                launch_control<kTextHeadDim, kTextRotaryDim, Heads, 0>(tpos, tx, tx, kTextTheta,
                                                                       stream);
            } else {
                ops::rope(tpos, kTextRotaryDim, kTextTheta, tx, execution.on_stream(stream));
            }
        },
        bytes);
    const std::string label = std::string("rope text ") + geometry + " " + role +
                              " axes=" + std::to_string(axes) + " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void run_d128(DeviceExecutionView execution, int tokens, bool control, bool profile) {
    const std::size_t q_elements =
        static_cast<std::size_t>(kFullHeadDim) * kFullHeadQHeads * tokens;
    const std::size_t k_elements =
        static_cast<std::size_t>(kFullHeadDim) * kFullHeadKHeads * tokens;
    DeviceBuffer positions = make_text_positions(tokens, 1);
    DeviceBuffer q         = make_bf16(q_elements, 107U);
    DeviceBuffer k         = make_bf16(k_elements, 109U);
    SavedBuffer initial_q(q);
    SavedBuffer initial_k(k);
    const auto restore = [&](cudaStream_t stream) {
        initial_q.restore(stream);
        initial_k.restore(stream);
    };
    Tensor tpos(positions.p, DType::I32, {tokens});
    Tensor tq(q.p, DType::BF16, {kFullHeadDim, kFullHeadQHeads, tokens});
    Tensor tk(k.p, DType::BF16, {kFullHeadDim, kFullHeadKHeads, tokens});
    const double bytes = 2.0 *
                         static_cast<double>((kFullHeadQHeads + kFullHeadKHeads) * kFullHeadDim) *
                         tokens * sizeof(__nv_bfloat16);
    const auto launch  = [&](cudaStream_t stream) {
        if (control) {
            launch_control<kFullHeadDim, kFullHeadDim, kFullHeadQHeads, kFullHeadKHeads>(
                tpos, tq, tk, kTextTheta, stream);
        } else {
            ops::rope(tpos, kFullHeadDim, kTextTheta, tq, tk, execution.on_stream(stream));
        }
    };
    if (profile) {
        for (int warmup = 0; warmup < 20; ++warmup) { launch(nullptr); }
        CUDA_CHECK(cudaDeviceSynchronize());
        restore(nullptr);
        launch(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile rope text d128 axes=1 T=%d%s\n", tokens, control ? " control" : "");
        return;
    }
    const Result result     = bench_loop_prepared(restore, launch, bytes);
    const std::string label = std::string("rope text d128 axes=1") + (control ? " control" : "") +
                              " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void run_d128_single_k(DeviceExecutionView execution, int tokens, bool control) {
    const std::size_t elements = static_cast<std::size_t>(kFullHeadDim) * kFullHeadKHeads * tokens;
    DeviceBuffer positions     = make_text_positions(tokens, 1);
    DeviceBuffer x             = make_bf16(elements, 111U);
    SavedBuffer initial_x(x);
    const auto restore = [&](cudaStream_t stream) { initial_x.restore(stream); };
    Tensor tpos(positions.p, DType::I32, {tokens});
    Tensor tx(x.p, DType::BF16, {kFullHeadDim, kFullHeadKHeads, tokens});
    const double bytes =
        2.0 * static_cast<double>(kFullHeadKHeads * kFullHeadDim) * tokens * sizeof(__nv_bfloat16);
    const Result result = bench_loop_prepared(
        restore,
        [&](cudaStream_t stream) {
            if (control) {
                launch_control<kFullHeadDim, kFullHeadDim, kFullHeadKHeads, 0>(tpos, tx, tx,
                                                                               kTextTheta, stream);
            } else {
                ops::rope(tpos, kFullHeadDim, kTextTheta, tx, execution.on_stream(stream));
            }
        },
        bytes);
    const std::string label = std::string("rope text d128 k axes=1") + (control ? " control" : "") +
                              " T=" + std::to_string(tokens);
    print_result(label.c_str(), result);
}

void run_vision(DeviceExecutionView execution, int patches, bool control) {
    constexpr int hidden   = kVisionHeadDim * kVisionHeads;
    constexpr int qkv      = hidden * 3;
    DeviceBuffer positions = make_vision_positions(patches);
    DeviceBuffer packed    = make_bf16(static_cast<std::size_t>(qkv) * patches, 113U);
    SavedBuffer initial_packed(packed);
    const auto restore = [&](cudaStream_t stream) { initial_packed.restore(stream); };
    Tensor tq(packed.p, DType::BF16, {kVisionHeadDim, kVisionHeads, patches});
    tq.nb[2]  = qkv * 2;
    Tensor tk = tq;
    tk.data   = static_cast<unsigned char*>(packed.p) + hidden * 2;
    Tensor tpos(positions.p, DType::I32, {patches, 2});
    const double bytes  = 2.0 * static_cast<double>(2 * kVisionHeads * kVisionHeadDim) * patches *
                          sizeof(__nv_bfloat16);
    const Result result = bench_loop_prepared(
        restore,
        [&](cudaStream_t stream) {
            if (control) {
                launch_control<kVisionHeadDim, kVisionHeadDim, kVisionHeads, kVisionHeads>(
                    tpos, tq, tk, kVisionTheta, stream);
            } else {
                ops::rope(tpos, kVisionHeadDim, kVisionTheta, tq, tk, execution.on_stream(stream));
            }
        },
        bytes);
    const std::string label =
        std::string("rope vision") + (control ? " control" : "") + " P=" + std::to_string(patches);
    print_result(label.c_str(), result);
}

struct Options {
    bool text            = false;
    bool vision          = false;
    bool control         = false;
    bool geometry_h24kv4 = false;
    bool geometry_h16kv2 = false;
    bool geometry_d128   = true;
    bool axes1           = true;
    bool axes3           = true;
    bool pair            = true;
    bool single_q        = false;
    bool single_k        = false;
    bool profile         = false;
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 128, 1024};
    std::vector<int> patches{8, 256, 4096, 49152, 65536};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const char* arg  = argv[index];
        const auto value = [&]() -> const char* {
            if (++index >= argc) { throw std::invalid_argument(std::string(arg) + " needs value"); }
            return argv[index];
        };
        if (!std::strcmp(arg, "--text")) {
            options.text = true;
        } else if (!std::strcmp(arg, "--vision")) {
            options.vision = true;
        } else if (!std::strcmp(arg, "--control")) {
            options.control = true;
        } else if (!std::strcmp(arg, "--geometry")) {
            const std::string selected(value());
            options.geometry_h24kv4 = selected == "h24kv4" || selected == "all";
            options.geometry_h16kv2 = selected == "h16kv2" || selected == "all";
            options.geometry_d128   = selected == "d128" || selected == "all";
            if (!options.geometry_h24kv4 && !options.geometry_h16kv2 && !options.geometry_d128) {
                throw std::invalid_argument("--geometry must be h24kv4, h16kv2, d128, or all");
            }
        } else if (!std::strcmp(arg, "--axes")) {
            const std::string selected(value());
            options.axes1 = selected == "1" || selected == "both";
            options.axes3 = selected == "3" || selected == "both";
            if (!options.axes1 && !options.axes3) {
                throw std::invalid_argument("--axes must be 1, 3, or both");
            }
        } else if (!std::strcmp(arg, "--form")) {
            const std::string selected(value());
            options.pair     = selected == "pair" || selected == "all";
            options.single_q = selected == "q" || selected == "all";
            options.single_k = selected == "k" || selected == "all";
            if (!options.pair && !options.single_q && !options.single_k) {
                throw std::invalid_argument("--form must be pair, q, k, or all");
            }
        } else if (!std::strcmp(arg, "--tokens")) {
            options.tokens = parse_csv(value());
        } else if (!std::strcmp(arg, "--patches")) {
            options.patches = parse_csv(value());
        } else if (!std::strcmp(arg, "--profile")) {
            options.profile = true;
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + arg);
        }
    }
    if (!options.text && !options.vision) { options.text = options.vision = true; }
    if (options.profile && (!options.text || options.vision || !options.geometry_d128 ||
                            options.geometry_h24kv4 || options.geometry_h16kv2 || !options.pair ||
                            options.single_q || options.single_k || options.tokens.size() != 1)) {
        throw std::invalid_argument("--profile requires one d128 Text pair token extent");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext device;
        const auto execution = device.execution_view();
        if (options.text) {
            if (options.geometry_d128 && options.axes1 && options.pair) {
                for (int tokens : options.tokens) {
                    run_d128(execution, tokens, options.control, options.profile);
                }
            }
            if (options.geometry_d128 && options.axes1 && options.single_k) {
                for (int tokens : options.tokens) {
                    run_d128_single_k(execution, tokens, options.control);
                }
            }
            for (int axes : {1, 3}) {
                if ((axes == 1 && !options.axes1) || (axes == 3 && !options.axes3)) { continue; }
                for (int tokens : options.tokens) {
                    if (options.geometry_h24kv4) {
                        if (options.pair) {
                            run_text<24, 4>(execution, tokens, axes, options.control, "h24kv4");
                        }
                        if (options.single_q) {
                            run_text_single<24>(execution, tokens, axes, options.control, "h24kv4",
                                                "q");
                        }
                        if (options.single_k) {
                            run_text_single<4>(execution, tokens, axes, options.control, "h24kv4",
                                               "k");
                        }
                    }
                    if (options.geometry_h16kv2) {
                        if (options.pair) {
                            run_text<16, 2>(execution, tokens, axes, options.control, "h16kv2");
                        }
                        if (options.single_q) {
                            run_text_single<16>(execution, tokens, axes, options.control, "h16kv2",
                                                "q");
                        }
                        if (options.single_k) {
                            run_text_single<2>(execution, tokens, axes, options.control, "h16kv2",
                                               "k");
                        }
                    }
                }
            }
        }
        if (options.vision) {
            for (int patches : options.patches) run_vision(execution, patches, options.control);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
    return 0;
}
