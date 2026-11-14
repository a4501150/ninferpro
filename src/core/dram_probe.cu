#include "core/dram_probe.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix {
namespace {

constexpr int kThreads = 512;
constexpr int kWarmups = 3;

// 16-byte loads in a grid-stride loop; the XOR of the words is stored only on an impossible match,
// so the loads cannot be elided and nothing is written.
__global__ void read_kernel(const uint4* data, std::size_t words, unsigned* sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < words;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(data + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9U && sink != nullptr) { *sink = acc; }
}

// Distinct words, so no memory compression or zero detection can shorten the reads.
__global__ void fill_kernel(uint4* data, std::size_t words) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < words;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto w = static_cast<unsigned>(i) * 2654435761U;
        data[i]      = make_uint4(w, w ^ 0x85ebca6bU, w ^ 0xc2b2ae35U, w ^ 0x27d4eb2fU);
    }
}

} // namespace

double measure_device_read_bytes_per_second(std::size_t bytes, int repetitions) {
    if (bytes < sizeof(uint4) || repetitions <= 0) {
        throw std::invalid_argument("device read probe needs bytes and repetitions");
    }
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const std::size_t words = bytes / sizeof(uint4);
    const int blocks        = sms * 4;
    cudaStream_t stream     = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    uint4* data = nullptr;
    double best = 0.0;
    const auto release = [&] {
        if (data != nullptr) { cudaFree(data); }
        if (start != nullptr) { cudaEventDestroy(start); }
        if (stop != nullptr) { cudaEventDestroy(stop); }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
    };
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMalloc(&data, words * sizeof(uint4)));
        fill_kernel<<<blocks, kThreads, 0, stream>>>(data, words);
        CUDA_CHECK(cudaGetLastError());
        for (int r = 0; r < kWarmups + repetitions; ++r) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            read_kernel<<<blocks, kThreads, 0, stream>>>(data, words, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            if (r < kWarmups) { continue; }
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms > 0.0F) {
                best = std::max(best, static_cast<double>(words * sizeof(uint4)) / (static_cast<double>(ms) * 1e-3));
            }
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    return best;
}

} // namespace infernix
