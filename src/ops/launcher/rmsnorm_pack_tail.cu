#include "ops/launcher/rmsnorm_pack_tail.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/kernel/rmsnorm_pack_tail.cuh"

namespace infernix::ops::detail {
namespace {

// Five BF16 pairs per thread at 5120 and 2560; four at 4096.
template <int Rows, int Threads>
void launch(const Tensor& input, const Tensor& weight, Tensor& output, float epsilon,
            cudaStream_t stream) {
    const dim3 grid(input.ne[1] - 1, input.ne[2]);
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), dim3(Threads), 0, stream},
                                    rmsnorm_pack_tail_kernel<Rows, Threads>,
                                    static_cast<const __nv_bfloat162*>(input.data),
                                    static_cast<const __nv_bfloat162*>(weight.data),
                                    static_cast<__nv_bfloat162*>(output.data), input.ne[1],
                                    epsilon));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, float epsilon,
                              Tensor& output, cudaStream_t stream) {
    if (input.ne[0] == 5120) {
        launch<5120, 512>(input, weight, output, epsilon, stream);
    } else if (input.ne[0] == 4096) {
        launch<4096, 512>(input, weight, output, epsilon, stream);
    } else {
        launch<2560, 256>(input, weight, output, epsilon, stream);
    }
}

} // namespace infernix::ops::detail
