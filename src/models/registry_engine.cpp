#include "models/qwen3_5/instance.h"
#include "models/qwen4_exp/instance.h"
#include "models/registry.h"

#include <stdexcept>

namespace infernix::models {

runtime::ConstructedBackend
construct_engine_backend(Architecture architecture, EngineOptions& options, DeviceContext& device) {
    switch (architecture) {
    case Architecture::Qwen3_5:
    case Architecture::Qwen3_5Moe:
        return runtime::construct_qwen3_5_backend(options, device);
    case Architecture::Qwen4Exp:
        return runtime::construct_qwen4_exp_backend(options, device);
    }
    throw std::invalid_argument("unknown model architecture");
}

} // namespace infernix::models
