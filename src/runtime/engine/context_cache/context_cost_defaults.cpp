#include "runtime/engine/context_cache/context_cost.h"

#include <array>
#include <vector>

namespace infernix::runtime {

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

// Accepted machine transfer defaults live only in this table and are compiled into the binary;
// each model family compiles its own prefill defaults. Runtime JSON presets are independent
// local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_transfer_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured transfer corpus (the corpus date is in the hybrid prefix
            // cache spec §6.2). A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
        },
    };
    return defaults;
}

} // namespace infernix::runtime
