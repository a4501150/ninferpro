#pragma once

// Host and GPU readings: NVML (loaded at run time, so the monitor still runs without a driver)
// and Linux /proc.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::monitor {

struct GpuInfo {
    unsigned index = 0;
    std::string name;
};

struct GpuReading {
    unsigned index = 0;
    std::optional<double> util;             // percent
    std::optional<std::uint64_t> mem_used;  // bytes
    std::optional<std::uint64_t> mem_total; // bytes
    std::optional<double> power;            // watts
    std::optional<double> power_limit;      // watts
    std::optional<double> temp;             // degrees C
    std::optional<unsigned> pcie_gen, pcie_gen_max, pcie_width;
};

struct HostReading {
    std::optional<double> cpu; // percent busy over the last interval
    std::optional<std::uint64_t> ram_used, ram_total;
    std::optional<double> disk_read_mb; // MB/s over the last interval, whole disks only
};

struct StaticInfo {
    std::vector<GpuInfo> gpus;
    std::string cpu_name, driver, cuda;
};

class SystemSampler {
public:
    SystemSampler();
    ~SystemSampler();
    SystemSampler(const SystemSampler&)            = delete;
    SystemSampler& operator=(const SystemSampler&) = delete;

    const StaticInfo& static_info() const noexcept { return static_; }
    std::vector<GpuReading> sample_gpus();
    // Rates are measured from the previous call; the first call has none.
    HostReading sample_host();

private:
    struct Nvml;
    Nvml* nvml_ = nullptr;
    StaticInfo static_;
    std::uint64_t cpu_busy_ = 0, cpu_total_ = 0;
    std::uint64_t sectors_read_ = 0;
    double disk_time_ = 0.0;
    bool have_cpu_ = false, have_disk_ = false;
};

} // namespace infernix::monitor
