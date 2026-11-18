#include "system_sampler.h"

#include <dlfcn.h>
#include <nvml.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace infernix::monitor {

struct SystemSampler::Nvml {
    void* library = nullptr;
    std::vector<nvmlDevice_t> devices;

    nvmlReturn_t (*shutdown)()                                                 = nullptr;
    nvmlReturn_t (*utilization)(nvmlDevice_t, nvmlUtilization_t*)              = nullptr;
    nvmlReturn_t (*memory)(nvmlDevice_t, nvmlMemory_v2_t*)                     = nullptr;
    nvmlReturn_t (*power)(nvmlDevice_t, unsigned*)                             = nullptr;
    nvmlReturn_t (*power_limit)(nvmlDevice_t, unsigned*)                       = nullptr;
    nvmlReturn_t (*temperature)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned*) = nullptr;
    nvmlReturn_t (*pcie_gen)(nvmlDevice_t, unsigned*)                          = nullptr;
    nvmlReturn_t (*pcie_gen_max)(nvmlDevice_t, unsigned*)                      = nullptr;
    nvmlReturn_t (*pcie_width)(nvmlDevice_t, unsigned*)                        = nullptr;

    template <typename Fn>
    void bind(Fn& fn, const char* name) {
        fn = reinterpret_cast<Fn>(dlsym(library, name));
    }
};

namespace {

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string cpu_model_name() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) { return line.substr(line.find_first_not_of(' ', colon + 1)); }
        }
    }
    return {};
}

std::string cuda_version(int version) {
    return std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
}

template <typename T>
std::optional<T> nvml_value(nvmlReturn_t (*fn)(nvmlDevice_t, T*), nvmlDevice_t device) {
    T value{};
    if (fn == nullptr || fn(device, &value) != NVML_SUCCESS) { return std::nullopt; }
    return value;
}

} // namespace

SystemSampler::SystemSampler() {
    static_.cpu_name = cpu_model_name();
    void* library    = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (library == nullptr) { return; }
    auto* nvml    = new Nvml;
    nvml->library = library;
    nvmlReturn_t (*init)()                                    = nullptr;
    nvmlReturn_t (*count)(unsigned*)                          = nullptr;
    nvmlReturn_t (*handle)(unsigned, nvmlDevice_t*)           = nullptr;
    nvmlReturn_t (*name)(nvmlDevice_t, char*, unsigned)       = nullptr;
    nvmlReturn_t (*driver)(char*, unsigned)                   = nullptr;
    nvmlReturn_t (*cuda)(int*)                                = nullptr;
    nvml->bind(init, "nvmlInit_v2");
    nvml->bind(count, "nvmlDeviceGetCount_v2");
    nvml->bind(handle, "nvmlDeviceGetHandleByIndex_v2");
    nvml->bind(name, "nvmlDeviceGetName");
    nvml->bind(driver, "nvmlSystemGetDriverVersion");
    nvml->bind(cuda, "nvmlSystemGetCudaDriverVersion_v2");
    nvml->bind(nvml->shutdown, "nvmlShutdown");
    nvml->bind(nvml->utilization, "nvmlDeviceGetUtilizationRates");
    nvml->bind(nvml->memory, "nvmlDeviceGetMemoryInfo_v2");
    nvml->bind(nvml->power, "nvmlDeviceGetPowerUsage");
    nvml->bind(nvml->power_limit, "nvmlDeviceGetEnforcedPowerLimit");
    nvml->bind(nvml->temperature, "nvmlDeviceGetTemperature");
    nvml->bind(nvml->pcie_gen, "nvmlDeviceGetCurrPcieLinkGeneration");
    nvml->bind(nvml->pcie_gen_max, "nvmlDeviceGetMaxPcieLinkGeneration");
    nvml->bind(nvml->pcie_width, "nvmlDeviceGetCurrPcieLinkWidth");
    unsigned devices = 0;
    if (init == nullptr || init() != NVML_SUCCESS || count == nullptr || count(&devices) != NVML_SUCCESS ||
        handle == nullptr) {
        dlclose(library);
        delete nvml;
        return;
    }
    nvml_ = nvml;
    for (unsigned i = 0; i < devices; ++i) {
        nvmlDevice_t device = nullptr;
        if (handle(i, &device) != NVML_SUCCESS) { continue; }
        nvml->devices.push_back(device);
        char buffer[NVML_DEVICE_NAME_V2_BUFFER_SIZE] = {};
        if (name != nullptr) { (void)name(device, buffer, sizeof(buffer)); }
        static_.gpus.push_back(GpuInfo{i, buffer});
    }
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    if (driver != nullptr && driver(version, sizeof(version)) == NVML_SUCCESS) { static_.driver = version; }
    int cuda_driver = 0;
    if (cuda != nullptr && cuda(&cuda_driver) == NVML_SUCCESS) { static_.cuda = cuda_version(cuda_driver); }
}

SystemSampler::~SystemSampler() {
    if (nvml_ == nullptr) { return; }
    if (nvml_->shutdown != nullptr) { (void)nvml_->shutdown(); }
    dlclose(nvml_->library);
    delete nvml_;
}

std::vector<GpuReading> SystemSampler::sample_gpus() {
    std::vector<GpuReading> readings;
    if (nvml_ == nullptr) { return readings; }
    for (std::size_t i = 0; i < nvml_->devices.size(); ++i) {
        nvmlDevice_t device = nvml_->devices[i];
        GpuReading r;
        r.index = static_.gpus[i].index;
        if (const auto u = nvml_value(nvml_->utilization, device)) { r.util = u->gpu; }
        // v2 counts the driver's reservation apart from used memory, as nvidia-smi does.
        nvmlMemory_v2_t memory{};
        memory.version = nvmlMemory_v2;
        if (nvml_->memory != nullptr && nvml_->memory(device, &memory) == NVML_SUCCESS) {
            r.mem_used  = memory.used;
            r.mem_total = memory.total;
        }
        if (const auto mw = nvml_value(nvml_->power, device)) { r.power = *mw / 1000.0; }
        if (const auto mw = nvml_value(nvml_->power_limit, device)) { r.power_limit = *mw / 1000.0; }
        unsigned temp = 0;
        if (nvml_->temperature != nullptr &&
            nvml_->temperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            r.temp = temp;
        }
        r.pcie_gen     = nvml_value(nvml_->pcie_gen, device);
        r.pcie_gen_max = nvml_value(nvml_->pcie_gen_max, device);
        r.pcie_width   = nvml_value(nvml_->pcie_width, device);
        readings.push_back(r);
    }
    return readings;
}

HostReading SystemSampler::sample_host() {
    HostReading host;
    {
        std::ifstream in("/proc/stat");
        std::string cpu;
        std::uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
        if (in >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal && cpu == "cpu") {
            const std::uint64_t idle_all = idle + iowait;
            const std::uint64_t total    = user + nice + system + idle_all + irq + softirq + steal;
            const std::uint64_t busy     = total - idle_all;
            if (have_cpu_ && total > cpu_total_) {
                host.cpu = 100.0 * static_cast<double>(busy - cpu_busy_) / static_cast<double>(total - cpu_total_);
            }
            cpu_busy_  = busy;
            cpu_total_ = total;
            have_cpu_  = true;
        }
    }
    {
        std::ifstream in("/proc/meminfo");
        std::string key, unit;
        std::uint64_t value = 0, total = 0, available = 0;
        while (in >> key >> value >> unit) {
            if (key == "MemTotal:") { total = value * 1024; }
            if (key == "MemAvailable:") { available = value * 1024; }
        }
        if (total != 0) {
            host.ram_total = total;
            host.ram_used  = total - available;
        }
    }
    {
        std::ifstream in("/proc/diskstats");
        std::string line;
        std::uint64_t sectors = 0;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            unsigned major = 0, minor = 0;
            std::string device;
            std::uint64_t reads = 0, merged = 0, read_sectors = 0;
            if (!(fields >> major >> minor >> device >> reads >> merged >> read_sectors)) { continue; }
            // Whole hardware disks only: partitions and device-mapper/loop devices would count
            // the same reads twice.
            std::error_code error;
            if (!std::filesystem::exists("/sys/block/" + device + "/device", error)) { continue; }
            sectors += read_sectors;
        }
        const double now = now_seconds();
        if (have_disk_ && now > disk_time_ && sectors >= sectors_read_) {
            host.disk_read_mb = static_cast<double>(sectors - sectors_read_) * 512.0 / 1e6 / (now - disk_time_);
        }
        sectors_read_ = sectors;
        disk_time_    = now;
        have_disk_    = true;
    }
    return host;
}

} // namespace infernix::monitor
