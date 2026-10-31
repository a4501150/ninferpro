#pragma once

// The VRAM monitor of the Qwen4Exp Program (design §19.3.7): one thread that reads the device's
// video memory when the OS changes this process's budget, or once a second, keeps the latest
// snapshot, and hands each to the Program's callback, which wakes an idle engine so it returns
// idle KV pages to the expert frames and reports pressure below the headroom. It mutates no CUDA
// state and resizes nothing: the client owns the placement.

#include "core/vram_budget.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace infernix::models::offload {

class VramMonitor {
public:
    // `on_snapshot` runs on the monitor thread after every reading.
    VramMonitor(int device, VramBudgetSource& source, std::function<void(const VramSnapshot&)> on_snapshot,
                std::chrono::milliseconds period = std::chrono::milliseconds(1000));
    ~VramMonitor();
    VramMonitor(const VramMonitor&)            = delete;
    VramMonitor& operator=(const VramMonitor&) = delete;

private:
    void run();

    int device_;
    VramBudgetSource& source_;
    std::function<void(const VramSnapshot&)> on_snapshot_;
    std::chrono::milliseconds period_;
    mutable std::mutex mutex_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
    VramSnapshot latest_;
    std::thread thread_;
};

} // namespace infernix::models::offload
