#pragma once

// Windows power throttling (EcoQoS) runs a process it considers to be in the background at reduced
// clock speed and steers its threads to efficiency cores: a server whose console is not the
// foreground window, or a job launched without a visible window. An inference engine's host threads
// (the CPU expert workers, the thread that submits CUDA graphs and polls the device) then slow the
// GPU down with them. The exemption is the whole gain; its measured before/after is in
// qwen3_8-flash-next-design.md §8.1. Outputs are unchanged: only where and how fast threads run.

namespace infernix {

// Opts the calling process out of execution-speed power throttling, once per process. A no-op on
// platforms without it (Linux) and when Windows refuses the request.
void exempt_process_from_power_throttling() noexcept;

} // namespace infernix
