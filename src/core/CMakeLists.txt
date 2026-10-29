if(WIN32)
  set(INFERNIX_READ_ONLY_FILE_SOURCE read_only_file_win32.cpp)
  set(INFERNIX_CUFILE_SOURCE cufile_io_unsupported.cpp)
else()
  set(INFERNIX_READ_ONLY_FILE_SOURCE read_only_file_posix.cpp)
  # The prefix cache's Disk tier (cuFile has no Windows build).
  set(INFERNIX_CUFILE_SOURCE cufile_io.cpp)
endif()

add_library(infernix_core STATIC
  ${INFERNIX_READ_ONLY_FILE_SOURCE}
  ${INFERNIX_CUFILE_SOURCE}
  arena.cu
  block_copy.cu
  copy_batch.cpp
  evictable_weight_pool.cu
  vmm_arena.cu
  device.cu
  direct_read_queue.cpp
  dtype.cpp
  host_load.cpp
  host_memory.cpp
  host_worker_pool.cpp
  link_probe.cu
  power_throttling.cpp
  vram_budget.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  decode_graph_gate.cu
)
infernix_internal_includes(infernix_core)
infernix_cuda_archive(infernix_core)
target_link_libraries(infernix_core PUBLIC infernix_cuda_runtime CUDA::nvtx3 CUDA::cuda_driver Threads::Threads)
if(NOT WIN32)
  target_link_libraries(infernix_core PRIVATE CUDA::cuFile)
endif()
if(WIN32)
  # DXGI: the OS video-memory budget and display outputs of the CUDA adapter (vram_budget.cpp).
  target_link_libraries(infernix_core PRIVATE dxgi)
endif()
