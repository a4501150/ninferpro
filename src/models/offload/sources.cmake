# The shared expert-offload tier (docs/maintainer/moe-offload-design.md), extracted from
# Qwen4Exp: the expert placement and store, the host expert cache (LFRU policy,
# frames, controller, saved state, RAM-level HostTier), the VRAM residency
# controller, the SSD tier's HostExpertTier, the route trace and the VRAM monitor. Host-only
# components each Program composes; the offloaded families' loading and runtime link it. The
# model-independent orchestration a family composes is the expert placement runtime
# (expert_placement_runtime), the startup resolution (expert_placement_startup) and the host
# RAM ledger and VRAM sizing (memory_plan); the families' hooks carry what stays with them.
add_library(infernix_offload STATIC
  "${CMAKE_CURRENT_LIST_DIR}/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_placement.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_placement_runtime.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_placement_startup.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_residency.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_expert_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/memory_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_trace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vram_monitor.cpp")
infernix_internal_includes(infernix_offload)
target_link_libraries(infernix_offload PUBLIC infernix_artifact infernix_ops infernix_runtime_support)
target_link_libraries(infernix_model_loading PUBLIC infernix_offload)
target_link_libraries(infernix_model_runtime PUBLIC infernix_offload)
