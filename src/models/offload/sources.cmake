# The shared expert-offload tier (glm5_3-flash-design.md and deepseek_v4_1-flash-design.md §8.2),
# extracted from Qwen4Exp: the expert placement and store, the host expert cache (LFRU policy,
# frames, controller, saved state, calibrated order, RAM-level HostTier), the VRAM residency
# controller, the SSD tier's HostExpertTier, the route trace and the VRAM monitor. Host-only
# components each Program composes; Qwen4Exp's loading and runtime link it.
add_library(infernix_offload STATIC
  "${CMAKE_CURRENT_LIST_DIR}/calibrated_order.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_placement.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_residency.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_expert_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_trace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vram_monitor.cpp")
infernix_internal_includes(infernix_offload)
target_link_libraries(infernix_offload PUBLIC infernix_artifact infernix_ops)
target_link_libraries(infernix_model_loading PUBLIC infernix_offload)
target_link_libraries(infernix_model_runtime PUBLIC infernix_offload)
