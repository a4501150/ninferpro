# The shared offload tier's host-only tests. The residency, tier and placement tests on the real
# artifacts stay with the family that exercises them (tests/models/qwen4_exp).

# The startup RAM ledger and VRAM sizing (models/offload/memory_plan.h), host only. It also
# checks the families' graph_bound (the family-specific part that stayed behind), so it links
# the model loading target that defines it.
infernix_add_test(infernix_offload_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES infernix_offload infernix_model_loading)

# The startup expert-placement resolution (models/offload/expert_placement_startup.h) on the
# family hooks, host only.
infernix_add_test(infernix_offload_expert_placement_startup_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_placement_startup.cpp"
  LIBRARIES infernix_offload)
