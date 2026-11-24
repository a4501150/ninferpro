# The reasoning-loop guard's repeated-passage measure (CPU only).
infernix_add_test(infernix_models_reasoning_loop_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reasoning_loop.cpp"
  LIBRARIES infernix_model_runtime)
