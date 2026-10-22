# The T1 round harness (AGENTS.md "Iteration tiers"): one Qwen4Exp generation round of a chosen
# shape through the production Forward and its CUDA Graph, every expert in a VRAM frame, without
# the Engine or the server.
add_executable(infernix_qwen4_exp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/round_bench.cpp")
infernix_internal_includes(infernix_qwen4_exp_round_bench)
target_compile_definitions(infernix_qwen4_exp_round_bench PRIVATE
  INFERNIX_BENCH_CORPUS="${PROJECT_SOURCE_DIR}/bench/fixtures/bench_corpus.ids")
target_link_libraries(infernix_qwen4_exp_round_bench PRIVATE
  infernix_model_runtime infernix_core)
