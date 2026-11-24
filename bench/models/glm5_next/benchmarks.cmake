# The T1 round harness (AGENTS.md "Iteration tiers"): GLM-5.3-Flash decode rounds of a chosen shape
# through the production Program over its expert placement, without the Engine or the server.
add_executable(infernix_glm5_next_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/round_bench.cpp")
infernix_internal_includes(infernix_glm5_next_round_bench)
target_link_libraries(infernix_glm5_next_round_bench PRIVATE infernix_engine infernix::json)

# Between T1 and T3: the public Engine with production's options in process, and bench_infernix.py's
# concurrent phase (make t2-bench FAMILY=glm5_next).
add_executable(infernix_glm5_next_engine_bench
  "${CMAKE_CURRENT_LIST_DIR}/engine_bench.cpp")
infernix_internal_includes(infernix_glm5_next_engine_bench)
target_link_libraries(infernix_glm5_next_engine_bench PRIVATE infernix_engine)
