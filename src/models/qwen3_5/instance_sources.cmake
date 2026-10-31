# The Qwen3.5 family instance and its option rules (the family's branches of the shared
# runtime/engine/engine_options.cpp). The option rules are artifact-free and join the loading
# target; the instance (which constructs the Program) joins the runtime target.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/options.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/instance.cpp"
)
