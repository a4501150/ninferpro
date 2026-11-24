# Model-independent frontend pieces every family's frontend uses unchanged: the tool-declaration
# contract, the reasoning-loop guard and SHA-256.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/reasoning_loop.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
)
