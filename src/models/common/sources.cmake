# Model-independent pieces every family uses unchanged: the tool-declaration contract, the
# reasoning-loop guard and SHA-256 (frontend), and the vision tower's offload (the pinned overlay
# layout joins infernix_model_loading, the weight stream infernix_model_runtime).
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vision/overlay_layout.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/reasoning_loop.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision/weight_stream.cpp"
)
