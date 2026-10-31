# The shared tier: the frontend, the n-gram archive, and the Vision pieces Qwen3.5 and Qwen4Exp
# share (the tower's config, parameters, pass, weight stream, control and overlay layout). The
# same compile owners as before: resource parsing, the shared tokenizer and the shared overlay
# layout join infernix_model_loading; the frontend, the archive and the Vision execution join
# infernix_model_runtime.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vision_config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/resources.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tokenizer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/vision_overlay.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ngram.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/reasoning_loop.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/processor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/media_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_grammar.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_control.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/block_keys.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision_tower.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision_weight_stream.cpp"
)
