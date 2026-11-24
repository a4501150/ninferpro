# Glm5Next (GLM-5.3-Flash): config, binding, loading, the option rules, resource parsing and the
# tokenizer join the loading tier; the native parameters, the Forward, the Program, the instance, chat template,
# output session, tool-call parsing and grammar join the runtime tier.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/options.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tokenizer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/resources.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/instance.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_stream.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/block_keys.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/construct.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/binding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/constraint.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/score.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/experts.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/encoding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_grammar.cpp"
)

