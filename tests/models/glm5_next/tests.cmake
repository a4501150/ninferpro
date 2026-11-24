infernix_add_test(infernix_glm5_next_tokenizer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer.cpp"
  LIBRARIES infernix_model_loading)

# The GLM-5.3-Flash tokenizer's exact behavior against the Hugging Face tokenizers oracle, over
# a committed corpus; the 20 MB tokenizer.json lives outside the repository and is read from
# GLM_TOKENIZER_RESOURCES.
infernix_add_test(infernix_glm5_next_tokenizer_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_model_loading infernix::json)

infernix_add_test(infernix_glm5_next_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_model_runtime)


infernix_add_test(infernix_glm5_next_tool_calls_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_calls.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_model_runtime)

# The load plan on the real artifact (INFERNIX_GLM_ARTIFACT, INFERNIX_GLM_DRAFT): every binding and
# the parsed config, without reading the weights.
infernix_add_test(infernix_glm5_next_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load_real.cpp"
  LIBRARIES infernix_model_loading)

# The forward against the FP64 reference forward (INFERNIX_GLM_ARTIFACT, INFERNIX_GLM_REFERENCE).
infernix_add_test(infernix_glm5_next_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  LIBRARIES infernix_model_runtime infernix_model_loading)

# The frontend on the real tokenizer and template (GLM_TOKENIZER_RESOURCES, like the tokenizer
# test): prompt ids against the tokenizers oracle and a reply's reasoning, content and call.
infernix_add_test(infernix_glm5_next_frontend_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_model_runtime infernix_model_loading)

# The Program on the real artifact over VRAM frames, the RAM tier and the artifact
# (INFERNIX_GLM_ARTIFACT, INFERNIX_GLM_REFERENCE): generation, a batched decode, causal scoring and
# teacher forcing.
infernix_add_test(infernix_glm5_next_program_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program_real.cpp"
  LIBRARIES infernix_model_runtime infernix_model_loading)

# The public Engine in process on the real artifact (INFERNIX_GLM_ARTIFACT, INFERNIX_GLM_REFERENCE):
# greedy generation alone and among concurrent requests, an expert-placement switch, DFlash2
# drafting (with INFERNIX_GLM_DRAFT), causal scoring.
infernix_add_test(infernix_glm5_next_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES infernix_engine)

# The prefix cache through the public Engine on the real artifact (INFERNIX_GLM_ARTIFACT): a
# shared-prefix resume and an exact repeat against cold runs, with the layer walk's in-place taps.
infernix_add_test(infernix_glm5_next_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_real.cpp"
  LIBRARIES infernix_engine)

# The T2 start-up and correctness smoke test (AGENTS.md "Iteration tiers"): the Engine with
# production's options on the real artifacts, then the default route against golden tokens.
infernix_add_test(infernix_glm5_next_smoke_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_smoke_real.cpp"
  LIBRARIES infernix_engine)

# The vision tower on the real artifact (INFERNIX_GLM_ARTIFACT) against the binary64 reference
# tower (INFERNIX_GLM_VISION_REFERENCE: tools/glm5_next/vision_reference.py --no-bf16-boundaries
# --hf-check --one-step), stage by stage and end to end, on the Hugging Face processor's pixels.
infernix_add_test(infernix_glm5_next_vision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_real.cpp"
  LIBRARIES infernix_model_runtime infernix_model_loading infernix::json)

# Media prompts through the public Engine on the real artifact (INFERNIX_GLM_ARTIFACT) with the
# examples' media: an image, an image and a video, concurrent media requests and prefix reuse.
infernix_add_test(infernix_glm5_next_vision_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_engine_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine)
