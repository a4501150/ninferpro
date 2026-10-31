target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_a8_pair.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_dispatch.cpp"
)
