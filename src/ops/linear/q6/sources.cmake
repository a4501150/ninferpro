target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q6_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q6_a16_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q6_a16_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q6_a16_sliced_k_mma.cu"
)
