# Glm5Next (GLM-5.3-Flash) linear shape tuning: the family owns its tuned (n, k) instantiations
# (the BF16 and Q8 projections of its recipe); each dtype's <dtype>_shapes.cpp lists the
# family's whole set in one table, which src/models/shapes.cpp registers with the ops dispatch;
# that reference pulls every shape file the table lists out of the archive. The compile target
# stays infernix_ops because the launch templates and RDC live there.
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n24576_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n288_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n320_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k12288.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k16384.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n4096_k8192.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n8192_k128.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n16384_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_shapes.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n16384_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n24576_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n4096_k12288.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n4096_k16384.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n4096_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n4096_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/n4096_k8192.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_shapes.cpp"
)
