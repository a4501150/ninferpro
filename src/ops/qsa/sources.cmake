# QSA pooled-key indexer, block selection and block-sparse attention
# (docs/maintainer/qwen3_8-flash-next-design.md §14.1, §3.1), and the fused prepare.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/qsa.cu" "${CMAKE_CURRENT_LIST_DIR}/qsa_select.cu"
                   "${CMAKE_CURRENT_LIST_DIR}/qsa_prompt.cu" "${CMAKE_CURRENT_LIST_DIR}/qsa_prepare.cu")
