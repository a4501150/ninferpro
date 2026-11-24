# The pooled-key indexer of sparse latent attention: the LayerNorm-fused append and pool
# completion, and the ReLU-of-dot scoring with exact top-pool selection
# (include/infernix/ops/dsa_indexer.h).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dsa_indexer.cu")
