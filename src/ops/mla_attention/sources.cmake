# Sparse multi-head latent attention over a paged shared latent cache: the latent append, the two
# absorbed projections and the split attention with fused output projection
# (include/infernix/ops/mla_sparse_attention.h).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/mla_attention.cu")
