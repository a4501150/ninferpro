# The iteration tiers' settings for GLM-5.3-Flash (included by the Makefile for FAMILY=glm5_next;
# docs/maintainer/round-bench.md). Production's artifacts (~/src/super-quant/configs/GLM-5.3-Flash/model.env).
ARTIFACT   ?= $(MODELS)/GLM-5.3-Flash-Uncensored-NVFP4-Infernix.infernix
DRAFTER    ?= $(MODELS)/GLM-5.3-Flash-DFlash2-Infernix.infernix
MODEL_ENV   = INFERNIX_GLM_ARTIFACT=$(ARTIFACT) INFERNIX_GLM_DRAFT=$(DRAFTER)
ROUND_ARGS  = --artifact $(ARTIFACT) --draft $(DRAFTER)
# The golden shapes: plain rounds of one, two and four rows, and a one-row round drafting three
# tokens (production's DFlash2 length). T1_EXACT is empty: the NVFP4 artifact has one route, and the
# GGUF artifact's only fast path (ggml-bf16-prefill) touches prefill alone; the round bench reports no
# device speed of light (the experts are tiered), so the SWEEP mode does not run.
T1_FAST    := 1x1,2x1,4x1,1x4
T1_EXACT   :=
SWEEP_SHAPE    ?= 1x1,4x1
SWEEP_CONTEXTS ?= 1024
SWEEP_CHUNK    ?= 4096
