# The iteration tiers' settings for Qwen3.8-Flash-Next (included by the Makefile for FAMILY=qwen4_exp;
# docs/maintainer/round-bench.md). Production's artifacts (configs/Qwen3.8-Flash-Next.toml).
ARTIFACT   ?= $(MODELS)/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.infernix
NGRAM      ?= $(MODELS)/Qwen3.8-Flash-Next-BF16.ngram
DRAFTER    ?= $(MODELS)/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
T3_CONFIG  ?= configs/Qwen3.8-Flash-Next.toml
MODEL_ENV   = INFERNIX_QWEN4_ARTIFACT=$(ARTIFACT) INFERNIX_QWEN4_NGRAM=$(NGRAM) INFERNIX_QWEN4_DFLASH2=$(DRAFTER)
ROUND_ARGS  = --artifact $(ARTIFACT) --ngram-volume $(NGRAM)
# The golden configurations: the fast route at decode and verification shapes, the exact route at two,
# and on both a mixed call (a verification round beside a prefill chunk; on the exact route it must
# equal the two calls it replaces).
T1_FAST    := 1x1,6x1,1x4,6x2,6x3,6x2+1024
T1_EXACT   := 6x1,6x2,6x2+1024
# The context sweep: uniform lengths, then one long lane beside short ones and six long lanes (the
# lanes a round does not use stay short, so every lane's KV fits beside the experts).
SWEEP_SHAPE    ?= 1x1,6x1,6x3
SWEEP_CONTEXTS ?= 1024 32768 131072,1024 131072x6,1024
SWEEP_CHUNK    ?= 8192
