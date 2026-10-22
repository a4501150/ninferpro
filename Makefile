# Iteration tiers (AGENTS.md "Iteration tiers"). Each target builds only what it runs, under the
# shared build lock and memory cap, and holds the GPU lock while it runs.
#
#   make t0 [R=regex]            unit and Op tests (ctest, everything but the real-artifact entries)
#   make t1 [SHAPE=6x1,6x2]      one round per shape on the real artifact: time and golden hashes
#   make t1-exact                the same on the exact route (--disable-fast-paths)
#   make t1-kernels [KSHAPE=6x2] per-kernel device time of one round (Nsight Systems)
#   make t1-record               record the golden hashes after an intended numerical change
#   make t2                      start-up and correctness smoke test: the Engine with production's
#                                options on the real artifacts, golden tokens on the exact route
#   make t2-record               record the golden tokens after an intended numerical change
#   make t2-bench                aggregate speed of concurrent requests through the Engine in process
#                                (bench_infernix.py's C6 phase without the server)
#   make t3 [PLACEMENTS=...]     promotion: bench_infernix.py against the server build (it holds
#                                both locks, so nothing else builds or boots while it measures)

BUILD     ?= build
# A build peaks near 1.75 GiB per nvcc job; MEM caps the whole build (an unbounded one has run the
# host out of memory). Lower both when production holds experts in host RAM.
JOBS      ?= 12
MEM       ?= 24G
ARGS      ?=
R         ?=
LOCKS     := $(HOME)/.cache/super-quant
CUDA_INC  := /usr/local/cuda-13.4/include
BUILD_CMD  = env CPATH=$(CUDA_INC) flock $(LOCKS)/build.lock systemd-run --user --scope -q -p MemoryMax=$(MEM) \
             cmake --build $(BUILD) -j$(JOBS)
GPU       := flock $(LOCKS)/gpu.lock
ROUND     := $(BUILD)/bench/infernix_qwen4_exp_round_bench
HASHES    := bench/models/qwen4_exp/round_hashes.txt
SMOKE     := $(BUILD)/tests/infernix_qwen4_exp_smoke_real_test
GOLDEN    := tests/models/qwen4_exp/smoke_golden.txt
ENGINE_BENCH := $(BUILD)/bench/infernix_qwen4_exp_engine_bench
# Production's artifacts (~/src/super-quant/configs/Qwen3.8-Flash-Next/model.env).
MODELS    ?= $(HOME)/models/infernix
ARTIFACT  ?= $(MODELS)/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.infernix
NGRAM     ?= $(MODELS)/Qwen3.8-Flash-Next-BF16.ngram
DRAFTER   ?= $(MODELS)/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
MODEL_ENV  = INFERNIX_QWEN4_ARTIFACT=$(ARTIFACT) INFERNIX_QWEN4_NGRAM=$(NGRAM) INFERNIX_QWEN4_DFLASH2=$(DRAFTER)
# The golden configurations: the fast route at decode and verification shapes, the exact route at two.
T1_FAST   := 1x1,6x1,1x4,6x2,6x3
T1_EXACT  := 6x1,6x2
SHAPE     ?= $(T1_FAST)
KSHAPE    ?= 6x2
# T3: the placements bench_infernix.py walks in one boot, and where its tables go.
PLACEMENTS ?= vram=100%
T3_OUT    ?= $(HOME)/src/super-quant/results/infernix_t3_$(shell git rev-parse --short HEAD)

.PHONY: t0 t1 t1-exact t1-kernels t1-record t2 t2-record t2-bench t3

t0:
	$(BUILD_CMD)
	$(GPU) ctest --test-dir $(BUILD) -LE real -j $(JOBS) --output-on-failure $(if $(R),-R '$(R)')

t1:
	$(BUILD_CMD) --target infernix_qwen4_exp_round_bench
	$(GPU) $(ROUND) --artifact $(ARTIFACT) --ngram-volume $(NGRAM) --shape $(SHAPE) --check $(HASHES) $(ARGS)

t1-exact:
	$(BUILD_CMD) --target infernix_qwen4_exp_round_bench
	$(GPU) $(ROUND) --artifact $(ARTIFACT) --ngram-volume $(NGRAM) --shape $(T1_EXACT) --exact --check $(HASHES) $(ARGS)

t1-kernels:
	$(BUILD_CMD) --target infernix_qwen4_exp_round_bench
	$(GPU) python3 tools/flash_next/t1_kernels.py --shape $(KSHAPE) --bench $(ROUND) -- --artifact $(ARTIFACT) --ngram-volume $(NGRAM) $(ARGS)

t1-record:
	$(BUILD_CMD) --target infernix_qwen4_exp_round_bench
	$(GPU) $(ROUND) --artifact $(ARTIFACT) --ngram-volume $(NGRAM) --shape $(T1_FAST) --replays 5 --record $(HASHES)
	$(GPU) $(ROUND) --artifact $(ARTIFACT) --ngram-volume $(NGRAM) --shape $(T1_EXACT) --replays 5 --exact --record $(HASHES)

t2:
	$(BUILD_CMD) --target infernix_qwen4_exp_smoke_real_test
	$(GPU) env $(MODEL_ENV) $(SMOKE) --check $(GOLDEN) $(ARGS)

t2-record:
	$(BUILD_CMD) --target infernix_qwen4_exp_smoke_real_test
	$(GPU) env $(MODEL_ENV) $(SMOKE) --record $(GOLDEN) $(ARGS)

t2-bench:
	$(BUILD_CMD) --target infernix_qwen4_exp_engine_bench
	$(GPU) env $(MODEL_ENV) $(ENGINE_BENCH) $(ARGS)

t3:
	$(BUILD_CMD) --target infernix-serve
	$(GPU) flock $(LOCKS)/build.lock python3 $(HOME)/src/super-quant/scripts/bench_infernix.py \
	    --binary $(BUILD)/apps/infernix-serve --placements '$(PLACEMENTS)' --out $(T3_OUT) $(ARGS)
