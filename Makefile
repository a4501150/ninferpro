# Iteration tiers (AGENTS.md "Iteration tiers"). Each target builds only what it runs, under the
# shared build lock and memory cap, and holds the GPU lock while it runs. FAMILY (default qwen4_exp)
# selects the model family: its benches and bench/models/$(FAMILY)/tiers.mk (artifacts, golden
# shapes; docs/maintainer/round-bench.md).
#
#   make t0 [R=regex]            unit and Op tests (ctest, everything but the real-artifact entries)
#   make t1 [SHAPE=6x1,6x2]      one round per shape on the real artifact: time, speed of light and
#                                golden hashes
#   make t1-exact                the same on the exact route (--disable-fast-paths)
#   make t1-kernels [KSHAPE=6x2] [METRICS=1]  per-kernel device time of one round (Nsight Systems);
#                                METRICS=1 adds the DRAM bandwidth left unused, per kernel
#   make t1-sweep [SWEEP_CONTEXTS="1024 131072,1024"] [SWEEP_SHAPE=6x1]  round time and speed of
#                                light against context length, skewed lane contexts included
#   make t1-record               record the golden hashes after an intended numerical change
#   make t2                      start-up and correctness smoke test: the Engine with production's
#                                options on the real artifacts, golden tokens on the exact route
#   make t2-record               record the golden tokens after an intended numerical change
#   make t2-bench                aggregate speed of concurrent requests through the Engine in process
#                                (bench_infernix.py's C6 phase without the server)
#   make t2-step                 where a decode step's time goes: the t2-bench workload under Nsight
#                                Systems, split into host phases, GPU idle and GPU work by launch site
#   make t3 [PLACEMENTS=...]     promotion: bench_infernix.py against the server build (it holds
#                                both locks, so nothing else builds or boots while it measures)
#   make t3-frontier             the serving frontier: one boot, a concurrency sweep of agentic
#                                mixed-length traffic, throughput against per-user decode speed

BUILD     ?= build
# A build peaks near 1.75 GiB per nvcc job; MEM caps the whole build (an unbounded one has run the
# host out of memory). Lower both when production holds experts in host RAM.
JOBS      ?= 8
MEM       ?= 24G
ARGS      ?=
R         ?=
LOCKS     := $(HOME)/.cache/super-quant
CUDA_INC  := /usr/local/cuda-13.4/include
BUILD_CMD  = env CPATH=$(CUDA_INC) flock $(LOCKS)/build.lock systemd-run --user --scope -q -p MemoryMax=$(MEM) \
             cmake --build $(BUILD) -j$(JOBS)
GPU       := flock $(LOCKS)/gpu.lock
FAMILY    ?= qwen4_exp
MODELS    ?= $(HOME)/models/infernix
include bench/models/$(FAMILY)/tiers.mk
ROUND_TARGET  := infernix_$(FAMILY)_round_bench
SMOKE_TARGET  := infernix_$(FAMILY)_smoke_real_test
ENGINE_TARGET := infernix_$(FAMILY)_engine_bench
ROUND     := $(BUILD)/bench/$(ROUND_TARGET)
HASHES    := bench/models/$(FAMILY)/round_hashes.txt
SMOKE     := $(BUILD)/tests/$(SMOKE_TARGET)
GOLDEN    := tests/models/$(FAMILY)/smoke_golden.txt
ENGINE_BENCH := $(BUILD)/bench/$(ENGINE_TARGET)
SHAPE     ?= $(T1_FAST)
KSHAPE    ?= 6x2
METRICS   ?=
# T3: the placements bench_infernix.py walks in one boot, and where its tables go.
PLACEMENTS ?= vram=100%
T3_OUT    ?= $(HOME)/src/super-quant/results/infernix_t3_$(shell git rev-parse --short HEAD)
T3_FRONTIER_OUT ?= $(HOME)/src/super-quant/results/infernix_frontier_$(shell git rev-parse --short HEAD)

.PHONY: t0 t1 t1-exact t1-kernels t1-sweep t1-record t2 t2-record t2-bench t2-step t3 t3-frontier

t0:
	$(BUILD_CMD)
	$(GPU) ctest --test-dir $(BUILD) -LE real -j $(JOBS) --output-on-failure $(if $(R),-R '$(R)')

t1:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(SHAPE) --check $(HASHES) $(ARGS)

t1-exact:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_EXACT) --exact --check $(HASHES) $(ARGS)

t1-kernels:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) python3 tools/tiers/kernels.py --shape $(KSHAPE) $(if $(METRICS),--gpu-metrics) --bench $(ROUND) -- $(ROUND_ARGS) $(ARGS)

t1-sweep:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) python3 tools/tiers/sweep.py --bench $(ROUND) --shape $(SWEEP_SHAPE) $(foreach c,$(SWEEP_CONTEXTS),--context $(c)) \
	    -- $(ROUND_ARGS) --prefill-chunk $(SWEEP_CHUNK) $(ARGS)

t1-record:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_FAST) --replays 5 --record $(HASHES)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_EXACT) --replays 5 --exact --record $(HASHES)

t2:
	$(BUILD_CMD) --target $(SMOKE_TARGET)
	$(GPU) env $(MODEL_ENV) $(SMOKE) --check $(GOLDEN) $(ARGS)

t2-record:
	$(BUILD_CMD) --target $(SMOKE_TARGET)
	$(GPU) env $(MODEL_ENV) $(SMOKE) --record $(GOLDEN) $(ARGS)

t2-bench:
	$(BUILD_CMD) --target $(ENGINE_TARGET)
	$(GPU) env $(MODEL_ENV) $(ENGINE_BENCH) $(ARGS)

t2-step:
	$(BUILD_CMD) --target $(ENGINE_TARGET)
	$(GPU) env $(MODEL_ENV) python3 tools/tiers/step.py --kernels 3 -- $(ENGINE_BENCH) --repeat 1 --profile $(ARGS)

t3:
	$(BUILD_CMD) --target infernix-serve
	$(GPU) flock $(LOCKS)/build.lock python3 $(HOME)/src/super-quant/scripts/bench_infernix.py \
	    --binary $(BUILD)/apps/infernix-serve --placements '$(PLACEMENTS)' --out $(T3_OUT) $(ARGS)

# bench_infernix_frontier.py holds the GPU lock for its server's whole life itself (an outer $(GPU)
# would wait on itself).
t3-frontier:
	$(BUILD_CMD) --target infernix-serve
	flock $(LOCKS)/build.lock python3 $(HOME)/src/super-quant/scripts/bench_infernix_frontier.py \
	    --binary $(BUILD)/apps/infernix-serve --out $(T3_FRONTIER_OUT) $(ARGS)
