# Iteration tiers (AGENTS.md "Iteration tiers"). Each target builds only what it runs, under the
# shared build lock and memory cap, and holds the GPU lock while it runs. FAMILY (default qwen4_exp)
# selects the model family: its benches and bench/models/$(FAMILY)/tiers.mk (artifacts, golden
# shapes; docs/maintainer/round-bench.md).
#
#   make t0 [R=regex]            unit and Op tests (ctest, everything but the real-artifact entries)
#   make t1 [SHAPE=6x1,6x2]      one round per shape on the real artifact: time, speed of light and
#                                golden hashes
#   make t1 SWEEP=1 [CONTEXTS="1024 131072,1024"]  the same against context length, skewed lane
#                                contexts included (no hash check)
#   make t1 PROFILE=1 [SHAPE=6x2] [METRICS=1]  per-kernel device time of one round (Nsight
#                                Systems); METRICS=1 adds the DRAM bandwidth left unused, per kernel
#   make t1-exact                the golden shapes on the exact route (--disable-fast-paths)
#   make t1-record               record the golden hashes after an intended numerical change
#   make t2                      start-up and correctness smoke test: the Engine with production's
#                                options on the real artifacts, golden tokens on the exact route
#   make t2-record               record the golden tokens after an intended numerical change
#   make t2-bench                aggregate speed of concurrent requests through the Engine in process
#                                (bench_infernix.py's C6 phase without the server)
#   make t2-bench PROFILE=1      where a decode step's time goes: one set under Nsight Systems, split
#                                into host phases, GPU idle and GPU work by launch site
#   make t3 [PLACEMENTS=...]     promotion: bench_infernix.py against the server build (it holds
#                                both locks, so nothing else builds or boots while it measures)
#   make t3-frontier             the serving frontier: one boot, a concurrency sweep of agentic
#                                mixed-length traffic, throughput against per-user decode speed

BUILD     ?= build
# The machine's 16 cores are the job count (SMT threads add little to compiles). MEM scales with
# it: a compile job peaks near 1.75 GiB and a debug-info link several GiB more (an unbounded nvcc
# build has run the host out of memory). While production serves, `JOBS=8` leaves half the cores
# to its host threads.
JOBS      ?= 16
MEM       ?= $(shell expr $(JOBS) \* 2 + 16)G
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
SWEEP     ?=
PROFILE   ?=
METRICS   ?=
CONTEXTS  ?= $(SWEEP_CONTEXTS)
# The shapes of a t1 run: SHAPE when given, else the mode's own (one shape per profile).
T1_SHAPE  := $(if $(filter-out undefined,$(origin SHAPE)),$(SHAPE),$(if $(PROFILE),6x2,$(if $(SWEEP),$(SWEEP_SHAPE),$(T1_FAST))))
# T3: the placements bench_infernix.py walks in one boot, and where its tables go.
PLACEMENTS ?= vram=100%
T3_OUT    ?= $(HOME)/src/super-quant/results/infernix_t3_$(shell git rev-parse --short HEAD)
T3_FRONTIER_OUT ?= $(HOME)/src/super-quant/results/infernix_frontier_$(shell git rev-parse --short HEAD)

.PHONY: t0 t1 t1-exact t1-record t2 t2-record t2-bench t3 t3-frontier

t0:
	$(BUILD_CMD)
	$(GPU) ctest --test-dir $(BUILD) -LE real -j $(JOBS) --output-on-failure $(if $(R),-R '$(R)')

t1:
	$(BUILD_CMD) --target $(ROUND_TARGET)
ifneq ($(PROFILE),)
	$(GPU) python3 tools/tiers/kernels.py --shape $(T1_SHAPE) $(if $(METRICS),--gpu-metrics) --bench $(ROUND) \
	    -- $(ROUND_ARGS) $(ARGS)
else ifneq ($(SWEEP),)
	$(GPU) python3 tools/tiers/sweep.py --bench $(ROUND) --shape $(T1_SHAPE) $(foreach c,$(CONTEXTS),--context $(c)) \
	    -- $(ROUND_ARGS) --prefill-chunk $(SWEEP_CHUNK) $(ARGS)
else
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_SHAPE) --check $(HASHES) $(ARGS)
endif

# A family without an exact route leaves T1_EXACT empty.
t1-exact:
	$(if $(T1_EXACT),,$(error FAMILY=$(FAMILY) has no exact route))
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_EXACT) --exact --check $(HASHES) $(ARGS)

t1-record:
	$(BUILD_CMD) --target $(ROUND_TARGET)
	$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_FAST) --replays 5 --record $(HASHES)
	$(if $(T1_EXACT),$(GPU) $(ROUND) $(ROUND_ARGS) --shape $(T1_EXACT) --replays 5 --exact --record $(HASHES))

t2:
	$(BUILD_CMD) --target $(SMOKE_TARGET)
	$(GPU) env $(MODEL_ENV) $(SMOKE) --check $(GOLDEN) $(ARGS)

t2-record:
	$(BUILD_CMD) --target $(SMOKE_TARGET)
	$(GPU) env $(MODEL_ENV) $(SMOKE) --record $(GOLDEN) $(ARGS)

t2-bench:
	$(BUILD_CMD) --target $(ENGINE_TARGET)
ifneq ($(PROFILE),)
	$(GPU) env $(MODEL_ENV) python3 tools/tiers/step.py --kernels 3 -- $(ENGINE_BENCH) --repeat 1 --profile $(ARGS)
else
	$(GPU) env $(MODEL_ENV) $(ENGINE_BENCH) $(ARGS)
endif

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
