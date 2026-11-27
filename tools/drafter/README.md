# GLM-5.3-Flash DFlash2 drafter fine-tuning

The incoai drafter (`incoai/GLM-5.3-Flash-DFlash2`, Apache-2.0) was trained against the official
checkpoint; production serves an uncensored edit. These steps fine-tune it on the served target's
own answers and on features computed by this engine, so training sees the serving numerics
(glm5_3-flash-design.md §10.4).

1. Prompts (`build_prompts.py`, in `.venv-convert`): UltraChat first turns (thinking on for 60 %),
   CodeAlpaca tasks (half as agentic tasks declaring `run_bash`; generation stops at the first
   assistant turn, so nothing is executed), and whole UltraChat conversations as off-policy text.

   ```sh
   hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --include "data/train_sft-*" --local-dir ultrachat
   hf download sahil2801/CodeAlpaca-20k --repo-type dataset --local-dir codealpaca
   .venv-convert/bin/python tools/drafter/build_prompts.py --ultrachat ultrachat/data \
     --codealpaca codealpaca/code_alpaca_20k.json --prompts prompts.jsonl --offpolicy offpolicy.jsonl
   ```

2. Answers (`infernix_glm5_next_drafter_generate`): the target samples at temperature 1.0, top-p
   0.95 (the checkpoint's defaults), plain decoding, up to eight requests at a time (default
   `--max-context 16384`, `--experts vram=50%,ram=50%`; use `vram=65%,ram=35%` for the GSQ-Q4
   artifact); the output ids are the target's own tokens.

   ```sh
   INFERNIX_GLM_ARTIFACT=~/models/infernix/GLM-5.3-Flash-Uncensored-NVFP4-Infernix.infernix \
     build/bench/infernix_glm5_next_drafter_generate --in prompts.jsonl --out gen.jsonl \
     --concurrency 8 --max-tokens 2048 --experts vram=50%,ram=50% --minutes 150
   ```

   Note: a single-GPU overnight run (~300k tokens) overfit after step 400 and did not raise
   accepted tokens per round (`glm5_3-flash-design.md` §10.4). Meaningful gains need tens of
   millions of on-policy tokens (e.g. generated on a multi-GPU node from
   `orcarouter/GLM-5.3-Flash-Uncensored-FP8` or the target artifact before feature capture).

3. Features (`infernix_glm5_next_drafter_capture`): each record prefilled once through the
   causal-scoring Program; per position the stream mean after layers 5, 14, 24, 33 and 42 and the
   final stream mean, BF16 (48 KiB a token).

   ```sh
   build/bench/infernix_glm5_next_drafter_capture --in gen.jsonl --out-dir cap-gen
   build/bench/infernix_glm5_next_drafter_capture --in offpolicy.jsonl --out-dir cap-off
   ```

4. Training (`train.py`): speculators' `DFlash2DraftModel` (rev `824125a`) warm-started from the
   incoai body, the target's embedding, final norm and head frozen. It needs a venv layered on
   `.venv-convert`'s torch with speculators installed `--no-deps` plus its light dependencies
   (`datasets loguru pydantic-settings rich typer click psutil tqdm aiohttp httpx openai protobuf`)
   and a stub `hs_connectors` package (speculators imports it for live capture, unused here).

   ```sh
   python tools/drafter/train.py --capture cap-gen:3 --capture cap-off:1 \
     --init ~/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/<rev> \
     --target ~/.cache/huggingface/hub/models--orcarouter--GLM-5.3-Flash-Uncensored-NVFP4/snapshots/<rev> \
     --out drafter-ft --minutes 90
   ```

5. Conversion: the output directory has the incoai layout (`config.json`, `model.safetensors`), so
   the drafter recipe converts it as it converts the original (`--source dflash2=drafter-ft`,
   `--components dflash2 --recipe glm5_3_flash_dflash2`; `~/models/infernix/README.local.md`).
