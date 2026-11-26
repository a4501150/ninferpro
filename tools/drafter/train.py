#!/usr/bin/env python3
"""Fine-tune the GLM-5.3-Flash DFlash2 drafter on features captured from the served target (README.md).

The model and objective are vllm-project/speculators' ``DFlash2DraftModel.forward`` (the reconstruction
of DFlash2 training the incoai drafter's layout matches): random anchors per row, an 8-slot block per
anchor (the clean anchor, then mask tokens), slot k taught the target's distribution for token a + k
from its final hidden at a + k - 1, a fixed exponential slot decay, and the candidate selector's CE.
The body starts from the incoai drafter; the embedding, final norm and LM head are the target's,
frozen. Features come from infernix_glm5_next_drafter_capture, so they carry the engine's serving
numerics.

Rows pack whole records (positions restart per record) up to --row-tokens. A longer record is cropped
to a random window whose first 2048 positions only give context (the drafter's window), so every
supervised anchor sees the context it would see when serving.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FEATURE_ROWS = 5 * 4096
HIDDEN = 4096
WINDOW = 2048


class Capture:
    """One capture directory: memory-mapped features, final hiddens, token ids and the record index."""

    def __init__(self, root, weight):
        root = Path(root)
        self.records = [json.loads(line) for line in open(root / "index.jsonl") if line.strip()]
        tokens = max(r["offset"] + r["n"] for r in self.records)
        self.features = np.memmap(root / "features.bin", dtype=np.uint16, mode="r", shape=(tokens, FEATURE_ROWS))
        self.final = np.memmap(root / "final.bin", dtype=np.uint16, mode="r", shape=(tokens, HIDDEN))
        self.ids = np.memmap(root / "ids.bin", dtype=np.int32, mode="r", shape=(tokens,))
        self.weight = weight
        self.name = root.name


def split(captures, holdout, seed):
    """Per capture: the last `holdout` share of records (in a fixed shuffle) for evaluation."""
    train, held = [], []
    for c in captures:
        order = list(range(len(c.records)))
        random.Random(seed).shuffle(order)
        k = int(len(order) * holdout) if c.weight > 0 else 0
        held += [(c, i) for i in order[:k]]
        train.append((c, order[k:]))
    return train, held


def piece(capture, index, budget, rng):
    """A record's (start, end, loss_from) window of at most `budget` tokens, absolute positions."""
    r = capture.records[index]
    n, loss_start = r["n"], r["loss_start"]
    if n <= budget:
        return 0, n, loss_start
    # A crop: the loss part sits past the window's first WINDOW positions (unless it starts at 0).
    lo = max(0, loss_start - WINDOW)
    start = rng.randint(lo, n - budget)
    return start, start + budget, max(loss_start, start + WINDOW if start > 0 else loss_start)


def make_row(entries, row_tokens, device):
    """Packs (capture, index, start, end, loss_from) entries into the forward's tensors."""
    hidden, final, ids, loss, docs, pos = [], [], [], [], [], []
    for d, (c, i, start, end, loss_from) in enumerate(entries):
        o = c.records[i]["offset"]
        hidden.append(torch.from_numpy(np.asarray(c.features[o + start:o + end])))
        final.append(torch.from_numpy(np.asarray(c.final[o + start:o + end])))
        ids.append(torch.from_numpy(np.asarray(c.ids[o + start:o + end]).astype(np.int64)))
        m = torch.zeros(end - start, dtype=torch.long)
        m[max(0, loss_from - start):] = 1
        loss.append(m)
        docs.append(torch.full((end - start,), d, dtype=torch.long))
        pos.append(torch.arange(start, end, dtype=torch.long))
    as_bf16 = lambda parts: torch.cat(parts).view(torch.bfloat16).unsqueeze(0).to(device, non_blocking=True)
    return dict(
        hidden_states=as_bf16(hidden),
        verifier_last_hidden_states=as_bf16(final),
        input_ids=torch.cat(ids).unsqueeze(0).to(device),
        loss_mask=torch.cat(loss).unsqueeze(0).to(device),
        document_ids=torch.cat(docs).unsqueeze(0).to(device),
        position_ids=torch.cat(pos).unsqueeze(0).to(device),
    )


def sample_row(train, row_tokens, rng):
    weights = [c.weight for c, _ in train]
    entries, used = [], 0
    while used < row_tokens:
        c, pool = rng.choices(train, weights=weights)[0]
        i = rng.choice(pool)
        start, end, loss_from = piece(c, i, row_tokens - used, rng)
        if end - start < 64 or loss_from + 8 >= end:
            if entries:
                break
            continue
        entries.append((c, i, start, end, loss_from))
        used += end - start
    return entries


def speculators_config(cfg):
    from speculators import DFlash2SpeculatorConfig, SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig
    from transformers import Qwen3Config

    d = cfg["dflash_config"]
    layers = cfg["num_hidden_layers"]
    transformer = Qwen3Config(
        hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"], num_hidden_layers=layers,
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], vocab_size=cfg["vocab_size"], rms_norm_eps=cfg["rms_norm_eps"],
        rope_parameters=cfg["rope_parameters"], sliding_window=cfg["sliding_window"], use_sliding_window=True,
        max_window_layers=layers, layer_types=["sliding_attention"] * layers,
        max_position_embeddings=cfg["max_position_embeddings"], tie_word_embeddings=False, attention_bias=False,
        initializer_range=cfg.get("initializer_range", 0.02))
    transformer.is_causal = False
    transformer._attn_implementation = "simple_flex_attention"
    return DFlash2SpeculatorConfig(
        transformer_layer_config=transformer, draft_vocab_size=cfg["vocab_size"], target_hidden_size=HIDDEN,
        block_size=d["block_size"], aux_hidden_state_layer_ids=list(d["target_layer_ids"]),
        mask_token_id=d["mask_token_id"], sliding_window_non_causal=True, sample_from_anchor=False,
        conv_kernel_size=d["conv_kernel_size"], conv_group_size=d["conv_group_size"],
        selector_rank=d["selector_rank"], selector_top_k=d["selector_top_k"],
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=d["block_size"] - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="zai-org/GLM-5.3-Flash",
                                    architectures=["Glm5NextForConditionalGeneration"])))


def target_tensors(root):
    index = json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]
    names = {"embed_tokens.weight": None, "lm_head.weight": None, "norm.weight": None}
    for key in index:
        if ".layers." in key or "visual" in key or "mtp" in key:
            continue
        for suffix in names:
            if key.endswith(suffix) and (suffix != "norm.weight" or key in (
                    "model.language_model.norm.weight", "model.norm.weight")):
                names[suffix] = key
    out = {}
    for suffix, key in names.items():
        assert key is not None, suffix
        with safe_open(Path(root) / index[key], "pt") as f:
            out[suffix] = f.get_tensor(key)
    return out


def build(init, target, device):
    from speculators import DFlash2DraftModel

    cfg = json.loads((Path(init) / "config.json").read_text())
    with torch.device("cpu"):
        model = DFlash2DraftModel(speculators_config(cfg))
    body = load_file(str(Path(init) / "model.safetensors"))
    remapped = model.remap_checkpoint_keys_for_load(body)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not unexpected, unexpected
    frozen = {"embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight", "verifier_norm.weight"}
    assert set(missing) <= frozen, set(missing) - frozen
    t = target_tensors(target)
    model.embed_tokens.weight = torch.nn.Parameter(t["embed_tokens.weight"].to(torch.bfloat16), requires_grad=False)
    model.lm_head.weight = torch.nn.Parameter(t["lm_head.weight"].to(torch.bfloat16), requires_grad=False)
    model.verifier_lm_head.weight = model.lm_head.weight
    model.verifier_norm.weight = torch.nn.Parameter(t["norm.weight"].float(), requires_grad=False)
    for name, p in model.named_parameters():
        if name not in frozen:
            p.data = p.data.float()
            p.requires_grad_(True)
    return model.to(device), cfg, sorted(body)


def ratios(metrics):
    out = {}
    for k, v in metrics.items():
        if k.endswith("_sum"):
            total = metrics.get(k[:-4] + "_total")
            if total is not None and float(total) > 0:
                out[k[:-4]] = float(v) / float(total)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="append", required=True, help="DIR:WEIGHT")
    p.add_argument("--init", required=True, help="the incoai drafter snapshot")
    p.add_argument("--target", required=True, help="the target checkpoint snapshot (embedding, norm, head)")
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--row-tokens", type=int, default=8192)
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--holdout", type=float, default=0.03)
    p.add_argument("--eval-rows", type=int, default=24)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--loss", default="kl_div")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda")
    captures = []
    for spec in a.capture:
        root, weight = spec.rsplit(":", 1)
        captures.append(Capture(root, float(weight)))
        print(f"capture {root}: {len(captures[-1].records)} records, {captures[-1].ids.shape[0]} tokens", flush=True)
    train, held = split(captures, a.holdout, a.seed)
    model, cfg, body_keys = build(a.init, a.target, device)
    model.train()

    from speculators.models.dflash2.core import DFlash2DraftModel
    trainer_kwargs, _ = DFlash2DraftModel.get_trainer_kwargs(loss_fn=a.loss, loss_implementation="fused",
                                                            max_anchors=a.anchors)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(x.numel() for x in params) / 1e9:.3f} B parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    # Fixed evaluation rows from the held-out records (one record a row, its first supervised window).
    eval_rng = random.Random(1234)
    eval_rows = []
    for c, i in held[:a.eval_rows]:
        start, end, loss_from = piece(c, i, a.row_tokens, eval_rng)
        eval_rows.append([(c, i, start, end, loss_from)])

    def evaluate():
        model.eval()
        totals = {}
        torch.manual_seed(99)  # the same anchors every evaluation
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for entries in eval_rows:
                _, _, metrics = model(**make_row(entries, a.row_tokens, device), **trainer_kwargs)
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
        model.train()
        return ratios(totals)

    # The checkpoint kept: the best held-out sum of the selector's teacher-forced accuracy at draft
    # positions 1-5 (the drafts a round of --draft-tokens auto may verify).
    best = {"score": -1.0, "step": 0, "state": None}

    def log_eval(step):
        r = evaluate()
        keys = [k for k in sorted(r) if any(s in k for s in ("accept", "loss", "acc", "recall"))]
        score = sum(r.get(f"position_{k}_acc", 0.0) for k in range(1, 6))
        print(f"eval step {step}: score {score:.4f}, " + ", ".join(f"{k}={r[k]:.4f}" for k in keys), flush=True)
        if score > best["score"]:
            best.update(score=score, step=step,
                        state={k: v.detach().to("cpu", torch.bfloat16, copy=True) for k, v in model.state_dict().items()
                               if k not in ("embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight",
                                            "verifier_norm.weight")})

    log_eval(0)
    start = time.time()
    deadline = start + 60 * a.minutes
    step = 0
    while step < a.max_steps and time.time() < deadline:
        total_steps = a.max_steps if a.max_steps < 100000 else None
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        if total_steps:
            lr *= 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
        else:
            frac = min(1.0, (time.time() - start) / (60 * a.minutes))
            lr *= 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        merged = {}
        for _ in range(a.grad_accum):
            row = make_row(sample_row(train, a.row_tokens, rng), a.row_tokens, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, metrics = model(**row, **trainer_kwargs)
            (loss / a.grad_accum).backward()
            for k, v in metrics.items():
                merged[k] = merged.get(k, 0.0) + float(v)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            r = ratios(merged)
            show = {k: v for k, v in r.items() if any(s in k for s in ("loss", "accept"))}
            print(f"step {step} lr {lr:.2e} {time.time() - start:.0f}s " +
                  ", ".join(f"{k}={v:.4f}" for k, v in sorted(show.items())), flush=True)
        if step % a.eval_every == 0:
            log_eval(step)
    log_eval(step)

    # The body under the incoai names, BF16, beside the incoai config.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from speculators.models.dflash.core import _BACKBONE_CHECKPOINT_PREFIXES
    from speculators.models.dflash.utils import rename_checkpoint_key
    back = {t: s for s, t in _BACKBONE_CHECKPOINT_PREFIXES.items()}
    state = {rename_checkpoint_key(k, back): v.contiguous() for k, v in best["state"].items()}
    state = {k: state[k] for k in body_keys}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {len(state)} tensors to {out}: step {best['step']} of {step}, score {best['score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
