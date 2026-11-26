#!/usr/bin/env python3
"""Prompt and off-policy records for the GLM-5.3-Flash DFlash2 drafter corpus (README.md).

Writes PROMPTS.jsonl (records the target answers: UltraChat first turns with and without thinking,
CodeAlpaca tasks, a share of them as agentic tasks that declare the run_bash tool) and
OFFPOLICY.jsonl (whole UltraChat conversations, captured teacher-forced without generation).
"""
import argparse
import glob
import json
import random

import pandas as pd

AGENT_SYSTEM = (
    "You are an engineering agent. You accomplish tasks by writing and running shell commands "
    "(typically python3 -c, or heredoc python) with the run_bash tool, inspecting their output, "
    "and iterating until the result is verified. Think first, then act.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ultrachat", required=True, help="directory of train_sft parquet shards")
    p.add_argument("--codealpaca", required=True, help="code_alpaca_20k.json")
    p.add_argument("--prompts", required=True)
    p.add_argument("--offpolicy", required=True)
    p.add_argument("--chat", type=int, default=4000)
    p.add_argument("--code", type=int, default=2000)
    p.add_argument("--conversations", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = random.Random(a.seed)

    frames = [pd.read_parquet(f, columns=["prompt_id", "messages"])
              for f in sorted(glob.glob(f"{a.ultrachat}/*.parquet"))]
    chats = pd.concat(frames).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    code = json.load(open(a.codealpaca))
    rng.shuffle(code)

    prompts = []
    for i, row in chats.head(a.chat).iterrows():
        user = row["messages"][0]["content"]
        prompts.append({"id": f"chat-{row['prompt_id'][:16]}", "user": user,
                        "thinking": rng.random() < 0.6, "tools": False})
    for i, item in enumerate(code[:a.code]):
        task = item["instruction"] + (("\n\nInput:\n" + item["input"]) if item["input"] else "")
        if rng.random() < 0.5:
            prompts.append({"id": f"agent-{i}", "system": AGENT_SYSTEM,
                            "user": task + "\nImplement it in Python and verify it with run_bash.",
                            "thinking": True, "tools": True})
        else:
            prompts.append({"id": f"code-{i}", "user": task, "thinking": rng.random() < 0.5, "tools": False})
    rng.shuffle(prompts)
    with open(a.prompts, "w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")

    n = 0
    with open(a.offpolicy, "w") as f:
        for _, row in chats.iloc[a.chat:].iterrows():
            messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()
            if len(messages) < 2 or messages[0]["role"] != "user":
                continue
            f.write(json.dumps({"id": f"off-{row['prompt_id'][:16]}", "messages": messages}) + "\n")
            n += 1
            if n >= a.conversations:
                break
    print(f"{len(prompts)} prompts, {n} off-policy conversations")


if __name__ == "__main__":
    main()
