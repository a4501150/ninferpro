"""The Hugging Face GLM-5.3-Flash processor's prompts and pixels, the oracle of the C++ frontend.

For every case (one user turn holding its media and a line of text) it writes the chat template's
text through Glm5NextProcessor: the prompt's token ids, its mm_token_type_ids, the media grids, and
the pixel values rounded to BF16 as the tower consumes them. A case is an image path, a video path
(.mp4, .webm, .mkv, .mov) or several of either joined with '+'.

    .venv-convert/bin/python tools/glm5_next/processor_reference.py --model SNAPSHOT \\
        --cases a.png,b.jpg,a.png+b.jpg,clip.mp4 --out DIR

DIR receives processor.json ({"cases": [{"media": [...], "kinds": [...], "text", "ids",
"token_types", "grids"}]}) and case-K.bf16 (each case's pixel rows, raw little-endian BF16 bits,
images then videos in prompt order). tests/models/glm5_next/test_frontend_real.cpp reads it through
GLM_PROCESSOR_REFERENCE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", required=True, help="comma-separated cases, media joined by '+'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.video_utils import load_video

    processor = AutoProcessor.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, case in enumerate(args.cases.split(",")):
        media = [Path(item) for item in case.split("+")]
        kinds = ["video" if path.suffix.lower() in VIDEO_SUFFIXES else "image" for path in media]
        content = [{"type": kind} for kind in kinds]
        content.append({"type": "text", "text": f"Describe case {index}."})
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        images = [Image.open(path).convert("RGB") for path, kind in zip(media, kinds)
                  if kind == "image"]
        # PyAV decodes every frame (Transformers' default backend, torchcodec, needs system FFmpeg
        # libraries it was built against); the processor samples them from the metadata.
        loaded = [load_video(str(path), backend="pyav") for path, kind in zip(media, kinds)
                  if kind == "video"]
        batch = processor(text=[text], images=images or None,
                          videos=[video for video, _ in loaded] or None,
                          video_metadata=[metadata for _, metadata in loaded] or None,
                          return_tensors="pt")
        rows = []
        grids = []
        if images:
            rows.append(batch["pixel_values"])
            grids += batch["image_grid_thw"].tolist()
        if loaded:
            rows.append(batch["pixel_values_videos"])
            grids += batch["video_grid_thw"].tolist()
        pixels = torch.cat(rows).to(torch.bfloat16).view(torch.int16).numpy()
        pixels.tofile(args.out / f"case-{index}.bf16")
        cases.append({
            "media": [str(path) for path in media],
            "kinds": kinds,
            "text": f"Describe case {index}.",
            "ids": batch["input_ids"][0].tolist(),
            "token_types": batch["mm_token_type_ids"][0].tolist(),
            "grids": grids,
        })
        print(f"case {index}: {len(cases[-1]['ids'])} tokens, grids {grids}")
    (args.out / "processor.json").write_text(json.dumps({"cases": cases}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
