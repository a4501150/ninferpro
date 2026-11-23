"""Read GGUF tensors (F16 or any ggml quant level) as a named conversion source.

`tools.convert` normally reads the base checkpoint straight from Safetensors
(see sources/safetensors.py). This module lets a recipe pull some or all
logical parameters from a GGUF file instead -- typically because that is the
only local copy of the weights (e.g. an F16 or Q4_K/Q5_K/Q6_K/Q8_0 export),
and downloading the full BF16/FP32 Safetensors checkpoint again is wasteful.

Parsing and dequantization are delegated to the `gguf` package (the ggml-org
reference reader used by llama.cpp itself) rather than reimplemented here --
GGUF's block-quant byte layouts are intricate and easy to get subtly wrong,
and `gguf.dequantize` is the maintained, tested implementation. `pip install
gguf` if it is not already available.

GGUF tensor names never match the HF Safetensors names a recipe's source
tensors are keyed by (e.g. "model.language_model.layers.0.self_attn.q_proj.weight"
vs. "blk.0.attn_q.weight"), and some exporters reorder and rescale tensors.
`HFAliasSource` bridges that gap with an explicit {hf_name: gguf_name} map plus
optional per-tensor value transforms, so it satisfies the same
has()/describe()/read_flat() surface as SafetensorsSource and can be dropped in
wherever a recipe accepts an alternate source (see the "Read another source"
section of docs/weight-conversion.md).

The family-specific name maps and transforms (which encode each model family's
export quirks, each checked against real checkpoint/GGUF pairs by comparing
every mapped tensor's dequantized values back to the HF original) live in the
family modules: tools/convert/qwen3_5_gguf.py and tools/convert/qwen_common.py.
This module is the generic reader; it knows no family. Inspect a file with
GGUFSource.list_tensors() (or `python -m tools.convert.sources.gguf FILE`) to
see exactly what tensor names it has, and verify any new map addition against
an unquantized companion tensor or a Q8_0/F16 GGUF for a tighter statistical
check. See docs/weight-conversion.md.

Nothing here changes what --model supplies: point it at a directory that
still has the checkpoint's config.json and frontend resources (tokenizer.json
etc. -- a few KB, no need to keep the multi-GB weight files) and use
`--source NAME=quantized.gguf` to supply weight values from GGUF instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable

import numpy as np
import torch

try:
    import gguf
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "reading a .gguf source requires the 'gguf' package: pip install gguf"
    ) from error


@dataclass(frozen=True, slots=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str  # ggml quantization type name, informational only


class GGUFSource:
    """Bounded flat reads over a GGUF file's own tensor names, dequantized to FP32.

    Exposes the same has()/describe()/read_flat()/close() surface as
    SafetensorsSource, keyed by the GGUF file's native tensor names (e.g.
    "blk.0.attn_q.weight"), so it can be used directly wherever a recipe
    expects that interface.

    A GGUF tensor's raw `ReaderTensor.data` is itself row-structured: its
    first axis always has exactly as many entries as the tensor's outer (HF-
    order) dimension, whether the tensor is quantized (raw per-row byte
    blocks) or plain float (already viewed to the right dtype) -- confirmed
    on real Q5_0/Q8_0 files on this machine (`gguf.dequantize` on a leading
    row-slice of `.data` reproduces exactly the same leading rows as
    dequantizing the whole tensor, byte for byte). read_flat() uses that to
    decode only the covering row range for the requested [begin, end)
    element range, rather than dequantizing the whole tensor on every call:
    this matters because tools.convert reads each tensor twice per
    parameter (a 1-element sample during recipe.prepare()'s preflight
    validation, then the real data row-chunk by row-chunk while writing),
    and GGUF's block quantization has no cheaper way to serve a narrow read
    than decoding whichever blocks it falls in -- so avoiding a wasted full
    decode on the 1-element preflight probe alone was roughly halving
    conversion time before this was fixed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = gguf.GGUFReader(self.path)
        self._tensors = {tensor.name: tensor for tensor in self._reader.tensors}
        self.config: dict = {}
        self.bytes_read = 0
        # Parallel conversion workers share this source; the memmap reads are
        # independent, only the byte counter is shared.
        self._count_lock = threading.Lock()

    def has(self, name: str) -> bool:
        return name in self._tensors

    def _tensor(self, name: str):
        try:
            return self._tensors[name]
        except KeyError as error:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}") from error

    def describe(self, name: str) -> TensorInfo:
        tensor = self._tensor(name)
        # GGUF's ne[] lists dimensions fastest-first; reverse to match the
        # row-major (fastest-last) shape convention Safetensors/PyTorch use.
        # The underlying bytes are in the same flat element order either way.
        shape = tuple(int(dim) for dim in reversed(tensor.shape.tolist()))
        return TensorInfo(shape, tensor.tensor_type.name)

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        tensor = self._tensor(name)
        shape = self.describe(name).shape
        elements = 1
        for dim in shape:
            elements *= dim
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        row_width = elements // shape[0] if shape else 1
        row_begin, row_end = begin // row_width, -(-end // row_width)
        decoded = gguf.dequantize(tensor.data[row_begin:row_end], tensor.tensor_type)
        flat = np.ascontiguousarray(decoded.reshape(-1), dtype=np.float32)
        offset = begin - row_begin * row_width
        result = flat[offset : offset + (end - begin)].copy()
        with self._count_lock:
            self.bytes_read += result.nbytes
        return torch.from_numpy(result)

    def list_tensors(self):
        """Yield (name, shape, ggml_type) for every tensor, for building a name_map."""
        for name in self._tensors:
            info = self.describe(name)
            yield name, info.shape, info.dtype

    def metadata(self) -> dict:
        """Raw GGUF key/value metadata (architecture, hyperparameters, tokenizer, ...)."""
        result = {}
        for key, field in self._reader.fields.items():
            try:
                result[key] = field.contents()
            except Exception:
                result[key] = None
        return result

    def close(self) -> None:
        pass

    def __enter__(self) -> "GGUFSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


Transform = Callable[[torch.Tensor, tuple[int, ...]], tuple[torch.Tensor, tuple[int, ...]]]


class HFAliasSource:
    """Expose a GGUFSource's tensors under caller-supplied HF-style names.

    tools.convert recipes capture the exact HF Safetensors tensor name at
    model-build time and later call describe()/read_flat() with that same
    string against whichever store a recipe assigns (see model.source() /
    docs/weight-conversion.md's "Read another source"). A GGUF file has no
    such names, and some tensors are reordered relative to their HF layout;
    this wrapper looks up an explicit map instead of guessing at either.

    `transforms` is an optional {hf_name: fn(flat_values, shape) -> (flat_values,
    shape)} map for tensors GGUF stores in a different element order or scale
    than HF (see the family name maps, e.g. tools/convert/qwen3_5_gguf.py).
    Most tensors need no transform: GGUF's ne[] is simply HF's shape reversed,
    with the same flat element order and value.

    `shapes` is an optional {hf_name: fn(gguf_shape) -> hf_shape} map for
    tensors whose *rank* also differs -- e.g. llama.cpp's exporter squeezes
    a PyTorch Conv1d weight's singleton (out_channels, 1, kernel_size) shape
    down to (out_channels, kernel_size) before writing it, so GGUF reports a
    2D tensor where the HF-side caller's captured shape is 3D even though the
    flat element order and count are identical. Most tensors need no entry
    here either.

    `composites` is an optional {hf_name: fn(GGUFSource) -> (flat_values, hf_shape)}
    map for the rarer case where ONE HF tensor was split across MULTIPLE GGUF
    tensors on export -- e.g. a vision tower's Conv3d patch-embed kernel
    (temporal depth 2) is written as two separate 2D kernels
    ("v.patch_embd.weight" and "v.patch_embd.weight.1"); the composite function
    reads both from the raw GGUFSource and stacks them back into the original 5D
    layout. A composite entry takes priority over `name_map`/`transforms`/`shapes`
    for that name.
    """

    def __init__(
        self,
        source: GGUFSource,
        name_map: dict[str, str],
        *,
        transforms: dict[str, Transform] | None = None,
        shapes: dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]] | None = None,
        composites: dict[str, Callable[[GGUFSource], tuple[torch.Tensor, tuple[int, ...]]]]
        | None = None,
    ) -> None:
        self._source = source
        self._names = dict(name_map)
        self._transforms = dict(transforms) if transforms else {}
        self._shapes = dict(shapes) if shapes else {}
        self._composites = dict(composites) if composites else {}
        self.path = source.path
        self.config: dict = {}
        # Transformed tensors (GDN's fused/reordered ones) need their whole
        # row group decoded regardless of the requested [begin, end) range,
        # since the reorder isn't expressible on a row subset alone. Cache
        # the last one so a tensor's own sequential produce() chunk reads
        # don't repeat that decode+transform per chunk; one slot per thread is
        # enough because each conversion worker processes one parameter at a
        # time and this is never the same tensor recipe.prepare()'s preflight
        # already decoded (that pass and the write pass are far enough apart in
        # tensor order that nothing would still be cached anyway).
        self._transform_cache = threading.local()

    def _resolve(self, hf_name: str) -> str:
        try:
            return self._names[hf_name]
        except KeyError as error:
            raise ValueError(
                f"no GGUF tensor mapped for {hf_name!r}; add it to the name_map "
                f"passed to HFAliasSource (inspect {self._source.path} with "
                "GGUFSource.list_tensors() to see the tensor names it actually has)"
            ) from error

    def has(self, name: str) -> bool:
        if name in self._composites:
            return True
        return name in self._names and self._source.has(self._names[name])

    def describe(self, name: str) -> TensorInfo:
        composite = self._composites.get(name)
        if composite is not None:
            values, shape = composite(self._source)
            return TensorInfo(shape, "composite")
        # Every transform here reorders or rescales values in place; none of
        # them change shape, so describe() never needs to run one -- only
        # consult `shapes` for the (rare) rank mismatch.
        info = self._source.describe(self._resolve(name))
        reshape = self._shapes.get(name)
        return info if reshape is None else TensorInfo(reshape(info.shape), info.dtype)

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        composite = self._composites.get(name)
        if composite is not None:
            values, shape = composite(self._source)
            elements = values.numel()
            end = elements if end is None else end
            if not 0 <= begin <= end <= elements:
                raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
            return values[begin:end].clone()
        gguf_name = self._resolve(name)
        transform = self._transforms.get(name)
        if transform is None:
            return self._source.read_flat(gguf_name, begin, end)
        shape = self._source.describe(gguf_name).shape
        cache = self._transform_cache
        if getattr(cache, "name", None) == name:
            values = cache.values
        else:
            values, _ = transform(self._source.read_flat(gguf_name), shape)
            cache.name, cache.values = name, values
        elements = values.numel()
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
        return values[begin:end].clone()

    def close(self) -> None:
        self._transform_cache = threading.local()

    def __enter__(self) -> "HFAliasSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def _main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List a GGUF file's tensor names, shapes and ggml types."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    with GGUFSource(args.path) as source:
        for name, shape, dtype in sorted(source.list_tensors()):
            print(f"{name}\t{tuple(shape)}\t{dtype}")


if __name__ == "__main__":
    _main()
