"""Generic ggml/GGUF logical sources over a GgufStore.

`sources.gguf_raw.GgufStore` reads a GGUF file's raw stored rows and plain tensors; this module
turns them into the `LogicalSource`s tools.convert recipes bind: `ggml_matrix` (a quantized
matrix's exact values and its stored rows as EncodedRows), `plain` (an F32/F16/BF16 tensor's
exact values), `ggml_or_plain` (whichever applies), `Bf16ExactSource` (a wrapper refusing
values that are not exact in BF16), and the routed-expert bank import (`GgmlExpertBank`,
`import_ggml_expert_bank`) the families' GGUF recipes assign.
"""

from __future__ import annotations

from math import prod

import numpy as np
import torch

from tools.artifact.codecs.ggml import (
    BY_GGML_ID,
    GGML_TYPES,
    GGUF_BF16,
    GGUF_F16,
    GGUF_F32,
    GGUF_Q8_0,
    decode_q8_0,
    decode_values,
    require_finite_scales,
)
from tools.artifact.layouts import ggml_expert_geometry

from ..methods import PrepareRequest, PreparedMethod
from .gguf_raw import GgufStore
from .logical import EncodedRows, LogicalSource


def ggml_matrix(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A quantized 2-D GGUF tensor: exact values, and its stored rows as EncodedRows."""

    t = store.tensor(name)
    if t.shape != tuple(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    k = shape[1]

    if t.ggml_type == GGUF_Q8_0:
        def encoded(begin: int, end: int) -> EncodedRows:
            codes, scales = decode_q8_0(np.ascontiguousarray(store.raw(name, begin, end)), end - begin, k)
            if (codes == -128).any():
                raise ValueError(f"{name}: Q8_0 code -128 has no q8_g32_fp16 encoding")
            return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.copy()).reshape(end - begin, k // 32, 32),
                               torch.from_numpy(scales.copy()).view(torch.float16))

        def values(begin: int, end: int) -> torch.Tensor:
            rows = encoded(begin // k, -(-end // k))
            v = rows.codes.to(torch.float64) * rows.scales.to(torch.float64)[..., None]
            flat = v.reshape(-1)
            start = begin - begin // k * k
            return flat[start : start + end - begin]

        return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)

    if t.ggml_type not in BY_GGML_ID:
        raise ValueError(f"{name}: GGUF type {t.ggml_type} is not a ggml block format this importer stores")
    fmt = BY_GGML_ID[t.ggml_type]

    def encoded(begin: int, end: int) -> EncodedRows:
        rows = np.array(store.raw(name, begin, end))  # a private copy of the stored bytes
        return EncodedRows(fmt.name, torch.from_numpy(rows), torch.empty(0, dtype=torch.uint8))

    def values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        rows = np.ascontiguousarray(store.raw(name, first, last))
        flat = torch.from_numpy(decode_values(fmt, rows, last - first, k)).reshape(-1)
        start = begin - first * k
        return flat[start : start + end - begin]

    return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)


def plain(store: GgufStore, name: str, shape: tuple[int, ...], *, minus_one: bool = False,
          bf16_exact: bool = True) -> LogicalSource:
    """An F32/F16/BF16 GGUF tensor's exact values (binary64), optionally g - 1; when ``bf16_exact``
    every value must be a BF16 value, so a BF16 binding stores exactly what the GGUF holds."""

    t = store.tensor(name)
    if t.ggml_type not in (GGUF_F32, GGUF_F16, GGUF_BF16):
        raise ValueError(f"{name}: expected an F32, F16 or BF16 tensor")
    if prod(t.shape) != prod(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    cache: list[torch.Tensor] = []

    def read(begin: int, end: int) -> torch.Tensor:
        if not cache:
            v = torch.from_numpy(store.plain(name).reshape(-1).astype(np.float64))
            if minus_one:
                v = v - 1.0  # exact in binary64
            if bf16_exact and not torch.equal(v.to(torch.bfloat16).to(torch.float64), v):
                raise ValueError(f"{name}: a value is not exact in BF16" + (" after undoing +1" if minus_one else ""))
            cache.append(v)
        return cache[0][begin:end]

    return LogicalSource(tuple(shape), f"{store.path}:{name}" + (" - 1" if minus_one else ""), read)


def ggml_or_plain(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A 2-D GGUF tensor: stored rows for ggml formats and Q8_0, exact values for BF16/F32/F16 (a
    BF16 binding: every value must be a BF16 value, so an F32 matrix is never rounded silently)."""

    t = store.tensor(name)
    if t.ggml_type in (GGUF_F32, GGUF_F16, GGUF_BF16):
        return plain(store, name, shape)
    return ggml_matrix(store, name, shape)


class Bf16ExactSource:
    """A source whose every value must be a BF16 value (a BF16 binding then stores it exactly)."""

    def __init__(self, source):
        self._source = source
        self.path = source.path
        self.config: dict = {}

    def has(self, name: str) -> bool:
        return self._source.has(name)

    def describe(self, name: str):
        return self._source.describe(name)

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> torch.Tensor:
        values = self._source.read_flat(name, begin, end)
        if not torch.equal(values.to(torch.bfloat16).to(values.dtype), values):
            raise ValueError(f"{self.path}: {name} holds a value that is not exact in BF16")
        return values


class GgmlExpertBank:
    """One layer's routed experts in the GGUF; only ``import_ggml_expert_bank`` reads it."""

    def __init__(self, store: GgufStore, layer: int, shape: tuple[int, int, int]):
        self.store = store
        self.shape = shape
        self.label = f"{store.path}:blk.{layer}.ffn_*_exps"
        experts, hidden, inter = shape
        self.gate = store.tensor(f"blk.{layer}.ffn_gate_exps.weight")
        self.up = store.tensor(f"blk.{layer}.ffn_up_exps.weight")
        self.down = store.tensor(f"blk.{layer}.ffn_down_exps.weight")
        if self.gate.dims != (hidden, inter, experts) or self.up.dims != self.gate.dims:
            raise ValueError(f"{self.label}: gate/up dims {self.gate.dims} differ from [{hidden}, {inter}, {experts}]")
        if self.down.dims != (inter, hidden, experts):
            raise ValueError(f"{self.label}: down dims {self.down.dims} differ from [{inter}, {hidden}, {experts}]")
        if self.gate.ggml_type != self.up.ggml_type or self.gate.ggml_type not in BY_GGML_ID or self.down.ggml_type not in BY_GGML_ID:
            raise ValueError(f"{self.label}: gate and up must share one ggml format")
        self.gate_up_format = BY_GGML_ID[self.gate.ggml_type].name
        self.down_format = BY_GGML_ID[self.down.ggml_type].name

    @property
    def layout(self) -> str:
        return "ggml_expert_" + self.down_format.removeprefix("ggml_") + "_v1"

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


_BANK_CHUNK = 32


def import_ggml_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts as ``ggml_expert_*_v1`` records: per expert its gate rows, up rows
    and down rows, each the GGUF's stored blocks, the record padded to 4 KiB."""

    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, GgmlExpertBank):
        raise ValueError("import_ggml_expert_bank requires one GGUF expert bank")
    bank = request.inputs[0].source
    if request.parameters:
        raise ValueError("import_ggml_expert_bank accepts no numerical parameters")
    if request.target.format != bank.gate_up_format or request.target.layout != bank.layout:
        raise ValueError(f"{bank.label}: target {request.target.format}/{request.target.layout} differs from the bank's "
                         f"{bank.gate_up_format}/{bank.layout}")
    if tuple(request.target.shape) != bank.shape:
        raise ValueError("expert bank target shape differs from its source")
    g = ggml_expert_geometry(bank.layout, bank.gate_up_format, bank.shape)
    experts, hidden, inter = bank.shape
    store = bank.store
    gate_up, down = GGML_TYPES[bank.gate_up_format], GGML_TYPES[bank.down_format]

    def produce(output):
        pad = bytes(g.record_stride - g.record_bytes)
        for first in range(0, experts, _BANK_CHUNK):
            count = min(_BANK_CHUNK, experts - first)
            part = bytearray()
            for e in range(first, first + count):
                for name, fmt, rows, k in ((bank.gate.name, gate_up, inter, hidden), (bank.up.name, gate_up, inter, hidden),
                                           (bank.down.name, down, hidden, inter)):
                    raw = store.raw(name, e * rows, (e + 1) * rows)
                    require_finite_scales(fmt, raw, rows, k, f"{name} expert {e}")
                    part += raw.tobytes()
                part += pad
            output.write_bytes(first * g.record_stride, part)

    return request.job(produce=produce)
