"""NVFP4 routed-expert banks: one layer's experts read as checkpoint words and written bit-exactly
as ``nvfp4_expert_rg16_v1`` records (docs/maintainer/storage-layouts.md, the expert record layout).

Two checkpoint representations: ModelOpt's W4A4 export (``weight`` codes with ``weight_scale_2``
and an ``input_scale`` per matrix) and the weight-only compressed-tensors export
(``weight_packed`` / ``weight_global_scale``, no activation scales: a W4A16 bank). The record
geometry follows the bank's [experts, hidden, intermediate] shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.layouts import expert_bank_geometry

from ..methods import PrepareRequest, PreparedMethod
from .modelopt import nvfp4_matrix_words, nvfp4_packed_matrix_words
from .safetensors import SafetensorsSource


@dataclass(frozen=True, slots=True)
class ExpertBankSource:
    """One layer's NVFP4 experts; only ``import_expert_bank`` can read it. ``packed`` selects the
    weight-only compressed-tensors export (``weight_packed`` / ``weight_global_scale``, no activation
    scales: a W4A16 bank) instead of ModelOpt's W4A4 one."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str  # "...mlp.experts." so expert e's gate is f"{prefix}{e}.gate_proj"
    packed: bool = False
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


def weight_only_bank(source) -> bool:
    """Whether a routed-expert parameter's source is a weight-only (W4A16) bank."""

    return isinstance(source, ExpertBankSource) and source.packed


_BANK_CHUNK = 32  # experts encoded per step; bounds conversion memory at ~0.4 GB


def import_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts bit-exactly in ``nvfp4_expert_rg16_v1``."""

    if request.target.format != "nvfp4_mul" or request.target.layout != "nvfp4_expert_rg16_v1":
        raise ValueError("import_expert_bank requires nvfp4_mul in nvfp4_expert_rg16_v1")
    if request.parameters:
        raise ValueError("import_expert_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, ExpertBankSource):
        raise ValueError("import_expert_bank requires exactly one expert-bank source")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    experts, hidden, intermediate = source.shape
    geometry = expert_bank_geometry("nvfp4_mul", source.shape)
    words = nvfp4_packed_matrix_words if source.packed else nvfp4_matrix_words

    def produce(output):
        multipliers = []
        for first in range(0, experts, _BANK_CHUNK):
            count = min(_BANK_CHUNK, experts - first)
            gc, gs, uc, us, dc, ds, mult = [], [], [], [], [], [], []
            for e in range(first, first + count):
                gate = words(source.store, f"{source.prefix}{e}.gate_proj", (intermediate, hidden))
                up = words(source.store, f"{source.prefix}{e}.up_proj", (intermediate, hidden))
                down = words(source.store, f"{source.prefix}{e}.down_proj", (hidden, intermediate))
                gc.append(gate.codes)
                gs.append(gate.scales)
                uc.append(up.codes)
                us.append(up.scales)
                dc.append(down.codes)
                ds.append(down.scales)
                mult.append(torch.stack((gate.weight_scale_2, up.weight_scale_2, down.weight_scale_2)))
            part = encode_nvfp4_expert_bank(
                torch.stack(gc), torch.stack(gs), torch.stack(uc), torch.stack(us),
                torch.stack(dc), torch.stack(ds), torch.stack(mult), (count, hidden, intermediate),
            )
            output.write_bytes(first * geometry.record_stride, memoryview(part)[: count * geometry.record_stride])
            multipliers.append(torch.stack(mult))
        plane = torch.cat(multipliers).contiguous()
        output.write_bytes(geometry.multiplier_offset, plane.numpy().tobytes())

    return request.job(produce=produce)
