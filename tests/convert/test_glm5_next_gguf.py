"""The Glm5Next GGUF import's checks against the checkpoint and its llama.cpp transforms undone."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from tools.convert.glm5_next_gguf import (kda_convolution, q8_bf16_rows, require_a_log, require_q8_rounding,
                                         require_shared_tensors)
from tools.convert.sources.gguf_raw import GgufStore


class _Checkpoint:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def describe(self, name):
        return SimpleNamespace(shape=tuple(self.tensors[name].shape))

    def read_flat(self, name, begin=0, end=None):
        return self.tensors[name].reshape(-1)[begin:end]


def _write(path, tensors: dict[str, tuple[np.ndarray, object]]) -> GgufStore:
    writer = gguf.GGUFWriter(str(path), "test")
    for name, (data, raw_dtype) in tensors.items():
        writer.add_tensor(name, data, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GgufStore([path])


def test_q8_rounding_accepts_llama_cpp_and_rejects_another_matrix(tmp_path):
    rng = np.random.default_rng(5)
    weight = torch.from_numpy(rng.standard_normal((8, 128)).astype(np.float32)).to(torch.bfloat16)
    q8 = gguf.quants.quantize(weight.float().numpy(), gguf.GGMLQuantizationType.Q8_0)
    other = weight.clone()
    other[3, 77] += 0.25
    with _write(tmp_path / "q8.gguf", {"w": (q8, gguf.GGMLQuantizationType.Q8_0)}) as store:
        require_q8_rounding(store, "w", _Checkpoint({"w": weight}), "w", (8, 128))
        with pytest.raises(ValueError, match="row 3"):
            require_q8_rounding(store, "w", _Checkpoint({"w": other}), "w", (8, 128))
        # The embedding's rows: d * code rounded once to BF16, read across a row boundary.
        rows = q8_bf16_rows(store, "w", (8, 128))
        codes = q8.reshape(8, 4, 34)
        d = codes[..., :2].copy().view(np.float16).astype(np.float64)
        values = codes[..., 2:].copy().view(np.int8).astype(np.float64) * d
        expected = torch.from_numpy(values.reshape(-1)).to(torch.bfloat16).to(torch.float64)
        assert torch.equal(rows.values(100, 300), expected[100:300])


def test_a_log_within_one_ulp(tmp_path):
    a_log = torch.tensor(np.random.default_rng(6).uniform(-3, 2, 16), dtype=torch.float32)
    a = (-np.exp(a_log.double().numpy())).astype(np.float32)
    near = np.nextafter(a, np.float32(0))
    far = a * np.float32(1 + 2.0**-18)
    with _write(tmp_path / "a.gguf", {"near": (near, None), "far": (far, None)}) as store:
        require_a_log(store, "near", _Checkpoint({"a": a_log}), "a", 16)
        with pytest.raises(ValueError):
            require_a_log(store, "far", _Checkpoint({"a": a_log}), "a", 16)


def test_kda_convolution_fuses_q_k_v_channels(tmp_path):
    width, taps = 6, 4
    kernels = {f: np.arange(width * taps, dtype=np.float32).reshape(width, 1, taps) + 100 * i
               for i, f in enumerate("qkv")}
    tensors = {f"blk.0.ssm_conv1d_{f}.weight": (k, None) for f, k in kernels.items()}
    with _write(tmp_path / "conv.gguf", tensors) as store:
        fused = kda_convolution(store, "blk.0.", taps, width).values().reshape(taps, 3 * width)
    expected = np.concatenate([kernels[f].reshape(width, taps).T for f in "qkv"], axis=1)
    assert np.array_equal(fused.numpy(), expected)


def test_shared_tensors_of_another_build(tmp_path):
    """The Q8_0 build's non-expert tensors: the Q4 build's bytes, another Q8_0 rounding of its Q8_0
    ones, or the Q8_0 rounding of its BF16 ones."""

    rng = np.random.default_rng(7)
    q8 = gguf.GGMLQuantizationType.Q8_0
    shared = gguf.quants.quantize(rng.standard_normal((4, 64)).astype(np.float32), q8)
    bf16 = torch.from_numpy(rng.standard_normal((4, 64)).astype(np.float32)).to(torch.bfloat16)
    bf16_raw = bf16.view(torch.int16).numpy().view(np.uint8).reshape(4, 128)
    rounded = gguf.quants.quantize(bf16.float().numpy(), q8)
    experts = rng.integers(0, 256, size=(2, 4, 34), dtype=np.uint8)
    main = {"a.weight": (shared, q8), "b.weight": (bf16_raw, gguf.GGMLQuantizationType.BF16),
            "blk.0.ffn_down_exps.weight": (experts, q8)}
    other = dict(main, **{"b.weight": (rounded, q8), "blk.0.ffn_down_exps.weight": (experts[:, :2], q8)})
    with _write(tmp_path / "main.gguf", main) as m, _write(tmp_path / "other.gguf", other) as o:
        require_shared_tensors(m, o)  # the experts may differ in every way
    # Another rounding of a Q8_0 tensor (a code one step off) passes; a code 40 steps off does not.
    rerounded = shared.copy()
    rerounded[2, 5] = np.uint8((int(rerounded[2, 5].view(np.int8)) + (1 if rerounded[2, 5].view(np.int8) < 127 else -1)) & 0xFF)
    with _write(tmp_path / "main1.gguf", main) as m, _write(tmp_path / "rr.gguf", dict(other, **{"a.weight": (rerounded, q8)})) as o:
        require_shared_tensors(m, o)
    changed = shared.copy()
    changed[2, 5] = np.uint8((int(changed[2, 5].view(np.int8)) + (40 if changed[2, 5].view(np.int8) < 0 else -40)) & 0xFF)
    with _write(tmp_path / "main2.gguf", main) as m, _write(tmp_path / "bad.gguf", dict(other, **{"a.weight": (changed, q8)})) as o:
        with pytest.raises(ValueError, match="row 2"):
            require_shared_tensors(m, o)
    far = bf16.clone()
    far[1, 3] += 0.5
    with _write(tmp_path / "main3.gguf", main) as m, \
         _write(tmp_path / "far.gguf", dict(other, **{"b.weight": (gguf.quants.quantize(far.float().numpy(), q8), q8)})) as o:
        with pytest.raises(ValueError, match="row 1"):
            require_shared_tensors(m, o)
