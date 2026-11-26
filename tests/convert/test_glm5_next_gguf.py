"""The Glm5Next GGUF import's checks against the checkpoint and its llama.cpp transforms undone."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from tools.convert.glm5_next_gguf import kda_convolution, q8_bf16_rows, require_a_log, require_q8_rounding
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
