"""Shared config-value validators: positive integers, F32 reals, and fixed fields."""

from __future__ import annotations

from math import isfinite
import struct


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _f32(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name}: not representable as F32") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected positive finite F32")
    return result


def _fixed(config, key, value, label):
    if key in config and (
        config[key] != value
        or isinstance(value, bool)
        and type(config[key]) is not bool
    ):
        raise ValueError(f"{label}.{key}: expected {value!r}, got {config[key]!r}")
