"""A tiny numpy CNN whose every number comes from a JSON file written by an LLM.

Weight-file format (JSON):

{
  "name": "...",
  "author": "...",
  "notes": "free-text reasoning about what the weights mean",
  "layers": [
    {"type": "avgpool", "size": 2},
    {"type": "conv2d", "padding": 1,
     "weights": [...],          # nested list, shape (out_ch, in_ch, k, k)
     "bias": [...],             # shape (out_ch,)  (optional, default zeros)
     "names": ["ink", ...]},    # optional, one label per output channel
    {"type": "relu"},
    {"type": "maxpool", "size": 2},
    {"type": "flatten"},
    {"type": "dense",
     "weights": [...],          # nested list; reshaped to (out, in). Any nesting
                                #   whose total size is out*in is accepted, e.g.
                                #   [10][C][H][W] in channel-major (C,H,W) order.
     "bias": [...]}             # shape (out,)
  ]
}

The input to the first layer is a batch of images shaped (N, 1, 28, 28) with
pixel values in [0, 1] (0 = background, 1 = ink). The last layer must produce
10 scores; the predicted digit is the argmax.

Supported layer types: conv2d, dense, relu, abs, tanh, sigmoid, maxpool,
avgpool, flatten, scale (multiply by a constant), shift (add a constant).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "abs": np.abs,
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}


@dataclass
class Layer:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None


def _pool(x: np.ndarray, size: int, op: str) -> np.ndarray:
    n, c, h, w = x.shape
    hh, ww = h // size, w // size
    x = x[:, :, : hh * size, : ww * size].reshape(n, c, hh, size, ww, size)
    return x.max(axis=(3, 5)) if op == "max" else x.mean(axis=(3, 5))


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int) -> np.ndarray:
    """Plain cross-correlation (like PyTorch's Conv2d), stride 1."""
    out_ch, in_ch, kh, kw = w.shape
    if x.shape[1] != in_ch:
        raise ValueError(f"conv2d expects {in_ch} input channels, got {x.shape[1]}")
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n, _, h, wd = x.shape
    oh, ow = h - kh + 1, wd - kw + 1
    # (N, in_ch, oh, ow, kh, kw) sliding windows, contracted in chunks of images
    # so deep stacks on big images never materialise more than ~200 MB at once
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    chunk = max(1, int(2.5e7 // max(1, in_ch * oh * ow * kh * kw)))
    w2 = w.reshape(out_ch, -1)
    out = np.empty((n, out_ch, oh, ow), dtype=np.result_type(x, w))
    for i in range(0, n, chunk):
        cols = win[i:i + chunk].transpose(0, 2, 3, 1, 4, 5).reshape(-1, in_ch * kh * kw)  # (n*oh*ow, c*k*k)
        out[i:i + chunk] = (cols @ w2.T).reshape(-1, oh, ow, out_ch).transpose(0, 3, 1, 2)
    return out + b[None, :, None, None]


class Network:
    def __init__(self, layers: list[Layer], meta: dict[str, Any]):
        self.layers = layers
        self.meta = meta

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "Network":
        layers: list[Layer] = []
        for i, raw in enumerate(spec["layers"]):
            t = raw["type"].lower()
            layer = Layer(type=t, params={k: v for k, v in raw.items() if k not in ("weights", "bias")})
            if t == "conv2d":
                w = np.asarray(raw["weights"], dtype=np.float64)
                if w.ndim == 3:  # (out, k, k) shorthand for single input channel
                    w = w[:, None]
                if w.ndim != 4 or w.shape[2] != w.shape[3]:
                    raise ValueError(f"layer {i}: conv2d weights must be (out, in, k, k), got {w.shape}")
                layer.weights = w
                layer.bias = np.asarray(raw.get("bias", np.zeros(w.shape[0])), dtype=np.float64)
            elif t == "dense":
                w = np.asarray(raw["weights"], dtype=np.float64)
                b = np.asarray(raw.get("bias", None) if raw.get("bias") is not None else np.zeros(w.shape[0]), dtype=np.float64)
                layer.weights = w.reshape(w.shape[0], -1)
                layer.bias = b
                if layer.bias.shape != (layer.weights.shape[0],):
                    raise ValueError(f"layer {i}: dense bias shape {layer.bias.shape} != ({layer.weights.shape[0]},)")
            elif t in ACTIVATIONS or t in ("maxpool", "avgpool", "flatten", "scale", "shift"):
                pass
            else:
                raise ValueError(f"layer {i}: unknown layer type {t!r}")
            layers.append(layer)
        meta = {k: v for k, v in spec.items() if k != "layers"}
        return cls(layers, meta)

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def forward(self, x: np.ndarray, return_intermediates: bool = False, check_output: bool = True):
        """x: (N, 28, 28) or (N, 1, 28, 28) floats in [0, 1]."""
        if x.ndim == 3:
            x = x[:, None]
        inter = []
        for layer in self.layers:
            t = layer.type
            if t == "conv2d":
                x = _conv2d(x, layer.weights, layer.bias, int(layer.params.get("padding", 0)))
            elif t == "dense":
                if x.ndim != 2:
                    x = x.reshape(x.shape[0], -1)
                if x.shape[1] != layer.weights.shape[1]:
                    raise ValueError(f"dense expects {layer.weights.shape[1]} inputs, got {x.shape[1]}")
                x = x @ layer.weights.T + layer.bias
            elif t in ACTIVATIONS:
                x = ACTIVATIONS[t](x)
            elif t == "maxpool":
                x = _pool(x, int(layer.params.get("size", 2)), "max")
            elif t == "avgpool":
                x = _pool(x, int(layer.params.get("size", 2)), "avg")
            elif t == "flatten":
                x = x.reshape(x.shape[0], -1)
            elif t == "scale":
                x = x * float(layer.params["value"])
            elif t == "shift":
                x = x + float(layer.params["value"])
            if return_intermediates:
                inter.append(x)
        if check_output and (x.ndim != 2 or x.shape[1] != 10):
            raise ValueError(f"network must output (N, 10) scores, got {x.shape}")
        return (x, inter) if return_intermediates else x

    def predict(self, x: np.ndarray, batch: int = 2000) -> np.ndarray:
        return np.concatenate([self.forward(x[i : i + batch]).argmax(1) for i in range(0, len(x), batch)])

    def n_params(self) -> int:
        return sum(l.weights.size + l.bias.size for l in self.layers if l.weights is not None)
