"""Print a weight file's kernels and dense-layer maps as text heatmaps, so a
human (or LLM) can read what each weight is meant to do.

    uv run ws-show weights/claude_handwritten_v1.json
"""

from __future__ import annotations

import argparse

import numpy as np

from .network import Network

SHADES = "▓▒░ ·+*#█"  # strongly negative ... 0 ... strongly positive


def heat(a: np.ndarray) -> str:
    m = np.abs(a).max() or 1.0
    idx = np.clip(np.round(a / m * 4).astype(int) + 4, 0, 8)
    return "\n".join("".join(SHADES[i] * 2 for i in row) for row in idx)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    args = p.parse_args(argv)
    net = Network.load(args.weights)
    print(f"{net.meta.get('name', args.weights)}  — {net.n_params()} params")
    shape_chw = None
    for i, layer in enumerate(net.layers):
        print(f"\n[{i}] {layer.type} {({k: v for k, v in layer.params.items() if k not in ('type', 'names')})}")
        if layer.type == "conv2d":
            names = layer.params.get("names") or [f"ch{j}" for j in range(layer.weights.shape[0])]
            for j, name in enumerate(names):
                for c in range(layer.weights.shape[1]):
                    print(f"  filter {j} '{name}' (in {c}), bias {layer.bias[j]:+.2f}")
                    print("    " + heat(layer.weights[j, c]).replace("\n", "\n    "))
            shape_chw = layer.weights.shape[0]
        if layer.type == "dense" and shape_chw:
            w = layer.weights
            hw = w.shape[1] // shape_chw
            side = int(round(hw ** 0.5))
            if side * side * shape_chw == w.shape[1]:
                for d in range(w.shape[0]):
                    maps = w[d].reshape(shape_chw, side, side)
                    print(f"  digit {d}  bias {layer.bias[d]:+.2f}")
                    rows = [heat(m).split("\n") for m in maps]
                    for r in range(side):
                        print("    " + "   ".join(rr[r] for rr in rows))


if __name__ == "__main__":
    main()
