"""Evaluate an LLM-written weight file on MNIST.

    uv run ws-eval weights/claude_handwritten_v1.json
    uv run ws-eval weights/*.json --split val
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import data
from .network import Network


def metrics(net: Network, x: np.ndarray, y: np.ndarray) -> dict:
    pred = net.predict(x)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per_class = (np.diag(cm) / np.maximum(cm.sum(1), 1)).round(4)
    return {
        "accuracy": float((pred == y).mean()),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": cm.tolist(),  # rows = true digit, cols = predicted
        "prediction_histogram": np.bincount(pred, minlength=10).tolist(),
    }


def format_report(m: dict) -> str:
    lines = [f"accuracy: {m['accuracy'] * 100:.2f}%", "", "per-class accuracy:"]
    lines += [f"  {d}: {a * 100:5.1f}%" for d, a in enumerate(m["per_class_accuracy"])]
    lines += ["", "confusion matrix (row = true, col = predicted):", "      " + "".join(f"{d:>6}" for d in range(10))]
    for d, row in enumerate(m["confusion_matrix"]):
        lines.append(f"  {d:>2}  " + "".join(f"{v:>6}" for v in row))
    return "\n".join(lines)


def load_split(split: str):
    if split == "val":
        return data.validation_split()
    return data.load(split)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", nargs="+", help="weight JSON file(s)")
    p.add_argument("--split", default="test", choices=["test", "train", "val"])
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = p.parse_args(argv)

    x, y = load_split(args.split)
    for path in args.weights:
        try:
            net = Network.load(path)
        except Exception as e:
            print(f"{path}: failed to load: {e}", file=sys.stderr)
            continue
        m = metrics(net, x, y)
        if args.json:
            print(json.dumps({"file": path, **m}))
        else:
            print(f"== {path}  ({net.n_params()} params, {args.split} split, n={len(y)})")
            print(format_report(m))
            print()


if __name__ == "__main__":
    main()
