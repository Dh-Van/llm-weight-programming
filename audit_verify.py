"""Cold-room verification of saved weight files.

Imports nothing from this repo. It rebuilds each network from the raw JSON
weight tensors using only standard torch.nn layers, downloads fresh copies of
the official MNIST test set and Imagenette validation set through torchvision,
and runs every sample once under torch.no_grad(). It reports the sample count,
per-class accuracy, the confusion matrix and top-1 accuracy.

    python audit_verify.py --data-root <empty folder> \
        mnist runs/explore-20261105-120713/best_weights.json \
        imagenette runs/explore-imagenette-20261105-193149/best_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)  # gentle on small machines

DEFAULTS = ["mnist", "weights/mnist_95.93.json",
            "imagenette", "weights/imagenette_67.77_5k-params.json",
            "imagenette", "weights/imagenette_67.90_57k-params.json"]


class Lambda(nn.Module):
    def __init__(self, fn, name):
        super().__init__()
        self.fn, self.name = fn, name

    def forward(self, x):
        return self.fn(x)

    def extra_repr(self):
        return self.name


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def build_layers(layers: list[dict], in_shape: tuple[int, ...]) -> nn.Sequential:
    mods, shape = [], in_shape  # shape excludes batch
    for i, L in enumerate(layers):
        ty = L["type"].lower()
        if isinstance(L.get("weights"), str) or isinstance(L.get("bias"), str):
            raise ValueError(f"layer {i}: unresolved array reference {L.get('weights')!r}")
        if ty == "conv2d":
            w = t(L["weights"])
            if w.dim() == 3:
                w = w[:, None]
            out_c, in_c, k, _ = w.shape
            pad = int(L.get("padding", 0))
            m = nn.Conv2d(in_c, out_c, k, padding=pad, bias=True)
            m.weight.data = w
            m.bias.data = t(L["bias"]) if L.get("bias") is not None else torch.zeros(out_c)
            mods.append(m)
            shape = (out_c, shape[1] + 2 * pad - k + 1, shape[2] + 2 * pad - k + 1)
        elif ty == "dense":
            w = t(L["weights"])
            w = w.reshape(w.shape[0], -1)
            if len(shape) != 1:
                mods.append(nn.Flatten())
            m = nn.Linear(w.shape[1], w.shape[0])
            m.weight.data = w
            m.bias.data = t(L["bias"]) if L.get("bias") is not None else torch.zeros(w.shape[0])
            mods.append(m)
            shape = (w.shape[0],)
        elif ty == "relu":
            mods.append(nn.ReLU())
        elif ty == "tanh":
            mods.append(nn.Tanh())
        elif ty == "sigmoid":
            mods.append(nn.Sigmoid())
        elif ty == "abs":
            mods.append(Lambda(torch.abs, "abs"))
        elif ty in ("maxpool", "avgpool"):
            s = int(L.get("size", 2))
            mods.append(nn.MaxPool2d(s) if ty == "maxpool" else nn.AvgPool2d(s))
            shape = (shape[0], shape[1] // s, shape[2] // s)
        elif ty == "flatten":
            mods.append(nn.Flatten())
            n = 1
            for d in shape:
                n *= d
            shape = (n,)
        elif ty == "scale":
            v = float(L["value"])
            mods.append(Lambda(lambda x, v=v: x * v, f"scale {v}"))
        elif ty == "shift":
            v = float(L["value"])
            mods.append(Lambda(lambda x, v=v: x + v, f"shift {v}"))
        else:
            raise ValueError(f"layer {i}: unknown type {ty}")
    mods.append(nn.Flatten())
    return nn.Sequential(*mods)


class Combined(nn.Module):
    """Feature files run in parallel, outputs concatenated, then one nn.Linear(k, 10)."""

    def __init__(self, spec: dict, in_shape):
        super().__init__()
        if "features" in spec:
            files = spec["feature_files"]
            self.branches = nn.ModuleList(build_layers(files[f]["layers"], in_shape) for f in spec["features"])
            with torch.no_grad():
                k = sum(b(torch.zeros(1, *in_shape)).shape[1] for b in self.branches)
            w = t(spec["weights"]).reshape(10, k)
            self.head = nn.Linear(k, 10)
            self.head.weight.data = w
            self.head.bias.data = t(spec.get("bias", [0.0] * 10))
        else:  # a plain single-network weight file
            self.branches = nn.ModuleList([build_layers(spec["layers"], in_shape)])
            self.head = nn.Identity()

    def forward(self, x):
        return self.head(torch.cat([b(x) for b in self.branches], 1))


def square_crop_resize(size):
    def f(im: Image.Image):
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return im.resize((size, size), Image.BICUBIC)
    return f


def load(dataset: str, root: str):
    if dataset == "mnist":
        ds = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
        return ds, [str(i) for i in range(10)], (1, 28, 28)
    ds = datasets.Imagenette(root, split="val", size="160px", download=True,
                             transform=transforms.Compose([square_crop_resize(64), transforms.ToTensor()]))
    names = [c[0] if isinstance(c, tuple) else c for c in ds.classes]
    return ds, names, (3, 64, 64)


def evaluate(dataset: str, path: str, root: str):
    ds, names, in_shape = load(dataset, root)
    spec = json.load(open(path, encoding="utf-8"))
    model = Combined(spec, in_shape).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 78}\n{dataset}: {path}")
    print(model)
    print(f"parameters: {n_params:,}")
    if dataset == "imagenette":
        print("torchvision class order:", getattr(ds, "wnids", None) or ds.class_to_idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=500, shuffle=False, num_workers=0)
    cm = torch.zeros(10, 10, dtype=torch.long)
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.double())
            assert out.shape == (len(y), 10), out.shape
            pred = out.argmax(1)
            cm += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
            seen += len(y)
    correct = cm.diag().sum().item()
    print(f"samples evaluated: {seen} (dataset length {len(ds)})")
    print("per-class accuracy:")
    for c in range(10):
        print(f"  {c} {names[c]:<18} {cm[c, c].item() / max(cm[c].sum().item(), 1) * 100:6.2f}%  (n={cm[c].sum().item()})")
    print("confusion matrix (row = true, col = predicted):")
    print("     " + "".join(f"{c:>6}" for c in range(10)))
    for c in range(10):
        print(f"  {c:>2} " + "".join(f"{v:>6}" for v in cm[c].tolist()))
    print(f"TOP-1 ACCURACY: {correct}/{seen} = {correct / seen * 100:.2f}%")
    return correct / seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".audit_data", help="folder for fresh torchvision downloads")
    p.add_argument("pairs", nargs="*", help="dataset path [dataset path ...] (default: the shipped weights)")
    a = p.parse_args()
    a.pairs = a.pairs or DEFAULTS
    if len(a.pairs) % 2:
        sys.exit("give dataset/path pairs")
    for ds, path in zip(a.pairs[::2], a.pairs[1::2]):
        evaluate(ds, path, a.data_root)


if __name__ == "__main__":
    main()
