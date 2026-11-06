"""Datasets for explore mode: MNIST, and Imagenette (real colour photos).

Each dataset gives the model the same kind of information: what the input
tensor looks like, the class names, sample images as text (and optionally as
real pictures), a validation split carved out of the training set for every
measurement, and a test split that is only ever scored.

Imagenette (fast.ai) is 10 easy-to-tell-apart ImageNet classes: tench (a
fish), English springer (a dog), cassette player, chain saw, church, French
horn, garbage truck, gas pump, golf ball, parachute. We use the 160px release
(~95 MB), centre-crop each photo to a square and shrink it to --size pixels
(default 64): big enough that a person can clearly say what each picture is,
unlike CIFAR-10's 32x32. The decoded arrays are cached in data/.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from . import data

DATA_DIR = data.DATA_DIR


class Dataset:
    name: str
    classes: list[str]
    shape: tuple[int, int, int]   # (C, H, W) as the network receives it
    input_note: str               # replaces the MNIST-specific facts in the system prompt
    pixel_formats: dict[str, str]
    default_pixel_format: str

    def train(self) -> tuple[np.ndarray, np.ndarray]: ...
    def test(self) -> tuple[np.ndarray, np.ndarray]: ...
    def grid(self, img: np.ndarray, fmt: str) -> str: ...

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def label(self, c: int) -> str:
        return f"{c} ({self.classes[c]})" if self.classes[c] != str(c) else str(c)

    def validation_split(self, n: int = 5000, seed: int = 0):
        x, y = self.train()
        idx = np.random.default_rng(seed).permutation(len(x))[:n]
        return x[idx], y[idx]

    def sample_ids(self, n: int, seed: int = 0, exclude_val: int = 0) -> list[int]:
        """n training images balanced across classes, nested across n. With
        exclude_val, images in the validation split are never picked."""
        x, y = self.train()
        banned = set()
        if exclude_val:
            banned = set(np.random.default_rng(0).permutation(len(x))[:exclude_val].tolist())
        rng = np.random.default_rng(seed)
        k = self.n_classes
        order = {d: [i for i in rng.permutation(np.flatnonzero(y == d)) if i not in banned] for d in range(k)}
        per = [n // k + (1 if d < n % k else 0) for d in range(k)]
        return [int(i) for d in range(k) for i in order[d][: per[d]]]

    def examples_prompt(self, n: int, fmt: str, seed: int = 0) -> tuple[str, list[int]]:
        x, y = self.train()
        ids = self.sample_ids(n, seed, exclude_val=5000 if self.name != "mnist" else 0)
        parts = [f"Here are {len(ids)} real {self.name} training images ({n // self.n_classes}+ per class). "
                 f"Pixel encoding: {self.pixel_formats[fmt]}."]
        for i in ids:
            parts.append(f"class {self.label(int(y[i]))} (#{i}):\n```\n{self.grid(x[i], fmt)}\n```")
        return "\n\n".join(parts), ids

    def contact_sheets(self, ids: list[int], per_sheet: int = 50, scale: int = 2) -> list[bytes]:
        """PNG pictures of the sample images, labelled '#id class', for models that can see images."""
        from PIL import Image, ImageDraw
        x, y = self.train()
        c, h, w = self.shape
        cell_w, cell_h, cols = w * scale, h * scale + 14, 10
        sheets = []
        for s in range(0, len(ids), per_sheet):
            chunk = ids[s: s + per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
            draw = ImageDraw.Draw(sheet)
            for k, i in enumerate(chunk):
                arr = (np.clip(x[i], 0, 1) * 255).astype(np.uint8)
                arr = np.repeat(arr, 3, 0) if c == 1 else arr
                im = Image.fromarray(arr.transpose(1, 2, 0)).resize((cell_w, h * scale), Image.NEAREST)
                ox, oy = (k % cols) * cell_w, (k // cols) * cell_h
                sheet.paste(im, (ox, oy + 14))
                draw.text((ox + 2, oy + 1), f"#{i} {self.classes[int(y[i])]}"[:22], fill="black")
            buf = io.BytesIO()
            sheet.save(buf, "PNG")
            sheets.append(buf.getvalue())
        return sheets


class MNIST(Dataset):
    name = "mnist"
    classes = [str(d) for d in range(10)]
    shape = (1, 28, 28)
    input_note = ""  # the original MNIST system prompt is used unchanged
    pixel_formats = data.PIXEL_FORMATS
    default_pixel_format = "hex"

    def train(self):
        return data.load("train")

    def test(self):
        return data.load("test")

    def validation_split(self, n: int = 5000, seed: int = 0):
        return data.validation_split(n, seed)  # identical to every earlier MNIST run

    def grid(self, img, fmt):
        return data.pixel_grid(img, fmt)

    def examples_prompt(self, n, fmt, seed=0):
        from . import generate  # the exact text every earlier MNIST run used
        return generate.examples_prompt(n, fmt, seed), generate.sample_ids(n, seed)


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_CLASSES = {  # folder (WordNet id) -> name, in label order
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball", "n03888257": "parachute",
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} (~95 MB) ...")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


class Imagenette(Dataset):
    name = "imagenette"
    classes = list(IMAGENETTE_CLASSES.values())
    default_pixel_format = "rgb16"

    def __init__(self, size: int = 64):
        self.size = size
        self.shape = (3, size, size)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        g = 16
        self.pixel_formats = {
            "rgb16": (f"the photo shrunk to {g}x{g} (each cell is the average of a {size // g}x{size // g} block): "
                      f"{g} rows of {g} cells separated by spaces, each cell 3 hex digits = red, green, blue, "
                      "0 (dark) to f (bright); the network itself sees the full-resolution photo"),
            "rgb32": ("the photo shrunk to 32x32 (each cell is the average of a "
                      f"{size // 32}x{size // 32} block): 32 rows of 32 cells, each 3 hex digits = red, green, "
                      "blue, 0 to f; the network itself sees the full-resolution photo"),
        }
        self.input_note = f"""- The task: classify colour photos into 10 classes:
  {", ".join(f"{i} = {n}" for i, n in enumerate(self.classes))}.
  These are ordinary photos (different backgrounds, angles, lighting, and the
  object can be anywhere in the frame and any size).
- The network receives a batch shaped (N, 3, {size}, {size}): channels are
  red, green, blue, each pixel from 0.0 (dark) to 1.0 (bright). Row 0 is the
  top, column 0 the left. Each photo was centre-cropped to a square and
  shrunk to {size}x{size}, so the conv/dense shapes you write must fit 3 input
  channels at {size}x{size} (an avgpool first is a cheap way to work at a
  lower resolution).
- You may be shown sample photos as text (encoding stated with them), and
  possibly also as pictures."""

    def _load(self, split: str):
        if split in self._cache:
            return self._cache[split]
        cache = DATA_DIR / f"imagenette-{self.size}.npz"
        if not cache.exists():
            self._build(cache)
        d = np.load(cache)
        for s in ("train", "val"):
            self._cache[s] = (d[f"x_{s}"].astype(np.float32) / 255.0, d[f"y_{s}"].astype(np.int64))
        return self._cache[split]

    def _build(self, cache: Path):
        from PIL import Image
        tgz = _download(IMAGENETTE_URL, DATA_DIR / "imagenette2-160.tgz")
        print(f"decoding photos to {self.size}x{self.size} ...")
        out = {}
        tmp = Path(tempfile.mkdtemp(prefix="imagenette-"))
        try:
            with tarfile.open(tgz) as t:
                t.extractall(tmp, filter="data")
            root = next(tmp.glob("imagenette2-160"))
            for split in ("train", "val"):
                xs, ys = [], []
                for label, wnid in enumerate(IMAGENETTE_CLASSES):
                    files = sorted((root / split / wnid).glob("*.JPEG"))
                    if not files:
                        raise RuntimeError(f"no photos for {wnid} in {split}")
                    for f in files:
                        im = Image.open(f).convert("RGB")
                        w, h = im.size
                        s = min(w, h)
                        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
                        im = im.resize((self.size, self.size), Image.BICUBIC)
                        xs.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
                        ys.append(label)
                out[f"x_{split}"], out[f"y_{split}"] = np.stack(xs), np.array(ys, dtype=np.uint8)
                print(f"  {split}: {len(ys)} photos")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        np.savez(cache, **out)

    def train(self):
        return self._load("train")

    def test(self):
        return self._load("val")  # Imagenette's official validation set is our held-out test set

    def grid(self, img, fmt):
        g = 32 if fmt == "rgb32" else 16
        c, h, w = img.shape
        small = img[:, : h // g * g, : w // g * g].reshape(c, g, h // g, g, w // g).mean(axis=(2, 4))
        q = np.clip(np.round(small * 15), 0, 15).astype(int)
        hexd = "0123456789abcdef"
        return "\n".join(" ".join(hexd[q[0, r, k]] + hexd[q[1, r, k]] + hexd[q[2, r, k]] for k in range(g))
                         for r in range(g))


def get(name: str, size: int = 64) -> Dataset:
    if name == "mnist":
        return MNIST()
    if name == "imagenette":
        return Imagenette(size)
    raise ValueError(f"unknown dataset {name!r}")


def main(argv=None):
    """uv run ws-data imagenette  -> download and cache a dataset ahead of time."""
    import argparse
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("dataset", choices=["mnist", "imagenette"])
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args(argv)
    ds = get(a.dataset, a.size)
    (xtr, ytr), (xte, yte) = ds.train(), ds.test()
    print(f"{ds.name}: train {xtr.shape}, test {xte.shape}; per class (train): {np.bincount(ytr).tolist()}")


if __name__ == "__main__":
    main()
