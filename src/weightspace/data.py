"""MNIST download + parsing. Used only for *evaluating* LLM-written weights
(and, optionally, for showing the LLM a handful of example digits)."""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    if dest.exists():
        return dest
    last_err = None
    for mirror in MIRRORS:
        try:
            print(f"downloading {mirror}{name} ...")
            urllib.request.urlretrieve(mirror + name, dest)
            return dest
        except Exception as e:  # try next mirror
            last_err = e
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {name}: {last_err}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Returns (images float32 (N,28,28) in [0,1], labels int64 (N,))."""
    x = _read_idx(_download(FILES[f"{split}_images"])).astype(np.float32) / 255.0
    y = _read_idx(_download(FILES[f"{split}_labels"])).astype(np.int64)
    return x, y


def validation_split(n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A fixed slice of the *training* set, used for feedback loops so the test
    set is never used to steer the LLM."""
    x, y = load("train")
    idx = np.random.default_rng(seed).permutation(len(x))[:n]
    return x[idx], y[idx]


PIXEL_FORMATS = {
    "int": "28 rows of 28 space-separated integers, 0 (background) to 255 (ink); the network sees value/255",
    "hex": "28 rows of 28 characters, one per pixel, hex digit 0 (background) to f (ink); "
           "the network sees roughly digit/15",
}


def pixel_grid(img: np.ndarray, fmt: str = "int") -> str:
    """Every pixel of a 28x28 image as text. 'int' is exact (0-255); 'hex' is
    one character per pixel (16 levels), ~2.4x shorter so many images fit."""
    if fmt == "hex":
        return "\n".join("".join("0123456789abcdef"[int(round(v * 15))] for v in row) for row in img)
    return "\n".join(" ".join(str(int(round(v * 255))) for v in row) for row in img)


def ascii_digit(img: np.ndarray) -> str:
    """Render a 28x28 image as text, for showing examples to an LLM."""
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[min(int(v * 9.999), 9)] for v in row) for row in img)
