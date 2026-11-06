"""Ablation: how much of the accuracy comes from the LLM's features, and how much from the LDA readout?

Takes the closed-form LDA recipe the model used in its calculator, unchanged, and applies it to
features nobody designed: raw pixels, pooled pixels, and random ReLU projections. It uses the
same 5,000-image training slice the harness measured on and scores on the same test sets.
Every baseline uses the LLM's own shrinkage setting; nothing is tuned.

It is CPU- and memory-heavy (Imagenette arrays are ~0.5 GB), so it pins numpy to one thread.
On a small machine, expect a few minutes.

    uv run ws-data imagenette          # once, if data/imagenette-64.npz is missing
    uv run python audit_ablation.py    # writes docs/data/ablation.json

Recipes, copied from the logged calculator snippets:
  MNIST       runs/explore-20261105-120713/calculator.jsonl, snippet 8
  Imagenette  runs/explore-imagenette-20261105-193149/calculator.jsonl, snippet 10
"""

from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # keep a small machine cool; set before numpy loads

import numpy as np

from weightspace import datasets

ROOT = Path(__file__).resolve().parent


def lda_mnist(F, y, a=0.1):
    """Snippet 8: total covariance minus between-class scatter (equal priors), shrunk 10% to its diagonal."""
    M = np.stack([F[y == k].mean(0) for k in range(10)])
    mu = M.mean(0)
    T = np.cov(F.T, bias=True)
    W = T - sum(0.1 * np.outer(M[k] - mu, M[k] - mu) for k in range(10))
    Ws = (1 - a) * W + a * np.diag(np.diag(W)) + 1e-5 * np.eye(F.shape[1])
    w = np.linalg.solve(Ws, M.T).T
    return w, -0.5 * np.sum(w * M, 1)


def lda_imagenette(F, y, lam=0.006):
    """Snippet 10: the same in standardised units, ridge lam, log class priors."""
    M = np.stack([F[y == k].mean(0) for k in range(10)])
    mean, std = F.mean(0), np.maximum(F.std(0), 1e-9)
    p = np.bincount(y, minlength=10) / len(y)
    C = np.corrcoef(F.T)
    C = np.nan_to_num(C)
    m = (M - mean) / std
    Rw = C - (m.T * p) @ m
    A = np.linalg.solve(Rw + lam * np.eye(F.shape[1]), m.T).T
    W = A / std
    b = -(A * (mean / std)).sum(1) - 0.5 * (A * m).sum(1) + np.log(p)
    return W, b


def acc(w, b, F, y):
    return float(((F @ w.T + b).argmax(1) == y).mean())


def pool(x, g):
    n, c, h, w = x.shape
    return x[:, :, : h // g * g, : w // g * g].reshape(n, c, g, h // g, g, w // g).mean((3, 5)).reshape(n, -1)


def random_relu(dim_in, k, seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((dim_in, k)) / np.sqrt(dim_in)
    c = rng.standard_normal(k) * 0.5
    return lambda X: np.maximum((X - X_MEAN) @ R + c, 0)


def run(name, ds, lda, n_llm, llm_acc):
    global X_MEAN
    xv, yv = ds.validation_split()
    xt, yt = ds.test()
    xv = xv.reshape(len(xv), *ds.shape)
    xt = xt.reshape(len(xt), *ds.shape)
    rows = []

    def add(label, kind, fv, ft):
        w, b = lda(fv, yv)
        a = acc(w, b, ft, yt)
        rows.append({"features": label, "kind": kind, "n_features": int(fv.shape[1]), "test_accuracy": a})
        print(f"  {label:<44} {fv.shape[1]:>5} features  test {a * 100:6.2f}%")

    print(name)
    side = ds.shape[1]
    grids = [side] if name == "mnist" else []
    grids += [7, 4] if name == "mnist" else [16, 8, 4]
    for g in grids:
        lab = "raw pixels" if g == side else f"pixels avg-pooled to {g}x{g}"
        add(lab, "pixels", pool(xv, g), pool(xt, g))
    base_v, base_t = (pool(xv, side), pool(xt, side)) if name == "mnist" else (pool(xv, 16), pool(xt, 16))
    X_MEAN = base_v.mean(0)
    for k in sorted({n_llm, 1000}):
        for s in range(3 if k == n_llm else 1):
            f = random_relu(base_v.shape[1], k, s)
            add(f"{k} random ReLU features (seed {s})", "random", f(base_v), f(base_t))
    rows.append({"features": f"{n_llm} LLM-designed features", "kind": "llm", "n_features": n_llm,
                 "test_accuracy": llm_acc})
    print(f"  {n_llm} LLM-designed features (from the run log)        test {llm_acc * 100:6.2f}%")
    return rows


X_MEAN = None

if __name__ == "__main__":
    out = {
        "mnist": run("mnist", datasets.get("mnist"), lda_mnist, 200, 0.9593),
        "imagenette": run("imagenette", datasets.get("imagenette"), lda_imagenette, 277, 0.6777070063694267),
    }
    p = ROOT / "docs" / "data" / "ablation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)
