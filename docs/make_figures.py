"""Regenerate the figures in README.md.

    uv run --with matplotlib python docs/make_figures.py

Reads docs/data/*.json, weights/*.json and results/*/, writes docs/img/*.png.
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, IMG = ROOT / "docs" / "data", ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

ablation = json.loads((DATA / "ablation.json").read_text())
verify = json.loads((DATA / "verify.json").read_text())
runs = json.loads((DATA / "all_runs.json").read_text())
IMNET = ["tench", "springer", "cassette", "chain saw", "church", "horn", "truck", "gas pump", "golf ball", "parachute"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def ablation_fig():
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, ds, title in [(axes[0], "mnist", "MNIST"), (axes[1], "imagenette", "Imagenette")]:
        rows = ablation[ds]
        groups = {}
        for r in rows:
            groups.setdefault(r["features"].split(" (seed")[0], []).append(r["test_accuracy"] * 100)
        names = list(groups)
        labels = [n.replace("pixels avg-pooled to ", "pixels pooled ").replace(" ReLU features", " ReLU feats")
                  .replace("LLM-designed features", "LLM features") for n in names]
        y = np.arange(len(names))[::-1]
        for yi, n in zip(y, names):
            v = groups[n]
            c = "C0" if "LLM" in n else "0.65"
            ax.barh(yi, np.mean(v), color=c, height=0.6,
                    xerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]] if len(v) > 1 else None, capsize=2)
            ax.text(np.mean(v) + 1, yi, f"{np.mean(v):.1f}", va="center", fontsize=8)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 105)
        ax.axvline(10, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("test accuracy (%)")
        ax.set_title(title)
    save(fig, "ablation.png")


def runs_fig():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    mn = sorted((r for r in runs if r["dataset"] == "mnist"),
                key=lambda r: re.search(r"\d{8}-\d{6}", r["run"]).group())
    style = {"direct": ("o", "0.6", "writes all weights directly"),
             "explore": ("s", "0.2", "explore + hand-written readout"),
             "lda": ("D", "C0", "explore + LDA readout")}
    seen = set()
    for i, r in enumerate(mn, 1):
        if r["best_test"] is None:
            continue
        k = "lda" if r["overlap_table"] else ("explore" if r["mode"] == "explore" else "direct")
        m, c, lab = style[k]
        ax.plot(i, r["best_test"] * 100, m, color=c, ms=5, label=None if k in seen else lab)
        seen.add(k)
    ax.set_xlabel("MNIST run (chronological)")
    ax.set_ylabel("best test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, len(mn) + 1))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "runs.png")


def kernels_mnist():
    spec = json.loads((ROOT / "weights" / "mnist_95.93.json").read_text())["feature_files"]
    def conv(fid):
        L = next(l for l in spec[fid]["layers"] if l["type"] == "conv2d")
        return np.array(L["weights"]), L.get("names")
    W, names = conv("R2F1")
    picks = [(names[k], W[k, 0]) for k in range(4)]
    W, names = conv("R2F3")
    picks += [(names[0], W[0, 0])]
    W, _ = conv("R3F1")
    picks += [("hole arm (up)", W[0, 0]), ("hole arm (left)", W[4, 0])]
    fig, axes = plt.subplots(1, len(picks), figsize=(10, 1.9))
    for ax, (lab, k) in zip(axes, picks):
        m = np.abs(k).max()
        ax.imshow(k, cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set_title(lab.replace("diag_", ""), fontsize=8)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if k.shape[0] <= 5:
            for (i, j), v in np.ndenumerate(k):
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 * m else "black")
    save(fig, "kernels_mnist.png")


def kernels_imagenette():
    spec = json.loads((ROOT / "weights" / "imagenette_67.77_5k-params.json").read_text())["feature_files"]
    picks = []
    for fid in ["R1F2", "R4F1"]:
        L = next(l for l in spec[fid]["layers"] if l["type"] == "conv2d")
        W = np.array(L["weights"])
        picks += [(n, W[k]) for k, n in enumerate(L["names"])]
    picks = picks[:16]
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for ax, (lab, w) in zip(axes.flat, picks):
        m = np.abs(w).max()
        ax.imshow(np.clip(0.5 + 0.5 * w.transpose(1, 2, 0) / m, 0, 1), interpolation="nearest")
        ax.set_title(lab.replace("_", " ")[:16], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "kernels_imagenette.png")


def telemetry():
    d = np.load(ROOT / "results" / "explore-20261105-120713" / "bank_table.npz")
    Z = (d["means"] - d["mean"]) / np.maximum(d["std"], 1e-9)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(10))
    ax.set_ylabel("digit")
    ax.set_xlabel("neuron")
    fig.colorbar(im, ax=ax, pad=0.01, label="class mean (std units)")
    save(fig, "telemetry_mnist.png")


def lda_head():
    W = np.array(json.loads((ROOT / "weights" / "mnist_95.93.json").read_text())["weights"]).reshape(10, -1)
    m = np.percentile(np.abs(W), 98)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(10))
    ax.set_ylabel("digit")
    ax.set_xlabel("neuron")
    fig.colorbar(im, ax=ax, pad=0.01, label="weight")
    save(fig, "lda_head_mnist.png")


def confusion():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, rec, classes, title in [(axes[0], verify[0], [str(i) for i in range(10)], "MNIST"),
                                    (axes[1], verify[1], IMNET, "Imagenette (5k params)")]:
        cm = np.array(rec["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=6,
                        color="white" if v / cm[i].sum() > 0.5 else "black")
        ax.set_xticks(range(10), classes, rotation=90 if len(classes[0]) > 1 else 0)
        ax.set_yticks(range(10), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}: {rec['correct']}/{rec['n']}")
    save(fig, "confusion.png")


if __name__ == "__main__":
    for f in IMG.glob("*.png"):
        f.unlink()
    ablation_fig()
    runs_fig()
    kernels_mnist()
    kernels_imagenette()
    telemetry()
    lda_head()
    confusion()
