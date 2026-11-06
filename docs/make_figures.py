"""Regenerate every figure in README.md from files in this repo.

    uv run --with matplotlib python docs/make_figures.py

Inputs: docs/data/*.json (ablation, cold-room verification, run summary), weights/*.json,
results/*/ (logs and the bank statistics table). Output: docs/img/*.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA, IMG = ROOT / "docs" / "data", ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

# palette (validated reference instance): one highlight hue, neutral baselines
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BLUE_DARK, ORANGE, GRAY_BAR, MID = "#2a78d6", "#184f95", "#eb6834", "#c9c8c2", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#9ec5f4", MID, "#f2a3a2", "#e34948", "#8f1f1e"])
MNIST_CLASSES = [str(i) for i in range(10)]
IMNET_CLASSES = ["tench", "springer", "cassette", "chain saw", "church", "French horn", "garbage truck", "gas pump",
                 "golf ball", "parachute"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def save(fig, name):
    fig.savefig(IMG / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def subtitle(ax, text):
    ax.text(0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=10)


def load(name):
    return json.loads((DATA / name).read_text())


ablation, verify, runs = load("ablation.json"), load("verify.json"), load("all_runs.json")


# 1. hero tiles -------------------------------------------------------------------------------------------
def hero():
    tiles = [("95.93%", "MNIST test top-1", "10,000 images · 6,779 params"),
             ("67.77%", "Imagenette test top-1", "3,925 photos · 5,304 params"),
             ("+9 / +29", "points over random features", "MNIST / Imagenette, same readout"),
             ("0", "gradient steps", "no backprop anywhere")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.3))
    for ax, (big, lab, sub) in zip(axes, tiles):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f4f3f0", edgecolor=GRID,
                                   linewidth=1, clip_on=False))
        ax.text(0.07, 0.62, big, fontsize=27, fontweight="bold", color=BLUE_DARK if big != "0" else INK,
                transform=ax.transAxes, va="center")
        ax.text(0.07, 0.30, lab, fontsize=11, color=INK, transform=ax.transAxes, va="center")
        ax.text(0.07, 0.13, sub, fontsize=9, color=INK2, transform=ax.transAxes, va="center")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "hero.png")


# 2/3. ablation bars --------------------------------------------------------------------------------------
def ablation_chart(ds, title, sub, fname, chance=0.10):
    rows = ablation[ds]
    groups, seen = [], set()
    for r in rows:
        key = r["features"].split(" (seed")[0]
        if key in seen:
            continue
        seen.add(key)
        accs = [x["test_accuracy"] for x in rows if x["features"].split(" (seed")[0] == key]
        groups.append((key + (f"  ({len(accs)} seeds)" if len(accs) > 1 else ""), r["kind"], min(accs), max(accs)))
    groups = groups[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(groups) + 1.6))
    for i, (lab, kind, lo, hi) in enumerate(groups):
        col = BLUE if kind == "llm" else GRAY_BAR
        ax.barh(i, hi, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
        if hi - lo > 1e-6:
            ax.plot([lo, hi], [i, i], color=INK2, lw=1.2)
            ax.plot([lo, lo], [i - 0.15, i + 0.15], color=INK2, lw=1.2)
        txt = f"{hi * 100:.2f}%" if hi - lo < 1e-6 else f"{lo * 100:.1f}–{hi * 100:.1f}%"
        ax.text(hi + 0.008, i, txt, va="center", fontsize=10.5, fontweight="bold" if kind == "llm" else "normal",
                color=INK)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    for t, g in zip(ax.get_yticklabels(), groups):
        if g[1] == "llm":
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.axvline(chance, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(chance + 0.006, -0.62, "chance", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("test top-1 accuracy (same closed-form LDA readout for every row)")
    ax.set_title(title)
    subtitle(ax, sub)
    save(fig, fname)


# 4. harness ladder (MNIST) -------------------------------------------------------------------------------
def ladder():
    by = {r["run"]: r for r in runs}
    steps = [
        (f"LLM writes every\nweight directly\n(best of {sum(r['mode'] in ('direct', 'sweep') for r in runs)} runs)",
         max(r["best_test"] or 0 for r in runs if r["mode"] in ("direct", "sweep")), GRAY_BAR),
        ("+ explore phase:\nper-class neuron\ntelemetry", by["explore-20261105-063218"]["best_test"], GRAY_BAR),
        ("+ calculator\nfor arithmetic", by["explore-20261105-111818"]["best_test"], GRAY_BAR),
        ("+ neuron overlap\nreport (|r| ≥ 0.7)", by["explore-20261105-114214"]["best_test"], GRAY_BAR),
        ("+ full stats table\nin calculator →\nmodel derives LDA", by["explore-20261105-120713"]["best_test"], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, (lab, v, c) in enumerate(steps):
        ax.bar(i, v, width=0.62, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(i, v + 0.012, f"{v * 100:.2f}%", ha="center", fontsize=10.5, fontweight="bold" if c == BLUE else "normal")
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("What moved MNIST from 70% to 96%")
    subtitle(ax, "Best test accuracy as harness features were added, one change at a time. The last step is where LDA enters.")
    save(fig, "ladder.png")


# 5. every run, in order -----------------------------------------------------------------------------------
def journey():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [20, 3]}, sharey=True)
    for ax, ds in zip(axes, ["mnist", "imagenette"]):
        import re
        rs = sorted((r for r in runs if r["dataset"] == ds),
                    key=lambda r: (re.search(r"\d{8}-\d{6}", r["run"]).group(), r["run"]))
        xs = range(1, len(rs) + 1)
        for x, r in zip(xs, rs):
            v = r["best_test"]
            if v is None:
                ax.plot(x, 0.02, marker="x", color=MUTED, ms=8, mew=1.5)
                continue
            lda = r["overlap_table"]
            col = BLUE if lda else (INK2 if r["mode"] == "explore" else GRAY_BAR)
            ax.plot(x, v, marker="o", ms=9, color=col, mec=SURFACE, mew=2, ls="none")
        ax.set_xticks(list(xs), [str(x) for x in xs], fontsize=8.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"{ds} run, in the order it was run")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2), [f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
    axes[0].set_title("All 23 runs, nothing hidden")
    subtitle(axes[0], "Best test accuracy per run. We report the best run per dataset, so these numbers are a best-of.")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", ms=9, color=c, mec=SURFACE, mew=2, label=l) for c, l in
         [(GRAY_BAR, "LLM writes all weights directly"), (INK2, "explore → LLM-written final layer"),
          (BLUE, "explore → closed-form LDA final layer")]] + [
        Line2D([], [], marker="x", ls="none", ms=8, mew=1.5, color=MUTED, label="no valid weights returned")]
    axes[0].legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.62), frameon=False, fontsize=9)
    save(fig, "journey.png")


# 6. combine rounds: pure LDA, then bias tuning ---------------------------------------------------------------
def combine_rounds():
    picks = [("explore-20261105-120713", "MNIST", "Revisions changed only the 10 biases\n(all 2,000 weights untouched)."),
             ("explore-imagenette-20261105-193149", "Imagenette 5k",
              "Revisions re-solved the LDA with class\npriors and shrinkage taken from the\nvalidation confusion matrix."),
             ("explore-imagenette-20261105-231021", "Imagenette 57k",
              "The revision re-solved the LDA on a\ndifferent subset of feature files.")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for ax, (run, lab, note) in zip(axes, picks):
        L = json.loads((ROOT / "results" / run / "log.json").read_text())
        rs = [r for r in L["rounds"] if r.get("val")]
        x = list(range(1, len(rs) + 1))
        v = [r["val"]["accuracy"] for r in rs]
        t = [r["test_accuracy"] for r in rs]
        ax.plot(x, v, color=MUTED, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="validation (training images)")
        ax.plot(x, t, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="test (never shown)")
        for xi, ti in zip(x, t):
            ax.text(xi, ti - 0.006, f"{ti * 100:.2f}", ha="center", va="top", fontsize=9, color=INK)
        ax.set_xticks(x, ["first\nLDA"] + [f"revision\n{i}" for i in range(1, len(x))], fontsize=9)
        lo = min(t + v)
        ax.set_ylim(lo - 0.03, max(t + v) + 0.012)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(lab, fontsize=11.5)
        ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8.6, color=INK2, va="top", wrap=True)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Combine rounds: a closed-form LDA readout, then revisions driven by validation feedback", x=0.06,
                 ha="left", fontweight="bold", fontsize=13, y=1.04)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "combine_rounds.png")


# 7. confusion matrices -------------------------------------------------------------------------------------
def confusion(rec, classes, title, fname):
    cm = np.array(rec["confusion"], dtype=float)
    rowp = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(rowp, cmap=SEQ, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            n = int(cm[i, j])
            if n:
                ax.text(j, i, str(n), ha="center", va="center", fontsize=8.5,
                        color="white" if rowp[i, j] > 0.55 else (INK if rowp[i, j] > 0.02 else MUTED))
    ax.set_xticks(range(10), classes, rotation=40 if len(classes[0]) > 1 or len(classes[-1]) > 1 else 0, ha="right" if len(classes[-1]) > 1 else "center")
    ax.set_yticks(range(10), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 10), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("share of the true class", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title)
    subtitle(ax, f"Cold-room PyTorch run, all {rec['n']:,} test images · {rec['correct']:,} correct "
                 f"({rec['correct'] / rec['n'] * 100:.2f}%) · counts in cells")
    save(fig, fname)


# 8. per-class accuracy --------------------------------------------------------------------------------------
def per_class():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, rec, classes, lab in [(axes[0], verify[0], MNIST_CLASSES, "MNIST (6,779 params)"),
                                  (axes[1], verify[1], IMNET_CLASSES, "Imagenette (5,304 params)")]:
        pc = np.array(rec["per_class"]) / 100
        ax.bar(range(10), pc, width=0.66, color=BLUE, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(pc):
            ax.text(i, v + 0.012, f"{v * 100:.0f}", ha="center", fontsize=9)
        overall = rec["correct"] / rec["n"]
        ax.axhline(overall, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(10), classes, rotation=0 if classes is MNIST_CLASSES else 35,
                      ha="center" if classes is MNIST_CLASSES else "right", fontsize=9.5)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{lab} · dashed line = overall {overall * 100:.2f}%", fontsize=11.5)
    fig.suptitle("Per-class test accuracy", x=0.06, ha="left", fontweight="bold", fontsize=13, y=1.03)
    save(fig, "per_class.png")


# 9. the LLM's own MNIST kernels -----------------------------------------------------------------------------
def kernels_mnist():
    spec = json.loads((ROOT / "weights" / "mnist_95.93.json").read_text())["feature_files"]
    picks = []
    for fid, names in [("R2F1", None), ("R3F3", None), ("R4F1", None), ("R2F3", None)]:
        L = next(l for l in spec[fid]["layers"] if l["type"] == "conv2d")
        W = np.array(L["weights"])
        for k in range(W.shape[0]):
            picks.append((f"{fid} · {L.get('names', [str(k)] * 99)[k]}", W[k, 0]))
    L = next(l for l in spec["R3F1"]["layers"] if l["type"] == "conv2d")
    W = np.array(L["weights"])
    for k in range(8):
        picks.append((f"R3F1 · hole arm {k}", W[k, 0]))
    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.25 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (lab, k) in zip(axes.flat, picks):
        m = np.abs(k).max() or 1
        ax.imshow(k, cmap=DIV, vmin=-m, vmax=m)
        ax.set_title(lab, fontsize=9, fontweight="normal", loc="center", pad=4)
        for (i, j), v in np.ndenumerate(k):
            if k.shape[0] <= 5:
                ax.text(j, i, f"{v:g}" if abs(v) >= 0.01 else "", ha="center", va="center", fontsize=6.3,
                        color="white" if abs(v) > 0.6 * m else INK)
    fig.suptitle("Conv kernels the LLM typed in by hand (MNIST)", x=0.02, ha="left", fontweight="bold", fontsize=13, y=1.02)
    fig.text(0.02, 0.955, "Stroke detectors (horizontal, vertical, \\ and /), endpoint and ink detectors, and the 9×9 arms of the "
             "hole detector. Blue = negative, red = positive. No number here was fitted.", color=INK2, fontsize=10)
    fig.subplots_adjust(top=0.87, hspace=0.35)
    save(fig, "kernels_mnist.png")


# 10. the LLM's own Imagenette kernels (RGB) ------------------------------------------------------------------
def kernels_imagenette():
    spec = json.loads((ROOT / "weights" / "imagenette_67.77_5k-params.json").read_text())["feature_files"]
    picks = []
    for fid in ["R1F2", "R4F1", "R3F1"]:
        L = next(l for l in spec[fid]["layers"] if l["type"] == "conv2d")
        W = np.array(L["weights"])
        names = L.get("names") or [str(i) for i in range(W.shape[0])]
        for k in range(W.shape[0]):
            picks.append((fid, names[k], W[k]))
    picks = picks[:24]
    cols = 8
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.0 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (fid, name, w) in zip(axes.flat, picks):
        m = np.abs(w).max() or 1
        rgb = np.clip(0.5 + 0.5 * w.transpose(1, 2, 0) / m, 0, 1)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{fid}\n{name}"[:34], fontsize=8.3, fontweight="normal", loc="center", pad=3)
    fig.suptitle("RGB kernels the LLM wrote for photos (Imagenette)", x=0.02, ha="left", fontweight="bold", fontsize=13, y=1.04)
    fig.text(0.02, 0.975, "Each 3×3 kernel drawn as colour: grey = 0, brighter channel = positive weight on that colour. "
             "Edge, blob, golf-ball-disc and red/yellow-blob detectors.", color=INK2, fontsize=10)
    fig.subplots_adjust(top=0.86, hspace=0.6)
    save(fig, "kernels_imagenette.png")


# 11. the telemetry the LLM saw ---------------------------------------------------------------------------------
def telemetry():
    d = np.load(ROOT / "results" / "explore-20261105-120713" / "bank_table.npz")
    M, mean, std = d["means"], d["mean"], np.maximum(d["std"], 1e-9)
    Z = (M - mean) / std
    ids = [str(x) for x in d["neuron_ids"]]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(Z, cmap=DIV, vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(10), [f"digit {i}" for i in range(10)], fontsize=9)
    starts = [i for i, n in enumerate(ids) if n.endswith(".0")]
    ax.set_xticks(starts, [ids[i].split(".")[0] for i in starts], fontsize=8.5)
    for s in starts[1:]:
        ax.axvline(s - 0.5, color=SURFACE, lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("class mean, in std units", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("200 neurons, grouped by the feature file that defines them")
    ax.set_title("The telemetry the LLM worked from (MNIST)")
    subtitle(ax, "Each neuron's mean activation per digit over 5,000 labelled training images: the main training-data "
                 "signal in the loop.")
    save(fig, "telemetry_mnist.png")


# 12. the LDA head -------------------------------------------------------------------------------------------------
def lda_head():
    b = json.loads((ROOT / "weights" / "mnist_95.93.json").read_text())
    W = np.array(b["weights"]).reshape(10, -1)
    m = np.percentile(np.abs(W), 98)
    fig, ax = plt.subplots(figsize=(13, 3.3))
    im = ax.imshow(W, cmap=DIV, vmin=-m, vmax=m, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(10), [f"digit {i}" for i in range(10)], fontsize=9)
    ax.set_xlabel("weight on each of the 200 neurons")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    ax.set_title("The 10 × 200 readout, from np.linalg.solve (MNIST)")
    subtitle(ax, f"Weights range from {W.min():.0f} to {W.max():.0f} and largely cancel each other: the signature of "
                 "an inverted covariance, not hand-picked numbers.")
    save(fig, "lda_head_mnist.png")


# 13. parameters vs accuracy -------------------------------------------------------------------------------------
def params_chart():
    pts = [(v["params"], v["correct"] / v["n"], v["dataset"], v["run"]) for v in verify]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for p, a, ds, run in pts:
        ax.plot(p, a, marker="o", ms=11, color=BLUE if ds == "mnist" else ORANGE, mec=SURFACE, mew=2, ls="none")
        ax.annotate(f"{a * 100:.2f}%  ·  {p:,} params", (p, a), xytext=(10, -4), textcoords="offset points", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(2500, 3e5)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("parameters (log scale)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=10, color=BLUE, label="MNIST"),
                       Line2D([], [], marker="o", ls="none", ms=10, color=ORANGE, label="Imagenette")],
              frameon=False, loc="lower right")
    ax.set_title("Tiny networks")
    subtitle(ax, "Every verified model. More Imagenette parameters (deeper, layered features) did not buy accuracy.")
    save(fig, "params.png")


if __name__ == "__main__":
    hero()
    ablation_chart("mnist", "MNIST: the LLM's features vs. features nobody designed",
                   "Same 5,000 training images, same closed-form LDA recipe the model used. Only the features change.",
                   "ablation_mnist.png")
    ablation_chart("imagenette", "Imagenette: the gap is much larger on real photos",
                   "Same 5,000 training photos, same LDA recipe. Random features and raw pixels stall near 40%.",
                   "ablation_imagenette.png")
    ladder()
    journey()
    combine_rounds()
    confusion(verify[0], MNIST_CLASSES, "MNIST confusion matrix", "cm_mnist.png")
    confusion(verify[1], IMNET_CLASSES, "Imagenette confusion matrix (5,304-param model)", "cm_imagenette.png")
    per_class()
    kernels_mnist()
    kernels_imagenette()
    telemetry()
    lda_head()
    params_chart()
