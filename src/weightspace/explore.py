"""Explore, then combine: Claude first works out what individual neurons do,
then writes a classifier from what it learned.

  explore rounds (--explore E, default 4)
      Claude writes "feature files": small networks whose outputs are named
      neurons, not 10 digit scores. Each neuron is run on the 5,000-image
      validation split and Claude gets back, per neuron: its mean activation
      for each digit, how spread out it is, how often it fires, and which
      digits make up its 50 strongest responses. Every neuron goes into a
      bank that persists across rounds. Claude keeps, fixes or replaces them.
      No classifier is built or scored in these rounds.
  combine rounds (--combine C, default 1)
      Claude picks feature files from the bank and writes the final layer:
      10 x (number of chosen neurons) weights + 10 biases. That network
      (Claude's feature files + Claude's final layer) is scored like any other.

Every weight is still written by Claude; the program only runs the network
and reports measurements. Nothing is trained or fitted. The test set is never
shown; the reported result is the combine round with the best validation score.

    uv run ws-explore                      # 100 samples, 4 explore + 3 combine
    uv run ws-explore --explore 3 --combine 2
    uv run ws-explore --dataset imagenette --sheets --calculator --overlap --overlap-table
    uv run ws-explore --backend deepseek ...   # DeepSeek API instead of Claude (your DeepSeek key)
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from . import backends, datasets
from .evaluate import format_report, metrics
from .generate import ROOT, LiveRound, build_parser as base_parser
from .network import Network, __doc__ as FORMAT_DOC

MAX_FILES = 3          # feature files per explore round
MAX_NEURONS = 40       # neurons per feature file
TOP_K = 50             # strongest responses summarised per neuron
OVERLAP_MIN = 0.7      # --overlap: report neuron pairs whose activations correlate at least this much
OVERLAP_MAX = 4        # --overlap: partners listed per neuron
OVERLAP_NOTE = f"""
Overlap measurements: with every measurement table you also get OVERLAP, the
other neurons in your bank whose activations move together with each neuron
across the 5,000 images (correlation |r| >= {OVERLAP_MIN}, strongest first, up to
{OVERLAP_MAX} per neuron).
"""

SYSTEM = f"""You are doing weight-space programming. You will write, by hand, every
weight of a small neural network that classifies MNIST handwritten digits.

Rules:
- There is no training. Nobody will run gradient descent on what you write.
- You have no tools and cannot run code. Every number in your answer is one
  you chose yourself by reasoning about what it should do.
- You may be shown real MNIST digits pixel by pixel (the encoding is stated
  with them). The network receives each pixel as a value from 0.0
  (background) to 1.0 (ink). Row 0 is the top, column 0 the left.
- Each digit is size-normalised into a 20x20 box and centred by centre of
  mass, so roughly pixels 4..23 hold the ink. Strokes are about 2-3 px wide.
- Set "author" to the name of the model writing the weights.

The network runtime and weight-file format:
{FORMAT_DOC}

How the process works. It has two phases.

1. EXPLORE rounds. Your job here is NOT to build a classifier. It is to find
   out exactly what individual neurons do on real data. Each round you submit
   up to {MAX_FILES} FEATURE FILES. A feature file uses the same format as above,
   but its last layer must output a flat vector of at most {MAX_NEURONS} values
   (one per neuron) instead of 10 digit scores, and it must list one label per
   neuron in "neurons": ["what this neuron should detect", ...]. Example:
   conv2d with K named filters -> relu -> maxpool with size equal to the map
   size -> flatten gives K neurons; or a dense layer straight from pixels.
   Every file is run on 5,000 validation images, and for every neuron you get
   back its mean activation per digit, its spread, how often it fires, and
   which digits caused its {TOP_K} strongest responses. Compare that with what
   you predicted. Every file goes into your BANK (id like "R2F1") and stays
   there; later rounds can refine neurons that misbehaved, replace dead or
   useless ones, and target digits the bank cannot yet tell apart.

2. COMBINE round(s). You write the final classifier from your bank, in this
   format (one ```json block):
   {{"name": "...", "author": "...", "notes": "...",
     "features": ["R1F1", "R3F2"],
     "weights": [[...], ... 10 rows ...],
     "bias": [10 numbers]}}
   The chosen feature files run unchanged; their neurons are concatenated in
   the order listed (each file's neurons in its own order). Row d of
   "weights" holds digit d's weight for every one of those neurons; the
   predicted digit is the argmax of weights . neurons + bias. Use the
   measured per-digit means and spreads, not just your intentions. After
   every combine round except the last you get its accuracy, confusion
   matrix and a breakdown of mistakes, so you can revise the final layer.

Each message tells you the round, the phase, and how many rounds are left.
Answer with a short summary of what you are testing or building and why,
then the JSON. The JSON must be strict (no comments, no trailing commas, no
expressions like 1/3 - write 0.333).
"""

def system_for(ds: datasets.Dataset) -> str:
    """MNIST keeps its original prompt word for word. Other datasets swap the
    MNIST facts for the dataset's own description and 'digit' for 'class'."""
    if ds.name == "mnist":
        return SYSTEM
    head, rest = SYSTEM.split("Rules:", 1)
    rules, rest = rest.split("The network runtime and weight-file format:", 1)
    head = head.replace("classifies MNIST handwritten digits.", f"classifies {ds.name} photos into 10 classes.")
    keep = [b for b in rules.strip().split("\n- ") if b.lstrip("- ").startswith(("There is no training",
                                                                                 "You have no tools", "Set \"author\""))]
    rules = "\n- " + "\n- ".join(b.lstrip("- ") for b in keep) + "\n" + ds.input_note + "\n\n"
    rest = (rest.replace("mean activation per digit", "mean activation per class")
                .replace("which digits caused", "which classes caused")
                .replace("target digits the bank", "target classes the bank")
                .replace("10 digit scores", "10 class scores")
                .replace("Row d of", "Row c of").replace("digit d's weight", "class c's weight")
                .replace("the\n   predicted digit is", "the\n   predicted class is")
                .replace("per-digit means", "per-class means")
                .replace("The input to the first layer is a batch of images shaped (N, 1, 28, 28) with\n"
                         "pixel values in [0, 1] (0 = background, 1 = ink). The last layer must produce\n"
                         "10 scores; the predicted digit is the argmax.",
                         f"The input to the first layer is a batch shaped (N, {', '.join(map(str, ds.shape))}) "
                         "(see the rules above).\nThe last layer must produce 10 scores; the predicted class "
                         "is the argmax."))
    assert "digit" not in rest.replace('"names": ["ink"', ""), "MNIST wording left in the prompt"
    return head + "Rules:" + rules + "The network runtime and weight-file format:" + rest


CALC_NOTE = """
You have one tool: a calculator (`python`). It runs a short Python snippet
(numpy, math and statistics available) and returns what it prints. It cannot
read files and cannot see any images or datasets: it only knows numbers you
copy into the code from this conversation, plus any variables a round message
says are preloaded. Use it for arithmetic, such as
working out weights from the measurement tables or checking sums. Do not use
it to run training or iterative optimisation (no gradient descent, no
fit-by-search loops); every weight must follow from reasoning or a formula you
can state. Every snippet is logged. Your final message must contain the
complete answer (summary and JSON) - text from earlier tool-using steps is not
read.
"""


LAYERED_NOTE = """
Building in layers. Three things help you build deeper features step by step:

- Stacking: a feature file may start from an earlier bank file with
  "base": "R2F1" and "base_layers": k. The first k layers of that file
  (exactly as stored in the bank, its own base included) run first, and this
  file's "layers" continue from their output. So you can keep a layer of
  detectors you have checked and build the next layer on its output maps,
  without re-writing them. With every measured file you get the output shape
  after each of its layers, so you know what the next layer receives.
- Saved arrays: in the calculator, keep(name, array) saves an array and
  kept(name) reads it back in a later snippet. In a feature file, any
  "weights" or "bias" may be the string "@name" instead of a nested list; it
  is replaced by that kept array (reshaped like a written list would be). So
  you can build large weight tensors from a formula you state, then
  reference them by name. The same rule applies: no training or fitting to
  data; every array must come from a formula you could explain.
- Strongest responses as pictures: after each explore round you also get one
  picture per new file. For each neuron, it shows the validation photos that
  made that neuron fire hardest (one row per neuron, strongest first, with
  their class names), so you can see what each neuron actually responds to.
"""


def resolve_spec(spec: dict, bank: dict[str, dict], store: Path | None) -> dict:
    """Expand "base"/"base_layers" and "@name" arrays into a plain weight file."""
    spec = json.loads(json.dumps(spec))
    def arr(v):
        if isinstance(v, str) and v.startswith("@"):
            f = (store / (v[1:] + ".npy")) if store else None
            if not f or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", v[1:]) or not f.exists():
                raise ValueError(f"no kept array {v!r} (save it in the calculator with keep({v[1:]!r}, array))")
            return np.load(f).tolist()
        return v
    for layer in spec.get("layers") or []:
        for key in ("weights", "bias"):
            if key in layer:
                layer[key] = arr(layer[key])
    if "base" in spec:
        bid = spec.pop("base")
        if bid not in bank:
            raise ValueError(f'"base" {bid!r} is not in the bank')
        base_layers = bank[bid]["spec"]["layers"]
        k = int(spec.pop("base_layers", len(base_layers)))
        if not 0 < k <= len(base_layers):
            raise ValueError(f'"base_layers" must be 1..{len(base_layers)} for {bid}')
        spec["layers"] = json.loads(json.dumps(base_layers[:k])) + (spec.get("layers") or [])
        spec["built_on"] = f"{bid} (first {k} layers)"
    return spec


def layer_shapes(net: Network, x: np.ndarray) -> str:
    _, inter = net.forward(x[:1], return_intermediates=True, check_output=False)
    return "; ".join(f"{i} {l.type} -> {tuple(v.shape[1:])}" for i, (l, v) in enumerate(zip(net.layers, inter)))


def top_sheet(acts: np.ndarray, x: np.ndarray, y: np.ndarray, names: list[str], classes: list[str],
              title: str, per: int = 6, px: int = 40) -> bytes:
    """One row per neuron: the `per` validation images it responds to most."""
    from PIL import Image, ImageDraw
    import io
    lab = 230
    rows = acts.shape[1]
    sheet = Image.new("RGB", (lab + per * (px + 2), 16 + rows * (px + 12)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 2), title[:90], fill="black")
    for j in range(rows):
        oy = 16 + j * (px + 12)
        d.text((2, oy + px // 2 - 6), f"{j}: {names[j]}"[:38], fill="black")
        for k, i in enumerate(np.argsort(-acts[:, j])[:per]):
            a = (np.clip(x[i], 0, 1) * 255).astype(np.uint8)
            a = np.repeat(a, 3, 0) if a.shape[0] == 1 else a
            im = Image.fromarray(a.transpose(1, 2, 0)).resize((px, px), Image.BILINEAR)
            ox = lab + k * (px + 2)
            sheet.paste(im, (ox, oy))
            d.text((ox, oy + px), classes[int(y[i])][:7], fill="gray")
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def feature_outputs(net: Network, x: np.ndarray, batch: int | None = None) -> np.ndarray:
    # conv windows are materialised per batch, so bigger images get smaller batches (MNIST stays at 2000)
    batch = batch or min(2000, max(16, int(1.6e6 // x[0].size)))
    out = [net.forward(x[i: i + batch], check_output=False) for i in range(0, len(x), batch)]
    out = np.concatenate([o.reshape(o.shape[0], -1) for o in out])
    return out


def check_feature_file(spec: dict, x: np.ndarray) -> tuple[Network, list[str]]:
    net = Network.from_dict(spec)
    k = feature_outputs(net, x[:4]).shape[1]
    if k > MAX_NEURONS:
        raise ValueError(f"outputs {k} neurons; the limit is {MAX_NEURONS} per file")
    names = [str(n) for n in spec.get("neurons") or []]
    names += [f"neuron {j}" for j in range(len(names), k)]
    return net, names[:k]


def neuron_stats(acts: np.ndarray, y: np.ndarray) -> list[dict]:
    """Per neuron: mean per digit, overall std, fraction > 0, digits of top-K responses."""
    stats = []
    for j in range(acts.shape[1]):
        a = acts[:, j]
        top = np.argsort(-a)[:TOP_K]
        counts = np.bincount(y[top], minlength=10)
        stats.append({"mean_per_digit": [float(a[y == d].mean()) for d in range(10)],
                      "std": float(a.std()), "min": float(a.min()), "max": float(a.max()),
                      "active": float((a > 1e-9).mean()),
                      "top_digits": {int(d): int(c) for d, c in enumerate(counts) if c}})
    return stats


def fmt_num(v: float) -> str:
    return f"{v:7.2f}" if abs(v) < 1000 else f"{v:7.0f}"


def bank_table(entries: list[dict], noun: str = "digit") -> str:
    """One row per neuron across the given bank files."""
    head = ("neuron              " + "".join(f"{'mean|' + str(d):>8}" for d in range(10))
            + f"     std  active  top-50 {'digits ' if noun == 'digit' else 'classes'}  label")
    lines = [head]
    for e in entries:
        for j, (name, s) in enumerate(zip(e["names"], e["stats"])):
            top = " ".join(f"{d}:{c}" for d, c in sorted(s["top_digits"].items(), key=lambda t: -t[1]))
            lines.append(f"{e['id'] + '.' + str(j):<20}" + "".join(f" {fmt_num(v)}" for v in s["mean_per_digit"])
                         + f" {fmt_num(s['std'])}  {s['active']*100:5.1f}%  {top:<15}  {name}")
    return "\n".join(lines)


PAIRS = 5        # most-confused digit pairs broken down per combine round
PAIR_NEURONS = 6  # neurons listed per breakdown


def combine_mistakes(net: "Combined", x: np.ndarray, y: np.ndarray, m: int, seed: int, fmt: str,
                     ds: datasets.Dataset) -> str:
    """Explain the final layer's mistakes in terms of its own neurons.
    For true digit t predicted as p, each neuron i adds (w[t,i] - w[p,i]) * value_i to
    score[t] - score[p]; negative = it pushed toward the wrong digit. These terms plus
    the bias gap sum exactly to the score gap."""
    f = net.features(x)
    s = f @ net.w.T + net.b
    pred = s.argmax(1)
    wrong = np.flatnonzero(pred != y)
    if len(wrong) == 0:
        return "No mistakes on the validation split."
    nid = lambda i: f"{net.ids[i]} ({net.labels[i]})"
    out = []
    # 1) Most-confused pairs: average per-neuron contribution over all those mistakes.
    pairs = {}
    for i in wrong:
        pairs.setdefault((int(y[i]), int(pred[i])), []).append(i)
    top = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:PAIRS]
    blocks = []
    for (t, p), idx in top:
        c = ((net.w[t] - net.w[p]) * f[idx]).mean(0)
        gap = (s[idx, t] - s[idx, p]).mean()
        worst = np.argsort(c)[:PAIR_NEURONS]
        good = np.argsort(-c)[:3]
        lines = [f"{ds.label(t)} predicted as {ds.label(p)}: {len(idx)} images. Average score[{t}] - score[{p}] = {gap:+.3f} "
                 f"(sum of neuron terms {c.sum():+.3f}, bias gap {net.b[t] - net.b[p]:+.3f})",
                 f"  neurons pushing toward {p} (average (w[{t}]-w[{p}]) x value):"]
        lines += [f"    {nid(k)}: value {f[idx, k].mean():.3f}, w[{t}] {net.w[t, k]:+.3f}, "
                  f"w[{p}] {net.w[p, k]:+.3f} -> {c[k]:+.3f}" for k in worst if c[k] < 0]
        lines.append(f"  neurons pushing toward {t}: " + "; ".join(f"{nid(k)} {c[k]:+.3f}" for k in good if c[k] > 0))
        blocks.append("\n".join(lines))
    out.append("Your most frequent confusions, explained through your final layer (averaged over all "
               "validation images with that confusion; the terms plus the bias gap sum to the score gap):\n```\n"
               + "\n\n".join(blocks) + "\n```")
    # 2) A few individual mistakes with their pixels.
    rng = np.random.default_rng(seed)
    picked = list(rng.choice(wrong, size=min(m, len(wrong)), replace=False))
    singles = []
    for i in picked:
        t, p = int(y[i]), int(pred[i])
        c = (net.w[t] - net.w[p]) * f[i]
        sc = ", ".join(f"{d}: {v:.2f}" for d, v in enumerate(s[i]))
        singles.append(f"true {ds.label(t)}, predicted {ds.label(p)}\nscores [{sc}]\npixels ({ds.pixel_formats[fmt]}):\n```\n"
                       f"{ds.grid(x[i], fmt)}\n```\nbiggest terms toward {p}: "
                       + "; ".join(f"{nid(k)} value {f[i, k]:.2f} -> {c[k]:+.3f}" for k in np.argsort(c)[:5] if c[k] < 0))
    out.append(f"{len(picked)} individual mistakes:\n\n" + "\n\n".join(singles))
    return "\n\n".join(out)


def overlap_lines(targets: list[dict], bank: dict[str, dict]) -> str:
    """For each neuron in `targets`, the bank neurons it correlates with most
    (|r| >= OVERLAP_MIN) across the validation images, any round included."""
    ids = [f"{e['id']}.{j}" for e in bank.values() for j in range(len(e["names"]))]
    A = np.concatenate([e["acts"] for e in bank.values()], axis=1).astype(np.float64)
    A = A - A.mean(0)
    norm = np.sqrt((A ** 2).sum(0))
    live = norm > 1e-9
    lines = []
    for e in targets:
        for j in range(len(e["names"])):
            me = f"{e['id']}.{j}"
            k = ids.index(me)
            if not live[k]:
                lines.append(f"{me}: never varies (constant output)")
                continue
            r = (A[:, k] @ A) / (norm[k] * np.where(live, norm, 1))
            r[k] = 0
            r[~live] = 0
            idx = [i for i in np.argsort(-np.abs(r)) if abs(r[i]) >= OVERLAP_MIN][:OVERLAP_MAX]
            if idx:
                lines.append(f"{me}: " + ", ".join(f"{ids[i]} r={r[i]:+.2f}" for i in idx))
    return "\n".join(lines) or f"no pairs with |r| >= {OVERLAP_MIN}"


def write_table(bank: dict[str, dict], y: np.ndarray, path: Path, noun: str = "digit") -> str:
    """Save the full bank statistics (including every pairwise correlation) for
    the calculator to preload, and return the message describing them."""
    ids = [f"{e['id']}.{j}" for e in bank.values() for j in range(len(e["names"]))]
    A = np.concatenate([e["acts"] for e in bank.values()], axis=1).astype(np.float64)
    means = np.stack([A[y == d].mean(0) for d in range(10)])
    std = A.std(0)
    C = A - A.mean(0)
    norm = np.sqrt((C ** 2).sum(0))
    live = norm > 1e-9
    corr = (C.T @ C) / np.outer(np.where(live, norm, 1), np.where(live, norm, 1))
    corr[~live, :] = 0
    corr[:, ~live] = 0
    np.fill_diagonal(corr, 1.0)
    np.savez(path, neuron_ids=np.array(ids), means=means, std=std, mean=A.mean(0), corr=corr)
    n = len(ids)
    return (f"Full overlap table: from now on your calculator has these variables preloaded (computed on the same "
            f"5,000 validation images as every other measurement; N = {n}, every neuron in your bank):\n"
            f"  NEURON_IDS  list of N ids in bank order ('R1F1.0', 'R1F1.1', ...)\n"
            f"  MEANS       array (10, N): mean activation of each neuron for each {noun} (the table above)\n"
            f"  MEAN        array (N,): mean activation over all images\n"
            f"  STD         array (N,): overall standard deviation (the table above)\n"
            f"  CORR        array (N, N): correlation between every pair of neurons across the images "
            f"(1 on the diagonal, 0 for neurons that never vary)")


class Combined:
    """Claude's chosen feature files + Claude's final dense layer."""

    def __init__(self, spec: dict, bank: dict[str, dict]):
        ids = spec.get("features") or []
        missing = [i for i in ids if i not in bank]
        if not ids or missing:
            raise ValueError(f'"features" must list bank ids; unknown: {missing}' if missing else '"features" is empty')
        self.parts = [bank[i]["net"] for i in ids]
        self.ids = [f"{i}.{j}" for i in ids for j in range(len(bank[i]["names"]))]
        self.labels = [n for i in ids for n in bank[i]["names"]]
        self.k = len(self.ids)
        w = np.asarray(spec["weights"], dtype=np.float64)
        if w.size != 10 * self.k:
            raise ValueError(f"weights must be 10 x {self.k} (the chosen files have {self.k} neurons), got {w.shape}")
        self.w = w.reshape(10, self.k)
        self.b = np.asarray(spec.get("bias", np.zeros(10)), dtype=np.float64).reshape(10)

    def features(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([feature_outputs(n, x) for n in self.parts], axis=1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.features(x) @ self.w.T + self.b

    def predict(self, x: np.ndarray, batch: int = 2000) -> np.ndarray:
        return self.forward(x).argmax(1)

    def n_params(self) -> int:
        return int(self.w.size + self.b.size + sum(n.n_params() for n in self.parts))


def json_blocks(text: str) -> list[tuple[dict | None, str | None]]:
    out = []
    for b in re.findall(r"```json\s*(.*?)```", text, flags=re.S):
        try:
            v = json.loads(b)
        except Exception as e:
            out.append((None, f"invalid JSON: {e}"))
            continue
        if isinstance(v, list) and v and all(isinstance(f, dict) for f in v):
            out += [(f, None) for f in v]  # several files written as one list
        elif isinstance(v, dict):
            out.append((v, None))
        else:
            out.append((None, "each ```json block must hold one JSON object (a file), not "
                              f"{type(v).__name__}"))
    return out


def build_parser():
    p = base_parser(__doc__)
    p.add_argument("--explore", type=int, default=4, help="explore rounds (default 4)")
    p.add_argument("--combine", type=int, default=3, help="combine rounds after exploring, each but the last with feedback (default 3)")
    p.add_argument("--calculator", action="store_true",
                   help="give Claude a Python calculator tool (no file or data access; every snippet logged to calculator.jsonl)")
    p.add_argument("--overlap", action="store_true",
                   help="also report which bank neurons fire together (correlation |r| >= 0.7) with every measurement")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "imagenette"],
                   help="mnist (default) or imagenette: 10 classes of real colour photos")
    p.add_argument("--size", type=int, default=64, help="imagenette: photo size the network sees (default 64)")
    p.add_argument("--sheets", action="store_true",
                   help="also show the sample images as real pictures (labelled contact sheets) in round 1; "
                        "only for backends that can see images (Claude)")
    backends.add_args(p)
    for a in p._actions:  # pixel formats depend on the dataset; resolved in run()
        if a.dest == "pixel_format":
            a.choices, a.default = None, None
            a.help = "how sample images are written as text (mnist: hex or int; imagenette: rgb16 or rgb32)"
    p.add_argument("--layered", action="store_true",
                   help="tools for building deeper features: files can stack on earlier bank files, the calculator "
                        "can save arrays referenced as \"@name\", per-layer shapes and top-response pictures are "
                        "reported (pictures only for backends that see images)")
    p.add_argument("--overlap-table", action="store_true",
                   help="in combine rounds, preload the full bank statistics (every pairwise correlation) into the "
                        "calculator; needs --calculator")
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


def run(args, run_dir: Path | None = None) -> dict:
    ds = datasets.get(args.dataset, args.size)
    args.pixel_format = args.pixel_format or ds.default_pixel_format
    if args.pixel_format not in ds.pixel_formats:
        sys.exit(f"--pixel-format for {ds.name} must be one of {', '.join(ds.pixel_formats)}")
    tag_name = "explore" if ds.name == "mnist" else f"explore-{ds.name}"
    run_dir = run_dir or ROOT / "runs" / (tag_name + "-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="ws-claude-")
    system, mcp_config = system_for(ds) + (OVERLAP_NOTE if args.overlap else ""), None
    store = run_dir / "kept" if args.layered else None
    if store:
        store.mkdir(exist_ok=True)
    if args.calculator:
        system += CALC_NOTE + (LAYERED_NOTE if args.layered else "")
        mcp_config = str(Path(workdir) / "mcp.json")
        Path(mcp_config).write_text(json.dumps({"mcpServers": {"calc": {
            "command": sys.executable, "args": ["-I", str(Path(__file__).with_name("calc_server.py"))],
            "env": {"WS_CALC_LOG": str(run_dir / "calculator.jsonl"),
                    # only exists once exploring is over, so explore rounds see nothing preloaded
                    **({"WS_CALC_DATA": str(run_dir / "bank_table.npz")} if args.overlap_table else {}),
                    **({"WS_CALC_STORE": str(store)} if store else {})}}}}),
            encoding="utf-8")
    elif args.overlap_table or args.layered:
        sys.exit("--overlap-table and --layered need --calculator")
    backend = backends.make(args, system, workdir, mcp_config, calc_log=run_dir / "calculator.jsonl",
                            calc_data=(run_dir / "bank_table.npz") if args.overlap_table else None, calc_store=store)
    xv, yv = ds.validation_split()
    xt, yt = ds.test()
    noun = "digit" if ds.name == "mnist" else "class"
    legend = "" if ds.name == "mnist" else ("Class numbers: " + ", ".join(
        f"{i} = {n}" for i, n in enumerate(ds.classes)) + ".\n")
    total = args.explore + args.combine

    bank: dict[str, dict] = {}
    combine_hist: list[tuple[int, float]] = []
    log = {"args": vars(args), "rounds": []}
    best = None
    session_id = None
    images = None
    prompt = "Start exploring: write feature files and predict what each neuron will respond to."
    if args.samples:
        ex_text, ex_ids = ds.examples_prompt(args.samples, args.pixel_format)
        prompt = ex_text + "\n\n" + prompt
        if args.sheets and backend.sees_images:
            images = ds.contact_sheets(ex_ids)
            (run_dir / "samples").mkdir(exist_ok=True)
            for k, png in enumerate(images):
                (run_dir / "samples" / f"sheet_{k + 1}.png").write_bytes(png)
            prompt = (f"The {len(images)} attached picture(s) show the same {len(ex_ids)} sample images, each "
                      "labelled with its #id and class.\n\n" + prompt)
        elif args.sheets:
            Console().print(f"[yellow]--sheets ignored: {backend.name} can't see images[/]")

    console = Console()
    progress = Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(bar_width=24),
                        MofNCompleteColumn(), TextColumn("{task.fields[status]}"), TimeElapsedColumn(), console=console)
    overall = progress.add_task("rounds", total=total, status="")
    round_task = progress.add_task("", total=None, status="", visible=False)
    progress.start()
    print = progress.console.print

    for r in range(1, total + 1):
        phase = "EXPLORE" if r <= args.explore else "COMBINE"
        left = total - r
        if phase == "EXPLORE":
            tag = (f"[Round {r} of {total}: EXPLORE ({args.explore - r} explore round(s) left after this one, "
                   f"then {args.combine} combine round(s)). Submit feature files only.]")
        else:
            tag = (f"[Round {r} of {total}: COMBINE. " + (f"{left} more combine round(s) with feedback after this one.]"
                   if left else "This is the LAST round: submit your final classifier.]"))
        prompt = tag + "\n\n" + prompt
        print(f"\n[bold cyan]===== round {r}/{total} · {phase} =====[/]  prompt {len(prompt):,} chars · "
              f"bank: {len(bank)} files, {sum(len(e['names']) for e in bank.values())} neurons")
        (run_dir / f"round_{r}_prompt.md").write_text(prompt, encoding="utf-8")
        entry = {"round": r, "phase": phase}
        progress.reset(round_task, total=None, visible=True, description=f"round {r}", status="waiting for Claude")
        live = LiveRound(progress, round_task, show_text=not args.quiet)
        t0 = time.time()
        try:
            out = backend.send(prompt, live, images=images)
            images = None  # pictures go with the message they belong to only
        except Exception as e:
            entry["error"] = str(e)
            log["rounds"].append(entry)
            print(Text(str(e), style="red"))
            break
        finally:
            progress.update(round_task, visible=False)
        text = out.get("result") or ""
        if re.search(r"hit your .*limit|usage limit", text, flags=re.I):
            entry["error"] = text
            log["rounds"].append(entry)
            print(Text(f"Stopping: {text}", style="bold red"))
            break
        u = out.get("usage") or {}
        extra = (f" · tokens so far: {u.get('prompt_tokens', 0):,} in ({u.get('prompt_cache_hit_tokens', 0):,} cached), "
                 f"{u.get('completion_tokens', 0):,} out") if args.backend == "deepseek" else ""
        print(f"[dim]({backend.name} finished in {time.time() - t0:.0f}s{extra})[/]")
        session_id = out.get("session_id", session_id)
        (run_dir / f"round_{r}_reply.md").write_text(text, encoding="utf-8")
        entry.update(session_id=session_id, usage=out.get("usage"), duration_ms=out.get("duration_ms"))
        blocks = json_blocks(text) or [(None, "no ```json block found in the reply")]
        fb: list[str] = []

        if phase == "EXPLORE":
            new, problems = [], []
            for k, (spec, err) in enumerate(blocks[:MAX_FILES], 1):
                fid = f"R{r}F{k}"
                try:
                    if err:
                        raise ValueError(err)
                    if args.layered:
                        spec = resolve_spec(spec, bank, store)
                    net, names = check_feature_file(spec, xv)
                    acts = feature_outputs(net, xv)
                    e = {"id": fid, "name": str(spec.get("name") or fid), "spec": spec, "net": net,
                         "names": names, "stats": neuron_stats(acts, yv), "acts": acts.astype(np.float32)}
                    bank[fid] = e
                    new.append(e)
                    (run_dir / f"{fid}.json").write_text(json.dumps(spec), encoding="utf-8")
                    if args.layered:
                        e["shapes"] = layer_shapes(net, xv)
                    print(f"  [green]{fid}[/] '{e['name']}': {len(names)} neurons, {net.n_params()} params"
                          + (f", {len(net.layers)} layers" + (f", built on {spec['built_on']}" if spec.get("built_on") else "")
                             if args.layered else ""))
                except Exception as ex:
                    problems.append(f"{fid}: could not be used: {ex}")
                    print(Text(f"  {fid}: could not be used: {ex}", style="red"))
            entry["files"] = [{"id": e["id"], "name": e["name"], "neurons": e["names"], "stats": e["stats"]} for e in new]
            if new:
                fb.append(legend + "Measurements for the files you just submitted, on 5,000 validation images. Columns: "
                          f"mean activation for each {noun} 0-9, overall std, % of images where the neuron is > 0, "
                          f"and how many of its {TOP_K} strongest responses came from each {noun}.\n```\n"
                          + bank_table(new, noun) + "\n```")
                if args.layered:
                    fb.append("Layer shapes (output shape after each layer, for one image):\n```\n" + "\n".join(
                        f"{e['id']}" + (f" (built on {e['spec']['built_on']})" if e["spec"].get("built_on") else "")
                        + f": {e['shapes']}" for e in new) + "\n```")
                    if backend.sees_images:
                        images = [top_sheet(e["acts"], xv, yv, e["names"], ds.classes,
                                            f"{e['id']} '{e['name']}': strongest responses per neuron") for e in new]
                        for e, png in zip(new, images):
                            (run_dir / f"{e['id']}_top.png").write_bytes(png)
                        fb.append(f"The {len(images)} attached picture(s) show, for each neuron of "
                                  + ", ".join(e["id"] for e in new) + ", the 6 validation photos it fired on most.")
                if args.overlap:
                    fb.append(f"Overlap: for each neuron above, the bank neurons (any round) whose activations "
                              f"correlate with it at |r| >= {OVERLAP_MIN} across the 5,000 images (strongest first, "
                              f"up to {OVERLAP_MAX}; neurons with none are not listed):\n```\n"
                              + overlap_lines(new, bank) + "\n```")
            fb += problems
            others = [e for e in bank.values() if e not in new]
            if others:
                fb.append("Your bank so far (files from earlier rounds): "
                          + ", ".join(f"{e['id']} '{e['name']}' ({len(e['names'])} neurons"
                                      + (f", {len(e['net'].layers)} layers" if args.layered else "") + ")" for e in others)
                          + ". Their measurements were reported in the round they were submitted.")
            fb.append("Use these measurements to plan the next step: which neurons behave as you predicted, "
                      f"which don't, and which {'digits' if noun == 'digit' else 'classes'} your bank still "
                      "cannot tell apart.")
            if r == args.explore:  # next round is the first combine round: hand over the whole bank
                fb = fb[:-1] + [
                    "Exploration is over. Here is your complete bank with every measurement:\n```\n"
                    + bank_table(list(bank.values()), noun) + "\n```",
                    *([f"Overlap across the whole bank (|r| >= {OVERLAP_MIN}, strongest first):\n```\n"
                       + overlap_lines(list(bank.values()), bank) + "\n```"] if args.overlap else []),
                    *([write_table(bank, yv, run_dir / "bank_table.npz", noun)] if args.overlap_table else []),
                    "Now write the final classifier in the COMBINE format: choose feature files by id and write "
                    "the 10 x N final-layer weights and 10 biases."]
        else:
            res = None
            for spec, err in blocks:
                try:
                    if err:
                        raise ValueError(err)
                    if "features" in spec:
                        net = Combined(spec, bank)
                    else:
                        net = Network.from_dict(spec)
                    mv, mt = metrics(net, xv, yv), metrics(net, xt, yt)
                    res = (spec, net, mv, mt)
                    break
                except Exception as ex:
                    fb.append(f"could not be used: {ex}")
                    print(Text(f"  could not be used: {ex}", style="red"))
            if res:
                spec, net, mv, mt = res
                (run_dir / f"round_{r}_combined.json").write_text(json.dumps(spec), encoding="utf-8")
                entry.update(val=mv, test_accuracy=mt["accuracy"], params=net.n_params())
                if best is None or mv["accuracy"] > best["val"]:
                    full = dict(spec, feature_files={i: bank[i]["spec"] for i in spec.get("features", [])})
                    best = {"round": r, "val": mv["accuracy"], "test": mt["accuracy"], "spec": full}
                print(f"[bold green]round {r}:[/] {net.n_params()} params · val {mv['accuracy']*100:.2f}% · "
                      f"[bold]test {mt['accuracy']*100:.2f}%[/]")
                print("  per digit: " + "  ".join(f"{d}:{a*100:3.0f}%" for d, a in enumerate(mv["per_class_accuracy"])))
                progress.update(overall, status=f"best: round {best['round']} · test {best['test']*100:.2f}%")
                combine_hist.append((r, mv["accuracy"]))
                fb.insert(0, legend + "Results of your classifier on the validation split:\n\n" + format_report(mv))
                if isinstance(net, Combined) and r < total:
                    fb.append(combine_mistakes(net, xv, yv, args.show_errors, seed=r, fmt=args.pixel_format, ds=ds))
            fb.append("Combine-round history (validation accuracy): "
                      + ", ".join(f"round {rr}: {a*100:.2f}%" for rr, a in combine_hist)
                      + (f". Best so far: round {best['round']}." if best else "."))
            fb.append("Revise the final layer (or choose different feature files) to fix the remaining confusions. "
                      "Submit the complete classifier again in the COMBINE format.")
        prompt = "\n\n".join(fb)
        log["rounds"].append(entry)
        (run_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        progress.advance(overall)

    progress.stop()
    shutil.rmtree(workdir, ignore_errors=True)
    if best:
        (run_dir / "best_weights.json").write_text(json.dumps(best["spec"]), encoding="utf-8")
        log["best"] = {k: v for k, v in best.items() if k != "spec"}
        (run_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        console.print(f"\n[bold]Result:[/] round {best['round']} (val {best['val']*100:.2f}%) -> "
                      f"[bold green]test {best['test']*100:.2f}%[/] · {ds.name}, {args.samples} sample images shown. "
                      "Saved as best_weights.json")
    console.print(f"saved to {run_dir}")
    return {"run_dir": str(run_dir), "best": None if not best else {k: v for k, v in best.items() if k != "spec"}}


if __name__ == "__main__":
    main()
