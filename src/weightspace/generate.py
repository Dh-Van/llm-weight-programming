"""Have Claude write MNIST classifier weights directly, from a few raw samples.

The idea being tested: an LLM looks at a handful of real digits as raw pixel
values plus its own raw weights, and edits the weights itself. No training
and no code.

Claude runs through Claude Code in headless mode (`claude -p`), so usage goes
on your Claude subscription, not on API tokens. ANTHROPIC_API_KEY and
ANTHROPIC_AUTH_TOKEN are removed from the subprocess environment so a stray
key can't switch you to per-token billing. All tools, MCP servers, claude.ai
connectors and skills are off, and Claude runs in an empty temp folder, so it
can't run code or read files.

What Claude sees:
  round 1   the network format, plus --samples N training digits (balanced
            across classes), every pixel of each (--pixel-format hex: one char
            per pixel, or int: exact 0-255). With --init, also the starting
            weight file to edit.
  round 2+  (--rounds R) its current weights as raw numbers, the accuracy and
            confusion matrix on a 5,000-image validation split of the TRAINING
            set, score history, best-so-far weights if it got worse, and
            --show-errors M images it got wrong: pixels, true and predicted
            digit, all 10 scores, and a per-cell breakdown of the final layer.
The test set is scored each round for the log only and is never shown.

    uv run ws-generate                              # 100 samples, 1 round
    uv run ws-generate --samples 200 --rounds 3     # samples + per-image feedback
    uv run ws-generate --samples 0 --show-errors 0  # pure zero-shot / aggregate-only
    uv run ws-sweep --start 100 --step 100 --stop 500   # accuracy vs. sample count

Everything (prompts, replies, weights, metrics) is saved under runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from . import data
from .evaluate import format_report, metrics
from .network import Network, __doc__ as FORMAT_DOC

ROOT = Path(__file__).resolve().parents[2]
MAX_CANDIDATES = 3  # weight files Claude may submit per round

SYSTEM = f"""You are doing weight-space programming. You will write, by hand, every
weight of a small neural network that classifies MNIST handwritten digits.

Rules:
- There is no training. Nobody will run gradient descent on what you write.
- You have no tools and cannot run code. Every number in your answer is one
  you chose yourself by reasoning about what it should detect.
- You may be shown real MNIST digits pixel by pixel (the encoding is stated
  with them). The network receives each pixel as a value from 0.0
  (background) to 1.0 (ink). Row 0 is the top, column 0 the left.
- Each digit is size-normalised into a 20x20 box and centred by centre of
  mass, so roughly pixels 4..23 hold the ink. Strokes are about 2-3 px wide.
- Make each weight mean something (stroke detectors, where a digit has ink,
  where it must not have ink, and so on). Explain your design in "notes".
- Watch calibration: a class whose template has much more (or more
  concentrated) positive weight than the others will win on most images.
  Use negative weights and biases to balance the classes.
- Keep the parameter count something you can write out in full (a few
  thousand numbers at most).
- Set "author" to the name of the model writing the weights.

The network runtime and weight-file format:
{FORMAT_DOC}

How the process works:
- This is a multi-round process. Each message tells you which round it is
  and how many rounds there are in total. After every round except the last,
  every weight file you submitted is run, and you get its results back.
- Your final result is the best full 10-digit weight file, by validation
  accuracy, from ANY round. So you do not have to solve everything at once:
  you can use early rounds to run experiments that set up later rounds.
- Each round you may submit up to {MAX_CANDIDATES} weight files, each in its own ```json
  block, each with a distinct "name". Each one is scored separately.
- A weight file may add "classes": [list of digits] to make it a probe. A
  probe is scored only on validation images of those digits, choosing only
  among those digits' outputs (the other outputs are ignored). Use probes to
  test one idea in isolation (e.g. separating two digits you keep confusing),
  then combine what works into a full network in a later round.
- A file without "classes" is a full 10-digit network and competes for your
  final result. Probes never do.
- Detailed per-image feedback (mistakes, final-layer breakdown) is given for
  your best full network of the round.

Answer with a short summary of your plan for this round (what each file is
for and what you changed), then the weight file(s). The JSON must be strict
(no comments, no trailing commas, no expressions like 1/3 — write 0.333).
"""


def find_claude() -> str:
    exe = shutil.which("claude")
    if not exe:
        for cand in (Path.home() / ".local" / "bin" / "claude.exe", Path.home() / ".local" / "bin" / "claude"):
            if cand.exists():
                exe = str(cand)
                break
    if not exe:
        sys.exit("Could not find the `claude` CLI. Install Claude Code, run `claude` once and /login, then retry.")
    return exe


def extract_json(text: str) -> dict:
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    if not blocks:
        raise ValueError("no ```json block found in the reply")
    return json.loads(blocks[-1])


def extract_candidates(text: str) -> list[tuple[str, dict | None, str | None]]:
    """Every ```json block in a reply, as (name, spec or None, error or None).
    Only the first MAX_CANDIDATES are used."""
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    out = []
    for k, b in enumerate(blocks[:MAX_CANDIDATES]):
        try:
            spec = json.loads(b)
            out.append((str(spec.get("name") or f"file {k + 1}"), spec, None))
        except Exception as e:
            out.append((f"file {k + 1}", None, f"invalid JSON: {e}"))
    return out


def probe_metrics(net: Network, x: np.ndarray, y: np.ndarray, classes: list[int]) -> dict:
    """Score a probe: only images of `classes`, argmax only among those outputs."""
    classes = sorted({int(c) for c in classes})
    if not classes or any(c < 0 or c > 9 for c in classes):
        raise ValueError(f'"classes" must list digits 0-9, got {classes}')
    keep = np.isin(y, classes)
    scores = scores_in_batches(net, x[keep])[:, classes]
    pred = np.array(classes)[scores.argmax(1)]
    yt = y[keep]
    cm = [[int(((yt == a) & (pred == b)).sum()) for b in classes] for a in classes]
    return {"classes": classes, "accuracy": float((pred == yt).mean()), "n": int(keep.sum()),
            "per_class_accuracy": [float((pred[yt == c] == c).mean()) for c in classes], "confusion_matrix": cm}


def format_probe(m: dict) -> str:
    cl = m["classes"]
    lines = [f"probe on digits {cl}: accuracy {m['accuracy']*100:.2f}% on {m['n']} images "
             f"(chance {100/len(cl):.0f}%)", "  per digit: " + "  ".join(
             f"{c}:{a*100:.1f}%" for c, a in zip(cl, m["per_class_accuracy"])),
             "  confusion (row = true, col = predicted):", "      " + "".join(f"{c:>6}" for c in cl)]
    lines += [f"  {c:>2}  " + "".join(f"{v:>6}" for v in row) for c, row in zip(cl, m["confusion_matrix"])]
    return "\n".join(lines)


def compact_json(spec: dict) -> str:
    """Weight file as JSON, one layer per line so the raw numbers stay readable."""
    head = {k: v for k, v in spec.items() if k != "layers"}
    lines = [json.dumps(l, separators=(",", ":")) for l in spec["layers"]]
    body = json.dumps(head, ensure_ascii=False)[:-1]
    return body + (', ' if head else '') + '"layers": [\n' + ",\n".join(lines) + "\n]}"


def sample_ids(n: int, seed: int = 0) -> list[int]:
    """n training images, balanced across the 10 digits (the first n % 10 digits
    get one extra) and nested: the first 100 of a 200-sample set are exactly the
    100-sample set, so a sweep only ever adds images."""
    _, y = data.load("train")
    rng = np.random.default_rng(seed)
    order = {d: rng.permutation(np.flatnonzero(y == d)) for d in range(10)}
    per = [n // 10 + (1 if d < n % 10 else 0) for d in range(10)]
    return [int(i) for d in range(10) for i in order[d][: per[d]]]


def examples_prompt(n: int, fmt: str, seed: int = 0) -> str:
    x, y = data.load("train")
    ids = sample_ids(n, seed)
    parts = [f"Here are {len(ids)} real MNIST training digits ({n // 10}+ per class). "
             f"Pixel encoding: {data.PIXEL_FORMATS[fmt]}."]
    for i in ids:
        parts.append(f"digit {y[i]} (#{i}):\n```\n{data.pixel_grid(x[i], fmt)}\n```")
    return "\n\n".join(parts)


def scores_in_batches(net: Network, x: np.ndarray, batch: int = 2000) -> np.ndarray:
    return np.concatenate([net.forward(x[i : i + batch]) for i in range(0, len(x), batch)])


MAX_FEATURES = 400  # only show final-layer inputs when they're small enough to read


def final_layer_inputs(net: Network, x: np.ndarray) -> tuple[np.ndarray, list[str] | None] | None:
    """The values that feed the last dense layer, for a few images, keeping the
    (C, H, W) shape when there is one. Returns (values, channel names) or None."""
    dense_idx = [i for i, l in enumerate(net.layers) if l.type == "dense"]
    if not dense_idx:
        return None
    k = dense_idx[-1]
    _, inter = net.forward(x, return_intermediates=True)
    j = k - 1
    while j >= 0 and net.layers[j].type == "flatten":  # report the shape before flattening
        j -= 1
    feats = x[:, None] if j < 0 else inter[j]
    names = None
    convs = [l for l in net.layers[: j + 1] if l.type == "conv2d"]
    if feats.ndim == 4 and convs and convs[-1].params.get("names"):
        names = list(convs[-1].params["names"])
    return feats, names


def format_features(f: np.ndarray, names: list[str] | None) -> str:
    if f.ndim == 3:  # (C, H, W)
        out = []
        for c in range(f.shape[0]):
            label = names[c] if names and c < len(names) else f"channel {c}"
            rows = "\n".join(" ".join(f"{v:.2f}" for v in row) for row in f[c])
            out.append(f"{label}:\n{rows}")
        return "\n".join(out)
    return " ".join(f"{v:.2f}" for v in f.ravel())


def contribution_block(net: Network, feats: np.ndarray, names: list[str] | None, true: int, pred: int) -> str:
    """Explain one mistake through the last dense layer. For every input cell:
    (w[true] - w[pred]) * input. Negative cells are where the true digit lost
    ground. Cells + bias gap sum exactly to score[true] - score[pred]."""
    dense = [l for l in net.layers if l.type == "dense"][-1]
    x = feats.ravel()
    diff = (dense.weights[true] - dense.weights[pred]) * x
    bias_gap = dense.bias[true] - dense.bias[pred]
    full = x.size <= MAX_FEATURES
    lines = [f"why {pred} beat {true}: (weights[{true}] - weights[{pred}]) x input, per cell "
             f"(negative = where {true} lost)" + (":" if full else f" — {x.size} cells, too many to list, "
                                                  "so only the totals and costliest cells:")]
    if full:
        lines.append(format_features(diff.reshape(feats.shape), names))
    lines.append(f"sum of cells {diff.sum():+.3f}, bias gap {bias_gap:+.3f}, "
                 f"total {diff.sum() + bias_gap:+.3f} = score[{true}] - score[{pred}]")
    worst = np.argsort(diff)[: 5 if full else 10]
    if feats.ndim == 3:
        c, h, w = feats.shape
        loc = lambda k: (f"{names[k // (h * w)] if names else f'channel {k // (h * w)}'} "
                         f"row {(k % (h * w)) // w} col {k % w}")
    else:
        loc = lambda k: f"input {k}"
    lines.append("costliest cells: " + "; ".join(
        f"{loc(k)}: input {x[k]:.2f}, w[{true}] {dense.weights[true][k]:+.2f}, "
        f"w[{pred}] {dense.weights[pred][k]:+.2f} -> {diff[k]:+.3f}" for k in worst))
    return "\n".join(lines)


def history_prompt(history: list[dict], best_round: int | None) -> str:
    lines = ["Score history on the validation split (your best round is marked *):",
             "round  val acc   " + "  ".join(f"  {d}" for d in range(10))]
    for h in history:
        mark = "*" if h["round"] == best_round else " "
        if h.get("val_accuracy") is None:
            lines.append(f"{h['round']:>4}{mark}   invalid weight file")
        else:
            pcs = "  ".join(f"{a*100:3.0f}" for a in h["per_class"])
            lines.append(f"{h['round']:>4}{mark}   {h['val_accuracy']*100:6.2f}%  {pcs}")
    return "\n".join(lines)


def errors_prompt(net: Network, x: np.ndarray, y: np.ndarray, m: int, seed: int, fmt: str = "int") -> str:
    """Up to m misclassified validation images, spread across the true classes,
    with exact pixels, all 10 output scores, and (when small) the values that
    fed the network's last layer."""
    scores = scores_in_batches(net, x)
    pred = scores.argmax(1)
    wrong = np.flatnonzero(pred != y)
    if m <= 0 or len(wrong) == 0:
        return ""
    rng = np.random.default_rng(seed)
    by_class = {d: list(rng.permutation(wrong[y[wrong] == d])) for d in range(10)}
    picked = []
    while len(picked) < m and any(by_class.values()):
        for d in range(10):  # round-robin over true digits
            if by_class[d] and len(picked) < m:
                picked.append(by_class[d].pop())
    feats = final_layer_inputs(net, x[picked])
    parts = [f"{len(picked)} validation images your network got wrong, with exact pixel values "
             "and the 10 output scores (index = digit)"
             + (", plus the exact values your last dense layer received for that image "
                "(its score for digit d = weights[d] · these values + bias[d])" if feats else "") + ":"]
    for n, i in enumerate(picked):
        sc = ", ".join(f"{d}: {s:.3f}" for d, s in enumerate(scores[i]))
        block = (f"true {y[i]}, predicted {pred[i]}\nscores [{sc}]\npixels ({data.PIXEL_FORMATS[fmt]}):\n"
                 f"```\n{data.pixel_grid(x[i], fmt)}\n```")
        if feats:
            f_n = feats[0][n]
            if f_n.size <= MAX_FEATURES and not (f_n.ndim == 3 and f_n.shape[1:] == (28, 28)):
                block += f"\ninput to the last dense layer:\n```\n{format_features(f_n, feats[1])}\n```"
            block += f"\n```\n{contribution_block(net, f_n, feats[1], int(y[i]), int(pred[i]))}\n```"
        parts.append(block)
    return "\n\n".join(parts)


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")


class LiveRound:
    """Turns Claude Code's stream-json events into live terminal output:
    reasoning text is printed as it arrives, the JSON weight block is counted
    (not printed), and a status line shows phase / elapsed / weights written."""

    def __init__(self, progress: Progress, task, show_text: bool):
        self.progress, self.task, self.show_text = progress, task, show_text
        self.text = ""
        self.printed_upto = 0
        self.phase = "waiting for Claude"
        self.tool_calls = 0

    def _update(self):
        j = self.text.find("```json")
        n = len(NUM_RE.findall(self.text[j:])) if j >= 0 else 0
        self.progress.update(self.task, status=f"{self.phase} · {n:,} numbers written · {len(self.text):,} chars")

    def _flush_text(self, final: bool = False):
        if not self.show_text:
            return
        j = self.text.find("```json")
        limit = j if j >= 0 else len(self.text)
        chunk = self.text[self.printed_upto:limit]
        if not final and j < 0:  # only print whole lines while streaming
            cut = chunk.rfind("\n")
            chunk = chunk[: cut + 1] if cut >= 0 else ""
        if chunk:
            for line in chunk.rstrip("\n").split("\n"):
                self.progress.console.print(Text(line, style="dim"))
            self.printed_upto += len(chunk)
        if j >= 0 and self.printed_upto <= j:
            self.progress.console.print(Text("```json  … (weights streaming, counted on the progress line)", style="dim cyan"))
            self.printed_upto = j + 1

    def handle(self, ev: dict):
        if ev.get("type") != "stream_event":
            return
        e = ev["event"]
        if e["type"] == "content_block_start":
            kind = e.get("content_block", {}).get("type")
            self.phase = "thinking" if kind == "thinking" else "writing"
            if kind == "tool_use":
                self.tool_calls += 1
                self.phase = f"using calculator (call {self.tool_calls})"
                if self.show_text:
                    self.progress.console.print(Text(f"[calculator call {self.tool_calls}]", style="dim magenta"))
        elif e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta":
            self.text += e["delta"]["text"]
            self.phase = "writing weights" if "```json" in self.text else "writing reasoning"
            self._flush_text()
        self._update()

    def finish(self):
        self._flush_text(final=True)


def call_claude(exe: str, args, prompt: str, workdir: str, session_id: str | None, live: LiveRound,
                system: str | None = None, mcp_config: str | None = None, images: list[bytes] | None = None) -> dict:
    cmd = [
        exe, "-p",
        "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        "--model", args.model,
        *(["--effort", args.effort] if args.effort else []),
        "--system-prompt", system or SYSTEM,
        "--tools", "",                  # no built-in tools: no code execution, no file reads
        "--strict-mcp-config",          # no MCP servers except the one passed below
        "--disable-slash-commands",     # no skills
    ]
    if mcp_config:  # only our calculator server, and only its one tool
        cmd += ["--mcp-config", mcp_config, "--allowedTools", "mcp__calc__python", "--max-turns", "60"]
    else:
        cmd += ["--disallowedTools", "mcp__*"]  # no connector tools
    if session_id:
        cmd += ["--resume", session_id]
    if images:  # send the prompt as one stream-json user message so pictures can ride along
        import base64
        cmd += ["--input-format", "stream-json"]
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(b).decode()}} for b in images]
        prompt = json.dumps({"type": "user", "message": {"role": "user",
                                                         "content": content + [{"type": "text", "text": prompt}]}}) + "\n"
    cmd += args.claude_arg
    # Strip API credentials so Claude Code can only use the claude.ai subscription
    # login, and turn off claude.ai connectors (they'd otherwise add tools).
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    env["ENABLE_CLAUDEAI_MCP_SERVERS"] = "false"
    # Prompt goes over stdin to stay clear of Windows' command-line length limit.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", cwd=workdir, env=env)
    timer = threading.Timer(args.timeout, proc.kill)
    timer.start()
    stderr_chunks: list[str] = []
    err_thread = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    err_thread.start()
    result = None
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                result = ev
            else:
                live.handle(ev)
        proc.wait()
    finally:
        timer.cancel()
        live.finish()
    err_thread.join(timeout=5)
    if result is None:
        raise RuntimeError(f"claude exited with {proc.returncode} and no result:\n{''.join(stderr_chunks).strip()[:2000]}")
    return result


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="opus", help="Claude Code model alias or id (default: opus)")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                   help="how hard Claude thinks each round (default: Claude Code's own default, ~1-3 min/round; "
                        "xhigh/max can take 20+ min per round)")
    p.add_argument("--rounds", type=int, default=1, help="number of write/revise rounds")
    p.add_argument("--samples", type=int, default=100,
                   help="training images shown in round 1, balanced across digits (0 = none, default 100)")
    p.add_argument("--pixel-format", default="hex", choices=sorted(data.PIXEL_FORMATS),
                   help="hex = 1 char per pixel, 16 levels (default, fits many images); int = exact 0-255")
    p.add_argument("--show-errors", type=int, default=10, help="misclassified val images (raw pixels + scores) per feedback round")
    p.add_argument("--init", help="start from this weight file and have Claude edit it")
    p.add_argument("--timeout", type=int, default=600, help="give up on a round after this many seconds (default 600)")
    p.add_argument("--claude-arg", action="append", default=[], help="extra flag passed to claude (repeatable)")
    p.add_argument("--quiet", action="store_true", help="don't stream Claude's reasoning text, just the progress bars")
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


def run(args, run_dir: Path | None = None) -> dict:
    """One generation run. Returns a summary with the best round (by validation)."""
    exe = find_claude()
    run_dir = run_dir or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="ws-claude-")  # empty folder: nothing for Claude to read
    xv, yv = data.validation_split()
    xt, yt = data.load("test")

    parts = []
    if args.samples:
        parts.append(examples_prompt(args.samples, args.pixel_format))
    if args.init:
        init_spec = json.loads(Path(args.init).read_text(encoding="utf-8"))
        init_net = Network.from_dict(init_spec)
        parts.append("Here is an existing weight file to start from, as raw numbers:\n```json\n"
                     + compact_json(init_spec) + "\n```")
        parts.append("Its results on a 5,000-image validation split:\n\n" + format_report(metrics(init_net, xv, yv)))
        if args.show_errors:
            parts.append(errors_prompt(init_net, xv, yv, args.show_errors, seed=0, fmt=args.pixel_format))
        parts.append("Edit these weights to improve them. Plan how to use your rounds.")
    else:
        parts.append("Build an MNIST classifier. Choose the architecture from the supported layer types, "
                     "and plan how to use your rounds.")
    prompt = "\n\n".join(parts)
    log = {"args": vars(args), "rounds": []}
    session_id = None
    seen = args.samples + (args.show_errors if args.init else 0)  # images shown to Claude so far
    best = None  # best round so far, chosen by validation accuracy
    history: list[dict] = []
    if args.init:  # the starting weights count as round 0
        mv0 = metrics(init_net, xv, yv)
        history.append({"round": 0, "val_accuracy": mv0["accuracy"], "per_class": mv0["per_class_accuracy"]})
        best = {"round": 0, "val": mv0["accuracy"], "test": metrics(init_net, xt, yt)["accuracy"], "spec": init_spec}

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )
    overall = progress.add_task("rounds", total=args.rounds, status="")
    round_task = progress.add_task("", total=None, status="", visible=False)
    progress.start()
    print = progress.console.print  # route all output above the live bars

    for r in range(1, args.rounds + 1):
        print(f"\n[bold cyan]===== round {r}/{args.rounds} =====[/]  prompt {len(prompt):,} chars · "
              f"images seen so far: {seen}")
        left = args.rounds - r
        prompt = (f"[Round {r} of {args.rounds}. "
                  + (f"{left} more round(s) with feedback after this one.]" if left
                     else "This is the LAST round: there is no more feedback, so submit your best full network.]")
                  + "\n\n" + prompt)
        (run_dir / f"round_{r}_prompt.md").write_text(prompt, encoding="utf-8")
        entry = {"round": r, "images_seen": seen}
        progress.reset(round_task, total=None, visible=True, description=f"round {r}", status="waiting for Claude")
        live = LiveRound(progress, round_task, show_text=not args.quiet)
        t0 = time.time()
        try:
            out = call_claude(exe, args, prompt, workdir, session_id, live)
            if "safeguards flagged" in (out.get("result") or ""):
                # False-positive safety flag (seen once as "reasoning_extraction"): retry once.
                print("[yellow]Claude's safety filter flagged this round (likely a false positive); retrying once...[/]")
                entry["flagged_retry"] = True
                live = LiveRound(progress, round_task, show_text=not args.quiet)
                out = call_claude(exe, args, prompt, workdir, session_id, live)
        except Exception as e:
            entry["error"] = str(e)
            log["rounds"].append(entry)
            print(Text(str(e), style="red"))
            break
        finally:
            progress.update(round_task, visible=False)
        if re.search(r"hit your .*limit|usage limit", out.get("result") or "", flags=re.I):
            entry["error"] = out.get("result")
            log["rounds"].append(entry)
            print(Text(f"Stopping: {out.get('result')}", style="bold red"))
            break
        print(f"[dim](Claude finished in {time.time() - t0:.0f}s)[/]")
        session_id = out.get("session_id", session_id)
        text = out.get("result") or ""
        (run_dir / f"round_{r}_reply.md").write_text(text, encoding="utf-8")
        entry.update(session_id=session_id, subtype=out.get("subtype"), num_turns=out.get("num_turns"),
                     usage=out.get("usage"), duration_ms=out.get("duration_ms"))
        # Score every weight file in the reply: full networks and probes.
        if out.get("is_error"):
            cands = [("reply", None, f"claude reported an error: {out.get('subtype')} {text[:500]}")]
        else:
            cands = extract_candidates(text) or [("reply", None, "no ```json block found in the reply")]
        reports, fulls, entry["candidates"] = [], [], []
        for k, (name, spec, err) in enumerate(cands, 1):
            c = {"name": name}
            try:
                if err:
                    raise ValueError(err)
                net = Network.from_dict(spec)
                (run_dir / f"round_{r}_file{k}.json").write_text(json.dumps(spec), encoding="utf-8")
                c["params"] = net.n_params()
                if spec.get("classes") is not None:
                    pm = probe_metrics(net, xv, yv, spec["classes"])
                    c["probe"] = pm
                    reports.append(f"### {name} (probe)\n" + format_probe(pm))
                    print(f"  [cyan]probe '{name}'[/] digits {pm['classes']}: {pm['accuracy']*100:.2f}%")
                else:
                    mv, mt = metrics(net, xv, yv), metrics(net, xt, yt)
                    c.update(val=mv, test_accuracy=mt["accuracy"])
                    fulls.append((mv["accuracy"], name, spec, net, mv, mt))
                    reports.append(f"### {name} (full network)\n" + format_report(mv))
                    print(f"  [green]full '{name}'[/] {net.n_params()} params · val {mv['accuracy']*100:.2f}% · "
                          f"test {mt['accuracy']*100:.2f}%")
            except Exception as e:
                c["error"] = str(e)
                reports.append(f"### {name}\ncould not be used: {e}")
                print(Text(f"  '{name}': could not be used: {e}", style="red"))
            entry["candidates"].append(c)

        if fulls:
            _, name, spec, net, mv, mt = max(fulls, key=lambda f: f[0])  # best full file this round, by val
            (run_dir / f"round_{r}_weights.json").write_text(json.dumps(spec), encoding="utf-8")
            entry.update(val=mv, test_accuracy=mt["accuracy"], params=net.n_params(), chosen=name)
            history.append({"round": r, "val_accuracy": mv["accuracy"], "per_class": mv["per_class_accuracy"]})
            if best is None or mv["accuracy"] > best["val"]:  # chosen on validation only, never test
                best = {"round": r, "val": mv["accuracy"], "test": mt["accuracy"], "spec": spec}
            print(f"[bold green]round {r}:[/] best full '{name}' · val {mv['accuracy']*100:.2f}% · "
                  f"[bold]test {mt['accuracy']*100:.2f}%[/] · images seen {seen}")
            print("  per digit: " + "  ".join(f"{d}:{a*100:3.0f}%" for d, a in enumerate(mv["per_class_accuracy"])))
        else:
            history.append({"round": r, "val_accuracy": None})
            print(Text(f"round {r}: no usable full 10-digit network this round", style="yellow"))
        if best:
            progress.update(overall, status=f"best: round {best['round']} · val {best['val']*100:.2f}% · "
                                            f"test {best['test']*100:.2f}%")

        fb = ["Results for each file you submitted (validation split of 5,000 training images):\n\n"
              + "\n\n".join(reports), history_prompt(history, best and best["round"])]
        if fulls:
            fb.insert(0, f"Your best full network this round was '{name}'. Its weights as raw numbers:\n"
                         "```json\n" + compact_json(spec) + "\n```")
        if best and (not fulls or best["round"] != r):
            fb.append(f"Your best full network overall is still from round {best['round']} "
                      f"(val {best['val']*100:.2f}%). Its weights:\n```json\n" + compact_json(best["spec"]) + "\n```")
        if fulls and args.show_errors and r < args.rounds:
            fb.append(f"Mistakes made by '{name}':\n\n"
                      + errors_prompt(net, xv, yv, args.show_errors, seed=r, fmt=args.pixel_format))
            seen += args.show_errors
        fb.append("Use these results to decide your next step: run more probes, combine what worked, "
                  "or revise your full network.")
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
        console.print(f"\n[bold]Result:[/] best round by validation = round {best['round']} "
                      f"(val {best['val']*100:.2f}%) → [bold green]test {best['test']*100:.2f}%[/] "
                      f"after seeing {seen} images. Saved as best_weights.json")
    console.print(f"saved to {run_dir}")
    return {"run_dir": str(run_dir), "images_seen": seen,
            "best": None if not best else {k: v for k, v in best.items() if k != "spec"}}


if __name__ == "__main__":
    main()
