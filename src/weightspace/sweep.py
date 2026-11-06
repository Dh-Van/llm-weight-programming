"""Accuracy vs. number of samples: run ws-generate once per sample count.

    uv run ws-sweep                                    # 100, 200, ..., 500 samples, 1 round each
    uv run ws-sweep --start 100 --step 100 --stop 1000 --rounds 2

Sample sets are nested (the 200 set contains the 100 set). Every other
ws-generate flag (--rounds, --show-errors, --model, ...) applies to each run.

For reference, the summary also shows what you get by mechanically averaging
the same N images per digit and picking the nearest average (no LLM). It's
reported for comparison only; it never feeds into Claude's weights.
"""

from __future__ import annotations

import json
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from . import data
from .generate import ROOT, build_parser, run, sample_ids


def averaging_baseline(n: int) -> float:
    xtr, ytr = data.load("train")
    xt, yt = data.load("test")
    ids = np.array(sample_ids(n))
    m = np.stack([xtr[ids[ytr[ids] == d]].reshape(-1, 784).mean(0) for d in range(10)])
    f = xt.reshape(-1, 784)
    d2 = (f ** 2).sum(1)[:, None] - 2 * f @ m.T + (m ** 2).sum(1)[None]
    return float((d2.argmin(1) == yt).mean())


def main(argv=None):
    p = build_parser(__doc__)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--stop", type=int, default=500)
    args = p.parse_args(argv)
    sweep_dir = ROOT / "runs" / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    console = Console()
    rows = []
    for n in range(args.start, args.stop + 1, args.step):
        console.rule(f"[bold]{n} samples")
        args.samples = n
        res = run(args, sweep_dir / f"n{n}")
        rows.append({"samples": n, **res, "averaging_baseline_test": averaging_baseline(n)})
        (sweep_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    t = Table(title=f"Accuracy vs. samples ({args.rounds} round(s) each)")
    for c in ["samples", "images seen", "best round", "val", "test (Claude)", "test (averaging, no LLM)"]:
        t.add_column(c, justify="right")
    for r in rows:
        b = r["best"]
        t.add_row(str(r["samples"]), str(r["images_seen"]),
                  str(b["round"]) if b else "-", f"{b['val']*100:.2f}%" if b else "failed",
                  f"{b['test']*100:.2f}%" if b else "failed", f"{r['averaging_baseline_test']*100:.2f}%")
    console.print(t)
    console.print(f"saved to {sweep_dir}")


if __name__ == "__main__":
    main()
