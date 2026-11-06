"""Weight-space programming for a 2D walker: Claude writes every weight of the
controller network for MuJoCo's Walker2d (gymnasium Walker2d-v5). No training.

Each round, every policy file Claude submits is run for several episodes. Claude
gets back the score and distance of each episode, why it ended (fell, tipped
over, or survived the full 1,000 steps), and a step-by-step trace of one episode:
the exact 17 sensor values the network received and the 6 torques it output.
Claude then edits its weights.

    uv run ws-walker                        # 5 rounds
    uv run ws-walker --rounds 8 --calculator
    uv run ws-walker-render runs/walker-.../best_policy.json   # save a GIF

Everything is saved under runs/walker-<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from .generate import ROOT, LiveRound, call_claude, compact_json, find_claude
from .network import Network

ENV_ID = "Walker2d-v5"
OBS = ["torso height z", "torso angle", "R thigh angle", "R knee angle", "R ankle angle",
       "L thigh angle", "L knee angle", "L ankle angle",
       "x velocity", "z velocity", "torso ang vel", "R thigh vel", "R knee vel", "R ankle vel",
       "L thigh vel", "L knee vel", "L ankle vel"]
OBS_SHORT = ["z", "torso", "thR", "knR", "anR", "thL", "knL", "anL",
             "vx", "vz", "vtor", "vthR", "vknR", "vanR", "vthL", "vknL", "vanL"]
ACT = ["R thigh", "R knee", "R ankle", "L thigh", "L knee", "L ankle"]
ACT_SHORT = ["aThR", "aKnR", "aAnR", "aThL", "aKnL", "aAnL"]
MAX_FILES = 3
CALC_NOTE = """
You have one tool: a calculator (`python`). It runs a short Python snippet
(numpy, math and statistics available) and returns what it prints. It cannot
read files and cannot run the simulator: it only knows numbers you copy into
the code from this conversation. Use it for arithmetic. Do not use it to run
training or iterative optimisation (no gradient descent, no fit-by-search
loops, no simulating the walker yourself); every weight must follow from
reasoning or a formula you can state. Every snippet is logged. Your final
message must contain the complete answer (summary and JSON) - text from
earlier tool-using steps is not read.
"""

SYSTEM = f"""You are doing weight-space programming. You will write, by hand, every
weight of a small neural network that controls a simulated 2D walking robot.

Rules:
- There is no training. Nobody will run gradient descent or any search on
  what you write. Every number in your answer is one you chose by reasoning.
- Set "author" to the name of the model writing the weights.

The robot (MuJoCo Walker2d, gymnasium "{ENV_ID}"):
- A planar biped: a torso, and two legs (right R, left L), each with a thigh
  (hip joint), a shin (knee joint) and a foot (ankle joint). It moves in the
  x (forward) / z (up) plane. Forward is +x. The foot points toward +x.
- Every control step lasts 0.008 s. An episode ends after 1,000 steps, or
  earlier when the robot is unhealthy: torso height z outside 0.8..2.0 (it
  starts at about 1.25) or torso angle outside -1..+1 rad.
- Reward per step = x velocity (m/s) + 1 (for staying healthy)
  - 0.001 * sum(torque^2). So standing still for 1,000 steps scores about
  1,000, and walking forward scores more. With zero torque the robot
  collapses after about 110 steps (score about 100).
- Angles are in radians. Measured sign conventions:
  - torso angle > 0: the torso leans forward (its top moves toward +x).
  - hip (thigh) angle: range -2.62..0. 0 = leg straight down along the
    torso; negative swings the knee backward (toward -x) relative to the
    torso.
  - knee angle: range -2.62..0. 0 = straight leg; negative bends the knee so
    the foot goes backward (toward -x) relative to the thigh.
  - ankle angle: range about -0.79..+0.79.
  - A positive torque increases its joint's angle; a negative torque
    decreases it.

The network:
- Input: a vector of 17 numbers, in this order (velocities are per second):
{chr(10).join(f"    {i:>2}  {n}" for i, n in enumerate(OBS))}
  If the file sets "clock_period": P (in steps), two more inputs are appended:
    17  sin(2*pi*t/P)   18  cos(2*pi*t/P)    (t = step number from 0)
- Output: 6 torques, in this order: {", ".join(f"{i} {a}" for i, a in enumerate(ACT))}.
  Each is clipped to -1..+1 (1 = full motor strength).
- Format (JSON):
  {{"name": "...", "author": "...", "notes": "...",
    "clock_period": 100,             (optional)
    "layers": [
      {{"type": "dense", "weights": [[...] x n_out], "bias": [n_out numbers]}},
      {{"type": "tanh"}},
      {{"type": "dense", "weights": [[...] x 6], "bias": [6 numbers]}}
    ]}}
  dense weights are (outputs x inputs). Activation layers: relu, tanh,
  sigmoid, abs; also scale/shift with "value". The last layer must output 6
  numbers. Keep it to at most a few thousand numbers.

How the process works:
- Multiple rounds; each message says which round it is and how many are left.
- Each round you may submit up to {MAX_FILES} policy files, each in its own
  ```json block with a distinct "name". Each is run for 5 episodes (random
  starting noise differs per episode) and scored separately.
- After each round you get every file's results and a step-by-step trace of
  your best file's first episode: the exact inputs it received and the
  torques it produced.
- Your final result is your best file, by average score, from ANY round, so
  you can use early rounds for experiments.

Answer with a short summary of what each file is for and what you changed,
then the JSON. The JSON must be strict (no comments, no trailing commas, no
expressions - write numbers).
"""


def policy_inputs(obs: np.ndarray, t: int, period: float | None) -> np.ndarray:
    if not period:
        return obs
    ph = 2 * np.pi * t / period
    return np.concatenate([obs, [np.sin(ph), np.cos(ph)]])


class Policy:
    def __init__(self, spec: dict):
        self.net = Network.from_dict(spec)
        self.period = spec.get("clock_period")
        if self.period is not None:
            self.period = float(self.period)
            if self.period <= 0:
                raise ValueError('"clock_period" must be positive')
        a = self.act(np.zeros(17), 0)
        if a.shape != (6,):
            raise ValueError(f"the network must output 6 torques, got {a.shape}")

    def act(self, obs: np.ndarray, t: int) -> np.ndarray:
        x = policy_inputs(obs, t, self.period)[None]
        out = self.net.forward(x, check_output=False)
        return np.clip(np.asarray(out, dtype=np.float64).reshape(-1), -1, 1)


def end_reason(obs: np.ndarray, terminated: bool) -> str:
    if not terminated:
        return "survived all 1000 steps"
    z, ang = obs[0], obs[1]
    if z < 0.8:
        return f"fell (torso height {z:.2f} < 0.8)"
    if z > 2.0:
        return f"jumped too high (torso height {z:.2f} > 2.0)"
    return f"tipped over (torso angle {ang:+.2f} outside -1..1, {'forward' if ang > 0 else 'backward'})"


def rollout(policy: Policy, seed: int, record: bool = False) -> dict:
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=seed)
    ret, rows, sat = 0.0, [], np.zeros(6)
    info = {"x_position": 0.0}
    t = 0
    for t in range(1000):
        a = policy.act(obs, t)
        sat += np.abs(a) >= 0.999
        if record:
            rows.append((t, obs.copy(), a.copy()))
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term or trunc:
            break
    env.close()
    steps = t + 1
    return {"seed": seed, "return": float(ret), "steps": steps, "x": float(info["x_position"]),
            "end": end_reason(obs, term), "saturation": (sat / steps).tolist(), "trace": rows, "final_obs": obs}


def evaluate(policy: Policy, seeds=range(5)) -> dict:
    eps = [rollout(policy, s, record=(i == 0)) for i, s in enumerate(seeds)]
    return {"mean_return": float(np.mean([e["return"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "mean_x": float(np.mean([e["x"] for e in eps])), "episodes": eps}


def format_result(name: str, res: dict) -> str:
    lines = [f"### {name}: average score {res['mean_return']:.1f}, average {res['mean_steps']:.0f} steps, "
             f"average distance {res['mean_x']:+.2f} m"]
    for e in res["episodes"]:
        lines.append(f"  episode {e['seed']}: score {e['return']:7.1f}, {e['steps']:4d} steps, "
                     f"x {e['x']:+.2f} m, {e['end']}")
    sat = np.mean([e["saturation"] for e in res["episodes"]], axis=0)
    lines.append("  share of steps each torque was at its +-1 limit: "
                 + ", ".join(f"{a} {s*100:.0f}%" for a, s in zip(ACT, sat)))
    return "\n".join(lines)


def format_trace(ep: dict, every: int, max_rows: int) -> str:
    rows = ep["trace"]
    keep = [r for r in rows if r[0] % every == 0][:max_rows]
    if rows and rows[-1][0] != keep[-1][0]:
        keep.append(rows[-1])  # always show the last step before the episode ended
    head = "  t " + "".join(f"{h:>7}" for h in OBS_SHORT) + " |" + "".join(f"{h:>6}" for h in ACT_SHORT)
    out = [head]
    for t, o, a in keep:
        out.append(f"{t:>4}" + "".join(f"{v:7.2f}" for v in o) + " |" + "".join(f"{v:6.2f}" for v in a))
    return "\n".join(out)


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="opus", help="Claude Code model alias or id (default: opus)")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--rounds", type=int, default=5, help="write/revise rounds (default 5)")
    p.add_argument("--trace-every", type=int, default=5, help="trace one row every N steps (default 5)")
    p.add_argument("--trace-rows", type=int, default=40, help="max trace rows per round (default 40)")
    p.add_argument("--calculator", action="store_true", help="give Claude a Python calculator (no simulator, no files)")
    p.add_argument("--timeout", type=int, default=600, help="give up on a round after this many seconds")
    p.add_argument("--claude-arg", action="append", default=[], help="extra flag passed to claude (repeatable)")
    p.add_argument("--quiet", action="store_true", help="don't stream Claude's reasoning text")
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


def run(args, run_dir: Path | None = None) -> dict:
    exe = find_claude()
    run_dir = run_dir or ROOT / "runs" / ("walker-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="ws-claude-")
    system, mcp_config = SYSTEM, None
    if args.calculator:
        system = SYSTEM + CALC_NOTE
        mcp_config = str(Path(workdir) / "mcp.json")
        Path(mcp_config).write_text(json.dumps({"mcpServers": {"calc": {
            "command": sys.executable, "args": ["-I", str(Path(__file__).with_name("calc_server.py"))],
            "env": {"WS_CALC_LOG": str(run_dir / "calculator.jsonl")}}}}), encoding="utf-8")

    log = {"args": vars(args), "rounds": []}
    best, history, session_id = None, [], None
    prompt = "Write a controller that makes the walker walk forward. Plan how to use your rounds."

    console = Console()
    progress = Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(bar_width=24),
                        MofNCompleteColumn(), TextColumn("{task.fields[status]}"), TimeElapsedColumn(), console=console)
    overall = progress.add_task("rounds", total=args.rounds, status="")
    round_task = progress.add_task("", total=None, status="", visible=False)
    progress.start()
    print = progress.console.print

    for r in range(1, args.rounds + 1):
        left = args.rounds - r
        prompt = (f"[Round {r} of {args.rounds}. " + (f"{left} more round(s) with feedback after this one.]" if left
                  else "This is the LAST round: there is no more feedback.]") + "\n\n" + prompt)
        print(f"\n[bold cyan]===== round {r}/{args.rounds} =====[/]  prompt {len(prompt):,} chars")
        (run_dir / f"round_{r}_prompt.md").write_text(prompt, encoding="utf-8")
        entry = {"round": r}
        progress.reset(round_task, total=None, visible=True, description=f"round {r}", status="waiting for Claude")
        live = LiveRound(progress, round_task, show_text=not args.quiet)
        t0 = time.time()
        try:
            out = call_claude(exe, args, prompt, workdir, session_id, live, system=system, mcp_config=mcp_config)
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
        print(f"[dim](Claude finished in {time.time() - t0:.0f}s)[/]")
        session_id = out.get("session_id", session_id)
        (run_dir / f"round_{r}_reply.md").write_text(text, encoding="utf-8")
        entry.update(session_id=session_id, usage=out.get("usage"), duration_ms=out.get("duration_ms"))

        blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)[:MAX_FILES] or [None]
        reports, scored, entry["files"] = [], [], []
        for k, b in enumerate(blocks, 1):
            name = f"file {k}"
            try:
                if b is None:
                    raise ValueError("no ```json block found in the reply")
                spec = json.loads(b)
                name = str(spec.get("name") or name)
                pol = Policy(spec)
                res = evaluate(pol)
                (run_dir / f"round_{r}_file{k}.json").write_text(json.dumps(spec), encoding="utf-8")
                scored.append((res["mean_return"], name, spec, res))
                reports.append(format_result(name, res))
                entry["files"].append({"name": name, "mean_return": res["mean_return"], "mean_steps": res["mean_steps"],
                                       "mean_x": res["mean_x"], "params": pol.net.n_params()})
                print(f"  [green]'{name}'[/] score {res['mean_return']:.1f} · {res['mean_steps']:.0f} steps · "
                      f"x {res['mean_x']:+.2f} m · {res['episodes'][0]['end']}")
            except Exception as e:
                reports.append(f"### {name}\ncould not be used: {e}")
                entry["files"].append({"name": name, "error": str(e)})
                print(Text(f"  '{name}': could not be used: {e}", style="red"))

        fb = ["Results (5 episodes per file):\n\n" + "\n\n".join(reports)]
        if scored:
            score, name, spec, res = max(scored, key=lambda s: s[0])
            history.append((r, score))
            if best is None or score > best["score"]:
                best = {"round": r, "name": name, "score": score, "steps": res["mean_steps"], "x": res["mean_x"], "spec": spec}
            ep = res["episodes"][0]
            every = max(args.trace_every, -(-ep["steps"] // args.trace_rows))  # spread rows over the whole episode
            fb.append(f"Trace of '{name}', episode 0 (one row every {every} steps, plus the last step; "
                      f"left of | = the 17 inputs it received, right = the 6 torques it output):\n```\n"
                      + format_trace(ep, every, args.trace_rows) + "\n```\nIt ended: " + ep["end"]
                      + f" after {ep['steps']} steps.")
            progress.update(overall, status=f"best: round {best['round']} · score {best['score']:.1f} · x {best['x']:+.2f} m")
        fb.append("Score history (best file per round): " + ", ".join(f"round {rr}: {s:.1f}" for rr, s in history)
                  + (f". Best so far: '{best['name']}' from round {best['round']} ({best['score']:.1f})." if best else "."))
        fb.append("Decide your next step: revise, combine what worked, or test a new idea.")
        prompt = "\n\n".join(fb)
        log["rounds"].append(entry)
        (run_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        progress.advance(overall)

    progress.stop()
    shutil.rmtree(workdir, ignore_errors=True)
    if best:
        (run_dir / "best_policy.json").write_text(json.dumps(best["spec"]), encoding="utf-8")
        log["best"] = {k: v for k, v in best.items() if k != "spec"}
        (run_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        console.print(f"\n[bold]Result:[/] best = '{best['name']}' from round {best['round']}: score "
                      f"[bold green]{best['score']:.1f}[/], {best['steps']:.0f} steps, {best['x']:+.2f} m. "
                      "Saved as best_policy.json")
    console.print(f"saved to {run_dir}")
    return {"run_dir": str(run_dir), "best": None if not best else {k: v for k, v in best.items() if k != "spec"}}


def render_main(argv=None):
    """Save a GIF of a policy file walking (episode 0)."""
    p = argparse.ArgumentParser(description="Render a walker policy to a GIF.")
    p.add_argument("policy")
    p.add_argument("--out", help="output .gif (default: next to the policy)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=4, help="keep one frame every N steps (default 4)")
    a = p.parse_args(argv)
    from PIL import Image
    pol = Policy(json.loads(Path(a.policy).read_text(encoding="utf-8")))
    env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=320)
    obs, _ = env.reset(seed=a.seed)
    frames, ret = [], 0.0
    for t in range(1000):
        if t % a.every == 0:
            frames.append(Image.fromarray(env.render()))
        obs, r, term, trunc, info = env.step(pol.act(obs, t))
        ret += r
        if term or trunc:
            frames.append(Image.fromarray(env.render()))
            break
    env.close()
    out = Path(a.out or Path(a.policy).with_suffix(".gif"))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * 0.008 * a.every), loop=0)
    print(f"score {ret:.1f}, {t + 1} steps, x {info['x_position']:+.2f} m -> {out}")


if __name__ == "__main__":
    main()
