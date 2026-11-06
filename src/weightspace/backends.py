"""Which model writes the weights.

  claude    (default) Claude Code in headless mode on your Claude subscription.
            API keys are stripped from its environment, so it never bills per token.
  deepseek  DeepSeek's API, billed per token to YOUR DeepSeek account. Only used
            when you pass --backend deepseek. The key is read from the
            DEEPSEEK_API_KEY environment variable, or from a line
            DEEPSEEK_API_KEY=... in a .env file in the project folder. It is
            never printed or saved in run logs.

Both backends get the same system prompt, the same round messages and the same
calculator (same guard rails, same calculator.jsonl log). DeepSeek models
can't see pictures, so --sheets is ignored for them.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import calc_server
from .generate import ROOT, call_claude, find_claude

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODELS = ["deepseek-v4-pro", "deepseek-flash"]
MAX_TOOL_STEPS = 60


def add_args(p):
    g = p.add_argument_group("model backend")
    g.add_argument("--backend", default="claude", choices=["claude", "deepseek"],
                   help="claude = Claude Code on your subscription (default); deepseek = DeepSeek API, "
                        "billed to your DeepSeek key")
    g.add_argument("--deepseek-model", default="deepseek-v4-pro",
                   help=f"DeepSeek model id (default deepseek-v4-pro; also {', '.join(DEEPSEEK_MODELS[1:])})")
    g.add_argument("--deepseek-effort", default="high", choices=["low", "high", "max"],
                   help="DeepSeek thinking effort (default high)")


def deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "DEEPSEEK_API_KEY":
                key = v.strip().strip('"').strip("'")
    if not key:
        sys.exit("--backend deepseek needs a key: set DEEPSEEK_API_KEY, or put DEEPSEEK_API_KEY=sk-... in "
                 f"{env_file.name} in the project folder")
    return key


def make(args, system: str, workdir: str, mcp_config: str | None, calc_log: Path, calc_data: Path | None,
         calc_store: Path | None = None):
    if args.backend == "deepseek":
        return DeepSeek(args, system, calculator=mcp_config is not None, calc_log=calc_log, calc_data=calc_data,
                        calc_store=calc_store)
    return Claude(args, system, workdir, mcp_config)


class Claude:
    name = "Claude"
    sees_images = True

    def __init__(self, args, system, workdir, mcp_config):
        self.exe, self.args, self.system, self.workdir, self.mcp = find_claude(), args, system, workdir, mcp_config
        self.session_id = None

    def send(self, prompt: str, live, images: list[bytes] | None = None) -> dict:
        out = call_claude(self.exe, self.args, prompt, self.workdir, self.session_id, live,
                          system=self.system, mcp_config=self.mcp, images=images)
        if "safeguards flagged" in (out.get("result") or ""):
            live.progress.console.print("[yellow]safety filter flagged this round (likely a false positive); "
                                        "retrying once...[/]")
            out = call_claude(self.exe, self.args, prompt, self.workdir, self.session_id, live,
                              system=self.system, mcp_config=self.mcp, images=images)
        self.session_id = out.get("session_id", self.session_id)
        return out


def _ev(event: dict) -> dict:
    return {"type": "stream_event", "event": event}


class DeepSeek:
    """OpenAI-format chat completions over plain HTTPS (no extra packages).
    The whole conversation is kept here and resent each request, including
    reasoning_content, which DeepSeek requires once tools are in play."""
    sees_images = False

    def __init__(self, args, system, calculator: bool, calc_log: Path, calc_data: Path | None,
                 calc_store: Path | None = None):
        self.calc_store = str(calc_store) if calc_store else None
        self.key = deepseek_key()
        self.args, self.model = args, args.deepseek_model
        self.name = f"DeepSeek ({self.model})"
        self.calculator, self.calc_log, self.calc_data = calculator, calc_log, calc_data
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
        self.tools = [{"type": "function", "function": {
            "name": calc_server.TOOL["name"], "description": calc_server.TOOL["description"],
            "parameters": calc_server.TOOL["inputSchema"]}}] if calculator else None

    def _request(self, live) -> dict:
        body = {"model": self.model, "messages": self.messages, "stream": True,
                "stream_options": {"include_usage": True},
                "thinking": {"type": "enabled"}, "reasoning_effort": self.args.deepseek_effort}
        if self.tools:
            body["tools"] = self.tools
        req = urllib.request.Request(DEEPSEEK_URL, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        msg = {"role": "assistant", "content": "", "reasoning_content": ""}
        calls: dict[int, dict] = {}
        usage = None
        kind = None
        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=self.args.timeout)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:1000]
                if e.code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"DeepSeek API error {e.code}: {detail}") from None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        if kind != "thinking":
                            kind = "thinking"
                            live.handle(_ev({"type": "content_block_start", "content_block": {"type": "thinking"}}))
                        msg["reasoning_content"] += d["reasoning_content"]
                    if d.get("content"):
                        if kind != "text":
                            kind = "text"
                            live.handle(_ev({"type": "content_block_start", "content_block": {"type": "text"}}))
                        msg["content"] += d["content"]
                        live.handle(_ev({"type": "content_block_delta",
                                         "delta": {"type": "text_delta", "text": d["content"]}}))
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                                 "function": {"name": "", "arguments": ""}})
                        c["id"] = tc.get("id") or c["id"]
                        f = tc.get("function") or {}
                        c["function"]["name"] += f.get("name") or ""
                        c["function"]["arguments"] += f.get("arguments") or ""
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        if usage:
            for k in self.usage:
                self.usage[k] += int(usage.get(k) or 0)
        return msg

    def send(self, prompt: str, live, images: list[bytes] | None = None) -> dict:
        self.messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        text = ""
        for _ in range(MAX_TOOL_STEPS):
            msg = self._request(live)
            self.messages.append(msg)
            text = msg["content"]
            if not msg.get("tool_calls"):
                break
            for c in msg["tool_calls"]:
                live.handle(_ev({"type": "content_block_start", "content_block": {"type": "tool_use"}}))
                try:
                    code = json.loads(c["function"]["arguments"] or "{}").get("code", "")
                except json.JSONDecodeError as e:
                    code, out, err = None, f"refused: tool arguments were not valid JSON ({e})", True
                if code is not None:
                    if c["function"]["name"] != calc_server.TOOL["name"]:
                        out, err = f"unknown tool {c['function']['name']!r}", True
                    else:
                        t1 = time.time()
                        out, err = calc_server.run_code(code, str(self.calc_data) if self.calc_data else None,
                                                        self.calc_store)
                        calc_server.log({"time": time.strftime("%H:%M:%S"), "code": code, "output": out,
                                         "error": err, "seconds": round(time.time() - t1, 2)}, str(self.calc_log))
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        live.finish()
        return {"result": text, "session_id": None, "usage": dict(self.usage),
                "duration_ms": int((time.time() - t0) * 1000)}
