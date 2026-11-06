"""A calculator for Claude: one MCP tool, `python`, over stdio. Standard library only.

It runs a short Python snippet in a separate process, from an empty temp folder,
with a time limit, and returns what it prints. It is meant for arithmetic on
numbers already in the conversation (tables, weights), not for touching data:

- imports are limited to math/numeric modules (numpy, math, statistics, ...);
- file, OS and introspection access is refused (open, os, eval/exec, names
  starting with "_", numpy's load/save/fromfile/...), so the MNIST files on disk
  can't be read;
- every snippet and its output is appended to the log file in $WS_CALC_LOG,
  so a run can be audited afterwards.

The checks are a guard rail, not a security sandbox; the log is the real audit.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time

ALLOWED_IMPORTS = {"math", "numpy", "statistics", "fractions", "decimal", "itertools", "functools", "operator", "cmath",
                   "json"}  # json can only format/parse strings here, since open() is refused
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                "delattr", "input", "breakpoint", "help", "memoryview", "exit", "quit"}
BANNED_ATTRS = {"load", "loads", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap", "save", "savez",
                "savez_compressed", "savetxt", "tofile", "DataSource", "lib", "ctypeslib", "testing", "f2py",
                "distutils", "show_config", "dump", "dumps"}
TIMEOUT = 30
MAX_OUT = 20000

TOOL = {
    "name": "python",
    "description": ("Calculator. Runs a short Python snippet and returns what it prints. Use it for arithmetic on "
                    "numbers you already have (copy them into the code). numpy, math and statistics are available. "
                    "It cannot read files or see any images or datasets. Print the results you need."),
    "inputSchema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code; print() the results"}},
                    "required": ["code"]},
}


def check(code: str) -> str | None:
    """Return a reason the snippet is refused, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in ALLOWED_IMPORTS or m.startswith("numpy.") and m.split(".")[1] in BANNED_ATTRS:
                return f"import of {m!r} is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"{node.id!r} is not allowed in the calculator"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRS):
            return f"attribute {node.attr!r} is not allowed in the calculator"
        if isinstance(node, ast.alias) and (node.name in BANNED_ATTRS or node.name.startswith("_")):
            return f"importing {node.name!r} is not allowed in the calculator"
    return None


MAX_KEEP = 2_000_000  # elements per kept array


def prelude(path: str | None = None, store: str | None = None) -> str:
    """If $WS_CALC_DATA points to an existing .npz, preload its arrays as
    UPPERCASE variables (the host decides what goes in it; user code still
    cannot read files itself). If $WS_CALC_STORE names a folder, define
    keep(name, array) / kept(name) so snippets can save weight arrays that
    feature files then reference as "@name"."""
    path = path or os.environ.get("WS_CALC_DATA")
    store = store or os.environ.get("WS_CALC_STORE")
    out = ""
    if path and os.path.exists(path):
        out += ("import numpy as _np\n"
                f"_d = _np.load(r'{path}')\n"
                "for _k in _d.files:\n"
                "    globals()[_k.upper()] = _d[_k].tolist() if _d[_k].dtype.kind == 'U' else _d[_k]\n"
                "del _d, _k\n")
    if store:
        out += (
            "def _mk(_np, _re, _os, _dir, _max):\n"
            "    def keep(name, array):\n"
            "        if not _re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,47}', str(name)):\n"
            "            raise ValueError('keep: name must be letters/digits/_ (max 48), starting with a letter')\n"
            "        a = _np.asarray(array, dtype=_np.float64)\n"
            "        if a.size > _max or not _np.isfinite(a).all():\n"
            "            raise ValueError(f'keep: array must be finite with at most {_max} numbers')\n"
            "        _np.save(_os.path.join(_dir, name + '.npy'), a)\n"
            "        print(f'kept @{name}: shape {a.shape}')\n"
            "    def kept(name):\n"
            "        p = _os.path.join(_dir, str(name).lstrip('@') + '.npy')\n"
            "        if not _re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,47}', str(name).lstrip('@')) or not _os.path.exists(p):\n"
            "            raise KeyError(f'no kept array {name!r}')\n"
            "        return _np.load(p)\n"
            "    return keep, kept\n"
            "import numpy as _np2, re as _re2, os as _os2\n"
            f"keep, kept = _mk(_np2, _re2, _os2, r'{store}', {MAX_KEEP})\n"
            "del _mk, _np2, _re2, _os2\n")
    return out


def run_code(code: str, data_path: str | None = None, store: str | None = None) -> tuple[str, bool]:
    reason = check(code)
    if reason:
        return f"refused: {reason}", True
    code = prelude(data_path, store) + code
    with tempfile.TemporaryDirectory(prefix="ws-calc-") as d:
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}  # no WS_* vars
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, env=env, capture_output=True,
                               text=True, encoding="utf-8", timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return f"timed out after {TIMEOUT}s", True
    out = (p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else "")).strip() or "(no output; use print)"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n... (truncated, {len(out):,} chars total)"
    return out, p.returncode != 0


def log(entry: dict, path: str | None = None):
    path = path or os.environ.get("WS_CALC_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, id_ = req.get("method"), req.get("id")
        if id_ is None:  # notification
            continue
        if method == "initialize":
            ver = (req.get("params") or {}).get("protocolVersion", "2025-06-18")
            reply(id_, {"protocolVersion": ver, "capabilities": {"tools": {}},
                        "serverInfo": {"name": "calc", "version": "0.1.0"}})
        elif method == "tools/list":
            reply(id_, {"tools": [TOOL]})
        elif method == "tools/call":
            params = req.get("params") or {}
            code = (params.get("arguments") or {}).get("code", "")
            if params.get("name") != "python":
                reply(id_, error={"code": -32602, "message": f"unknown tool {params.get('name')}"})
                continue
            t0 = time.time()
            out, is_err = run_code(code)
            log({"time": time.strftime("%H:%M:%S"), "code": code, "output": out, "error": is_err,
                 "seconds": round(time.time() - t0, 2)})
            reply(id_, {"content": [{"type": "text", "text": out}], "isError": is_err})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
