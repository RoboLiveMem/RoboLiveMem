#!/usr/bin/env python3
"""Evaluate a coding agent (Claude Code or Codex, headless) as the RoboLiveMem-Bench agent.

    python3 run_agent.py --backend claude --jobs jobs.jsonl --root <workdir>/evals/<run> [--model claude-opus-5-5]
    python3 run_agent.py --backend codex  --jobs jobs.jsonl --root <workdir>/evals/<run> --model gpt-6-astra --effort medium

Codex runs inside its own workspace-write sandbox (network enabled only so that `rlm` reaches the local server).

jobs.jsonl rows: {"scenario": "s02_annas_mug", "seed": 0, "clock": "live", "url": "http://127.0.0.1:8701"}

For every job: POST /reset to the scenario server, create an isolated work directory with the `rlm`
client, run `claude -p` there (allowed tools: Bash(rlm ...) and Read of ./view/), then POST /end and
store result.json next to the Claude Code transcript.

Credentials: CLAUDE_CODE_OAUTH_TOKEN is read from --secret (a chmod-600 KEY=VALUE file outside any
repository) and passed only through the child environment. It is never printed or written elsewhere.
Usage limits: if Claude Code reports a usage / rate limit, the episode is discarded, the harness waits
(probing every --probe-every seconds with a one-word prompt) and then retries the same job.
"""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT_RE = re.compile(r"usage limit|rate.?limit|limit reached|too many requests|\b429\b|overloaded|quota", re.I)


MODEL_TAG = {"claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet", "gpt-6-luna": "luna", "gpt-6.1-sol": "sol"}


def post(url, path, body=None, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def read_secret(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return env


def child_env(secret, workdir, url, cfgdir):
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME", "SHELL", "TMPDIR")}
    env["PATH"] = f"{workdir / 'bin'}:" + env.get("PATH", "/usr/bin:/bin")
    env["HOME"] = os.environ.get("HOME", str(workdir))
    env["CLAUDE_CONFIG_DIR"] = str(cfgdir)
    env["RLM_URL"] = url
    env["RLM_VIEW"] = str(workdir / "view")
    env["DISABLE_AUTOUPDATER"] = "1"
    env["DISABLE_TELEMETRY"] = "1"
    env["CLAUDE_CODE_OAUTH_TOKEN"] = secret["CLAUDE_CODE_OAUTH_TOKEN"]
    env["MCP_TOOL_TIMEOUT"] = "3600000"   # ms; a skill can take minutes of wall time when the simulator is slow
    return env


TOOLS = ["robot_spec", "robot_observe", "robot_act", "robot_wait", "robot_frames", "robot_done"]
MCP = HERE.parent / "client" / "rlm_mcp.py"


def prompt_for(spec, backend="claude", variant=""):
    if variant == "nostream":
        note = spec["clock_note"].split(" Every response includes")[0]
        if " Safety interlock:" in spec["clock_note"]:
            note += " Safety interlock:" + spec["clock_note"].split(" Safety interlock:")[1]
        instr = spec["instruction"].replace("its head camera kept recording the whole time, and your first observation includes that "
                                            "recording.", "its camera recording of that time is not available to you.")
        return (HERE / "prompt_nostream.md").read_text().format(title=spec["title"], setting=spec["setting"], instruction=instr,
                                                                clock_note=note, budget=spec["manipulation_budget"])
    return (HERE / "prompt.md").read_text().format(title=spec["title"], setting=spec["setting"], instruction=spec["instruction"],
                                                   clock_note=spec["clock_note"], budget=spec["manipulation_budget"])


def codex_env(args, workdir, url):
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME", "SHELL", "TMPDIR", "HOME")}
    env["PATH"] = f"{workdir / 'bin'}:" + env.get("PATH", "/usr/bin:/bin")
    env["CODEX_HOME"] = args.codex_home
    env["RLM_URL"] = url
    env["RLM_VIEW"] = str(workdir / "view")
    return env


def run_codex(args, workdir, url, prompt, log_path, cwd=None):
    logdir = workdir
    workdir = cwd or workdir      # v2s: the agent works in an opaque directory (its path must not reveal scenario or seed)
    kv = [("RLM_URL", url), ("RLM_LOG", str(logdir / "tools.jsonl")), ("RLM_IMGDIR", str(logdir / "imgs"))] + ([("RLM_NOSTREAM", "1")] if getattr(args, "variant", "") == "nostream" else [])
    envs = "{" + ", ".join(f'{k}="{v}"' for k, v in kv) + "}"
    cmd = [args.codex, "exec", "--skip-git-repo-check", "--json", "-m", args.model, "-C", str(workdir),
           "-c", f"model_reasoning_effort={args.effort or 'medium'}", "--sandbox", "read-only",
           "-c", 'mcp_servers.rlm.command="python3"', "-c", f'mcp_servers.rlm.args=["{MCP}"]',
           "-c", f"mcp_servers.rlm.env={envs}", "-c", 'mcp_servers.rlm.default_tools_approval_mode="approve"',
           # a skill can take minutes of wall time when the simulator is slow; never let the client give up on it
           "-c", "mcp_servers.rlm.tool_timeout_sec=3600", prompt]
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            p = subprocess.run(cmd, cwd=workdir, env=codex_env(args, workdir, url), stdout=lf, stderr=subprocess.STDOUT,
                               timeout=args.timeout, stdin=subprocess.DEVNULL)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, time.time() - t0


def summarize_codex(log_path):
    n_tool = 0; errs = []; final = None; usage = None; non_rlm = []
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        try:
            m = json.loads(line)
        except Exception:
            if LIMIT_RE.search(line):
                errs.append(line[:300])
            continue
        t = m.get("type", "")
        item = m.get("item") or {}
        if t == "item.completed" and item.get("type") in ("command_execution", "mcp_tool_call"):
            n_tool += 1
            if item.get("type") == "command_execution":
                non_rlm.append(str(item.get("command", ""))[:120])
        if t == "item.completed" and item.get("type") == "agent_message":
            final = item.get("text")
        if t == "turn.completed":
            usage = m.get("usage")
        if t in ("error", "turn.failed"):
            errs.append(json.dumps(m)[:300])
    done_ok = usage is not None
    limited = any(LIMIT_RE.search(e) for e in errs) and not done_ok
    return dict(result_subtype="success" if done_ok else ("error" if errs else None), is_error=bool(errs) and not done_ok,
                cost_usd=None, usage=usage, tool_calls=n_tool, limited=limited, final_text=(final or "")[:2000],
                non_rlm_commands=non_rlm[:10], errors=errs[:3])


def run_claude(args, secret, workdir, url, prompt, log_path, cwd=None):
    cfgdir = Path(args.root) / "claude_config"
    cfgdir.mkdir(parents=True, exist_ok=True)
    logdir = workdir
    workdir = cwd or workdir      # v2s: the agent works in an opaque directory (its path must not reveal scenario or seed)
    view = workdir / "view"
    mcp_cfg = logdir / "mcp.json"
    mcp_cfg.write_text(json.dumps({"mcpServers": {"rlm": {"command": "python3", "args": [str(MCP)],
                                                          "env": dict({"RLM_URL": url, "RLM_LOG": str(logdir / "tools.jsonl"), "RLM_IMGDIR": str(logdir / "imgs")},
                                                                      **({"RLM_NOSTREAM": "1"} if getattr(args, "variant", "") == "nostream" else {}))}}}))
    env = child_env(secret, workdir, url, cfgdir)
    if getattr(args, "max_out", ""):
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(args.max_out)
    cmd = [args.claude, "-p", prompt, "--model", args.model, "--output-format", "stream-json", "--verbose",
           "--max-turns", str(args.max_turns), "--mcp-config", str(mcp_cfg), "--strict-mcp-config",
           "--allowedTools", ",".join(f"mcp__rlm__{t}" for t in TOOLS),
           "--disallowedTools", "Bash,Read,Write,Edit,NotebookEdit,WebFetch,WebSearch,Task,Glob,Grep"]
    if args.effort:
        cmd += ["--effort", args.effort]
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            p = subprocess.run(cmd, cwd=workdir, env=env, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=args.timeout, stdin=subprocess.DEVNULL)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, time.time() - t0


def summarize_log(log_path):
    res = None; n_tool = 0; errs = []
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        try:
            m = json.loads(line)
        except Exception:
            if LIMIT_RE.search(line):
                errs.append(line[:300])
            continue
        if m.get("type") == "result":
            res = m
        if m.get("type") == "assistant":
            for c in (m.get("message", {}).get("content") or []):
                if c.get("type") == "tool_use":
                    n_tool += 1
        if m.get("type") in ("system",) and m.get("subtype") in ("api_error", "error"):
            errs.append(json.dumps(m)[:300])
    limited = False
    if res is not None and res.get("is_error") and LIMIT_RE.search(json.dumps(res)):
        limited = True
    if any(LIMIT_RE.search(e) for e in errs) and (res is None or res.get("is_error")):
        limited = True
    return dict(result_subtype=None if res is None else res.get("subtype"), is_error=None if res is None else res.get("is_error"),
                cost_usd=None if res is None else res.get("total_cost_usd"), num_turns=None if res is None else res.get("num_turns"),
                duration_ms=None if res is None else res.get("duration_ms"), tool_calls=n_tool, limited=limited,
                final_text=None if res is None else str(res.get("result"))[:2000])


def probe(args, secret):
    if args.backend == "codex":
        tmp = Path(args.root) / "probe"; tmp.mkdir(exist_ok=True); (tmp / "bin").mkdir(exist_ok=True)
        try:
            p = subprocess.run([args.codex, "exec", "--skip-git-repo-check", "-m", args.model, "Reply with the single word: ok"],
                               cwd=tmp, env=codex_env(args, tmp, "http://127.0.0.1:1"), capture_output=True, text=True, timeout=300)
            out = (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return False, "probe timeout"
        return (p.returncode == 0 and not LIMIT_RE.search(out)), re.sub(r"eyJ[A-Za-z0-9._-]{20,}", "<redacted>", out[-300:])
    cfgdir = Path(args.root) / "claude_config"; cfgdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.root) / "probe"; tmp.mkdir(exist_ok=True); (tmp / "bin").mkdir(exist_ok=True)
    try:
        p = subprocess.run([args.claude, "-p", "Reply with the single word: ok", "--model", args.model, "--output-format", "json"],
                           cwd=tmp, env=child_env(secret, tmp, "http://127.0.0.1:1", cfgdir), capture_output=True, text=True, timeout=300)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "probe timeout"
    ok = p.returncode == 0 and not LIMIT_RE.search(out) and '"is_error":false' in out.replace(" ", "")
    return ok, re.sub(r"sk-ant-[A-Za-z0-9_-]+", "<redacted>", out[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backend", default="claude", choices=["claude", "codex"])
    ap.add_argument("--secret", default=os.path.expanduser("~/.config/rlm_claude/env"))
    ap.add_argument("--codex", default=shutil.which("codex") or "codex")
    ap.add_argument("--codex-home", default=os.path.expanduser("~/.config/rlm_codex/home"))
    ap.add_argument("--claude", default=shutil.which("claude") or "claude")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="")
    ap.add_argument("--max-out", default="", help="Claude Code CLAUDE_CODE_MAX_OUTPUT_TOKENS (empty: Claude Code default)")
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--probe-every", type=int, default=900)
    ap.add_argument("--record", type=int, default=1)
    ap.add_argument("--url", default="", help="worker override: serve every job from this scenario server")
    ap.add_argument("--server-out", default="", help="worker override: episode output dir inside that server")
    args = ap.parse_args()
    args.root = str(Path(args.root).resolve())   # v2s: the agent runs in another cwd, so every path must be absolute
    secret = {}
    secrets = [p for p in args.secret.split(",") if p]
    sidx = 0; limited_since_ok = set()
    if args.backend == "claude":
        secret = read_secret(secrets[sidx])
        assert "CLAUDE_CODE_OAUTH_TOKEN" in secret, "secret file must define CLAUDE_CODE_OAUTH_TOKEN"
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.jsonl"
    done = set()
    if summary.exists():
        for line in summary.read_text().splitlines():
            r = json.loads(line)
            if r.get("valid"):
                done.add((r.get("backend", "claude"), r.get("model"), r["scenario"], r["seed"], r["clock"], r.get("variant", ""), r.get("harness", "minimal")))
    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    client = HERE.parent / "client" / "rlm"
    base_args = vars(args).copy()
    def refresh_done():
        # several workers (one per scenario server) may share one root: re-read what is already finished
        if summary.exists():
            for line in summary.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("valid"):
                    done.add((r.get("backend", "claude"), r.get("model"), r["scenario"], r["seed"], r["clock"], r.get("variant", ""), r.get("harness", "minimal")))

    claims = root / "claims"; claims.mkdir(exist_ok=True)
    for job in jobs:
        job = dict(job)
        if args.url:
            job["url"] = args.url
        if args.server_out:
            job["server_out"] = args.server_out
        for k in ("backend", "model", "effort", "max_out"):
            setattr(args, k, job.get(k, base_args[k]))
        args.variant = job.get("variant", "")
        harness = job.get("harness", "minimal")
        key = (args.backend, args.model, job["scenario"], job["seed"], job["clock"], args.variant, harness)
        if args.backend == "codex" and (root / "NO_CODEX").exists():
            continue
        refresh_done()
        if key in done:
            continue
        try:   # claim the job so that another worker on the same root skips it
            os.close(os.open(str(claims / "_".join(str(x) for x in key).replace("/", "-")), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        if args.backend == "claude" and not secret:
            secret = read_secret(secrets[sidx])
        attempt = 0
        while True:
            if args.backend == "codex" and (root / "NO_CODEX").exists():
                break
            attempt += 1
            tag = MODEL_TAG.get(args.model, "")      # default models keep the short names of earlier runs
            name = f"{args.backend}{'_' + tag if tag else ''}{'_' + args.variant if args.variant else ''}{'_commit' if harness == 'commit' else ''}_{job['scenario']}_{job['clock']}_s{job['seed']}_a{attempt}"
            wd = root / "episodes" / name
            if wd.exists():
                shutil.rmtree(wd)
            (wd / "bin").mkdir(parents=True); (wd / "view").mkdir()
            import uuid
            cwd = root / "ws" / uuid.uuid4().hex[:12]
            (cwd / "bin").mkdir(parents=True); (cwd / "view").mkdir()
            (wd / "agent_cwd.txt").write_text(str(cwd))
            try:
                spec = post(job["url"], "/reset", dict(scenario=job["scenario"], seed=job["seed"], clock=job["clock"],
                                                  out=job.get("server_out", "") + f"/{name}" if job.get("server_out") else None,
                                                  record=bool(args.record), visible=(args.variant == "visible"),
                                                  hard=(args.variant == "hard"), commit=(harness == "commit")))
            except Exception as e:   # main_1115: a failed reset must not kill the worker
                err = getattr(e, "read", lambda: b"")()
                print(time.strftime("%H:%M:%S"), "RESET FAILED", name, str(e)[:200], err[-600:], flush=True)
                time.sleep(60)
                if attempt >= 4:
                    break
                continue
            (wd / "spec.json").write_text(json.dumps(spec, indent=1))
            prompt = prompt_for(spec, args.backend, args.variant)
            (wd / "prompt.md").write_text(prompt)
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            if args.backend == "claude":
                rc, wall = run_claude(args, secret, wd, job["url"], prompt, wd / "agent.jsonl", cwd=cwd)
            else:
                rc, wall = run_codex(args, wd, job["url"], prompt, wd / "agent.jsonl", cwd=cwd)
            final = post(job["url"], "/end")
            (wd / "result.json").write_text(json.dumps(final, indent=1))
            s = summarize_log(wd / "agent.jsonl") if args.backend == "claude" else summarize_codex(wd / "agent.jsonl")
            row = dict(backend=args.backend, model=args.model, effort=(args.effort or None), max_out=(args.max_out or None),
                       variant=args.variant, harness=harness, episode=name,
                       secret_idx=(sidx if args.backend == "claude" else None), scenario=job["scenario"], seed=job["seed"], clock=job["clock"],
                       attempt=attempt, rc=rc, wall_s=round(wall, 1),
                       success=final.get("success"), reason=final.get("reason"), end=final.get("end"), world_t=final.get("world_t"),
                       agent_t=final.get("agent_t"), manipulation_actions=final.get("manipulation_actions"),
                       latencies=[round(x["latency"], 1) for x in final.get("timeline", []) if x.get("kind") == "think"], **s)
            row["valid"] = not s["limited"] and rc != "timeout" and s["result_subtype"] is not None
            with open(summary, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(time.strftime("%H:%M:%S"), "end", name, "success" if row["success"] else "fail", row["reason"],
                  "valid" if row["valid"] else "INVALID", flush=True)
            if row["valid"]:
                limited_since_ok.clear()
                break
            if s["limited"] and args.backend == "claude" and len(secrets) > 1:
                limited_since_ok.add(sidx)
                if len(limited_since_ok) < len(secrets):
                    sidx = (sidx + 1) % len(secrets)
                    while sidx in limited_since_ok:
                        sidx = (sidx + 1) % len(secrets)
                    secret = read_secret(secrets[sidx])
                    print(time.strftime("%H:%M:%S"), "usage limit: switching to secret", sidx, flush=True)
                    continue
                limited_since_ok.clear()
            if s["limited"]:
                while True:
                    print(time.strftime("%H:%M:%S"), "usage limit: waiting", args.probe_every, "s", flush=True)
                    time.sleep(args.probe_every)
                    ok, msg = probe(args, secret)
                    print(time.strftime("%H:%M:%S"), "probe", ok, msg.replace("\n", " ")[:200], flush=True)
                    if ok:
                        break
            if attempt >= 4:
                break


if __name__ == "__main__":
    main()
