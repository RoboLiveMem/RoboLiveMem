#!/usr/bin/env python3
"""Realistic tier, API models: the same scenario servers, prompt, tools and observations as the Claude Code / Codex
runs (harness/run_agent.py, client/rlm_mcp.py), with the model called directly through robolivemem.models.Chat
(OpenAI, Anthropic, Gemini, Bedrock or any OpenAI-compatible server; keys from environment variables, see .env.example).
Tool calls are written as one JSON object at the end of each reply.

Clock: the live clock is charged only the measured time of the model requests (meta.latency_s) since the previous
server response, passed to the server as the request's `latency`; harness overhead is not charged. Failed model calls
are retried and not charged; an episode with a call that still fails is invalid.

usage: python3 run_agent_api.py --jobs J.jsonl --root R --url http://127.0.0.1:PORT --server-out /out/<session>/episodes
jobs.jsonl rows: {"scenario", "seed", "clock", "backend": "api", "model": "anthropic:claude-sonnet-5-5", "max_tokens": 8192}
("model" is any robolivemem.models spec or a paper alias such as "grok-4.7")"""
import argparse, json, os, re, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_agent as RA                                     # post(), prompt_for(): identical task prompt
sys.path.insert(0, str(HERE.parent / "client"))
os.environ.setdefault("RLM_URL", "http://127.0.0.1:1")
import rlm_mcp as MCP                                      # obs_content(), call(): identical observations
sys.path.insert(0, str(HERE.parents[2]))                   # the repository root (robolivemem package)
from robolivemem.models import Chat                        # noqa: E402

PROTOCOL = """

HOW TO CALL A TOOL
Each of your replies must end with exactly one tool call written as a JSON object on its own line, for example
{"tool": "robot_observe"}
{"tool": "robot_act", "skill": "pick", "mark": "2"}
{"tool": "robot_wait", "seconds": 5}
{"tool": "robot_frames", "t_from": 12.0, "t_to": 20.0}
{"tool": "robot_spec"}
{"tool": "robot_done"}
You may write brief reasoning before the JSON. The tool result (text and images) comes back as the next message."""
LIMIT_RE = re.compile(r"throttl|too many requests|\b429\b|quota|rate.?exceeded|limit exceeded", re.I)


_CHATS = {}


def relay(body, timeout=400):
    """One model call -> {"text", "meta"}; meta["latency_s"] is the measured request time (what the live clock charges)."""
    key = (body["model"], body.get("effort"), int(body.get("max_tokens", 4096)))
    if key not in _CHATS:
        _CHATS[key] = Chat(body["model"], max_tokens=key[2], effort=body.get("effort"), timeout_s=timeout)
    text, meta = _CHATS[key](body.get("system", ""), body["turns"])
    return {"text": text, "meta": meta}


def wait_relay():
    """Kept for the episode loop: direct API calls have no relay to wait for."""
    return


def parse_call(text):
    """Last top-level JSON object in the reply that names a tool."""
    best = None
    for m in re.finditer(r"\{", text or ""):
        depth = 0
        for j in range(m.start(), len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(text[m.start():j + 1])
                        if isinstance(o, dict) and isinstance(o.get("tool"), str):
                            best = o
                    except Exception:
                        pass
                    break
    return best


def to_parts(content):
    parts = []
    for c in content:
        if c["type"] == "text":
            parts.append({"text": c["text"]})
        elif c["type"] == "image":
            parts.append({"jpeg": c["data"]})
    return parts


def limit_images(turns, keep):
    """Keep the `keep` most recent images; older ones become a short text note (the conversation stays in order)."""
    n = 0; out = []
    for role, parts in reversed(turns):
        np_ = []
        for p in reversed(parts):
            if "jpeg" in p:
                n += 1
                np_.append(p if n <= keep else {"text": "[an earlier image, omitted to save context]"})
            else:
                np_.append(p)
        out.append([role, list(reversed(np_))])
    return list(reversed(out))


def run_episode(args, job, wd, url, prompt):
    """One episode. Returns (summary dict, wall seconds)."""
    MCP.URL = url
    system = prompt + PROTOCOL
    turns = [["user", [{"text": "Begin. Call one robot_* tool."}]]]
    agent_log = open(wd / "agent.jsonl", "w"); tools_log = open(wd / "tools.jsonl", "w")
    n_tool = n_calls = n_bad = 0; errs = []; limited = False; pending = 0.0; t_start = time.time(); final_text = ""
    try:
        RA.post(url, "/hello")
    except Exception:
        pass
    over = False
    for turn in range(args.max_turns):
        if time.time() - t_start > args.timeout:
            errs.append("episode wall timeout"); break
        body = dict(model=job["model"], max_tokens=int(job.get("max_tokens", 4096)), effort=job.get("effort"), system=system,
                    turns=limit_images(turns, args.keep_images))
        out = None
        for k in range(12):
            try:
                out = relay(body)
            except Exception as e:
                out = {"text": "", "meta": {"error": "relay unreachable: " + type(e).__name__}}
                agent_log.write(json.dumps(dict(turn=turn, try_=k, meta=out["meta"])) + "\n"); agent_log.flush()
                wait_relay()          # the tunnel came back: retry this call (the episode stays valid only if it succeeds)
            meta = out.get("meta") or {}
            agent_log.write(json.dumps(dict(turn=turn, try_=k, meta=meta, reply=(out.get("text") or "")[:4000])) + "\n"); agent_log.flush()
            if not meta.get("error") and meta.get("http_status", 200) == 200:
                break
            if LIMIT_RE.search(str(meta.get("error"))):
                limited = True
            time.sleep(min(120, 5 * 2 ** k))
        meta = out.get("meta") or {}
        if meta.get("error") or meta.get("http_status", 200) != 200:
            errs.append(str(meta.get("error"))[:200]); break
        n_calls += 1; pending += float(meta.get("latency_s") or 0.0)
        text = out.get("text") or ""
        turns.append(["assistant", [{"text": text or "(empty reply)"}]])
        call = parse_call(text)
        if call is None:
            n_bad += 1
            turns.append(["user", [{"text": "No tool call found. End your reply with exactly one JSON tool call, e.g. "
                                            "{\"tool\": \"robot_observe\"}."}]])
            if n_bad >= 6:
                errs.append("no parseable tool call"); break
            continue
        name = call.pop("tool")
        if name == "robot_spec":      # /spec is never charged by the server: carry the time to the next charged request
            lat = 0.0
        else:
            lat = round(pending, 3); pending = 0.0
        # charge the measured API time on this request (rlm_mcp.http sends no latency; pass it explicitly)
        t0 = time.time()
        orig = MCP.http
        MCP.http = lambda path, b=None, _o=orig, _l=lat: _o(path, dict(b or {}, latency=_l))
        try:
            content, is_err = MCP.call(name, call)
        except Exception as e:
            content, is_err = [{"type": "text", "text": f"ERROR: {e}"}], True
        finally:
            MCP.http = orig
        n_tool += 1
        tools_log.write(json.dumps(dict(t=round(t0, 3), dt=round(time.time() - t0, 3), tool=name, args=call, charged=lat,
                                        text=[c.get("text") for c in content if c["type"] == "text"])) + "\n"); tools_log.flush()
        turns.append(["user", to_parts(content) or [{"text": "(no output)"}]])
        txt = " ".join(c.get("text", "") for c in content if c["type"] == "text")
        if name == "robot_done" or "EPISODE OVER" in txt:
            final_text = text[-2000:]; over = True; break
    agent_log.close(); tools_log.close()
    ok = over and not errs
    return dict(result_subtype="success" if ok else ("error" if errs else "max_turns"), is_error=bool(errs), tool_calls=n_tool,
                model_calls=n_calls, unparsed=n_bad, limited=limited, final_text=final_text, errors=errs[:3]), time.time() - t_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--url", required=True); ap.add_argument("--server-out", default="")
    ap.add_argument("--max-turns", type=int, default=80); ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--keep-images", type=int, default=12); ap.add_argument("--record", type=int, default=1)
    args = ap.parse_args()
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.jsonl"; claims = root / "claims"; claims.mkdir(exist_ok=True)

    def done_keys():
        d = set()
        if summary.exists():
            for line in summary.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("valid"):
                    d.add((r.get("backend"), r.get("model"), r["scenario"], r["seed"], r["clock"], r.get("variant", ""), r.get("harness", "minimal")))
        return d

    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    resets_failed = 0
    for job in jobs:
        variant = job.get("variant", ""); harness = job.get("harness", "minimal")
        key = ("api", job["model"], job["scenario"], job["seed"], job["clock"], variant, harness)
        if key in done_keys():
            continue
        try:
            os.close(os.open(str(claims / "_".join(str(x) for x in key).replace("/", "-")), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        for attempt in range(1, 5):
            if (root / "STOP_API").exists():
                print(time.strftime("%H:%M:%S"), "STOP_API present: stopping", flush=True); return
            name = f"api_{job['model']}{'_' + variant if variant else ''}_{job['scenario']}_{job['clock']}_s{job['seed']}_a{attempt}"
            wd = root / "episodes" / name
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            try:
                spec = RA.post(args.url, "/reset", dict(scenario=job["scenario"], seed=job["seed"], clock=job["clock"],
                                                        out=(args.server_out + f"/{name}") if args.server_out else None,
                                                        record=bool(args.record), visible=(variant == "visible"),
                                                        hard=False, commit=(harness == "commit")))
            except Exception as e:
                print(time.strftime("%H:%M:%S"), "RESET FAILED", name, str(e)[:200], flush=True)
                resets_failed += 1
                if resets_failed >= 2:   # the server is gone: give the job back and stop this worker
                    try: os.remove(str(claims / "_".join(str(x) for x in key).replace("/", "-")))
                    except OSError: pass
                    print(time.strftime("%H:%M:%S"), "server unreachable: released", name, "and stopping", flush=True); return
                time.sleep(60); continue
            resets_failed = 0
            (wd / "spec.json").write_text(json.dumps(spec, indent=1))
            prompt = RA.prompt_for(spec, "api", variant)
            (wd / "prompt.md").write_text(prompt + PROTOCOL)
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
            s, wall = run_episode(args, job, wd, args.url, prompt)
            final = RA.post(args.url, "/end")
            (wd / "result.json").write_text(json.dumps(final, indent=1))
            row = dict(backend="api", model=job["model"], effort=job.get("effort"), max_out=job.get("max_tokens"), variant=variant,
                       harness=harness, episode=name, secret_idx=None, scenario=job["scenario"], seed=job["seed"],
                       clock=job["clock"], attempt=attempt, rc=0, wall_s=round(wall, 1), success=final.get("success"),
                       reason=final.get("reason"), end=final.get("end"), world_t=final.get("world_t"), agent_t=final.get("agent_t"),
                       manipulation_actions=final.get("manipulation_actions"),
                       latencies=[round(x["latency"], 1) for x in final.get("timeline", []) if x.get("kind") == "think"], **s)
            row["valid"] = not s["is_error"] and not s["limited"]
            with open(summary, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(time.strftime("%H:%M:%S"), "end", name, "success" if row["success"] else "fail", row["reason"],
                  "valid" if row["valid"] else "INVALID " + "; ".join(s["errors"]), flush=True)
            if row["valid"]:
                break
            if s["limited"]:
                time.sleep(300)


if __name__ == "__main__":
    main()
