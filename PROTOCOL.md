# The RoboLiveMem protocol

RoboLiveMem separates the **environment** from the **agent**. Each tier runs as a server that owns the simulator and the
clock. An agent is anything that reads an observation and answers with one command. It can be:

- a bare model behind our reference harness;
- a model inside an existing harness (Claude Code, Codex, any MCP client);
- your own system with its own memory, planning or tools.

The environment only sees which commands arrive and when they arrive.

```
             reset(task, seed, condition, clock)                 ┌────────────────────────────┐
  agent  ───────────────────────────────────────────────────────▶│  tier server               │
 (model,      observation: instructions, text, images, time     │  simulator + world clock   │
  harness, ◀──────────────────────────────────────────────────── │  people / robots / machines│
  method)     act(command)          ... until the episode ends  │  keep moving while the     │
         ───────────────────────────────────────────────────────▶│  agent thinks (live clock) │
              result: success, variant, decisions, latency       └────────────────────────────┘
```

## 1. Episodes

An episode is one task, one seed and one condition:

| Condition | Decisive state | Clock | Reported as |
|---|---|---|---|
| `P` | hidden | paused | what current benchmarks measure |
| `L` | hidden | live (or `fixed:<s>`) | what a running world costs |
| `V` | kept in view (sticker, open lid, token) | live (or `fixed:<s>`) | the visible twin: the same change without memory |

The seed decides the scene, the timing and the variant:

- **event**: the decisive state changes while the agent works;
- **decoy**: the same actor and motion, but nothing decisive changes;
- **null**: nothing changes.

The tier reports the variant of every episode. Nothing in the instructions says whether or when a change happens.

## 2. HTTP calls

These are served by `python -m robolivemem.serve` (tabletop), `tiers/semi/server.py` and `tiers/controlled/server.py`:

| Call | Body | Returns |
|---|---|---|
| `POST /reset` | `{"task", "seed", "condition": "P"\|"L"\|"V", "clock": "live"\|"fixed:<s>", "commit": false}` | the first observation |
| `POST /act` | `{"episode", "command": {...}, "latency": <optional s>, "text": <optional reply>}` | the next observation, or the result |
| `POST /observe` | `{"episode"}` | the current observation again (no clock effect) |
| `POST /end` | `{"episode"}` | stops the episode; returns the result |
| `GET /info` | – | tier, tasks, conditions, command vocabulary |

The realistic tier runs one BEHAVIOR-1K server per scene and exposes the same episode through tools. The Python client
(`robolivemem.env.connect("realistic", "house_v3=http://...,...")`) presents it with the same `reset` / `act`
interface.

## 3. Observations

```json
{"episode": "3f2a...", "step": 4, "done": false, "world_time": 12.5,
 "system":   "instructions: the setting, the commands, the reply format",
 "content":  [{"type": "text", "text": "..."}, {"type": "image", "media_type": "image/png", "data": "<base64>"}],
 "dialogue": [["user", [...]], ["assistant", [...]], ...]}
```

- `content` is what is new at this decision. Per tier:

  | Tier | `content` holds |
  |---|---|
  | Tabletop | the task (first step), the result of the last command, a contact sheet of the frames recorded since the previous decision, and the current view with numbered marks |
  | Semi-realistic | the task, the robot's commands so far with their results, up to 30 frames (2 Hz) since the previous decision, and the current view with numbered marks |
  | Controlled | the demonstration (first step), the frames since the previous decision tiled into a few images, the result of the last command, and the current view with numbered marks |
  | Realistic | what the last tool returned: the head camera now with numbered marks, a contact sheet of the stream since the previous request, results and text |

- `dialogue` is optional. It is the exact context the paper's reference harness sent to its model at this decision. You
  may use it, or keep your own memory from `content`.
- The last observation has `"done": true` and `"result"`: `success`, `variant`, `status`, `decisions`,
  `mean_latency_s`, `latency_source`.

The agent never receives simulator state, the seed's variant or the timing of a change.

## 4. Commands

Each decision is one JSON object in the tier's vocabulary. The `system` text of every observation repeats it.

| Tier | Commands |
|---|---|
| Tabletop | `{"action": "pick", "mark": n}`, `{"action": "wait", "seconds": s}`, `{"action": "done"}` |
| Semi-realistic | `{"tool": "pick" \| "place" \| "press", "mark": n}`, `{"tool": "wait", "seconds": s}`, `{"tool": "done"}`, optional `"note"` |
| Controlled | `{"choice": "<listed command label>", "mark": n}`, `{"choice": "wait", "wait": s}` |
| Realistic | `{"tool": "robot_observe"}`, `{"tool": "robot_act", "skill": "pick", "mark": "2"}`, `{"tool": "robot_wait", "seconds": s}`, `{"tool": "robot_frames", "t_from": a, "t_to": b}`, `{"tool": "robot_spec"}`, `{"tool": "robot_done"}` |

An invalid command is reported back and costs a decision, but never a manipulation.

## 5. The clock (what is charged)

- **paused**: nothing moves between the observation and the next command. The world moves only while the robot executes
  or waits.
- **live**: before the command runs, the world advances by the agent's latency. By default the latency is the wall
  time the server measures between delivering the observation and receiving `/act`. Everything inside the agent
  counts: model calls, harness logic, tool calls and network.
- **fixed:L**: every decision is charged `L` seconds, to vary latency on its own.

A client may declare `"latency"` with a command. The server then charges that value instead of the wall time; the
paper's reference agents declare the measured duration of their model request. Every result records
`latency_source`:

- `wall`: measured by the server;
- `declared`: reported by the agent;
- `mixed`: both within one episode.

Results with declared latency must say so.

Under the live clock the world also moves while the agent thinks. People, robots and machines act only while the robot
holds still, as in a real shared workspace, and a safety interlock keeps the robot out of a place another actor is
handling. Answering fast therefore cannot skip a change: the agent has to notice it in the recorded frames.

## 6. Connecting an agent

1. **A model, with our reference harness**: `--agent chat:<provider>:<model>`.
   - Providers: OpenAI, Anthropic, Gemini, Amazon Bedrock, or any OpenAI-compatible server through `OPENAI_BASE_URL`.
   - Keys are read from environment variables (`.env.example`).
2. **Your own agent in Python**: subclass `robolivemem.agents.Agent` and implement `act(obs) -> command`, then pass
   `--agent my_agent.py:MyAgent`. `act` may return `(command, text, latency)` to declare a latency. See
   `examples/my_agent.py`.
3. **An external harness or program**: `--agent "cmd:<command>"` starts it once per episode with `RLM_ENV_URL` and
   `RLM_TIER` set.
   - It can act through MCP tools: add `python -m robolivemem.mcp` to the harness's MCP servers. The tools are
     `robot_instructions`, `robot_act` and `robot_end`; the realistic tier has its own tools in
     `tiers/realistic/client/rlm_mcp.py`.
   - Or it can call the HTTP API directly (`examples/external_harness.py`).
   - The clock starts when the harness first reads the observation.
4. **Directly over HTTP**, from any language (section 2).

## 7. Reporting

The unit of a result is **model + harness**. The same model can score differently in another harness, because harness
logic shapes both what the model sees and how long a decision takes. A report states:

- the model and its reasoning setting;
- the harness: ours (`chat:`), Claude Code, Codex, or yours;
- the latency source (`wall` or `declared`);
- the tiers, tasks and seeds;
- whether the commit check was on.

`python -m robolivemem.eval` writes one row per episode (`--tier all --envs envs.json` runs every tier listed in the
file). `python -m robolivemem.report` turns rows into the paper's
quantities for every tier and every task:

| Column | Definition |
|---|---|
| `P` / `L` | success on event seeds, paused and live |
| `none` | live success on null seeds |
| `L − P` | the live drop, paired over task and seed, with a 95% bootstrap interval |
| `V`, memory `V − L` | on the families with a visible twin, event and decoy seeds |

Class E families are execution controls and are left out of the tier summary.

Settings used in the paper:

- **Realistic tier**:
  - the eight Claude and GPT models ran in Claude Code or Codex (`tiers/realistic/harness/run_agent.py`), and the
    server measured wall time;
  - the other eight ran in our minimal API agent with the same tools and prompt, declaring the model time.
- **Semi-realistic and controlled tiers**: every model ran in our one-call-per-step harness (`chat:`), declaring the
  model time.

## 8. Rules

- Interact only through the protocol. Do not read the simulator, the server's files, the seed or the logs.
- One episode at a time per server: a new `/reset` ends the running episode.
- Report every episode you ran, including failures and invalid episodes, and rerun invalid episodes with the same seed.
