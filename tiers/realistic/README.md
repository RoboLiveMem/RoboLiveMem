# Realistic tier: 16 BEHAVIOR-1K scenes

People, other robots and machines change the decisive state while the agent thinks. The agent works as a tool-using
agent (Claude Code, Codex, or our own agent for any API model) and decides when to look, act or wait.

| Scene | Class | Actor | Scenario file |
|---|---|---|---|
| Grandpa's mug | U | son | `v3_mugs` |
| Express tote | U | picker | `v3_totes` |
| Car keys | U | teenager | `v3_keys` |
| Gauge in bins | U | technician | `v3_bins` |
| Mom's tea | W | daughter | `v3_sugar` |
| Catalyst dose | W | technician | `v3_reagent` |
| Put away | U | helper robot | `s01_put_away` |
| Anna's mug | U | helper robot | `s02_annas_mug` |
| Vacuum and ball | U | robot vacuum | `s03_vacuum_ball` |
| Salted soup | W | helper robot | `s04_salted_soup` |
| Rotary reject | U | rotary table | `s05_rotary_reject` |
| Conveyor lanes | U | conveyor | `s06_conveyor_lanes` |
| Press lamp | W | press | `s07_press_lamp` |
| Parcel locker | U | courier robot | `s08_parcel_locker` |
| Kitting partner | W | partner robot | `s09_kitting_partner` |
| Cloche pass | U | expo robot | `s10_cloche_pass` |

Scene files (`server/scenes/`): `house_v3.json` (Grandpa's mug, Car keys, Mom's tea), `hall_v3.json` (Express tote,
Gauge in bins, Catalyst dose), `house.json` (s01-s04), `restaurant.json` (s10), `industrial.json` (s05-s09).

## Layout

```
server/            runs inside the BEHAVIOR-1K container (OmniGibson 3.9.2, Isaac Sim 5.1)
  bench_server.py  loads one scene and its scenarios, serves the agent protocol over HTTP
  core.py          Episode: paused / live clock, head-camera stream, recording, verification, commit check
  v3.py            templates of the people scenes (TrackOne, Stash, DoseOnce)
  common.py        StationScenario: slots, containers (lids, hinged doors), path movers, generic skills
  station.py       kinematic arm motions;  props.py  primitive-built machines and furniture
  scenarios/       one file per scenario;  scenes/  scene specs
  launch.py        docker launcher (explicit GPU mounts, one published port)
client/rlm_mcp.py  MCP stdio server with the robot_* tools (what Claude Code and Codex use)
client/rlm         the same calls on the command line (stdlib only)
harness/           run_agent_api.py (any API model), run_agent.py (Claude Code or Codex, headless, MCP tools only),
                   scripted_agent.py (stale / privileged references), aggregate*.py, prompt.md
video/             paused | live side-by-side videos
tests_dev/         a mock simulator to exercise scenario logic without a GPU (development aid)
```

## Requirements

A Linux machine with an NVIDIA RTX GPU, Docker, the image `stanfordvl/behavior:3.9.2` and the BEHAVIOR-1K assets
(licensed separately, see the BEHAVIOR-1K website).

## Run

```bash
# 1. start a scenario server (one per scene), with this folder mounted as /work
mkdir -p $WORK && cp -r . $WORK/bench
python3 server/launch.py --root $WORK --assets $BEHAVIOR_ASSETS --gpu 0 --name rlm_house --port 8701 -- \
  python /work/server/bench_server.py --session /out/house --scene /work/server/scenes/house_v3.json \
  --scenarios v3_mugs,v3_keys,v3_sugar --port 8701 --code /work/server

# 2a. any API model (keys from environment variables, see ../../.env.example)
echo '{"scenario": "v3_mugs", "seed": 0, "clock": "live", "model": "anthropic:claude-sonnet-5-5", "max_tokens": 8192}' > jobs.jsonl
python3 harness/run_agent_api.py --jobs jobs.jsonl --root $WORK/evals/api --url http://127.0.0.1:8701 \
  --server-out /out/house/episodes

# 2b. Claude Code or Codex as the agent (headless, MCP tools only)
python3 harness/run_agent.py --backend claude --model claude-opus-5-5 --jobs jobs.jsonl --root $WORK/evals/agents \
  --url http://127.0.0.1:8701 --server-out /out/house/episodes --secret <file exporting CLAUDE_CODE_OAUTH_TOKEN>
# (--backend codex --model <model> --codex-home <CODEX_HOME> for Codex)

# 3. tables
python3 harness/aggregate.py $WORK
```

A job row is `{"scenario", "seed", "clock": "paused" | "live", "model", "effort", "variant": "" | "visible",
"harness": "minimal" | "commit"}` (for `run_agent.py` also `"backend": "claude" | "codex"`; the command-line
`--backend`, `--model` and `--effort` are the defaults). Seeds follow `seed % 3`: event, decoy, null. In the paper, event seeds 0, 3, 6 and 9
run paused and live, null seeds 2, 5, 8 and 11 run live, visible twins (`"variant": "visible"`) run live, and
`"harness": "commit"` turns on the commit check.

## Agent protocol

`rlm spec | obs | act <skill> <mark> | wait <s> | frames <t0> <t1> | done`. Every response carries the current
head-camera view with numbered marks (fixed places) and a contact sheet of the frames recorded since the previous
request. Under the live clock the model time between a response and the next request is simulated before the request
is served. The change is due a few seconds after the first observation (drawn from the seed), and a safety interlock refuses manipulation
while another actor works in the shared workspace (no budget used), so an agent cannot beat the change by answering
fast: it has to wait and then decide from the recorded frames. Design rules: `DESIGN.md`; 56 further scenario ideas
(text only): `SCENARIOS_TEXT.md`.
