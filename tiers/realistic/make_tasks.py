#!/usr/bin/env python3
"""Generate Embodied-First-Exam-style task directories for RoboLiveMem-Bench.

    python3 make_tasks.py --out tasks/robolivemem

For every scenario three task directories are written:
  <sid>-privileged         paused clock, observation includes the true object states (oracle / debugging)
  <sid>-standard           paused clock, images only
  <sid>-standard-realtime  live clock (latency-charged), images only

Each directory: task.toml, instruction.md, environment/instance.json, tests/test_outcome.py, solution/solve.sh.
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCEN = {
    # sid: (scene server, title, setting, instruction, success description, budget)
    "s01_put_away": ("house", "Put-away", "home kitchen", "Put the apple into the fruit bowl (mark 5). Right now the apple lies on the table (mark 4). The helper robot tidies this table from the other side and puts loose food away into the storage bins (marks 1 to 3). Opening a bin, picking and placing each use one of your 3 manipulation actions.", "apple_at == 'bowl'", 3),
    "s02_annas_mug": ("house", "Anna's mug", "home dining room", "Anna's mug is the one at mark 1 right now (the two mugs look the same). Put Anna's mug on Anna's tray (mark 4). The household helper robot is setting the table from the other side.", "annas_mug_at == 'tray'", 2),
    "s03_vacuum_ball": ("house", "Robot vacuum", "home living room", "Put the soccer ball into the toy box (mark 4). Right now the ball lies on the rug (mark 1). The robot vacuum cleans this room on its own schedule. Marks 2 to 3 are places under the furniture that you can reach into.", "ball_at == 'box'", 2),
    "s04_salted_soup": ("house", "Salted soup", "home kitchen", "The soup on the stove (mark 1) needs exactly ONE portion of salt. When you start, nobody has salted it yet. Use the skill `salt 1` to add one portion. Too much salt ruins it. The helper robot is cooking next to you and also seasons dishes. When the soup has exactly one portion of salt, run `done`.", "salt_portions == 1", 1),
    "s05_rotary_reject": ("industrial", "Rotary reject", "factory work cell", "A vision station inspects the part in the nest under its lamp. When you start, its lamp is RED: the part under the red lamp is defective. Put that defective part into the red reject bin (mark 7). Do not reject good parts. The indexing table keeps cycling on its own schedule; it never moves while your hand is in the cell.", "rejected == [defective]", 2),
    "s06_conveyor_lanes": ("industrial", "Conveyor lanes", "warehouse conveyor", "The tote at the pick station under the scanner (mark 1) is order #17. Put the order #17 tote on the packing table (mark 5). The conveyor restarts on its own schedule and sorts totes into the spur lanes (marks 2 to 4); all totes look the same. The conveyor never moves while your hand is at the belt.", "order_tote_at == 'pack'", 2),
    "s07_press_lamp": ("industrial", "Press ready lamp", "factory work cell", "The press is running a curing cycle on the part inside. When the cycle ends, the green READY lamp on top (mark 2) lights up for a few seconds and then goes dark again; nothing else about the press changes. Unload the part (skill `unload`) after the cycle has ended, and no later than 90 seconds after it ended. Unloading during the cycle ruins the part. You have one unload.", "part == 'unloaded'", 1),
    "s08_parcel_locker": ("industrial", "Parcel locker", "office mailroom", "Dr. Lee's parcel (a cardboard box) is waiting for pickup in the mailroom. Bring it to the reception desk (mark 8). The courier robot also works in this mailroom and puts uncollected parcels away into the lockers (marks 1 to 6). Opening a locker, picking and placing each use one of your 3 manipulation actions.", "parcel_at == 'desk'", 3),
    "s09_kitting_partner": ("industrial", "Kitting partner", "warehouse kitting rack", "Kit the four totes on the rack (marks 1 to 4): every tote must end up with exactly ONE part. A tote with two parts is a kitting error. Your partner robot works the same rack from the other side and may kit some totes too. The lids are closed, and you cannot see inside a closed tote. Use the skill `kit <mark>` for each tote that still needs a part, then `done`.", "parts_per_tote == [1, 1, 1, 1]", 4),
    "s10_cloche_pass": ("restaurant", "Cloche pass", "restaurant kitchen pass", "Three plates wait under identical cloches on the pass (marks 1 to 4 are the pass spots). The chef said the VEGETARIAN plate is the one at mark 1 right now. Put the vegetarian plate on the tray for table 4 (mark 5). The expo robot on the kitchen side regroups plates on the pass for other tables.", "veg_at == 'tray'", 2),
}
MEMTYPE = {"s01_put_away": "location-in-container", "s02_annas_mug": "identity", "s03_vacuum_ball": "location-occluded",
           "s04_salted_soup": "invisible-state", "s05_rotary_reject": "identity-machine", "s06_conveyor_lanes": "location-machine",
           "s07_press_lamp": "transient-event", "s08_parcel_locker": "location-in-container", "s09_kitting_partner": "task-progress",
           "s10_cloche_pass": "identity-covered"}
# instructions exactly as the scenario servers state them (harness/instructions.json, dumped from the scenario code)
_INSTR = json.loads((HERE / "harness" / "instructions.json").read_text()) if (HERE / "harness" / "instructions.json").exists() else {}
SCEN = {k: (v[0], v[1], v[2], _INSTR.get(k, v[3]), v[4], v[5]) for k, v in SCEN.items()}
INTERLOCK = (" Safety interlock: while another robot or a machine is working in the shared workspace, manipulation commands are "
             "refused (nothing happens and no manipulation action is used); waiting is always allowed.")
CLOCK_NOTE = {
    "paused": "PAUSED clock: the world waits while you think. It only moves while the robot executes a skill or waits.",
    "live": "LIVE clock: the world keeps running while you think. Other robots, machines and people keep moving. Every response includes the robot camera frames recorded since your previous request, so you can see what happened while you were thinking.",
}
TEST = '''"""Verifier for {tid}: reads the episode result written by the scenario server and checks the success predicate."""
import json, os
from pathlib import Path

RESULT = Path(os.environ.get("RLM_RESULT", "result.json"))


def predicate(s, hidden):
{pred}


def test_outcome():
    r = json.loads(RESULT.read_text())
    assert r["scenario"] == "{sid}" and r["clock"] == "{clock}"
    ok = predicate(r["final_state"], r["hidden"])
    assert ok == r["success"], "server verdict and test predicate disagree"
    assert ok, r["reason"]
'''
PRED = {
    "s01_put_away": "    return s['apple_at'] == 'bowl'",
    "s02_annas_mug": "    return s['annas_mug_at'] == 'tray'",
    "s03_vacuum_ball": "    return s['ball_at'] == 'box'",
    "s04_salted_soup": "    return s['salt_portions'] == 1",
    "s05_rotary_reject": "    return s['rejected'] == [hidden['defective']]",
    "s06_conveyor_lanes": "    return s['order_tote_at'] == 'pack'",
    "s07_press_lamp": "    return s['part'] == 'unloaded'",
    "s08_parcel_locker": "    return s['parcel_at'] == 'desk'",
    "s09_kitting_partner": "    return s['parts_per_tote'] == [1, 1, 1, 1]",
    "s10_cloche_pass": "    return s['veg_at'] == 'tray'",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "tasks" / "robolivemem"))
    a = ap.parse_args(); out = Path(a.out)
    prompt_t = (HERE / "harness" / "prompt.md").read_text()
    for sid, (scene, title, setting, instr, succ, budget) in SCEN.items():
        for mode, clock, priv in [("privileged", "paused", True), ("standard", "paused", False), ("standard-realtime", "live", False)]:
            tid = f"{sid}-{mode}"; d = out / tid
            for sub in ("environment", "tests", "solution"):
                (d / sub).mkdir(parents=True, exist_ok=True)
            mem = MEMTYPE[sid]
            (d / "task.toml").write_text(f"""schema_version = "1.3"
artifacts = [{{ source = "/out/result.json", service = "sim" }},
  {{ source = "/out/frames.jsonl", service = "sim" }},
  {{ source = "/out/main.mp4", service = "sim" }},          # spectator video of the live episode
  {{ source = "/out/head.mp4", service = "sim" }},          # the robot's own camera (5 Hz, world time)
  {{ source = "/out/server.log", service = "sim" }},
  "/app/output/",
]

[task]
name = "robolivemem/{tid}"
description = "{mode.upper()} ({'clock pauses while the agent thinks' if clock == 'paused' else 'clock does not stop: deliberation is charged as world time'}), RoboLiveMem-Bench on BEHAVIOR-1K v3.9.2: {title}. {instr.replace('"', "'")}"
keywords = ["robotics", "behavior", "omnigibson", "isaac-sim", "r1", "live-memory", "{mem}", "{setting.split()[0]}"]

[metadata]
task_type = "episode"
simulator = "behavior"
robot = "r1"
family = "{sid}"
dynamics = "dynamic"              # another robot or a machine changes the scene while the agent deliberates
protocol = "robolivemem/1.0"      # own runtime (EFE protocol README section 3): skills at numbered marks, MCP / HTTP client
mode = "{mode}"
clock = "{clock}"
memory_type = "{mem}"
oracle = "scripted"
oracle_evidence = "privileged oracle (solution/solve.sh) succeeds on every dev seed under both clocks; a stale-memory policy succeeds under the paused clock and fails under the live clock"
manipulation_budget = {budget}
seeds_dev = [0, 1, 2, 3, 4]
seeds_test = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
expert_time_estimate_hours = 0.1

[verifier]
timeout_sec = 300.0
environment_mode = "separate"
network_mode = "no-network"

[[verifier.collect]]
service = "sim"
command = "python3 /srv/finalize.py"   # end the episode (POST /end) and write /out/result.json
timeout_sec = 120

[agent]
timeout_sec = 1500.0
network_mode = "allowlist"
allowed_hosts = ["chatgpt.com", "*.chatgpt.com", "auth.openai.com", "api.openai.com", "*.openai.com", "api.anthropic.com", "*.anthropic.com", "pypi.org", "files.pythonhosted.org"]

[environment]
build_timeout_sec = 1800.0
cpus = 8
memory_mb = 24576
storage_mb = 10240
gpu_types = ["RTX 6000 Ada", "L40S", "L40", "RTX 4090"]
""")
            (d / "instruction.md").write_text(prompt_t.format(title=title, setting=setting, instruction=instr, clock_note=CLOCK_NOTE[clock] + INTERLOCK, budget=budget))
            (d / "environment" / "sim").mkdir(parents=True, exist_ok=True)
            (d / "environment" / "sim" / "instance.json").write_text(json.dumps(dict(scenario=sid, scene_server=scene, clock=clock, privileged=priv,
                                                                                     seed="${RLM_SEED:-100}"), indent=1))
            (d / "environment" / "docker-compose.yaml").write_text(f"""# The simulator runs in the `sim` sidecar (agent cannot open it); the agent reaches it only through port 8000
# with the shipped client (client/rlm_mcp.py: MCP tools robot_*; client/rlm: the same calls on the command line).
services:
  main:
    depends_on: [sim]
    environment:
      RLM_URL: http://localhost:8000
  sim:
    image: docker.io/stanfordvl/behavior:3.9.2
    volumes:
      - ${{RCB_BEHAVIOR_DATA:?set RCB_BEHAVIOR_DATA to the licensed BEHAVIOR dataset}}:/data:ro
      - ../../../../server:/srv/server:ro
    command: python /srv/server/bench_server.py --session /out --scene /srv/server/scenes/{scene}.json --scenarios {sid} --port 8000 --code /srv/server
    environment:
      OMNIGIBSON_HEADLESS: "1"
      OMNIGIBSON_DATA_PATH: /data
      RLM_EPISODE: '{{"scenario": "{sid}", "clock": "{clock}", "privileged": {str(priv).lower()}}}'
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
""")
            (d / "environment" / "client-requirements.txt").write_text("# none: the clients use the Python standard library only\n")
            (d / "tests" / "test.sh").write_text("""#!/bin/bash
# Verifier (separate sandbox): grades the sim sidecar's own record (/out/result.json), never the agent's files.
mkdir -p /logs/verifier
RLM_RESULT=${RLM_RESULT:-/out/result.json} python3 /tests/test_outcome.py > /logs/verifier/grade.log 2>&1
""")
            (d / "tests" / "test_outcome.py").write_text(TEST.format(tid=tid, sid=sid, clock=clock, pred=PRED[sid]) +
                                                         "\n\nif __name__ == '__main__':\n    test_outcome(); print('reward 1')\n")
            (d / "README.md").write_text(f"# {title} ({mode})\n\n{instr}\n\nMemory type: {mem}. Clock: {clock}. "
                                         f"Success: `{succ}` on the final state recorded by the simulator service.\n")
            (d / "solution" / "solve.sh").write_text(f'''#!/usr/bin/env bash
# Reference solution: privileged oracle (waits until helpers and machines are idle, then acts on the true state).
set -euo pipefail
python3 "$(dirname "$0")/../../../../harness/scripted_agent.py" --url "${{RLM_URL:-http://127.0.0.1:8000}}" \\
  --scenario {sid} --seed "${{SEED:-0}}" --clock {clock} --policy oracle --think "${{THINK:-20}}"
''')
    print("wrote", len(SCEN) * 3, "task directories under", out)


if __name__ == "__main__":
    main()
