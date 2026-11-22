# RoboLiveMem-Bench: live-memory scenarios in BEHAVIOR-1K

## 1. What is being tested

An agent that deliberates for L seconds can only keep a correct memory of the world if the world is static during those L seconds. Under the usual **paused** evaluation this is guaranteed: the simulator waits. In a real home, warehouse or factory it is not: while the agent thinks, other robots, machines and people keep changing things. **Live memory** is the ability to keep the belief about unobserved state correct across one's own deliberation, by integrating what the robot's own camera recorded while the agent was thinking.

Every scenario is built so that:

1. **The change happens while the agent deliberates.** It is scheduled in world time 0.5 to 1.5 s after the first observation (inside the first deliberation of any current model), never triggered by the agent's thinking. A **safety interlock** refuses manipulation commands while another robot or a machine works in the shared workspace (nothing happens, no budget is used, the robot holds still for 1 s), so a fast agent cannot act on the first view before the change lands: it has to wait, then decide from what its camera recorded. Under the paused clock the first command arrives at world time 0, before the change starts, and no helper or machine starts while the agent's robot executes a skill, so under the paused clock the change can only happen while the agent chooses to `wait`: the paused clock stays the change-free control. (The first protocol started the change 4 to 7 s after the first observation. Opus 5.5 and GPT-6 Astra both answer in 1 to 4 s, acted before anything moved and solved 11 of 11 live episodes without needing memory. Early onset plus the interlock close that loophole.)
2. **The change is visible in the robot's own camera stream.** No god view is needed: the robot's head camera records the other robot / machine doing it. The frames are delivered to the agent with its next response.
3. **At action time the current view does not decide the task.** The changed state is inside a closed container, among identical objects, out of view behind furniture, or an invisible property (salted / filled / finished). Looking again does not help; remembering the stream does.
4. **The outcome is verifiable.** A predicate on the privileged final state (object identity at a goal, container counts, process state) decides success. Each task has a manipulation budget, so searching every container is not an option.
5. **No fixed policy wins.** The hidden outcome is randomized per seed (which bin, which lane, how many totes, whether the helper salted at all), so an agent must read the stream rather than guess.
6. **It is solvable.** A privileged oracle and a stream-reading reference policy succeed in every seed under both clocks; under the paused clock the change does not occur before a prompt agent finishes, so a memory that is never updated is enough.

## 2. Clocks

| Clock | World advances during | Use |
|---|---|---|
| `paused` | skill execution and `wait` only | standard evaluation; memory cannot go stale |
| `live` | additionally, all wall time between a response and the agent's next request (the clock starts at the agent's first request, `robot_spec` included) | real-time evaluation; latency-charged lazy simulation, identical in outcome to a world that kept running, because the robot holds still between requests |

Robot camera frames (5 Hz) are recorded whenever the world advances. Every response carries a contact sheet of the frames since the previous delivery (`stream`), and `rlm frames t0 t1` shows any recorded window in detail. The agent is told which clock it is under.

## 3. Agent interface

HTTP server inside the OmniGibson container; command-line client `rlm` for code agents (Claude Code, Codex), the same JSON for API agents.

| Call | Meaning | Cost |
|---|---|---|
| `rlm spec` | task, skills, numbered marks, budget, clock | free |
| `rlm obs` | current head-camera view with numbered marks + stream since the previous request | free (time passes under `live`) |
| `rlm act <skill> <mark>` | `pick`, `place`, `open`, `close`, or a task skill (`salt`, `kit`, `unload`, `reject`) at a numbered place | one manipulation action |
| `rlm wait <s>` | hold still, world runs | free (world time) |
| `rlm frames <t0> <t1>` | recorded frames in a window | free |
| `rlm done` | end the episode | - |

Marks are fixed places (counter spots, bins, lanes, nests, lockers), not objects, so they never leak identity. A skill acts on whatever is at the place when it runs. Feedback is physical only ("grasped one item", "nothing to grasp there"), never the hidden state.

## 4. Scenarios (rendered and evaluated)

| ID | Setting | Who changes the world | Hidden at action time because | Memory | Task and success predicate | Budget |
|---|---|---|---|---|---|---|
| S01 put-away | home kitchen | helper robot tidies the apple into one of three lidded bins | lid closed, bins identical | location (container) | apple in fruit bowl | 3 |
| S02 Anna's mug | home dining room | helper robot moves one of two identical mugs aside to make room for the cutting board | mugs identical | identity + location | Anna's mug on Anna's tray | 2 |
| S03 robot vacuum | home living room | robot vacuum pushes the ball under one of two sofas | out of view under furniture | location (occluded) | ball in toy box | 2 |
| S04 salted soup | home kitchen | helper robot either salts the soup or only stirs it | salt is invisible in soup | process state (progress) | exactly one portion of salt | 1 |
| S05 rotary reject | factory cell | indexing table rotates 1 to 5 steps after the vision station flags a part | six identical parts, flag lamp off | identity under machine motion | flagged part in reject bin, nothing else | 2 |
| S06 conveyor lanes | warehouse | sorter diverts three identical totes into lanes A/B/C | totes identical | location under machine motion | order tote on packing table | 2 |
| S07 press lamp | factory cell | heat press finishes at a random time (8 to 40 s); the READY lamp lights for 6 s | press looks the same before and after | transient event (missed write) | part unloaded after the cycle and within 90 s | 1 |
| S08 parcel locker | office mailroom | courier robot stores the parcel in one of six lockers | doors closed, lockers identical | location (container) | parcel on the reception desk | 3 |
| S09 kitting partner | warehouse | partner robot fills a random subset of four kitting totes | tote contents not visible from the robot | task progress of another robot | every tote holds exactly one part | 4 |
| S10 cloche pass | restaurant | expo robot regroups three plates under identical cloches (two moves) | covers identical | identity under covers | vegetarian plate at table 4's tray | 2 |

Controls available for every scenario: **static** (the helper / machine does not act), **visible twin** (the changed object is distinguishable, e.g. Anna's mug red, totes numbered, lids transparent), **privileged** (observation includes the object states; EFE's privileged mode), **no stream** (agent ablation: the MCP client returns only the current view, no contact sheet and no `robot_frames`; `RLM_NOSTREAM=1`, job field `"variant": "nostream"`). The agent can still watch by waiting in short steps, but then it has to remember what it saw across its own calls.

## 5. Task format (Embodied First Exam style)

```
tasks/robolivemem/<scenario>-<clock>/
  task.toml            # metadata: suite, scenario, clock, seeds, budget, deadline, timeouts
  instruction.md       # what the agent is told (task, clock, interface)
  environment/
    scenario.json      # scene, server entry point, scenario id, seeds
  tests/
    test_outcome.py    # verifier: reads result.json, checks the success predicate and the action log
  solution/
    solve.py           # reference policy (privileged oracle) through the same rlm client
```
