# Controlled tier: 5 RoboMME-scene tasks in ManiSkill

Scripted object motions change the decisive state while the agent thinks. One model call per step, with the frames
recorded since the previous step. The live clock advances the simulator `ceil(latency x 20 Hz)` control steps while the
robot holds its joint targets; then the command runs (`rlm_controlled/live.py`).

| Task (hidden / visible twin) | Class | What changes |
|---|---|---|
| `SwapStreamW` / `SwapStreamWV` | U | containers hiding cubes swap three or four times |
| `InterruptSwapW` / `InterruptSwapWV` | U | the same, interrupting the robot's own sequence |
| `RepickStreamW` / `RepickStreamWV` | U | identical cubes trade places after a demonstration |
| `HighlightLateW` / `HighlightLateWV` | W | targets highlighted once after hand-over |
| `RevealLateW` / `RevealLateWV` | W | cubes revealed once after hand-over |

## Install

```bash
git clone https://github.com/RoboMME/robomme_benchmark.git robomme_src
pip install "mani-skill @ git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"
pip install -e robomme_src -e ../..        # RoboMME and the robolivemem package (model API)
```

CPU rendering works (`VK_ICD_FILENAMES` pointing at Mesa's lavapipe, `CUDA_VISIBLE_DEVICES=` empty), a GPU is faster.

## Run as an environment server

```bash
RLM_STRICT_TASKS=1 python server.py --port 8802     # CPU: VK_ICD_FILENAMES=<Mesa lavapipe icd>, CUDA_VISIBLE_DEVICES=

# from the repository root, any agent (see ../../PROTOCOL.md):
python -m robolivemem.eval --tier controlled --env http://127.0.0.1:8802 --agent chat:anthropic:claude-sonnet-5-5 \
    --out results/controlled_sonnet.jsonl
python -m robolivemem.eval --tier controlled --env http://127.0.0.1:8802 --agent server:oracle --seeds 0-5 \
    --out results/controlled_check.jsonl          # privileged reference policy, no model
```

Every episode runs in a fresh worker process with the paper's settings: up to 30 new frames and 30 demonstration
frames per decision, tiled into at most 24 images, 25 decisions and a 1300-step budget. The agent receives the new
turn (`content`) and the dialogue the paper's models received (`dialogue`), and answers
`{"choice": "<listed label>" | "wait", "mark": n, "wait": s}`. `--commit` turns on the commit check.

## Batch runs without the server (as in the paper)

```bash
export RLM_STRICT_TASKS=1
OPTS="--max-tokens 4096 --max-new-frames 30 --max-demo-frames 30 --max-images 62 --tile 24 --retry --eps 0-39"
python -m rlm_controlled.run_agent --model anthropic:claude-sonnet-5-5 $OPTS \
  --tasks SwapStreamW,RepickStreamW,HighlightLateW,RevealLateW,InterruptSwapW --modes paused,live \
  --hints fullgoal,noevent,cleanframe,twinnote --out results/hidden.jsonl
python -m rlm_controlled.run_agent --model anthropic:claude-sonnet-5-5 $OPTS \
  --tasks SwapStreamWV,RepickStreamWV,InterruptSwapWV,HighlightLateWV,RevealLateWV --modes live \
  --hints fullgoal,noevent,cleanframe,twinnote --out results/twins.jsonl
# commit check: add "commit" to --hints;  fixed clock: --modes fixed2,fixed8
```
