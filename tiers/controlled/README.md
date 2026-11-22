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

## Run

```bash
cd tiers/controlled
export RLM_STRICT_TASKS=1          # registers the world-time task variants used in the paper
OPTS="--max-tokens 4096 --max-new-frames 30 --max-demo-frames 30 --max-images 62 --tile 24 --retry --eps 0-39"

# hidden tasks, paused and live (any robolivemem.models spec or paper alias)
python -m rlm_controlled.run_agent --model anthropic:claude-sonnet-5-5 $OPTS \
  --tasks SwapStreamW,RepickStreamW,HighlightLateW,RevealLateW,InterruptSwapW --modes paused,live \
  --hints fullgoal,noevent,cleanframe,twinnote,privlog --out results/hidden.jsonl

# visible twins, live
python -m rlm_controlled.run_agent --model anthropic:claude-sonnet-5-5 $OPTS \
  --tasks SwapStreamWV,RepickStreamWV,InterruptSwapWV --modes live \
  --hints fullgoal,noevent,cleanframe,twinnote,privlog --out results/twins.jsonl

# commit check: add "commit" to --hints;  fixed clock: --modes fixed2,fixed8
# privileged / stale reference policies (no API):
python -m rlm_controlled.run_oracle_strict --tasks SwapStreamW --eps 0-39 --lat 0,3 --oracle stale --out results/stale.jsonl
```

Seeds (`--eps`) follow `episode % 3`: event, decoy, null. `--shard i/n` splits a run over processes.
