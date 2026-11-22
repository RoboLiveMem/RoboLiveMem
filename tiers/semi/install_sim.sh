#!/bin/bash
# Install the simulator of the semi-realistic tier: Embodied Safety Exam (ESE 0.5.6, MuJoCo 3.8.1) from the
# Embodied First Exam repository, into ./venv next to this script. Asset packs are fetched with huggingface_hub
# (set HF_TOKEN if the packs require it).
set -e
cd "$(dirname "$0")"
[ -d efe ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/mll-lab-nu/embodied-first-exam efe
python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q -r efe/images/ese/requirements.txt
pip install -q -e efe/images/ese/context/ese-src
pip install -q huggingface_hub requests
pip install -q -e ../..            # the robolivemem package (model API)
(cd efe && python3 images/ese/assets.py fetch)
python -c "import mujoco, embodied_safety_exam; print('mujoco', mujoco.__version__, 'ok')"
