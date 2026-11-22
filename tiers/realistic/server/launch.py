#!/usr/bin/env python3
"""Launch one BEHAVIOR-1K (OmniGibson) container for a RoboLiveMem scenario server (one port published on localhost).

For hosts without the NVIDIA Container Toolkit: the selected GPU's device nodes and the loaded driver's libraries are
bind-mounted explicitly. BEHAVIOR-1K assets (licensed separately, see https://behavior.stanford.edu) are mounted
read-only from --assets; outputs go to <root>/runs, and this tier's code is mounted from <root>/bench.
(With the NVIDIA Container Toolkit you can instead run the image with --gpus and the same mounts.)
"""
import argparse, glob, json, subprocess, xml.etree.ElementTree as ET
from pathlib import Path

IMAGE = "docker.io/stanfordvl/behavior:3.9.2@sha256:2702900192741904e69a573aa7869ea51c592586f9791050796efc37faa893fb"


def build(root, gpu, name, args, port=None, assets=None):
    env0 = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{json .Config.Env}}"], text=True)) or []
    env0 = dict(s.split("=", 1) for s in env0 if "=" in s)
    cmd = ["docker", "run", "--rm", "--name", name, "--shm-size=8g", "--cpus=16", "--memory=48g"]
    if port:
        cmd += ["--publish", f"127.0.0.1:{port}:{port}"]
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "cache" / f"gpu_{gpu}").mkdir(parents=True, exist_ok=True)
    cmd += ["--mount", f"type=bind,src={assets},dst=/data,readonly",
            "--mount", f"type=bind,src={root / 'cache' / f'gpu_{gpu}'},dst=/cache",
            "--mount", f"type=bind,src={root / 'runs'},dst=/out",
            "--mount", f"type=bind,src={root / 'bench'},dst=/work,readonly"]
    env = {"OMNIGIBSON_HEADLESS": "1", "OMNIGIBSON_DATA_PATH": "/data", "OMNIGIBSON_APPDATA_PATH": "/cache/appdata",
           "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMP_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1",
           "TORCHDYNAMO_DISABLE": "1"}
    row = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=driver_version,uuid,pci.bus_id", "--format=csv,noheader"], text=True).strip()
    driver, uuid, bus = [s.strip() for s in row.split(",")]
    minor = ET.fromstring(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "-q", "-x"], text=True)).findtext("gpu/minor_number")
    for dev in [f"/dev/nvidia{minor}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]:
        if Path(dev).exists():
            cmd += ["--device", dev]
    dom, bd, fn = bus.lower().split(":")
    pci = f"{int(dom, 16):04x}:{bd}:{fn}"
    for sfx in ("card", "render"):
        link = Path(f"/dev/dri/by-path/pci-{pci}-{sfx}")
        if link.exists():
            cmd += ["--device", str(link.resolve())]
    seen = set()
    for pat in ["libcuda.so*", "libnvidia-*.so*", "libnvoptix.so*", "libnvcuvid.so*", "libGLX_nvidia.so*", "libEGL_nvidia.so*",
                "libGLESv1_CM_nvidia.so*", "libGLESv2_nvidia.so*"]:
        for f in sorted(glob.glob("/usr/lib/x86_64-linux-gnu/" + pat)):
            real = Path(f).resolve()
            if driver not in real.name or not real.is_file():
                continue
            tgt = "/usr/local/nvidia/lib64/" + Path(f).name
            if tgt not in seen:
                cmd += ["--mount", f"type=bind,src={real},dst={tgt},readonly"]
                seen.add(tgt)
    for f in ["/usr/share/vulkan/icd.d/nvidia_icd.json", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "/usr/bin/nvidia-smi"]:
        if Path(f).is_file():
            cmd += ["--mount", f"type=bind,src={f},dst={f},readonly"]
    env.update({"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:" + env0.get("LD_LIBRARY_PATH", ""),
                "CUDA_VISIBLE_DEVICES": uuid, "NVIDIA_VISIBLE_DEVICES": uuid,
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics,display,video"})
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    return cmd + [IMAGE] + args


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="working folder: bench/ (this tier's code), runs/, cache/")
    p.add_argument("--assets", type=Path, required=True, help="BEHAVIOR-1K asset folder (OMNIGIBSON_DATA_PATH)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    raise SystemExit(subprocess.call(build(a.root.resolve(), a.gpu, a.name, args, a.port, a.assets.resolve())))
