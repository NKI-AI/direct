#!/bin/bash
#SBATCH --job-name=uniform_keys
#SBATCH --partition=a6000
#SBATCH --qos=a6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/direct/multiorgan/base_vsharp/jobs/keys_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate direct

python3 <<'PY'
import torch
from pathlib import Path
from direct.environment import setup_inference_environment

old = torch.load("/projects/direct/multiorgan/base_vsharp/model_686000.pt", map_location="cpu", weights_only=False)
conv = torch.load("/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt", map_location="cpu", weights_only=False)

old_keys = list(old["model"].keys())
print("OLD samples (up/conv/init):")
for k in old_keys:
    if any(x in k for x in ("up_conv", "up_transpose", "conv_out", "initializer.out", "denoiser_blocks.0.conv")):
        print(" ", k)

print("\nCONVERTED samples:")
for k in conv["model"].keys():
    if any(x in k for x in ("up_conv", "up_transpose", "conv_out", "initializer.out")):
        print(" ", k)

cfg = Path.home() / "projects/direct/projects/UNIFORM/configs/inference/inference_brain.yaml"
exp = Path("/projects/direct/multiorgan/base_vsharp/jobs/key_test")
exp.mkdir(parents=True, exist_ok=True)
env = setup_inference_environment("", exp, "cuda:0", 0, False, cfg)
expected = set(env.engine.model.state_dict().keys())
have = set(conv["model"].keys())
missing = sorted(expected - have)
unexpected = sorted(have - expected)
print(f"\nMISSING ({len(missing)}):")
for k in missing[:30]:
    print(" ", k)
print(f"\nUNEXPECTED ({len(unexpected)}):")
for k in unexpected[:30]:
    print(" ", k)
PY
