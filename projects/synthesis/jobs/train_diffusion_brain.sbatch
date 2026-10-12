#!/bin/bash
#SBATCH --job-name=synth-diff-brain
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
CFG="${DIRECT_ROOT}/projects/synthesis/diffusion_brain.yaml"
TRAIN_ROOT=/mnt/rhea/data_public/fastmri/brain/multicoil_train
VAL_ROOT=/mnt/rhea/data_public/fastmri/brain/multicoil_val
EXP_DIR=/projects/direct/synthesis_diffusion

mkdir -p "${EXP_DIR}"
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct

cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-}"
echo "EXP_DIR=${EXP_DIR}"
echo "python=$(which python) branch=$(git branch --show-current)"
python -c "import direct; print('direct=', direct.__file__)"
nvidia-smi -L || true

python - <<PY
import sys
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

sys.argv = [
    "direct",
    "train",
    "${EXP_DIR}",
    "--cfg",
    "${CFG}",
    "--training-root",
    "${TRAIN_ROOT}",
    "--validation-root",
    "${VAL_ROOT}",
    "--num-gpus",
    "1",
    "--num-workers",
    "8",
    "--device",
    "cuda",
    "--name",
    "diffusion_brain",
    "--mixed-precision",
]
main()
PY
