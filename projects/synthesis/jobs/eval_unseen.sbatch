#!/bin/bash
#SBATCH --job-name=synth-eval-unseen
#SBATCH --partition=a6000
#SBATCH --qos=a6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/eval_unseen/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/eval_unseen/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
EXP_DIR=/projects/direct/synthesis_diffusion
OUT_DIR="${EXP_DIR}/eval_unseen"
mkdir -p "${OUT_DIR}"

source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-}"
echo "python=$(which python) branch=$(git branch --show-current)"
python -c "import direct; print('direct=', direct.__file__)"
nvidia-smi -L || true

latest_ckpt() {
  local run_dir=$1
  local iter
  iter="$(tr -d '[:space:]' < "${run_dir}/last_model.txt")"
  echo "${run_dir}/model_${iter}.pt"
}

eval_anatomy() {
  local name=$1 cfg=$2 ckpt=$3 root=$4 lst=$5
  local od="${OUT_DIR}/${name}"
  mkdir -p "${od}"
  echo "=== ${name} ckpt=${ckpt} root=${root} lst=${lst} ==="
  python - <<PY
import sys
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

sys.argv = [
    "direct",
    "predict",
    "${od}",
    "--cfg",
    "${cfg}",
    "--checkpoint",
    "${ckpt}",
    "--data-root",
    "${root}",
    "--filenames-filter",
    "${lst}",
    "--experiment_directory",
    "${EXP_DIR}/${name}",
    "--num-gpus",
    "1",
    "--num-workers",
    "4",
    "--device",
    "cuda",
]
main()
PY
  echo "=== ${name} metrics ==="
  python - <<PY
import json
from pathlib import Path
p = Path("${od}") / "metrics_inference.json"
print(p.read_text() if p.exists() else f"missing {p}")
PY
}

BRAIN_CKPT="$(latest_ckpt "${EXP_DIR}/diffusion_brain")"
BREAST_CKPT="$(latest_ckpt "${EXP_DIR}/diffusion_breast")"
echo "BRAIN_CKPT=${BRAIN_CKPT}"
echo "BREAST_CKPT=${BREAST_CKPT}"

eval_anatomy \
  diffusion_brain \
  "${DIRECT_ROOT}/projects/synthesis/diffusion_brain.yaml" \
  "${BRAIN_CKPT}" \
  /mnt/rhea/data_public/fastmri/brain/multicoil_val \
  "${DIRECT_ROOT}/projects/synthesis/lists/brain/test.lst"

eval_anatomy \
  diffusion_breast \
  "${DIRECT_ROOT}/projects/synthesis/diffusion_breast.yaml" \
  "${BREAST_CKPT}" \
  /mnt/rhea/data_public/fastmri/breast/derived/cartesian \
  "${DIRECT_ROOT}/projects/synthesis/lists/breast/test.lst"

echo "=== ALL DONE ==="
find "${OUT_DIR}" -name metrics_inference.json -print -exec cat {} \;
