#!/bin/bash
#SBATCH --job-name=synth-smoke-prostate
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
CFG="${DIRECT_ROOT}/projects/synthesis/synthesize_prostate_smoke.yaml"
OUT=/projects/direct/synthesis_diffusion/synthesize_prostate_smoke
PHASE=/projects/direct/synthesis_diffusion/flow_matching_multi/model_290000.pt
SENS=/projects/direct/synthesis_diffusion/sensitivity_maps_multi_500k/model_110000.pt
DATA=/mnt/rhea/data_public/fastmri/prostate/training/T2
LIST="${DIRECT_ROOT}/projects/synthesis/lists/prostate/s_val_smoke.lst"

mkdir -p "${OUT}" /projects/direct/synthesis_diffusion
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct

cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-}"
echo "PHASE=${PHASE}"
echo "SENS=${SENS}"
python -c "import direct; print('direct=', direct.__file__)"
nvidia-smi -L || true

# Expand vars into a real launcher file (spawn-safe).
LAUNCH="${OUT}/launch_synthesize_prostate_smoke.py"
cat > "${LAUNCH}" <<EOF
#!/usr/bin/env python3
import sys
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

sys.argv = [
    "direct",
    "synthesize",
    "${OUT}",
    "--cfg",
    "${CFG}",
    "--phase-model",
    "${PHASE}",
    "--sensitivity-model",
    "${SENS}",
    "--data-root",
    "${DATA}",
    "--filenames-filter",
    "${LIST}",
    "--num-gpus",
    "1",
    "--num-workers",
    "2",
    "--device",
    "cuda",
    "--name",
    "synthesize_prostate_smoke",
    "--experiment_directory",
    "${OUT}",
]
main()
EOF

python "${LAUNCH}"

echo "Plots:"
ls -la "${OUT}/plots" || true
echo "H5:"
ls -la "${OUT}"/*.h5 || true
