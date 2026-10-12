#!/bin/bash
#SBATCH --job-name=synth-sens-smooth
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
CFG="${DIRECT_ROOT}/projects/synthesis/sensitivity_maps_multi_smooth.yaml"
EXP_DIR=/projects/direct/synthesis_diffusion

mkdir -p "${EXP_DIR}"
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct

cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-}"
echo "EXP_DIR=${EXP_DIR}"
echo "CFG=${CFG}"
echo "python=$(which python) branch=$(git branch --show-current)"
python -c "import direct; print('direct=', direct.__file__)"
nvidia-smi -L || true

python "${DIRECT_ROOT}/projects/synthesis/jobs/launch_sensitivity_maps_multi_smooth.py"
