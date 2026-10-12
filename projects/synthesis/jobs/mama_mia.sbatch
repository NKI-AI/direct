#!/bin/bash
#SBATCH --job-name=synth-mama-mia
#SBATCH --partition=a6000
#SBATCH --qos=a6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/mama_mia/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/mama_mia/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
EXP_DIR=/projects/direct/synthesis_diffusion
OUT_DIR="${EXP_DIR}/mama_mia"
RUN_DIR="${EXP_DIR}/diffusion_breast"
mkdir -p "${OUT_DIR}"

source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

python -m pip install -q nibabel

ITER="$(tr -d '[:space:]' < "${RUN_DIR}/last_model.txt")"
CKPT="${RUN_DIR}/model_${ITER}.pt"

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-}"
echo "CKPT=${CKPT}"
echo "python=$(which python) branch=$(git branch --show-current)"
python -c "import direct, nibabel; print('direct=', direct.__file__, 'nibabel=', nibabel.__version__)"
nvidia-smi -L || true

python "${DIRECT_ROOT}/projects/synthesis/tools/synthesize_mama_mia.py" \
  --checkpoint "${CKPT}" \
  --output "${OUT_DIR}" \
  --steps 50 \
  --seed 0 \
  --batch-size 2 \
  --device cuda

echo "=== summary ==="
cat "${OUT_DIR}/summary.json"
ls -lh "${OUT_DIR}" "${OUT_DIR}/plots"
