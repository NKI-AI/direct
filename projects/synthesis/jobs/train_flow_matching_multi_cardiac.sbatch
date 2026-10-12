#!/bin/bash
#SBATCH --job-name=synth-flow-card
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

BUILD_DIR="${DIRECT_ROOT}/build/cp312"
LOCK=/tmp/direct-meson-rebuild.lock

echo "Host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-} start=$(date -Is)"
# Ensure C extensions exist once; launch script then bypasses editable ninja rebuilds.
flock -x "${LOCK}" "${CONDA_PREFIX}/bin/ninja" -C "${BUILD_DIR}"
python - <<'PY'
from direct.nn.synthesis.synthesis import ORGAN_NAMES, organ_id_from_filename
assert len(ORGAN_NAMES) == 6
assert organ_id_from_filename("y2024_P001_cine_sax.mat") == 4
assert organ_id_from_filename("odd.nii") == 5
print("flow matching organ setup ok", ORGAN_NAMES)
PY
nvidia-smi -L || true
python "${DIRECT_ROOT}/projects/synthesis/jobs/launch_flow_matching_multi_cardiac.py"
