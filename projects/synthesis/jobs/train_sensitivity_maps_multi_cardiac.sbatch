#!/bin/bash
#SBATCH --job-name=synth-sens-cardiac
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
# Must finish before maint_oct_26 StartTime=2026-10-05T00:00:00.
#SBATCH --time=4-00:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) start=$(date -Is) maint=2026-10-05T00:00:00"
python -c "import direct; print('direct=', direct.__file__)"
python - <<'PY'
from direct.nn.synthesis.synthesis import ORGAN_NAMES, organ_id_from_filename, UNKNOWN_ORGAN_ID
assert len(ORGAN_NAMES) == 6 and UNKNOWN_ORGAN_ID == 5
assert organ_id_from_filename("y2024_P001_cine_sax.mat") == 4
assert organ_id_from_filename("/mnt/rhea/.../knee/multicoil_train/file1000002.h5") == 1
print("organ ids ok", ORGAN_NAMES)
PY
nvidia-smi -L || true
python "${DIRECT_ROOT}/projects/synthesis/jobs/launch_sensitivity_maps_multi_cardiac.py"
