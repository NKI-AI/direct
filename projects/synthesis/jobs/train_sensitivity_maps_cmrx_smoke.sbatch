#!/bin/bash
#SBATCH --job-name=synth-cmrx-smoke
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Walltime must end before maint_oct_26 (2026-10-05T00:00:00).
#SBATCH --time=00:45:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail

DIRECT_ROOT=/home/g.yiasemis/projects/direct
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r

echo "Host=$(hostname) start=$(date -Is) deadline=2026-10-05T00:00:00"
python -c "import direct; print('direct=', direct.__file__)"
python - <<'PY'
from pathlib import Path
from direct.nn.synthesis.synthesis import organ_id_from_filename
smoke = Path('projects/synthesis/lists/cardiac/s_train_smoke.lst').read_text().split()
assert smoke, 'empty smoke list'
for name in smoke:
    link = Path('/projects/direct/synthesis_diffusion/cmrxrecon_cine/train') / name
    assert link.exists(), link
    assert organ_id_from_filename(str(link)) == 4, name
print('smoke files', smoke)
PY
nvidia-smi -L || true
python "${DIRECT_ROOT}/projects/synthesis/jobs/launch_sensitivity_maps_cmrx_smoke.py"
echo "done=$(date -Is)"
