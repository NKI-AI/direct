#!/bin/bash
#SBATCH --job-name=eval-msd-brain
#SBATCH --account=aifo
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-01:00:00
#SBATCH --output=/projects/direct/synthesis_diffusion/%x-%j.out
#SBATCH --error=/projects/direct/synthesis_diffusion/%x-%j.err

set -euo pipefail
DIRECT_ROOT=/home/g.yiasemis/projects/direct
source /home/g.yiasemis/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd "${DIRECT_ROOT}"
unset PYTHONPATH
hash -r
export MESONPY_EDITABLE_VERBOSE=1
flock -x /tmp/direct-meson-rebuild.lock "${CONDA_PREFIX}/bin/ninja" -C "${DIRECT_ROOT}/build/cp312" || true
nvidia-smi -L || true
python "${DIRECT_ROOT}/projects/synthesis/tools/synthesize_msd_brain_maps_phase.py" \
  --nifti /mnt/rhea/data_public/MSD_Brain_Tumour/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
  --modality 0 \
  --sens-ckpt /projects/direct/synthesis_diffusion/sensitivity_maps_multi_cardiac_500k/model_25000.pt \
  --phase-ckpt /projects/direct/synthesis_diffusion/flow_matching_multi_cardiac/model_10000.pt \
  --out-dir /projects/direct/synthesis_diffusion/eval_msd_brain/BRATS_001_flair \
  --steps 30 \
  --device cuda
