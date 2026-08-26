#!/bin/bash
#SBATCH --job-name=uniform_pred
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/direct/multiorgan/base_vsharp/jobs/predict_uniform_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd ~/projects/direct

# Force-disable cuDNN for every Python process (including `direct` CLI).
NOCUDNN_DIR=/projects/direct/multiorgan/base_vsharp/jobs/nocudnn_sitecustomize
mkdir -p "$NOCUDNN_DIR"
cat > "$NOCUDNN_DIR/sitecustomize.py" <<'PY'
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
PY
export PYTHONPATH="${NOCUDNN_DIR}:${PYTHONPATH:-}"

SRC=/projects/direct/multiorgan/base_vsharp/model_686000.pt
CKPT=/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt
CFG=~/projects/direct/projects/UNIFORM/configs/inference
LISTS=/projects/direct/multiorgan/base_vsharp/lists/test
OUT=/projects/direct/multiorgan/base_vsharp/jobs/predictions
mkdir -p "$OUT" "$LISTS"

echo "=== Convert checkpoint ==="
python3 projects/UNIFORM/tools/convert_uniform_checkpoint.py \
  "$SRC" "$CKPT" --force -v \
  --note "UNIFORM base_vsharp model_686000 (OpenReview I13Y1nU6gs)"

echo "=== Strict load check ==="
python3 <<'PY'
from pathlib import Path
from direct.environment import setup_inference_environment
from direct.checkpointer import Checkpointer
import torch
print("cudnn.enabled=", torch.backends.cudnn.enabled, flush=True)
ckpt = Path("/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt")
cfg = Path.home() / "projects/direct/projects/UNIFORM/configs/inference/inference_brain.yaml"
exp = Path("/projects/direct/multiorgan/base_vsharp/jobs/load_test")
exp.mkdir(parents=True, exist_ok=True)
env = setup_inference_environment("", exp, "cuda:0", 0, False, cfg)
Checkpointer(save_directory=exp, save_to_disk=False, model=env.engine.model, **env.engine.models).load_models_from_file(ckpt)
print("LOAD OK", flush=True)
PY

predict() {
  local name=$1 cfg=$2 root=$3 lst=$4
  local od="${OUT}/${name}_4x"
  mkdir -p "$od"
  echo "=== ${name} root=${root} ==="
  direct predict "$od" \
    --cfg "$cfg" \
    --checkpoint "$CKPT" \
    --data-root "$root" \
    --filenames-filter "$lst" \
    --num-gpus 1 \
    --num-workers 4
}

predict brain "${CFG}/inference_brain.yaml" \
  /mnt/rhea/data_public/fastmri/brain/multicoil_test/fully_sampled "$LISTS/brain_4x.lst"
predict knee "${CFG}/inference_knee.yaml" \
  /mnt/rhea/data_public/fastmri/knee/multicoil_val "$LISTS/knee_4x.lst"

PROSTATE_ROOT=""
for d in /mnt/rhea/data_public/fastmri/prostate/training_T2_{1..9}; do
  if [[ -f "${d}/file_prostate_AXT2_0265.h5" ]]; then PROSTATE_ROOT="$d"; break; fi
done
if [[ -n "$PROSTATE_ROOT" ]]; then
  predict prostate "${CFG}/inference_prostate.yaml" "$PROSTATE_ROOT" "$LISTS/prostate_4x.lst"
else
  echo "WARNING: prostate files not found, skipping"
fi

# Cardiac: use ValidationSet FullSample (ky=246), NOT TrainingSet (ky=162).
# Flatten patient/view mats so CMRxRecon lists can use P038_cine_sax.mat style names.
CARDIAC_FLAT=/projects/direct/multiorgan/base_vsharp/jobs/cardiac_val_flat
CARDIAC_BASE=/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Cine
mkdir -p "$CARDIAC_FLAT"
ln -sfn "$CARDIAC_BASE/ValidationSet/FullSample/P038/cine_sax.mat" "$CARDIAC_FLAT/P038_cine_sax.mat"
ln -sfn "$CARDIAC_BASE/ValidationSet/FullSample/P038/cine_lax.mat" "$CARDIAC_FLAT/P038_cine_lax.mat"
printf "%s\n" "P038_cine_sax.mat" "P038_cine_lax.mat" > "$LISTS/cardiac_val.lst"
predict cardiac "${CFG}/inference_cardiac.yaml" "$CARDIAC_FLAT" "$LISTS/cardiac_val.lst"

echo "=== ALL METRICS ==="
find "$OUT" -name metrics_inference.json -print -exec cat {} \;
