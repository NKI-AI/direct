#!/bin/bash
#SBATCH --job-name=uniform_pred
#SBATCH --partition=a6000
#SBATCH --qos=a6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/direct/multiorgan/base_vsharp/jobs/predict_uniform_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd ~/projects/direct

CKPT=/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt
SRC=/projects/direct/multiorgan/base_vsharp/model_686000.pt

echo "=== Converting checkpoint from ${SRC} ==="
python3 projects/UNIFORM/tools/convert_uniform_checkpoint.py \
  "$SRC" "$CKPT" --force \
  --note "UNIFORM base_vsharp model_686000 (OpenReview I13Y1nU6gs)"

CFG=~/projects/direct/projects/UNIFORM/configs/inference
LISTS=/projects/direct/multiorgan/base_vsharp/lists/test
OUT=/projects/direct/multiorgan/base_vsharp/jobs/predictions
mkdir -p "$OUT" "$LISTS"

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

predict brain    "${CFG}/inference_brain.yaml"    /mnt/rhea/data_public/fastmri/brain/multicoil_test/fully_sampled "$LISTS/brain_4x.lst"
predict knee     "${CFG}/inference_knee.yaml"     /mnt/rhea/data_public/fastmri/knee/multicoil_val              "$LISTS/knee_4x.lst"

PROSTATE_ROOT=""
for d in /mnt/rhea/data_public/fastmri/prostate/training_T2_{1..5}; do
  if [[ -f "${d}/file_prostate_AXT2_0265.h5" ]]; then PROSTATE_ROOT="$d"; break; fi
done
if [[ -n "$PROSTATE_ROOT" ]]; then
  predict prostate "${CFG}/inference_prostate.yaml" "$PROSTATE_ROOT" "$LISTS/prostate_4x.lst"
else
  echo "WARNING: prostate files not found, skipping"
fi

CARDIAC_ROOT=""
for d in \
  /mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallengeData/multi_coil_val \
  /mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallengeData/multi_coil_train; do
  if [[ -f "${d}/P038_cine_sax.mat" ]]; then CARDIAC_ROOT="$d"; break; fi
done
if [[ -n "$CARDIAC_ROOT" ]]; then
  predict cardiac "${CFG}/inference_cardiac.yaml" "$CARDIAC_ROOT" "$LISTS/cardiac_4x.lst"
else
  echo "WARNING: cardiac root not found, skipping"
fi

echo "=== ALL METRICS ==="
find "$OUT" -name metrics_inference.json -print -exec cat {} \;
