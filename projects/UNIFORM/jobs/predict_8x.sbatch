#!/bin/bash
#SBATCH --job-name=uniform_8x
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/direct/multiorgan/base_vsharp/jobs/predict_8x_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd ~/projects/direct

NOCUDNN_DIR=/projects/direct/multiorgan/base_vsharp/jobs/nocudnn_sitecustomize
export PYTHONPATH="${NOCUDNN_DIR}:${PYTHONPATH:-}"

CKPT=/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt
CFG=~/projects/direct/projects/UNIFORM/configs/inference
LISTS=/projects/direct/multiorgan/base_vsharp/lists/test
OUT=/projects/direct/multiorgan/base_vsharp/jobs/predictions_8x
mkdir -p "$OUT" "$LISTS"

# Confirm biases present
python3 - <<'PY'
import re, torch
sd=torch.load("/projects/direct/multiorgan/base_vsharp/uniform_vsharp.pt", map_location="cpu", weights_only=False)["model"]
n=sum(1 for k in sd if re.search(r"denoiser_blocks\.\d+\.conv_out\.bias$", k))
print("ckpt conv_out.bias count", n, "nkeys", len(sd), flush=True)
assert n==12
PY

predict() {
  local name=$1 cfg=$2 root=$3 lst=$4
  local od="${OUT}/${name}_8x"
  mkdir -p "$od"
  echo "=== predict ${name} 8x ==="
  direct predict "$od" --cfg "$cfg" --checkpoint "$CKPT" --data-root "$root" \
    --filenames-filter "$lst" --num-gpus 1 --num-workers 4
}

printf "%s\n" "file_brain_AXT2_200_6002174.h5" "file_brain_AXT2_200_2000124.h5" > "$LISTS/brain_4x.lst"
printf "%s\n" "file1002155.h5" "file1000323.h5" > "$LISTS/knee_4x.lst"
printf "%s\n" "file_prostate_AXT2_0265.h5" "file_prostate_AXT2_0180.h5" > "$LISTS/prostate_4x.lst"
printf "%s\n" "P038_cine_sax.mat" "P038_cine_lax.mat" > "$LISTS/cardiac_val.lst"

predict brain "${CFG}/inference_brain_8x.yaml" \
  /mnt/rhea/data_public/fastmri/brain/multicoil_test/fully_sampled "$LISTS/brain_4x.lst"
predict knee "${CFG}/inference_knee_8x.yaml" \
  /mnt/rhea/data_public/fastmri/knee/multicoil_val "$LISTS/knee_4x.lst"

PROSTATE_ROOT=""
for d in /mnt/rhea/data_public/fastmri/prostate/training_T2_{1..9} /mnt/rhea/data_public/fastmri/prostate/test/T2; do
  if [[ -f "${d}/file_prostate_AXT2_0265.h5" ]]; then PROSTATE_ROOT="$d"; break; fi
done
predict prostate "${CFG}/inference_prostate_8x.yaml" "$PROSTATE_ROOT" "$LISTS/prostate_4x.lst"

CARDIAC_FLAT=/projects/direct/multiorgan/base_vsharp/jobs/cardiac_val_flat
predict cardiac "${CFG}/inference_cardiac_8x.yaml" "$CARDIAC_FLAT" "$LISTS/cardiac_val.lst"

echo "=== COMPARE 8x TO REFS ==="
python3 - <<'PY'
import json
from pathlib import Path

base = Path("/projects/direct/multiorgan/base_vsharp")
out = base / "jobs/predictions_8x"
refs = {
    "brain": base / "metrics_val_test_brain_8x_random_420001.json",
    "knee": base / "metrics_val_test_knee_8x_equip_420001.json",
    "prostate": base / "metrics_val_test_prostate_8x_equip_420001.json",
    "cardiac": base / "metrics_val_test_cardiac_8x_420001.json",
}
for anat, refp in refs.items():
    pred = json.load(open(out / f"{anat}_8x" / "metrics_inference.json"))
    ref = json.load(open(refp))
    print("===", anat, "8x ===")
    for k, v in pred.items():
        rk = k if k in ref else next((x for x in ref if Path(x).name == Path(k).name), None)
        r = ref.get(rk, {})
        for m in ["fastmri_ssim_metric", "fastmri_psnr_metric", "fastmri_nmse_metric"]:
            rv = r.get(m, float("nan"))
            print(f"  {k} {m}: {v[m]:.5f} vs {rv:.5f} d={v[m]-rv:+.5f}")
PY
