#!/bin/bash
#SBATCH --job-name=uniform_biasfix
#SBATCH --partition=a100
#SBATCH --qos=four_a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/direct/multiorgan/base_vsharp/jobs/reconvert_predict_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate direct
cd ~/projects/direct

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
OUT=/projects/direct/multiorgan/base_vsharp/jobs/predictions_biasfix
mkdir -p "$OUT" "$LISTS"

echo "=== Convert once (keep conv_out.bias) ==="
python3 - <<PY
import importlib.util
import re
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "convert_uniform_checkpoint",
    str(Path.home() / "projects/direct/projects/UNIFORM/tools/convert_uniform_checkpoint.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

src = Path("$SRC")
ckpt = Path("$CKPT")
print("loading", src, flush=True)
raw = torch.load(src, map_location="cpu", weights_only=False)
sd = raw["model"]
bias_keys = sorted(k for k in sd if re.search(r"denoiser_blocks\.\d+\.up_conv\.\d+\.1\.bias$", k))
print("legacy denoiser head biases:", len(bias_keys), flush=True)
for k in bias_keys:
    v = sd[k].float()
    print(f"  {k}: absmax={float(v.abs().max()):.4f} mean={float(v.mean()):.4f}", flush=True)

mod.convert_checkpoint(src, ckpt, "UNIFORM model_686000; keep denoiser conv_out.bias")
out = torch.load(ckpt, map_location="cpu", weights_only=False)
kept = [k for k in out["model"] if re.search(r"denoiser_blocks\.\d+\.conv_out\.bias$", k)]
print("converted nkeys", len(out["model"]), "kept conv_out.bias", len(kept), flush=True)
assert len(kept) == 12, kept
PY

echo "=== Strict load ==="
python3 - <<PY
from pathlib import Path

from direct.checkpointer import Checkpointer
from direct.environment import setup_inference_environment

ckpt = Path("$CKPT")
cfg = Path.home() / "projects/direct/projects/UNIFORM/configs/inference/inference_brain.yaml"
exp = Path("/projects/direct/multiorgan/base_vsharp/jobs/load_test_bias")
exp.mkdir(parents=True, exist_ok=True)
env = setup_inference_environment("", exp, "cuda:0", 0, False, cfg)
Checkpointer(save_directory=exp, save_to_disk=False, model=env.engine.model, **env.engine.models).load_models_from_file(ckpt)
b = env.engine.model.denoiser_blocks[0].conv_out.bias
print("LOAD OK bias0 absmax", float(b.detach().abs().max()), flush=True)
PY

predict() {
  local name=$1 cfg=$2 root=$3 lst=$4
  local od="${OUT}/${name}_4x"
  mkdir -p "$od"
  echo "=== predict ${name} ==="
  direct predict "$od" --cfg "$cfg" --checkpoint "$CKPT" --data-root "$root" \
    --filenames-filter "$lst" --num-gpus 1 --num-workers 4
}

printf "%s\n" "file_brain_AXT2_200_6002174.h5" "file_brain_AXT2_200_2000124.h5" > "$LISTS/brain_4x.lst"
printf "%s\n" "file1002155.h5" "file1000323.h5" > "$LISTS/knee_4x.lst"
printf "%s\n" "file_prostate_AXT2_0265.h5" "file_prostate_AXT2_0180.h5" > "$LISTS/prostate_4x.lst"
printf "%s\n" "P038_cine_sax.mat" "P038_cine_lax.mat" > "$LISTS/cardiac_val.lst"

predict brain "${CFG}/inference_brain.yaml" \
  /mnt/rhea/data_public/fastmri/brain/multicoil_test/fully_sampled "$LISTS/brain_4x.lst"
predict knee "${CFG}/inference_knee.yaml" \
  /mnt/rhea/data_public/fastmri/knee/multicoil_val "$LISTS/knee_4x.lst"

PROSTATE_ROOT=""
for d in /mnt/rhea/data_public/fastmri/prostate/training_T2_{1..9} /mnt/rhea/data_public/fastmri/prostate/test/T2; do
  if [[ -f "${d}/file_prostate_AXT2_0265.h5" ]]; then PROSTATE_ROOT="$d"; break; fi
done
if [[ -n "$PROSTATE_ROOT" ]]; then
  predict prostate "${CFG}/inference_prostate.yaml" "$PROSTATE_ROOT" "$LISTS/prostate_4x.lst"
fi

CARDIAC_FLAT=/projects/direct/multiorgan/base_vsharp/jobs/cardiac_val_flat
CARDIAC_BASE=/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Cine
mkdir -p "$CARDIAC_FLAT"
ln -sfn "$CARDIAC_BASE/ValidationSet/FullSample/P038/cine_sax.mat" "$CARDIAC_FLAT/P038_cine_sax.mat"
ln -sfn "$CARDIAC_BASE/ValidationSet/FullSample/P038/cine_lax.mat" "$CARDIAC_FLAT/P038_cine_lax.mat"
predict cardiac "${CFG}/inference_cardiac.yaml" "$CARDIAC_FLAT" "$LISTS/cardiac_val.lst"

echo "=== COMPARE TO REFS ==="
python3 - <<PY
import json
from pathlib import Path

base = Path("/projects/direct/multiorgan/base_vsharp")
refs = {
    "brain": base / "metrics_val_test_brain_4x_random_420001.json",
    "knee": base / "metrics_val_test_knee_4x_equip_420001.json",
    "prostate": base / "metrics_val_test_prostate_4x_equip_420001.json",
    "cardiac": base / "metrics_val_test_cardiac_4x_420001.json",
}
out = Path("$OUT")
for anat, refp in refs.items():
    pred_p = out / f"{anat}_4x" / "metrics_inference.json"
    if not pred_p.exists():
        print(anat, "MISSING")
        continue
    pred = json.load(open(pred_p))
    ref = json.load(open(refp))
    print("===", anat, "===")
    for k, v in pred.items():
        rk = k if k in ref else next((x for x in ref if Path(x).name == Path(k).name), None)
        r = ref.get(rk, {})
        for m in ["fastmri_ssim_metric", "fastmri_psnr_metric", "fastmri_nmse_metric"]:
            if m in v:
                rv = r.get(m, float("nan"))
                print(f"  {k} {m}: {v[m]:.5f} vs {rv:.5f} d={v[m]-rv:+.5f}")
PY
