#!/usr/bin/env python3
"""Pre-generate the small synthetic-recon store: 320x320 magnitude + a K-phase bank per R_train slice.

Magnitude is extracted exactly as the synthesizer saw it (SYNTHESIS transform: 320 crop, FULL_RSS,
p99 scale, coils compressed to 30). Phases come from the flow-matching model (EMA weights); K samples
per slice with fixed seeds, stored as fp16 angle. Sensitivity maps are NOT stored (generated on the
fly during recon). One HDF5 per input file under ``--out-root/<dataset>/<file>.h5``.

    python gen_synth_store.py --dataset cardiac --split r_train --device cuda
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, "/home/g.yiasemis/projects/direct")  # use this checkout, not an editable-install path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
from direct.checkpointer import load_ema_weights
from direct.data import datasets_config as C
from direct.data import transforms as T
from direct.data.datasets import build_dataset_from_input
from direct.data.mri_transforms import build_mri_transforms
from direct.nn.synthesis.synthesis import PhaseFlowMatching, organ_id_from_filename
from direct.utils import dict_flatten, remove_keys

PROJ = pathlib.Path("/home/g.yiasemis/projects/direct/projects/synthesis")
E = "/projects/direct/synthesis_diffusion"
PHASE_CKPT = f"{E}/flow_matching_multi_cardiac_v2/model_155000.pt"
ROOTS = {
    "brain": ("FastMRI", "/mnt/rhea/data_public/fastmri/brain/multicoil_train", {}),
    "knee": ("FastMRI", "/mnt/rhea/data_public/fastmri/knee/multicoil_train", {}),
    "prostate": ("FastMRI", "/mnt/rhea/data_public/fastmri/prostate/training/T2", {}),
    "breast": ("FastMRI", "/mnt/rhea/data_public/fastmri/breast/derived/cartesian", {}),
    "cardiac": ("CMRxRecon", f"{E}/cmrxrecon_cine/train",
                {"kspace_key": "kspace_full", "compute_mask": False, "kspace_context": None}),
    "mosaic": ("MosaicMRI", "/mnt/kronos/data_public/MosaicMRI/multicoil_train", {}),
}

def synth_transform():
    tf = {"transforms_type": "SYNTHESIS", "use_seed": True, "masking": None,
          "cropping": {"crop": "[320, 320]", "image_center_crop": True},
          "augmentation": {"pad": [320, 320]},
          "sensitivity_map_estimation": {"estimate_sensitivity_maps": True, "sensitivity_maps_type": "FULL_RSS",
                                         "sensitivity_maps_gaussian": None},
          "normalization": {"scaling_key": "kspace", "scale_percentile": 0.99},
          "synthesis_calibration": {"synthesis_extract_phase_and_maps": False},  # only need magnitude
          "compress_coils": 30}
    return build_mri_transforms(forward_operator=T.fft2, backward_operator=T.ifft2, mask_func=None,
                                **dict_flatten(dict(remove_keys(tf, "masking"))))

def load_phase(device):
    m = PhaseFlowMatching(num_filters=128, num_pool_layers=4, num_integration_steps=50,
                          eval_integration_steps=30, num_organs=15, organ_embed_dim=16, organ_dropout_prob=0.0)
    blob = torch.load(PHASE_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict({k.removeprefix("module."): v for k, v in blob["model"].items()})
    assert load_ema_weights(m, PHASE_CKPT), "no EMA weights"
    return m.to(device).eval()

@torch.no_grad()
def run(args):
    device = args.device
    dsname = args.dataset
    name, root, extra = ROOTS[dsname]
    root = pathlib.Path(root)
    files = (PROJ / f"lists/{dsname}/{args.split}.lst").read_text().split()
    if args.limit: files = files[: args.limit]
    out_root = pathlib.Path(args.out_root) / dsname
    out_root.mkdir(parents=True, exist_ok=True)
    tr = synth_transform()
    phase = load_phase(device)
    dcfg = OmegaConf.merge(OmegaConf.structured(getattr(C, f"{name}Config")), {"name": name, **extra})

    def real_coils(path):
        with h5py.File(path, "r") as h:
            if "kspace" in h:
                return int(h["kspace"].shape[1])
            k = h["kspace_full"]  # CMRxRecon (t, slice, coil, H, W)
            return int(k.shape[-3])
    for fi, f in enumerate(files):
        out = out_root / (pathlib.Path(f).stem + ".h5")
        if out.exists() and not args.overwrite:
            continue
        ncoils = min(real_coils(root / f), 30)
        ds = build_dataset_from_input(transforms=tr, dataset_config=dcfg, data_root=root, filenames_filter=[root / f])
        organ = organ_id_from_filename(str(root / f))
        mag_list, slice_nos = [], []
        for i in range(len(ds)):
            s = ds[i]
            m = s["magnitude"]
            m = m if m.ndim == 4 else m.unsqueeze(0)  # (1,1,H,W)
            mag_list.append(m[0])  # (1,H,W)
            slice_nos.append(int(s.get("slice_no", i)))
        mags_t = torch.stack(mag_list).to(device).float()  # (S,1,H,W)
        S, K = mags_t.shape[0], args.k
        # Batch all (slice x K) magnitudes through the phase ODE in chunks; one generator per file,
        # so each (slice,k) element gets its own noise -> K distinct phases per slice.
        rep = mags_t.repeat_interleave(K, dim=0)  # (S*K,1,H,W)
        org = torch.full((rep.shape[0],), organ, device=device, dtype=torch.long)
        g = torch.Generator(device=device).manual_seed(args.seed_base + fi)
        ang = torch.empty(rep.shape[0], rep.shape[2], rep.shape[3], dtype=torch.float16)
        for a in range(0, rep.shape[0], args.chunk):
            b = slice(a, a + args.chunk)
            ph = phase.sample_phase(rep[b], g, num_steps=args.steps, organ_id=org[b])  # (n,H,W,2)
            ang[b] = torch.atan2(ph[..., 1], ph[..., 0]).half().cpu()
        banks = ang.reshape(S, K, rep.shape[2], rep.shape[3]).numpy()  # (S,K,H,W)
        mags = mags_t[:, 0].cpu().numpy().astype(np.float32)  # (S,H,W)
        tmp = out.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as h:
            h.create_dataset("magnitude", data=mags, compression="gzip", compression_opts=1)
            h.create_dataset("phase_bank_angle", data=banks, compression="gzip", compression_opts=1)
            h.attrs["organ_id"] = organ
            h.attrs["num_coils"] = ncoils
            h.attrs["k_phases"] = args.k
            h.attrs["source_file"] = str(root / f)
            h.create_dataset("slice_no", data=np.array(slice_nos))
        tmp.rename(out)
        if (fi + 1) % 20 == 0:
            print(f"  {dsname}/{args.split}: {fi+1}/{len(files)} files", flush=True)
    print(f"RESULT {dsname}/{args.split}: {len(files)} files -> {out_root}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ROOTS))
    p.add_argument("--split", default="r_train")
    p.add_argument("--out-root", default=f"{E}/synth_store")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())

if __name__ == "__main__":
    main()
