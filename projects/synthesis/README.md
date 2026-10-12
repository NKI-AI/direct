# Magnitude images to synthetic multicoil k-space

Turns a magnitude image into plausible multicoil k-space, so reconstruction
networks can be trained on data that only exists as magnitude (DICOM, NIfTI).
The output is a draw from a conditional distribution, not a recovery of the
discarded scanner measurements.

```
k_c = FFT( S_c · m · e^{iφ} ),   Σ_c |S_c|² = 1  ⇒  RSS(IFFT(k)) = m exactly
```

| Network | Predicts | Conditioning |
|---|---|---|
| `SensitivityMapSynthesizer` | complex maps `S` for any coil count `N` (U-Net on a 64×64 grid, hard RSS normalisation) | magnitude, `N`, coil index, organ |
| `PhaseFlowMatching` | unit phase `φ` (OT conditional flow matching, Euler ODE, 30 steps) | magnitude, time, organ |

Training labels are computed from fully sampled k-space in the dataset
transforms (`transforms_type: SYNTHESIS`). Maps are `FULL_RSS`
(`S_c = IFFT(k_c) / RSS`, gauge-fixed to the strongest coil). The phase label is
the SENSE combination anchored at a bright pixel. Organ codes are
`brain, knee, prostate, breast, cardiac, unknown` (`num_organs=6`); 15 % of
training samples are relabelled `unknown` (classifier-free organ dropout), and
unseen anatomy is synthesised with `unknown` at apply time.

## Configs

| Config | Run | Notes |
|---|---|---|
| `sensitivity_maps_multi_cardiac.yaml` | `sensitivity_maps_multi_cardiac_500k` | v1, all five anatomies |
| `flow_matching_multi_cardiac.yaml` | `flow_matching_multi_cardiac` | v1, all five anatomies |
| `sensitivity_maps_multi_cardiac_v2.yaml` | (new) | v1 + active smoothness term, 150k cosine, EMA 0.999, 20-volume validation, 600 s loader timeout |
| `flow_matching_multi_cardiac_v2.yaml` | (new) | v1 + EMA 0.9999, 200k cosine, 20-volume validation, 600 s loader timeout |
| `single_organ/{sensitivity_maps,flow_matching}_{organ}_v2.yaml` | (new) | the v2 config restricted to one organ: single-organ baselines |
| `reconstruction.yaml` | | RecurrentVarNet on exported synthetic HDF5 |

Each v2 file lists its changes in its header. The v2 options are standard
`training` keys: `lr_scheduler: cosine`, `ema_decay` (validation and
`direct synthesize` / `direct predict` then use the EMA weights stored under
`ema` in the checkpoint) and `dataloader_timeout`.

The configs set `data_root` and `filenames_lists_root` per dataset, so run them
without `--training-root`; adjust both paths to your environment.

```bash
direct train <experiment_dir> --cfg projects/synthesis/sensitivity_maps_multi_cardiac_v2.yaml \
  --num-gpus 1 --num-workers 16 --device cuda --name sensitivity_maps_multi_cardiac_v2 --mixed-precision

direct train <experiment_dir> --cfg projects/synthesis/flow_matching_multi_cardiac_v2.yaml \
  --num-gpus 2 --num-workers 8 --device cuda --name flow_matching_multi_cardiac_v2 --mixed-precision
```

## Splits

Patient-level, seed 42, recorded in `lists/paper_split_manifest.json`. No file
appears in two splits.

| Split | Role | Data |
|---|---|---|
| **S_train** | train the synthesizers | real k-space (labels: SENSE phase + FULL_RSS maps) |
| **S_val** | synthesizer checkpoint selection | real k-space |
| **R_train** | train reconstruction models, same patients in both arms | real arm: real k-space; synthetic arm: magnitude only |
| **R_val** | reconstruction model selection | real k-space only |
| **R_test** | frozen reconstruction test | real k-space only |

Fractions:

- FastMRI brain, knee, prostate T2: official train → S_train 55 % / S_val 5 % / R_train 40 %; official val → R_val 50 % / R_test 50 %.
- FastMRI breast (no official val; split by patient, `fastMRI_breast_{id}_*` stay together): 45 / 5 / 40 / 5 / 5.
- CMRxRecon cine (2023 + 2024 FullSample): train patients 55 / 5 / 40 stratified by year × view (sax / lax / lvot); ValidationSet → R_val / R_test 50 / 50.
- Brain is stratified by contrast (AXFLAIR, AXT1, AXT1PRE, AXT1POST, AXT2), knee by PD / PDFS. S_train and R_train are then split per coil count.
- Excluded truncated files: `file_brain_AXFLAIR_200_6002462.h5` (brain val), `2023/P004/cine_sax.mat` (cardiac).

Volumes per split (cardiac counts are cine files):

| Anatomy | S_train | S_val | R_train | R_val | R_test | Coils |
|---|---|---|---|---|---|---|
| Brain | 2458 | 223 | 1788 | 689 | 688 | 2–28 (mostly 4 / 16 / 20) |
| Knee | 535 | 48 | 390 | 100 | 99 | 15 |
| Prostate T2 | 120 | 11 | 87 | 24 | 24 | 10–30 (mostly 20) |
| Breast | 269 | 30 | 239 | 29 | 30 | 16 |
| Cardiac cine | 427 | 38 | 306 | 145 | 143 | 10 |

### Lists

All lists are under `lists/<anatomy>/`, one filename per line, relative to the
dataset root.

| File | Content |
|---|---|
| `s_train.lst`, `s_val.lst` | synthesizer train / validation volumes |
| `s_train_{N}_coils.lst` | S_train split by coil count (one training dataset per `N`, so batches share a coil count) |
| `s_val_sub.lst` | 20-volume evenly spaced subset of `s_val.lst` (all 11 for prostate), used by the v2 configs |
| `r_train.lst`, `r_val.lst`, `r_test.lst` | reconstruction train / validation / test volumes |
| `r_train_{N}_coils.lst` | R_train split by coil count |

Coil counts with a list: brain S_train 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28 and R_train 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;
knee 15; prostate S_train 10, 14, 16, 20, 24, 26, 30 and R_train 14, 16, 20, 24, 26, 30; breast 16; cardiac 10.

Dataset roots used by the configs:

| Anatomy | S_train / S_val / R_train | R_val / R_test |
|---|---|---|
| Brain | `fastmri/brain/multicoil_train` | `fastmri/brain/multicoil_val` |
| Knee | `fastmri/knee/multicoil_train` | `fastmri/knee/multicoil_val` |
| Prostate | `fastmri/prostate/training/T2` | `fastmri/prostate/validation/T2` |
| Breast | `fastmri/breast/derived/cartesian` | same |
| Cardiac | CMRxRecon cine `train/` (flat symlinks `y{year}_PXXX_cine_*.mat`, `kspace_key: kspace_full`) | CMRxRecon cine `val/` |

## Apply

`direct synthesize` takes a config with the phase model under `model`, the map
model under `additional_models.sensitivity_model`, a `synthesize` block and an
`inference.dataset`:

```bash
direct synthesize <output_dir> --cfg <synthesize.yaml> \
  --phase-model <flow checkpoint> --sensitivity-model <map checkpoint> \
  --data-root <fastmri root> --device cuda
```

## Code

| Path | Content |
|---|---|
| `direct/nn/synthesis/` | models, engines, losses, model configs |
| `direct/synthesis/` | physics (`synthesize`, `normalize_sensitivities`), HDF5 export, evaluation |
| `direct/synthesize.py` | `direct synthesize` |

```bash
python -m pytest tests/tests_synthesis -q
```

See `research_design.md` for the design rationale and evaluation protocol.
