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

Every dataset is split into the same five roles, at patient (or study) level,
seed 42. No patient appears in two roles.

| Role | Used for | Data seen |
|---|---|---|
| **S_train** | training the map and phase synthesizers | real k-space (labels: SENSE phase + FULL_RSS maps) |
| **S_val** | synthesizer checkpoint selection | real k-space |
| **R_train** | training reconstruction models; the same patients in every arm | real arm: real k-space; synthetic arms: magnitude only |
| **R_val** | reconstruction model selection | real k-space only |
| **R_test** | final reconstruction test, used once | real k-space only |

How each dataset maps onto the roles:

| Dataset | Unit | Official train → | Official val / test → |
|---|---|---|---|
| FastMRI brain, knee, prostate T2 | volume | S_train 55 % / S_val 5 % / R_train 40 % | val → R_val 50 % / R_test 50 % |
| FastMRI breast | patient (`fastMRI_breast_{id}_*`) | single folder: 45 / 5 / 40 / 5 / 5 | (in the same split) |
| CMRxRecon cine 2023 + 2024 | patient | S_train 55 % / S_val 5 % / R_train 40 %, stratified by year × view | ValidationSet → R_val 50 % / R_test 50 % |
| MosaicMRI | study (ISMRMRD `StudyLOID`) | S_train 55 % / S_val 5 % / R_train 40 %, stratified by anatomy | val → R_val, test → R_test (kept as released) |

Brain is stratified by contrast (AXFLAIR, AXT1, AXT1PRE, AXT1POST, AXT2) and
knee by PD / PDFS. MosaicMRI has no patient ID; its official splits are
study-disjoint (914 studies), and studies are the grouping unit here. Excluded
truncated files: `file_brain_AXFLAIR_200_6002462.h5`, CMRxRecon `2023/P004/cine_sax.mat`.

Volumes per role (cardiac: cine files; MosaicMRI: volumes, with studies in brackets):

| Dataset | S_train | S_val | R_train | R_val | R_test | Coils |
|---|---|---|---|---|---|---|
| Brain | 2458 | 223 | 1788 | 689 | 688 | 2–28 (mostly 4 / 16 / 20) |
| Knee | 535 | 48 | 390 | 100 | 99 | 15 |
| Prostate T2 | 120 | 11 | 87 | 24 | 24 | 10–30 (mostly 20) |
| Breast | 269 | 30 | 239 | 29 | 30 | 16 |
| Cardiac cine | 427 | 38 | 306 | 145 | 143 | 10 |
| MosaicMRI (10 MSK anatomies) | 1230 (349) | 101 (33) | 835 (251) | 465 (138) | 465 (143) | 4–46 (mostly 12 / 16) |

MosaicMRI per anatomy (volumes):

| Anatomy | S_train | S_val | R_train | R_val | R_test |
|---|---|---|---|---|---|
| ankle | 43 | 4 | 33 | 19 | 17 |
| elbow | 28 | 4 | 11 | 10 | 8 |
| foot | 43 | 3 | 33 | 18 | 17 |
| hip | 104 | 9 | 77 | 41 | 43 |
| knee | 95 | 7 | 69 | 36 | 36 |
| leg_tibfib | 9 | 1 | 3 | 2 | 4 |
| pelvis_pubalgia | 33 | 5 | 16 | 11 | 11 |
| shoulder | 173 | 14 | 126 | 67 | 66 |
| spine | 631 | 51 | 421 | 236 | 237 |
| wrist_hand_thumb | 71 | 3 | 46 | 25 | 26 |

### Lists

Lists are under `lists/<dataset>/` (`brain`, `knee`, `prostate`, `breast`,
`cardiac`, `mosaic`), one file per line, relative to the dataset root. MosaicMRI
entries are `<anatomy>/<file>.h5`.

| File | Content |
|---|---|
| `s_train.lst`, `s_val.lst` | synthesizer train / validation volumes |
| `s_train_{N}_coils.lst` | S_train split by coil count (one training dataset per `N`, so batches share a coil count) |
| `s_val_sub.lst` | small S_val subset for cheap validation in the v2 configs: 20 evenly spaced volumes (all 11 for prostate); MosaicMRI 2 per anatomy |
| `r_train.lst`, `r_val.lst`, `r_test.lst` | reconstruction train / validation / test volumes |
| `r_train_{N}_coils.lst` | R_train split by coil count |
| `paper_split_manifest.json` (top level), `mosaic/mosaic_split_manifest.json` | seeds, roots and counts per role, contrast and coil count |

Dataset roots used by the configs:

| Dataset | S_train / S_val / R_train | R_val | R_test |
|---|---|---|---|
| Brain | `fastmri/brain/multicoil_train` | `fastmri/brain/multicoil_val` | same as R_val |
| Knee | `fastmri/knee/multicoil_train` | `fastmri/knee/multicoil_val` | same as R_val |
| Prostate | `fastmri/prostate/training/T2` | `fastmri/prostate/validation/T2` | same as R_val |
| Breast | `fastmri/breast/derived/cartesian` | same | same |
| Cardiac | CMRxRecon cine `train/` (flat symlinks `y{year}_PXXX_cine_*.mat`, `kspace_key: kspace_full`) | CMRxRecon cine `val/` | same as R_val |
| MosaicMRI | `MosaicMRI/multicoil_train` | `MosaicMRI/multicoil_val` | `MosaicMRI/multicoil_test` |

## Experiment

Question: can reconstruction networks be trained on k-space synthesised from
magnitude images alone, instead of raw k-space?

1. **Train the synthesizers** on S_train of brain, knee, prostate, breast and
   cardiac (`*_multi_cardiac_v2.yaml`), select on S_val. MosaicMRI is not used
   for synthesizer training; its S_val subset only monitors how the synthesizers
   generalise to unseen anatomies (organ code `unknown`, knee excepted).
2. **Synthesise R_train.** For every R_train slice, take only the RSS magnitude,
   draw maps (map synthesizer with the slice's coil count) and phase (flow
   model), and form `k = FFT(S · m · e^{iφ})`. The real R_train k-space is never
   read by the synthetic arms.
3. **Train reconstruction models** (vSHARP, RecurrentVarNet) on R_train with the
   same patients, masks, accelerations, iteration budget and ACS-estimated maps
   in every arm:
   - real k-space (upper reference),
   - synthetic: learned maps + learned phase (main method),
   - controls: zero phase; smooth random phase; simulated coils (birdcage / surface / Biot–Savart) with learned phase,
   - real + synthetic mixed.
4. **Select on R_val and report on R_test**, real k-space only: NMSE, PSNR,
   SSIM per volume, patient-level bootstrap intervals, per dataset and per
   MosaicMRI anatomy.
5. **MosaicMRI as the unseen-anatomy test.** The synthesizers never saw
   musculoskeletal raw data. Synthesising Mosaic R_train from magnitudes and
   testing on real Mosaic R_test measures whether magnitude-only data of a new
   anatomy can replace its raw data. Ablation: add Mosaic S_train to synthesizer
   training to get the in-domain upper bound.
6. **Single-organ baselines** (`single_organ/*_v2.yaml`) show what the shared
   multi-organ synthesizer gains or loses per organ.

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
