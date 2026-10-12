# Paper splits: synthetic k-space vs real k-space for reconstruction

Seed **42**.
Manifest: `projects/synthesis/lists/paper_split_manifest.json`.

## Scientific roles

| Split | Role | Real k-space? |
|---|---|---|
| **S_train** | Train synthesizer \(p(\varphi \mid m, \text{organ})\) | Yes (SENSE phase + FULL_RSS maps) |
| **S_val** | Synthesizer checkpointing | Yes |
| **R_train** | Train two recon models on the **same patients** | Real arm: yes. Synth arm: magnitude + simulated maps |
| **R_val** | Recon early stopping | Yes, real only |
| **R_test** | Frozen recon test | Yes, real only |

No filename is in two splits. Breast files from the same patient (`fastMRI_breast_{id}_*`) stay together.

## Fractions

Official FastMRI **train** (brain, knee, prostate T2):

- S_train **55%** / S_val **5%** / R_train **40%**

Official FastMRI **val** (brain, knee, prostate T2):

- R_val **50%** / R_test **50%**

Breast has no official val (one cartesian folder). Patients:

- S_train **45%** / S_val **5%** / R_train **40%** / R_val **5%** / R_test **5%**

Brain is stratified by contrast (`AXFLAIR`, `AXT1`, `AXT1PRE`, `AXT1POST`, `AXT2`).
Knee is stratified by PD vs PDFS. S_train and R_train are then split **per coil count**.

Skipped truncated file: `file_brain_AXFLAIR_200_6002462.h5` in brain val.

## File counts (volumes)

| Anatomy | S_train | S_val | R_train | R_val | R_test |
|---|---|---|---|---|---|
| Brain | 2458 | 223 | 1788 | 689 | 688 |
| Knee | 535 | 48 | 390 | 100 | 99 |
| Prostate T2 | 120 | 11 | 87 | 24 | 24 |
| Breast | 269 | 30 | 239 | 29 | 30 |
| Cardiac Cine (CMRxRecon 2023+2024) | 427 | 38 | 306 | 145 | 143 |

Lists live under `projects/synthesis/lists/{brain,knee,prostate,breast,cardiac}/`:

- `s_train.lst`, `s_val.lst`, `r_train.lst`, `r_val.lst`, `r_test.lst`
- `s_train_{n}_coils.lst`, `r_train_{n}_coils.lst`

Roots:

- Brain: `/mnt/rhea/data_public/fastmri/brain/multicoil_{train,val}/`
- Knee: `/mnt/rhea/data_public/fastmri/knee/multicoil_{train,val}/`
- Prostate: `/mnt/rhea/data_public/fastmri/prostate/{training,validation}/T2/`
- Breast: `/mnt/rhea/data_public/fastmri/breast/derived/cartesian/`
- Cardiac: CMRxRecon **2023 + 2024** Cine FullSample via flat symlinks
  `/projects/direct/synthesis_diffusion/cmrxrecon_cine/{train,val}/`
  (`y{year}_PXXX_cine_*.mat`).
  Loaded with DIRECT ``CMRxRecon`` dataset (`kspace_key=kspace_full`). Same 55/5/40 on train
  patients (stratified by year × sax/lax/lvot), ValidationSet → R_val/R_test 50/50. All 10-coil.
  Skipped truncated `2023/P004/cine_sax.mat`.

## Organ conditioning

AdaIN organ codes: `brain, knee, prostate, breast, cardiac, unknown` (`num_organs=6`).
Unrecognized paths → `unknown` (not a silent neighbor). Sensitivity-map training uses
**15% organ dropout** onto `unknown` (classifier-free). At apply time, OOD anatomy uses `unknown`.

## Synthesizer (phase job)

- Model: `PhaseFlowMatching`, 128-filter / 4-pool U-Net, AdaIN on time **and organ**.
- Labels: full-kspace RSS maps + SENSE phase. Velocity loss only. 30 ODE eval steps.
- Train data: **S_train coil lists only** (original four anatomies for the finished FM run).
- Config: `projects/synthesis/flow_matching_multi_cardiac.yaml`
- Job: 2×A100, no `--training-root` (each dataset has `data_root`).
- Experiment dir: `/projects/direct/synthesis_diffusion/flow_matching_multi`

## Sensitivity maps (cardiac + unknown)

- Model: `SensitivityMapSynthesizer`, nearest+conv upsample, ∇S smooth, organ dropout 0.15.
- Data: paper **S_train/S_val** for brain/knee/prostate/breast **plus** cardiac cine.
- Config: `projects/synthesis/sensitivity_maps_multi_cardiac.yaml`
- Experiment dir: `/projects/direct/synthesis_diffusion/sensitivity_maps_multi_cardiac_500k`

## Phase flow matching (cardiac + unknown)

- Same data/organ policy as the map synthesizer (`num_organs=6`, dropout 0.15).
- Config: `projects/synthesis/flow_matching_multi_cardiac.yaml`
- Fresh run (do not resume old 4-organ `flow_matching_multi` checkpoints).
- Experiment dir: `/projects/direct/synthesis_diffusion/flow_matching_multi_cardiac`


## Next (not this job)

1. Apply the synthesizer to **R_train magnitudes**. Maps at apply time come from DIRECT's existing coil simulator (`birdcage`, `surface`, `biot_savart`) or `empirical` reuse of real maps — not from R_train k-space.
2. Train matched recon models (RecurrentVarNet, then vSHARP) on synthetic vs real R_train k-space, same masks, ACS maps at recon time.
3. Report recon metrics on **R_val** then frozen **R_test** (real k-space only).
