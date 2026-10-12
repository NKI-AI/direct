# Magnitude images to synthetic multicoil k-space

This project turns a magnitude image into plausible multicoil k-space so that
reconstruction networks can be trained on data that only exists as magnitude
(DICOM, NIfTI). The output is a draw from a conditional distribution, not a
recovery of the discarded scanner measurements.

```
k_c = FFT( S_c · m · e^{iφ} ),   Σ_c |S_c|² = 1  ⇒  RSS(IFFT(k)) = m exactly
```

Two learned networks supply the factors:

| Network | Predicts | Conditioning | Config |
|---|---|---|---|
| `SensitivityMapSynthesizer` | complex maps `S` for any coil count `N` | magnitude, `N`, coil index, organ | `sensitivity_maps_multi_cardiac.yaml` |
| `PhaseFlowMatching` | unit phase `φ` (OT conditional flow matching, Euler ODE) | magnitude, time, organ | `flow_matching_multi_cardiac.yaml` |

Training labels come from fully sampled k-space inside the dataset transforms
(`transforms_type: SYNTHESIS`): maps are `FULL_RSS` (`S_c = IFFT(k_c) / RSS`,
gauge-fixed to the strongest coil) and the phase label is the SENSE combination
anchored at a bright pixel. Organ codes are `brain, knee, prostate, breast,
cardiac, unknown`; 15 % of training samples are relabelled `unknown` so the
models also work on anatomy they have not seen.

## Layout

| Path | Content |
|---|---|
| `direct/nn/synthesis/` | models, engines, losses, configs (registered with DIRECT) |
| `direct/synthesis/` | physics (`synthesize`, `normalize_sensitivities`), HDF5 export, evaluation helpers, legacy CLI |
| `direct/synthesize.py`, `direct/cli/synthesize.py` | `direct synthesize`: apply phase + map checkpoints to a dataset |
| `projects/synthesis/lists/` | paper split lists and `paper_split_manifest.json` |
| `projects/synthesis/paper_splits.md` | split roles, fractions and counts |
| `projects/synthesis/tools/` | split generators, config validator, apply-time and figure scripts |

## Paper configs

| Config | Role |
|---|---|
| `sensitivity_maps_multi_cardiac.yaml` | map synthesizer, 5 anatomies, S_train / S_val |
| `flow_matching_multi_cardiac.yaml` | phase flow matching, 5 anatomies, S_train / S_val |
| `synthesize_prostate_smoke.yaml` | `direct synthesize` smoke test (4-organ checkpoints) |
| `sensitivity_maps_cmrx_smoke.yaml` | 200-iteration CMRxRecon smoke run |
| `reconstruction.yaml` | RecurrentVarNet on exported synthetic HDF5 |

The other YAMLs are earlier single-anatomy or ablation runs (`*_brain`,
`*_breast`, `*_multi`, `*_smooth*`, `diffusion_*`, `phase_*`, `complex_*`,
`factorized_*`) and the `*_smoke` / `*_e2e` pipeline checks. Configs with
`filenames_lists_root` set must be run without `--training-root`.

Validate configs before submitting (schema, model + engine construction,
transforms, list files and, with `--check-files`, every listed volume):

```bash
python projects/synthesis/tools/validate_configs.py projects/synthesis/*.yaml --check-files
```

## Train

```bash
direct train /projects/direct/synthesis_diffusion \
  --cfg projects/synthesis/flow_matching_multi_cardiac.yaml \
  --num-gpus 2 --num-workers 8 --device cuda --name flow_matching_multi_cardiac --mixed-precision
```

The logged `loss` is the sum of optimised terms only; entries ending in
`_metric` are monitors.

## Apply

```bash
direct synthesize /path/to/output \
  --cfg projects/synthesis/synthesize_prostate_smoke.yaml \
  --phase-model <flow checkpoint> --sensitivity-model <map checkpoint> \
  --data-root /mnt/rhea/data_public/fastmri/prostate/training/T2 \
  --filenames-filter projects/synthesis/lists/prostate/s_val_smoke.lst --device cuda
```

For NIfTI / NRRD magnitude volumes (MSD, OASIS, prostate biopsy) use
`tools/synthesize_msd_brain_maps_phase.py --input <file> --organ <name> --num-coils N`.

## Tests

```bash
python -m pytest tests/tests_synthesis -q
ruff check direct/nn/synthesis direct/synthesis tests/tests_synthesis projects/synthesis
```

---

# Legacy standalone CLI

The sections below describe `python -m direct.synthesis.cli`, the first
standalone implementation (baseline, factorized and DDPM generators trained
outside the DIRECT engine). It is kept for its tests and DICOM ingestion; new
work should use the DIRECT configs above. See
[research_design.md](research_design.md) for the original design rationale.

## Installation

Use the repository's normal Python 3.12+ setup and enable the small DICOM extra:

```bash
uv sync --extra synthesis
uv run python -m direct.synthesis.cli --help
```

In an existing working DIRECT environment, install `pydicom>=3,<4` and use
`python` in place of `uv run python` below. DIRECT's native extensions must
already be built. The synthesis code adds no generative-model dependency.

## Training with DIRECT

The primary path is the same as reconstruction: YAML config + `direct train` /
`direct predict`. TensorBoard tags match recon (`train/target`,
`val/{dataset}/prediction`, `val/{dataset}/loss`, per-case `*_metric` JSON).

```bash
direct train experiments/synthesis_diffusion \
  --cfg projects/synthesis/diffusion_e2e.yaml \
  --training-root /path/to/fastmri/knee \
  --validation-root /path/to/fastmri/knee/val \
  --device cuda --name diffusion_knee

direct predict experiments/synthesis_diffusion/generated \
  --cfg projects/synthesis/diffusion_e2e.yaml \
  --checkpoint experiments/synthesis_diffusion/diffusion_knee/model_24000.pt \
  --data-root /path/to/fastmri/knee/val \
  --experiment_directory experiments/synthesis_diffusion/diffusion_knee \
  --device cuda --name diffusion_gen
```

Configs: `phase_e2e.yaml` / `complex_e2e.yaml` (supervised), `diffusion_e2e.yaml`
(conditional DDPM), `flow_matching_e2e.yaml` (OT-CFM). Smoke YAMLs run a few
iterations to check the pipeline boots.

The `python -m direct.synthesis.cli` helpers below are optional (DICOM/H5 prepare).

## 1. Establish the data split

Split **patients**, before extracting slices or estimating maps. Keep donor
coils, generator training patients, reconstruction training patients, validation,
and the final real acquisition test set auditable. Multiple examinations from
one patient belong to the same split. Subject IDs should be study pseudonyms.

Create a JSON manifest with these exact fields; relative paths are relative to
the manifest. This example is a schema example, not a supplied dataset:

```json
[
  {
    "path": "/data/raw/train_volume.h5",
    "subject_id": "subject_001",
    "split": "train",
    "domain": "prostate_t2_3t_axial"
  },
  {
    "path": "/data/raw/validation_volume.h5",
    "subject_id": "subject_002",
    "split": "validation",
    "domain": "prostate_t2_3t_axial"
  }
]
```

Input HDF5 files must contain **fully sampled**, centered native-complex
`kspace` of shape `(slices, coils, H, W)`. Calibration is never evidence that an
acquisition is fully sampled. Verify acquisition completeness externally,
including partial Fourier, GRAPPA-filled data and readout padding. No automatic
cropping, resizing, coil compression or removal of readout oversampling occurs.

```bash
uv run python -m direct.synthesis.cli prepare \
  --manifest manifest.json --output prepared \
  --fully-sampled --acs-size 24 --max-projection-nmse 0.1
```

The preparation step reuses `EstimateSensitivityMapModule`, fixes the local
coil-map gauge, removes global receiver phase, and records the single-map
projection NMSE. The input image is the full-coil RSS. Calibration support
weights suppress unreliable phase supervision. Slices exceeding the configured
projection-error threshold are logged and excluded. The default threshold is a
starting screening criterion, not an established MRI quality threshold.

Output directories are new-only. Each volume is read slice-by-slice and each
HDF5 file is published after its write completes. If preparation stops, earlier
completed volumes remain; remove or relocate the incomplete run directory
before restarting. Do not silently accept a large rejection rate: inspect the
calibration region, noise, FOV, support and single-map approximation.

## 2. Run the baseline immediately

```bash
uv run python -m direct.synthesis.cli generate \
  --method baseline --dicom image.dcm --output synthetic/image_001.h5 \
  --domain prostate_t2_3t_axial --num-coils 16 --seed 42
```

For empirical sensitivities, replace simulated coils with a selected training
donor. The donor maps must share the spatial array size; physical FOV and
orientation matching remain a curation requirement:

```bash
uv run python -m direct.synthesis.cli generate \
  --method baseline --dicom image.dcm --output synthetic/image_002.h5 \
  --domain prostate_t2_3t_axial --seed 42 \
  --maps prepared/train/volume_000000.h5 --map-slice 0
```

`--phase-scale` is the coarse random phase scale in radians. The simulator is
an inexpensive Gaussian coil-layout approximation with smooth phase ramps,
not an electromagnetic field simulation. Tune its distribution against real
training acquisitions, and use held-out data to assess transfer.

## 3. Train the learned methods

```bash
uv run python -m direct.synthesis.cli train \
  --method factorized --train prepared/train --validation prepared/validation \
  --output runs/factorized --device cuda --epochs 100 --seed 42

uv run python -m direct.synthesis.cli train \
  --method diffusion --train prepared/train --validation prepared/validation \
  --output runs/diffusion --device cuda --epochs 100 --timesteps 1000 --seed 42
```

The default batch size is one, so image shapes and, for diffusion, coil counts
can vary between examples. Larger batches require matching shapes and coil
counts. Factorized synthesis always requires a fixed coil count **and consistent
coil ordering/configuration**, which must be curated upstream. Do not equate
the same coil count across unrelated arrays with matching configurations.

The phase U-Net requires `min(H, W) >= 2**(num_pool_layers + 1)`. The sensitivity
branch predicts a `map_size × map_size` grid (default 32). This reduces spatial
bandwidth; normalization and gauge conventions do not guarantee physical coil
fields. Both branches use DIRECT's existing `UnetModel2d`.

Each run records configuration, cohort labels, source hashes and subject IDs.
Training writes epoch metrics, `last.pt`, and `best.pt` when validation improves.
The diffusion checkpoint criterion is validation noise-prediction loss, not
proven downstream reconstruction utility. Model comparison must use the
separate real reconstruction protocol.

Resume at an epoch boundary into a new run directory, keeping model arguments
the same. The checkpoint restores optimizer, loader and random-generator states.
CUDA reproducibility can still depend on hardware and kernel determinism.

```bash
uv run python -m direct.synthesis.cli train \
  --method diffusion --train prepared/train --validation prepared/validation \
  --output runs/diffusion_continued --device cuda --epochs 200 --timesteps 1000 \
  --seed 42 --resume runs/diffusion/last.pt
```

The validation best score carries across runs. If no new best is reached, the
previous run's `best.pt` remains the best checkpoint. `last.pt` is always written
in the resumed run after a completed epoch. Keep seed, batch size and data
curation unchanged for a reproducible continuation. Training is single-process;
distributed orchestration is not implemented for the generators.

## 4. Generate from DICOM

```bash
uv run python -m direct.synthesis.cli generate \
  --method factorized --checkpoint runs/factorized/best.pt \
  --dicom image.dcm --output synthetic/factorized_001.h5 \
  --domain prostate_t2_3t_axial --device cuda

uv run python -m direct.synthesis.cli generate \
  --method diffusion --checkpoint runs/diffusion/best.pt \
  --dicom image.dcm --output synthetic/diffusion_001.h5 \
  --domain prostate_t2_3t_axial --device cuda --seed 42 \
  --maps prepared/train/volume_000000.h5 --map-slice 0
```

The CLI processes one 2D DICOM at a time. Generate multiple realizations with
different seeds and donor maps; sample donors only from the allowed training
pool. Record which input patient each exported file belongs to in your study
manifest. Use unique output paths: exports never overwrite files.

DICOM ingestion applies the modality rescale/LUT, removes declared padding and
normalizes by the maximum valid intensity. It does **not** apply window/level.
It records input hash, scale and available spacing/orientation, without copying
patient tags. Missing component metadata requires `--assume-magnitude` after
review. Explicit phase, real/imaginary, ADC and SWI declarations are rejected.
Only classic MR Image Storage, single-frame, monochrome magnitude images are
supported. Enhanced MR, Siemens mosaics and compressed transfer syntaxes without
an installed decoder require an explicit preprocessing workflow.

The reader is not a general DICOM de-identification or diagnostic-image
classification tool. A magnitude label does not exclude denoising, sharpening,
interpolation, bias correction, vendor scaling or other nonlinear processing.
Series selection and spatial/acquisition compatibility must be reviewed.

## 5. Train an existing DIRECT reconstruction model

Exports contain:

- `kspace`: native complex64 `(slices, coils, H, W)`;
- `reconstruction_rss`: float32 conditioning magnitude;
- `synthesis/sensitivity_map` and `synthesis/phase`: diagnostic factors;
- provenance, a synthetic-data flag and physical-consistency metrics.

DIRECT's existing `FastMRIDataset` reads these files unchanged. No dataset
registry, reconstruction engine or model API was replaced. Synthetic ground
truth sensitivities are not loaded into normal reconstruction samples: estimate
sensitivities from the **undersampled ACS** as for real data.

```bash
uv run direct train runs/reconstruction \
  --cfg projects/synthesis/reconstruction.yaml \
  --training-root synthetic \
  --validation-root /data/real_validation \
  --num-gpus 1
```

The supplied config uses RecurrentVarNet and ordinary supervised RSS loss.
Keep acquisition-grid and crop choices matched across real and synthetic cohorts;
this example disables cropping and is not an official fastMRI leaderboard recipe.
Use the existing vSHARP or VarNet configs for additional reconstruction ablations,
retaining their architectures and loss settings across training-data comparisons.
Mixed real/synthetic datasets and real-data fine-tuning can use DIRECT's existing
multiple-dataset and checkpoint-initialization workflows.

The Python API also permits generation directly before masking, avoiding storage
of expanded coil data. The CLI provides offline HDF5 generation; it does not
register an online DICOM dataset in DIRECT. For diffusion, offline generation or
caching latent factors usually avoids an expensive reverse chain each iteration.

## Evaluation helpers

`synthesis_diagnostics(output)` checks RSS and FFT consistency. These constraints
are enforced by construction and are not quality or realism scores.

`kspace_features(kspace)` reports radial energy, coil covariance eigenvalues and
effective coil rank. Compare distributions within matching acquisition strata.
`reconstruction_metrics(prediction, reference)` reuses DIRECT's fastMRI NMSE,
PSNR and SSIM on identically cropped/scaled volumes. Aggregate by patient, not
by treating adjacent slices as independent observations.

See [research_design.md](research_design.md) for required real-data controls,
DICOM-specific evaluation, limitations and experiment sequencing.

## Tests

```bash
uv run --extra synthesis python -m pytest tests/tests_synthesis -q
uv run ruff check direct/synthesis tests/tests_synthesis
uv run ruff format --check direct/synthesis tests/tests_synthesis
uv run ty check direct/synthesis
```

The tests use small synthetic phantoms and temporary DICOM files. They cover
physics, gauge transformations, RNG behavior, gradients, checkpoint resume,
input rejection, HDF5 roundtrips, patient overlap, and an actual RecurrentVarNet
forward/backward pass with ACS-estimated maps. They do not replace experiments
on scanner acquisitions. Check [VALIDATION.md](VALIDATION.md) for the checks
performed on this implementation.
