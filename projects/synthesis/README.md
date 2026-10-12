# Magnitude images to synthetic multicoil k-space

Turns a magnitude image into plausible multicoil k-space, so reconstruction
networks can be trained on data that only exists as magnitude (DICOM, NIfTI).
The output is a draw from a conditional distribution, not a recovery of the
discarded scanner measurements.

```
k_c = FFT( S_c · m · e^{iφ} ),   Σ_c |S_c|² = 1  ⇒  RSS(IFFT(k)) = m exactly
```

| Network | Predicts | Conditioning | Config |
|---|---|---|---|
| `SensitivityMapSynthesizer` | complex maps `S` for any coil count `N` | magnitude, `N`, coil index, organ | `sensitivity_maps_multi_cardiac.yaml` |
| `PhaseFlowMatching` | unit phase `φ` (OT conditional flow matching, Euler ODE) | magnitude, time, organ | `flow_matching_multi_cardiac.yaml` |

Training labels are computed from fully sampled k-space in the dataset
transforms (`transforms_type: SYNTHESIS`). Maps are `FULL_RSS`
(`S_c = IFFT(k_c) / RSS`, gauge-fixed to the strongest coil). The phase label is
the SENSE combination anchored at a bright pixel. Organ codes are
`brain, knee, prostate, breast, cardiac, unknown`; 15 % of training samples are
relabelled `unknown` so the models also apply to unseen anatomy.

`reconstruction.yaml` trains RecurrentVarNet on exported synthetic HDF5 files.

## Layout

| Path | Content |
|---|---|
| `direct/nn/synthesis/` | models, engines, losses, model configs |
| `direct/synthesis/` | physics (`synthesize`, `normalize_sensitivities`), HDF5 export, evaluation |
| `direct/synthesize.py` | `direct synthesize`: apply phase + map checkpoints to a dataset |
| `lists/` | paper split lists and `paper_split_manifest.json` (see `paper_splits.md`) |

## Train

The configs set `data_root` and `filenames_lists_root` per dataset, so run them
without `--training-root`. Adjust both to your environment.

```bash
direct train <experiment_dir> --cfg projects/synthesis/sensitivity_maps_multi_cardiac.yaml \
  --num-gpus 1 --num-workers 4 --device cuda --name sensitivity_maps --mixed-precision

direct train <experiment_dir> --cfg projects/synthesis/flow_matching_multi_cardiac.yaml \
  --num-gpus 2 --num-workers 8 --device cuda --name flow_matching --mixed-precision
```

## Apply

`direct synthesize` takes a config with the phase model under `model`, the map
model under `additional_models.sensitivity_model`, a `synthesize` block and an
`inference.dataset`:

```bash
direct synthesize <output_dir> --cfg <synthesize.yaml> \
  --phase-model <flow checkpoint> --sensitivity-model <map checkpoint> \
  --data-root <fastmri root> --device cuda
```

## Tests

```bash
python -m pytest tests/tests_synthesis -q
```

See `research_design.md` for the design rationale and evaluation protocol.
