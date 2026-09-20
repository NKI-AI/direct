# Implementation validation

Base repository: `NKI-AI/direct`, commit
`490a14631e6a252690c3356e0093dff3f884e12f`.

Local branch: `feature/magnitude-kspace-synthesis`.

## Checks completed

- **49 tests passed**: 31 synthesis tests and 18 selected existing DIRECT
  transform tests; 132 unrelated transform cases were deselected.
- Ruff lint passed for `direct/synthesis` and `tests/tests_synthesis`.
- Ruff formatting check passed for all seven Python files.
- `ty` type checking passed for `direct/synthesis`.
- The reconstruction YAML passed the existing model, dataset, training and
  validation configuration schemas.
- `uv lock --check --offline` passed. The optional synthesis dependency adds
  pydicom without changing existing locked package versions.
- `git diff --check` passed.

Test command:

```bash
python -m pytest tests/tests_synthesis tests/tests_data/transforms_test.py -q \
  -k 'synthesis or fft2 or complex_multiplication or root_sum_of_squares'
```

The test environment used Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.5.3,
h5py 3.16.0, pydicom 3.0.2 and DIRECT 2.2.0 built from the base checkout.
DIRECT's native extensions were built locally. Type checking used the active
interpreter and its user-package search path because this environment did not
use the repository's default `.venv` location.

## What the tests establish

- Centered FFT output agrees with an independently expressed PyTorch inverse
  FFT, including odd and rectangular grids.
- Synthetic RSS reproduces input magnitude and coil power is normalized.
- Joint phase/map gauge changes preserve k-space. Equivalent global receiver
  phase rotations produce consistent calibrated labels inside reliable support.
- Seeds, including zero, reproduce stochastic outputs; the legacy coil
  simulator's NumPy RNG state is restored.
- Both learned methods produce finite losses/gradients and update parameters.
- The CLI prepares raw-like phantom data, trains both methods, resumes each
  checkpoint and generates HDF5 output from temporary magnitude DICOM files.
- Diffusion samples are repeatable for a fixed seed and differ across seeds.
- DICOM modality rescaling and padding are handled without display windowing;
  explicitly incompatible image types are rejected.
- Existing HDF5 files cannot be overwritten; interrupted HDF5 writes are not
  published as completed outputs.
- The configured DIRECT transform pipeline accepts exported files.
- FastMRIDataset, DIRECT masks, ACS sensitivity estimation and RecurrentVarNet
  execute an actual reconstruction forward/backward pass with exported data.
- Distribution features are invariant to coil permutation and global scaling.
- Manifest patient overlap is rejected before preparation writes data.

## What was not established

No real MRI training cohort or patient DICOM files were supplied. Tests used
small generated phantoms and temporary DICOM objects. No clinically trained
synthesis checkpoints, reconstruction benchmark improvements, external-domain
generalization results or clinical validity claims are included.

CUDA execution, distributed generator training, the full DIRECT test suite,
full reconstruction-engine training, compressed DICOM decoder combinations,
3D/temporal coherence and scanner-specific preprocessing were not tested.
The reconstruction integration check exercises existing modules and transforms;
it is not a completed downstream learning study.
