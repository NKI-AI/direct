# Research design: multicoil synthesis from reconstructed magnitude MRI

## The target is a conditional distribution

A reconstructed magnitude image cannot uniquely specify the acquired phase,
coil configuration, noise, sampling trajectory, receiver scaling or acquisition
matrix. Postprocessing can remove information as well. A suitable objective is
to approximate a useful distribution of acquisitions given the available image
and a declared acquisition domain:

\[
 (\phi,S) \sim p(\phi,S\mid m,\text{domain}),\qquad
 \tilde{k}_c=\mathcal F\{S_c m e^{i\phi}\}.
\]

With `sum_c |S_c|² = 1`, the coil-image RSS equals `m`. This makes `m` an RSS
surrogate, not necessarily a quantitative body-coil magnitude. A vendor DICOM
may use adaptive coil combination and nonlinear processing rather than RSS.
The current code deliberately preserves those input intensities. It does not
learn to recover the preprocessed, pre-combination object magnitude.

There is also a gauge freedom: `S_c -> S_c exp(-i psi)` and
`phase -> phase exp(i psi)` produce exactly the same coil data. Factor labels
must be assigned a convention; their phase values are not unique physical truth.

## Three methods and the decision behind them

### A. Smooth phase with simulated or empirical coils

Sample a low-resolution Gaussian phase field, interpolate it smoothly, and add
a global phase. Use DIRECT's existing sensitivity simulator with coil-specific
smooth phase ramps, or draw calibrated maps from real training acquisitions.
Normalize coil power and apply the existing expansion operator and FFT.

This has no learned components and runs immediately. It is an essential
baseline because learned phase need not improve reconstruction more than broad
augmentation. Its assumptions are weak: random smooth phase misses susceptibility
interfaces, flow, wraps with sharp spatial structure, sequence-specific phase
and realistic noise. Simulated Gaussian coil layouts are especially approximate.
Empirical maps improve the anatomical/scanner match only when FOV, position,
orientation and receiver-array geometry are compatible.

### B. Supervised factorized prediction

A DIRECT U-Net predicts two Cartesian phase channels; unit normalization avoids
the discontinuity of directly regressing wrapped radians. A second DIRECT U-Net
predicts `2 * num_coils` map channels on a small grid, using magnitude plus
spatial coordinates. Bilinear interpolation and coil-power normalization produce
full-size sensitivities. MRI physics produces k-space in one feed-forward pass.

Supervision combines support-weighted phase error, map error and complex
coil-factor error. These are calibrated pseudo-label losses, not direct evidence
of recovered physical fields. The loss is

\[
 L=L_{\mathrm{coil}}+0.25(L_{\mathrm{phase}}+L_{\mathrm{maps}}).
\]

The coil term compares `S * phase`, weighted by squared magnitude support. Under
fixed magnitude and orthonormal FFT, the corresponding magnitude-weighted coil
image error is equivalent to k-space squared error by Parseval's identity;
duplicating both adds no independent supervision.

This is the fast learned baseline. It is limited to a fixed, consistently ordered
coil configuration. Receiver phase and geometry are incompletely determined by
magnitude, so deterministic regression can average incompatible factors. Gauge
alignment removes a nuisance ambiguity but cannot solve that missing information.
The learned map branch is not guaranteed to remain a smooth Maxwell-consistent
field after normalization. Inspect its spectra and parallel-imaging conditioning.

### C. Conditional phase diffusion with empirical maps

Diffuse the two Cartesian unit-phase coordinates in ordinary Euclidean space.
The noise predictor is DIRECT's U-Net with magnitude concatenation and its
existing AdaIN layers conditioned on a sinusoidal time embedding. Training uses
epsilon prediction and a cosine schedule. Sampling uses the full ancestral DDPM
chain; the final sample is normalized to a unit complex phase. An empirical map
realization supplies coil factors before the explicit Fourier transform.

This supports multiple phase samples for one magnitude and variable coil counts
without requiring consistent coil-channel order inside the generator. It is the
preferred learned experiment, provided the baseline is established first.
It is a new implementation of a conventional conditional DDPM, not an exact
reproduction of the papers below. It does not implement VE-SDE sampling, DDIM,
classifier-free guidance or a joint diffusion distribution over coil maps.

The implementation approximates the joint distribution by sampling phase from
`p(phi | magnitude, domain)` and choosing maps from a domain-matched empirical
bank. Phase and maps can still be dependent within a domain, and calibration
can leave anatomy in the map estimates. Domain matching is an explicit
approximation, not a learned conditional coil model. A future extension should
condition on coil descriptors or model joint gauge-consistent factors if the
independence approximation limits downstream performance.

## Calibration and training labels

Start from the available fully sampled multicoil acquisitions. Inverse FFT gives
coil images, and RSS gives the magnitude input. Reuse DIRECT's ACS sensitivity
estimator. Choose the coil with largest integrated map power, and remove its
pointwise phase from all maps. SENSE combination then gives an object phase in
this reference-coil gauge. Finally align object phase at a reliable bright pixel
to remove global receiver phase; this changes all synthesized coil data only by
a global unit complex factor.

The code uses DIRECT's RSS ACS estimator. ESPIRiT is already available in DIRECT
and is a useful calibration ablation, but is not exposed by this new preparation
CLI. Compare calibration methods before interpreting map prediction accuracy.
Multiple ESPIRiT map sets and regions where one map is insufficient need an
extended forward model, not a larger phase network.

Reproject calibrated factors onto the acquired coil images and record normalized
error. This measures the mismatch introduced by the single-map model. Reject
poor labels, inspect where they fail, and report rejection rates per domain.
Weight phase losses by image support and reference-coil confidence to avoid
teaching models arbitrary background phase. Current coil targets describe the
calibrated approximation; noise and residual model mismatch are not synthesized.

The initial generator training uses the user's proposed `raw -> RSS -> factors`
construction. A matched `raw + clinical DICOM` subset is the most useful next
dataset: it allows the conditioning input to reflect actual vendor processing
while retaining phase/map targets from raw data. Spatial registration, crop/FOV
and intensity handling must be explicit. Merely pairing two images from the
same patient is insufficient. Training augmentation can later mimic measured
gamma, blur, bias fields and quantization, but none of these are assumed to
invert unknown scanner processing. Such augmentation is not implemented here.

## Iterative synthesis and reconstruction networks

The baseline and factorized model generate in one pass; diffusion is iterative.
No reconstruction network is needed to produce fully sampled synthetic data.
An iterative data-consistency loop cannot recover missing measurements from a
DICOM because there are no acquired complex samples to enforce.

Use RecurrentVarNet as a tractable downstream control, then vSHARP and an
end-to-end VarNet as architecture checks. Keep capacity, masks, real-data access,
optimization budget and augmentation identical between synthesis methods.
The supplied configuration uses RecurrentVarNet; all exports use the existing
FastMRIDataset interface, so the established reconstruction configs remain usable.

Do not initially train a generator and reconstruction model only with a cycle
loss. They can learn an artificially easy encoding that reconstructs their own
synthetic data while transferring poorly to real acquisitions. The exact RSS
constraint already prevents magnitude alteration in this implementation, but
does not prevent unrealistic phase or coil encodings. If adding a reconstruction
utility loss later, freeze the reconstruction evaluator, retain real calibration
constraints and use a separate, untouched real acquisition test set.

## Evaluation that can answer the research question

### 1. Code and physical consistency

Check FFT conventions, complex axis placement, map-power normalization, gauge
invariance, preserved magnitude, finite gradients, coil-count behavior and seed
reproducibility. These are implementation gates. Perfect RSS agreement is
guaranteed by construction and cannot rank the methods scientifically.

### 2. Held-out raw data with pseudo-DICOM inputs

Hide the complex information and condition on RSS or a carefully defined
postprocessing surrogate. Assess circular phase differences in reliable support
after gauge alignment, map smoothness, radial k-space spectra, coil covariance
eigenvalues, spatial phase-gradient statistics, and sensitivity conditioning.
For stochastic models, compare distributions and repeated samples; do not
expect a sampled phase to reproduce the discarded phase pixel for pixel.

The implementation includes radial power and covariance features. It does not
include g-factor calculations, phase-distribution hypothesis tests or clinical
feature assessment. Those require a study-specific evaluation pipeline.

### 3. Actual clinical DICOM inputs

Without paired raw data, raw-domain accuracy is unobservable. Audit image-type
selection, processing, spacing/orientation, crop and FOV. Inspect synthetic
artifacts and phase/map distributions against matched real training acquisitions.
Review pathology-containing slices. Reconstructing undersampled synthetic data
back to the same DICOM is a useful implementation check but is a circular
evaluation of generalization. Use a paired raw/DICOM subset or independent
held-out real raw acquisitions for stronger evidence.

### 4. Primary endpoint: real reconstruction performance

Train otherwise matched reconstruction models using:

| Arm | Reconstruction training data |
|---|---|
| Real-only control | Available real training acquisitions |
| Zero-phase control | Magnitude with identical coil policy and zero phase |
| Smooth-phase baseline | Method A, same source magnitude cases |
| Supervised factors | Method B, same source magnitude cases |
| Diffusion phase | Method C, same source magnitude cases |
| Mixed and fine-tuned variants | Each synthetic method plus the same real-data budget |

For source-controlled comparisons, use the same magnitude images and coil policy
where possible. In particular compare smooth and diffusion phase with the **same
empirical maps**, otherwise phase benefits are confounded with coil realism.
Compare factorized maps separately against empirical maps. Match update count
as well as patient count, and report the compute spent generating synthetic data.

Evaluate on untouched real k-space with prespecified acceleration/ACS settings
and held-out masks, scanners and sequences. Report volume-level NMSE, PSNR and
SSIM, pathology/region-specific errors, and failure cases. Use patient-level
bootstrap intervals and multiple training seeds. Synthetic variants from one
DICOM are augmentations, not independent subjects.

Use estimated ACS maps during reconstruction for every arm. Providing exact
synthesis maps only to synthetic training/evaluation gives privileged information
that is not normally available in deployment. Keep a separate oracle-map
ablation if needed, labelled accordingly.

An empirical success criterion is improved real-data reconstruction over the
real-only and smooth-phase controls at a fixed real-data budget, without worse
pathology preservation or out-of-domain failure rates. No numerical improvement
is assumed in advance. None has been measured by this implementation session.

## Organ, sequence and scanner specificity

Start with one well-characterized domain, for example axial prostate T2-weighted
3T acquisitions if an adequate paired cohort is available. This is a suggested
first experiment, not an assumption that such data is present in this workspace.
Do not initially pool EPI diffusion, gradient echo, spin echo, cine and different
coil geometries without a conditioning or stratification plan.

Anatomy matters through FOV, coil loading, geometry and susceptibility. Sequence,
TE, field strength, phase-encoding direction, motion and receiver hardware can
matter at least as much. The CLI's `domain` string is a curation/checkpoint
compatibility label; it is **not** a learned metadata embedding. Train separate
checkpoints initially. Pooling domains later requires actual metadata inputs and
leave-one-domain-out validation; this is not implemented by changing the label.

## Priority beyond this implementation

1. Measure transfer with the baseline before spending substantial compute on
   learned synthesis. Compare real-data fractions and synthetic pretraining.
2. Assemble a matched raw/DICOM subset and quantify the vendor-processing gap.
3. Evaluate empirical-map gauge stability, leakage of anatomy into maps, coil
   covariance and map conditioning. Freeze these choices before comparing phase.
4. Train diffusion and compare against the factorized baseline. Do not infer
   multicoil realism from single-coil phase results alone.
5. Add calibrated complex noise, residual coil-model variability and explicit
   acquisition metadata only where real-data mismatch warrants them. For noise,
   separate clean targets from noisy inputs and use measured receiver covariance.
6. Extend to volume/time coherence and learned joint phase/map priors only after
   the independent 2D experiment shows useful real-data transfer.

## Relevant evidence and scope of the literature search

- [Deveshwar et al., *Synthesizing Complex-Valued Multicoil MRI Data from
  Magnitude-Only Images*, Bioengineering 2023](https://doi.org/10.3390/bioengineering10030358)
  is directly relevant prior work on learned phase and synthetic multicoil data.
  Its bibliographic record was located, but full-text access was blocked during
  this implementation; no specific quantitative result is relied upon here.
- [Luo et al., *Generative Priors for MRI Reconstruction Trained from
  Magnitude-Only Images Using Phase Augmentation*](https://arxiv.org/html/2308.02340v2)
  describes using phase augmentation to train complex-image priors and evaluating
  them in linear and nonlinear reconstruction. It supports considering complex
  prior learning as an alternative to training a monolithic k-space translator.
- [Rempe et al., *PhaseGen*, 2025 preprint](https://arxiv.org/html/2504.07560v1)
  describes magnitude-conditioned complex phase generation. Its reconstruction
  experiments use the fastMRI single-coil knee dataset. It therefore does not
  by itself validate the multicoil sensitivity generation required here.
- [Sahin et al., *Phase-map synthesis from magnitude-only MR images*, 2026](https://arxiv.org/html/2605.01185v1)
  presents conditional score-based phase generation and downstream reconstruction
  comparisons. The manuscript identifies itself as accepted to a CVPR workshop.
  It uses single-channel knee data and compresses brain multicoil data to one
  channel, which limits direct evidence for the present multicoil factor model.

Search coverage: a scoped search on 19 September 2026 across primary preprints,
publisher/index records and repository/proceedings searches, using magnitude-only
MRI, synthetic complex k-space, phase generation, coil sensitivity and
reconstruction-transfer terms, including limitations and inverse-crime queries.
Primary full text for the three accessible preprints above was inspected. Some
publisher/index pages were inaccessible or returned incomplete records. This is
an implementation-focused evidence review, not an exhaustive literature review
or a novelty claim for magnitude-to-k-space synthesis.
