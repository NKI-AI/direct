# Data notes: corrupt / excluded files

A full per-frame scan of every R_train dataset (88,639 slices/frames across
brain, knee, prostate, breast, MosaicMRI, cardiac) was run on 2026-10-21,
checking each frame for all-zero (empty) or non-finite k-space / target.

## Corrupt files

| File | Dataset / role | Problem | Action |
|---|---|---|---|
| `y2024_P100_cine_sax.mat` | CMRxRecon cardiac, **R_train only** | 17 of 180 cine frames have all-zero k-space (empty target), scattered across slices/phases | Removed from `lists/cardiac/r_train.lst` and `lists/cardiac/r_train_10_coils.lst` (commit d83f30d) |

The same patient's `y2024_P100_cine_lax.mat` and `y2024_P100_cine_lvot.mat`
are clean and kept.

All other datasets scanned clean (0 corrupt frames): brain (28,284), knee
(13,926), prostate (2,647), breast (19,881), MosaicMRI (23,901), rest of
cardiac (18,955).

## Why it mattered

An all-zero target makes the SSIM loss data-range 0 (NaN), and on the
near-constant / zero-padded region the SSIM gradient blows up to Inf. In
reconstruction training this poisoned the weights on a single step and the run
went NaN (vSHARP first died at iter ~140 then ~5880; VarNet at iter 124). This
file supplied those empty frames.

## Robustness (independent of the data fix)

Three engine guards now tolerate such frames even if they reappear in new data:
- SSIM `data_range` floored at 1e-6 (no NaN on empty target) — commit daa0b23
- vSHARP engine raises `NonFiniteLossError` before backward — commit 32b72fa
- optimizer step skipped on non-finite gradients — commit 04ab0cc

Rescan: `/home/g.yiasemis/corr_scan/corrupt.py` (FastMRI/Mosaic) and
`corrupt_cardiac.py` (CMRxRecon).
