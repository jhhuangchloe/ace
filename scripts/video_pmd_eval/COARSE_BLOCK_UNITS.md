# Coarse-block units offset in two-block inference

## The problem

Two-block video PMD models (`two_block: true`) assemble their output as

```
x_hat = U(I + r_hat) + Pi(d_hat)
```

and then convert it to physical units with the **fine** statistics (`normalization`). But
`I` (the coarse baseline) and `r_hat` are built from the 100 km input normalized with the
**coarse** statistics (`coarse_normalization`). When the two sets of statistics differ, every
coarse cell of the output is shifted by

```
offset(c) = (c - mean_f) * (std_c / std_f - 1) + (mean_c - mean_f)
```

per channel, where `c` is the cell's value in physical units (log1p space for precipitation).
`d_hat` cannot correct this, because it averages to zero over each coarse cell, and the
network never learns to, because `r` is defined in the same coarse units in training.

The offset depends on the value. With the configs used since the v3 baseline:

| Channel | Offset |
|---|---|
| T2m | +0.34 K at 280 K, **+0.58 K at 301 K** (tropics) |
| PRMSL | about -0.07 hPa near 1014 hPa |
| u10, v10 | a few hundredths of a m/s |
| PRATEsfc | about -0.07 in log1p(mm/day) |

On the S2022 typhoon box it accounts for about 60% of the T2m MSE of every two-block model
(v3-ou, vortex, two-block). The 100 km input itself matches the 25 km truth to within
±0.07 K there.

**Affected:** every two-block run with a `coarse_normalization` that differs from
`normalization`. **Not affected:** RSTT/STVSR, which uses the fine statistics for both.

## Fix for new inference runs

Add to the `model:` block of the inference config:

```yaml
model:
  coarse_block_in_fine_units: true
```

Generation then converts `I + r_hat` to fine units before assembling the output. It works with
existing checkpoints and needs no retraining: the network and its inputs are unchanged. It is
off by default, so old configs reproduce their old outputs. See
`VideoDiffusionModelConfig.coarse_block_in_fine_units` in `fme/downscaling/video_models.py`.

## Fix for output that already exists

`correct_coarse_block_units.py` removes the offset from an inference zarr, so a finished run
does not need to be regenerated. The detail term averages to zero over each coarse cell, so
`c` is recovered exactly as the 4x4 block mean of the written output.

- **Exact** for T2m, PRMSL, u10 and v10 (float32 rounding only).
- **Approximate** for PRATEsfc: inference clamps negative precipitation to 0, which loses the
  log1p value of near-dry pixels. Most of the offset is still removed, but up to about 0.1 in
  log1p(mm/day) can remain in some cells. Use the inference flag if exact precipitation
  matters.

The script refuses to run if the store is already corrected, if the model ran with
`coarse_block_in_fine_units: true`, or if the model isn't two-block. After writing, it checks
that at the first clip endpoint the block means match the 100 km input, and stops if they
don't.

```bash
# corrected copy next to the original (original untouched)
python scripts/video_pmd_eval/correct_coarse_block_units.py OUTPUT.zarr INFERENCE.yaml

# corrected copy written to a chosen path
python scripts/video_pmd_eval/correct_coarse_block_units.py OUTPUT.zarr INFERENCE.yaml --out FIXED.zarr

# overwrite: write + verify a corrected copy, then rename the original to
# OUTPUT-uncorrected.zarr and the copy to OUTPUT.zarr
python scripts/video_pmd_eval/correct_coarse_block_units.py OUTPUT.zarr INFERENCE.yaml --replace
```

Pass the inference config that produced the store: it supplies the normalization statistics
and the 100 km store used for the check. The corrected store has the attribute
`coarse_block_units_corrected`. It needs as much free disk as the original until you delete the
backup.

## To do: v3-ou full test-set run

Run: `configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/test_inference_ou_detail.yaml`
(tmux `v3-ou-test`, GPUs 4 and 5), written without the fix. Once it has finished:

1. Check it is complete: every rank logged `batch 23/23 written` and the log ends with
   `Completed inference` (`/data/pmd/runs/eval-v3-ou-detail/inference.log`).
2. Correct it in place (about 200 GB; keep that much free until step 4):
   ```bash
   cd ~/Projects/ace
   conda run --no-capture-output -n fme python scripts/video_pmd_eval/correct_coarse_block_units.py \
       /data/pmd/runs/eval-v3-ou-detail/test-2023-2024-global-ens4.zarr \
       configs/experiments/2026-10-15-video-pmd-two-block-v3-baseline/test_inference_ou_detail.yaml \
       --replace
   ```
   All four exact channels should print a verification error below 1e-3.
3. Score the corrected store. Report PRATEsfc as approximately corrected.
4. Delete `/data/pmd/runs/eval-v3-ou-detail/test-2023-2024-global-ens4-uncorrected.zarr`.

For future runs, add `coarse_block_in_fine_units: true` to the inference config instead.
Older diffusion test outputs (vortex, two-block, the 2026-09 runs) have the same offset and can
be corrected the same way before they are compared with the fixed v3-ou or RSTT outputs.
