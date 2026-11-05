# Static Stokes I KINE in PyTorch

Based on https://github.com/mariannafoschi/kine3c345, particularly code/kine/model.py, helpers.py, training.py and params/params_closure_m87.yml. GPL-3.0; see KINE_LICENSE.txt.

## Run in WSL

```bash
~/.local/bin/uv sync
~/.local/bin/uv run python run.py
# Explicit GPU:
~/.local/bin/uv run python run.py --device cuda
# Fast execution check (not scientific reconstruction):
~/.local/bin/uv run python run.py --npix 16 --width 32 --init-steps 5 --steps 5 --out results/aligned_smoke
~/.local/bin/uv run python -m utils.check_alignment
```

CUDA 13.0 PyTorch wheels are configured in pyproject.toml. Default device auto selects CUDA when available. Dense Fourier matrices are cached on the chosen device; reduce npix for memory constraints. Existing results/kine_static files belong to the previous simplified implementation and are preserved. New default output: results/kine_aligned.

## Alignment

- Uses upstream ehtim default UVFITS -> Stokes I loading, 120s coherent averaging, then add_fractional_noise(0.01). Unlike the previous strict paired-hand variant, this uses ehtim fallback behavior where only one parallel hand exists (assumes negligible Stokes V there).
- Uses exactly ehtim.chisqdata: pol=I, natural weighting, direct transform, debias=False, maxset=False, with the same default image pixel pulse. Targets, errors, pixel centers, handedness and Fourier sign now come from that API, not a separate implementation.
- Input grid uses linspace(0,1) and ij indexing. 4+1 hidden layers, width256, zero positional encoding by default, He uniform initialization, BatchNorm, approximate GELU, first-layer skip, shifted Softplus (shift7). No hard flux normalization or centroid penalty.
- Initializes to a 60 uas diameter uniform disk blurred by 20 uas Gaussian. Flux from median visibility amplitude on shortest baseline, matching upstream static flux estimate. This is an initialization, not an ongoing flux constraint.
- Initialization uses unscaled image MSE for 10000 optimizer steps. Reconstruction uses all closure phases and log closure amplitudes for 10000 steps. No random closure minibatches.
- Adamax, lr0.01, decay factor0.01 at global optimizer steps10000 and20000, including initialization. Optimizer state is retained across phases, as upstream.
- Output image.npy / FITS is Jy per pixel. PNG uses ehtim sky orientation; it differs from the previous l/m display. Checkpoint contains config, weights and optimizer state. Metrics distinguish initialization and reconstruction and report full-data closure losses.

## Intentional scope / numerical differences

One static frame fits the entire single input file; no multi-epoch/time network, temporal splitting, polarized channels, post-fit self-calibration or absolute-visibility data loss. These are the requested scope changes. The original M87 preset is dynamic; this is not a reproduction of its video experiment.
PyTorch and JAX PRNG/initial draws and floating point reductions differ. Flax uses biased running variance while PyTorch BatchNorm updates running variance differently; fitting and image export use full-grid batch statistics, avoiding a running-stat inference mismatch. Loss formulas are checked against ehtim; a good closure fit alone does not ensure a unique ring.

Model/model.py contains the neural field. utils/closure.py prepares products and differentiable losses. run.py handles training and export.

## Initialization comparison

Keep data, seed, steps and all other settings identical:

```bash
uv run python run.py --device cuda --init-kind disk --init-size 60 --init-blur 20 --seed 1 --out results/init_disk60
uv run python run.py --device cuda --init-kind gauss --init-size 60 --init-blur 0 --seed 1 --out results/init_gauss60
```

init-size means disk diameter or Gaussian FWHM, not radius. init-blur is additional Gaussian FWHM. Each output stores init_target (prescribed target), pretrained (network after warmup), and image (closure reconstruction) as PNG/NPY/FITS. metrics.json includes the initialization settings. Reconstruction has no ground-truth training image and does not retain an initialization MSE penalty. Defaults preserve the upstream-aligned disk setup. Keep init-steps at 10000 for this comparison since learning-rate boundaries count global optimizer steps.
