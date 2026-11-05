# KINE static PyTorch audit

## Scope and evidence
Four runs: disk diameters 40 and 100 uas, seeds 1 and 4396; all other recorded training/data settings identical except output path. Same 1.2244794 Jy initialization flux, 64x64 grid, 120s average, 1% fractional uncertainty, 10000 warmup + 10000 reconstruction steps, Adamax full-batch, initial lr0.01 -> 0.0001 at global step10000.

| Diameter | Seed | phase loss | logamp loss | final Jy |
|---|---|---|---|---|
|40|1|1.114717|0.785441|0.041635|
|40|4396|1.093181|0.773449|0.051227|
|100|1|1.103105|0.801032|0.343383|
|100|4396|1.093919|0.781635|0.468914|

After unit-flux normalization and integer translation registration optimized at common 20 uas Gaussian resolution, relative L2 differences: 40uas seed comparison26.1% native/17.1% smoothed; 100uas seed comparison43.9%/25.7%; seed1 diameter40 vs100 comparison67.2%/44.2%. Not parameter-estimation confidence intervals. Registration is integer-grid, not an exhaustive continuous morphological fit. Smoothed images use a Gaussian observation kernel for comparison only.

## Numerical verification
Identical weights transferred from PyTorch into the actual original model.py through a mock config module: reduced width32, depth4,64 random coordinates. Forward max abs6.52e-8; all-parameter gradient relative L2 3.30e-6. Identical supplied-gradient Adamax sequences (20 steps, includes gradients near epsilon): max parameter discrepancy4.17e-7. Current available JAX0.10.2,Optax0.2.8,PyTorch2.14.1+cu130; historical versions are not pinned by this audit. These tests establish local parity, not identical 20000-step trajectories.

Closure pixel gradients vs independent ehtim analytical gradients: relative L2 3.37e-13 (phase),1.51e-15(logamp), float64. CPU/CUDA float32 loss check passes. See utils/check_alignment.py and utils/audit_torch_parity.py (the latter needs the temporary /tmp/kine_audit_deps installation).

Full disk100 seed1 repeat completed on CUDA; see same_seed_repeat_audit.json. Original outputs untouched. This repeat directly tests repeatability under the present environment, not every CUDA platform.

## Confirmed limitations / differences
- No total-flux regularizer, despite the paper Methods describing one for closure amplitude imaging. Released M87 YAML has reglcurve=False; this follows YAML but not that textual method description. Scale drift is expected; it can also affect optimization conditioning, not enough evidence it causes morphology differences.
- Fixed stopping, no implementation of paper convergence criterion (100-step moving average, <1% improvement over1000 iterations, each term ~1). Logs only every100 updates cannot rigorously evaluate that criterion.
- Only10000 data-fit steps; counting10000 image-warmup steps as equivalent to20000 paper data-fit updates is unjustified. Longer fitting must actually be tested.
- Last iterate exported; no best-checkpoint or per-step convergence trace. No intermediate parameter checkpoints at phase transition; current final checkpoints cannot isolate optimizer reset without redoing warmup.
- Initialization dimensions40/100 differ from released60; valid robustness tests but cannot assert every such scale meets the paper approximate support assumption.
- Static frame, unlike released dynamic multi-day run, lacks cross-frame information sharing. Cannot attribute static implementation result directly to published dynamic method.
- Different random weights produce different pretrained Jacobians and optimizer states. That is a mechanism, not proof of the cause of observed instability.
- No random minibatches, dropout or stochastic added visibility noise. add_fractional_noise inflates uncertainties only.
- BatchNorm plotting in original helpers.py uses train=True; current full-grid training-mode export aligns with that. Previous suggestion of an original eval-vs-train output mismatch does not apply here.
- No wrong residual connection/depth conversion found: both implementations use depth+1 hidden layers and add first hidden output before final Dense. Identical softplus shift7, GELU and He-uniform families.
- GPU nondeterminism must not be invoked to explain different-seed outputs without evidence.

## Valid conclusion
This static PyTorch experiment shows material initialization/seed sensitivity at the tested budget and constraints, despite similar closure data losses. It does not establish an inherent failure of original KINE or numerical PyTorch port corruption. Recommended wording: single-frame PyTorch adaptation of KINE, not a verbatim execution of original paper code.

## Next controlled investigations
Implement full per-update convergence trace and checkpoints at warmup end; retain vs reset Adamax on an identical warmup checkpoint; longer data fitting with explicit schedule; paper-style flux constraint as a separately labelled variant; native original static-reference run. Keep baseline code/results unchanged while investigating.
