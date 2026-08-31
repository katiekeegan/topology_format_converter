# Generative Topology Optimization

Research code for learning generative priors over 3D topology-optimization
designs from the SELTO dataset. SELTO is the Sample-Efficient Learned Topology
Optimization dataset, published on Zenodo and exposed by DL4TO through
`dl4to.datasets.SELTODataset`. The main representation in this repository is a
signed distance field (SDF): SELTO density grids are converted to sampled
surface point clouds and SDF query points, a VAE/SDF decoder learns the shape
space, and a VE score model is trained over the VAE latent space.

## Repository Layout

- `models.py`: PointNet encoder, VAE, SDF decoder, and modulation wrapper.
- `utils/preprocess_data.py`: DL4TO/SELTO density grid to surface points, query
  points, normalized SDF values, and optional problem-conditioning grids.
- `trainer.py`: trains the modulation module (`VAE + SDFNetwork`).
- `trainer_diffusion.py`: trains a variance-exploding (VE) score model on VAE
  latents.
- `sample_sdf_obj.py`: samples from the VAE Gaussian prior and exports an OBJ.
- `sample_diffusion_model.py`: samples through the trained VE score model and
  exports an OBJ.
- `trainer_posterior.py`: experimental RealNVP posterior training for one fixed
  SELTO problem using a score prior and a dl4to PDE energy.
- `gto_format_converter/`: reusable Python package and CLI for SELTO, voxel,
  SDF cache, and mesh conversions.
- `scripts/`: diagnostics and sanity checks.
- `dl4to/`: vendored DL4TO package used for SELTO data access and PDE solves.

Generated datasets, checkpoints, meshes, and W&B outputs are not part of the
source workflow and are ignored by `.gitignore`.

`utils/csvMaker.py` and `utils/objMaker.py` are legacy conversion helpers. They
refer to local mesh/export folders and are not the canonical way to obtain
SELTO. For this project, use DL4TO's `SELTODataset` loader.

## Environment

On NERSC-style systems, use the module stack from the Slurm scripts:

```bash
module load python
module load pytorch/2.6.0
export PYTHONPATH=dl4to:.
```

The code expects at least:

```text
torch
numpy
scipy
pandas
tqdm
trimesh
scikit-image
requests
```

Optional mesh-repair/conversion helpers use `pymeshfix` and
`point_cloud_utils`.

## Format Converter Package

This repository includes an installable conversion package:

```bash
pip install -e .
```

On HPC systems where dependencies come from environment modules, use:

```bash
pip install -e . --no-deps
```

After installation, use the CLI as `gto-convert`. Without installing, call it as
a Python module:

```bash
PYTHONPATH=dl4to:. python -m gto_format_converter.cli --help
```

Supported v1 conversions:

```text
SELTO/DL4TO sample -> OBJ/STL/PLY mesh
SELTO/DL4TO sample -> SDF training cache (.npz)
.npy/.npz/.pt volume -> OBJ/STL/PLY mesh
.npy/.npz/.pt volume -> SDF training cache (.npz)
mesh -> mesh format supported by trimesh
```

Every mesh/cache export writes a metadata sidecar by default, for example
`sample.obj.json` or `sample_cache.npz.json`. The sidecar records source,
dataset/split/index when available, density threshold, voxel shape, spacing,
coordinate mode, and SDF sign convention.

Python API example:

```python
from gto_format_converter import (
    density_to_training_cache,
    export_mesh,
    load_selto_sample,
    save_training_cache,
)
from gto_format_converter.selto import selto_sample_to_mesh

sample = load_selto_sample(root=".", dataset="sphere_complex", split="train", index=0)
mesh = selto_sample_to_mesh(sample, threshold=0.5, coordinate_mode="training")
export_mesh(mesh, "sphere_complex_0.obj", metadata=sample.metadata)

cache = density_to_training_cache(sample.density, threshold=0.5)
save_training_cache(cache, "sphere_complex_0_cache.npz")
```

CLI examples:

```bash
gto-convert selto-to-mesh \
  --root . \
  --dataset sphere_complex \
  --split train \
  --index 0 \
  --coordinate-mode training \
  --out sphere_complex_0.obj

gto-convert selto-to-cache \
  --root . \
  --dataset disc_simple \
  --index 0 \
  --num-surface-points 10000 \
  --num-query-points 10000 \
  --out disc_simple_0_cache.npz

gto-convert volume-to-mesh density.npy \
  --threshold 0.5 \
  --coordinate-mode unit-box \
  --out density.stl
```

Coordinate modes:

- `voxel`: preserve voxel-index units, optionally with `--spacing DX DY DZ`.
- `unit-box`: map each axis independently into `[-1, 1]`.
- `training`: use the normalization convention expected by the SDF training
  code.

## Data

The training scripts load the SELTO dataset through DL4TO's
`dl4to.datasets.SELTODataset` class:

```python
SELTODataset(root=".", name="sphere_complex", train=True)
```

If the requested dataset directory is empty, DL4TO downloads the corresponding
SELTO archive from Zenodo and converts CSV files into `.pt` samples under:

```text
<dataset-name>/train/
<dataset-name>/test/
```

Common dataset names are:

```text
disc_simple
disc_complex
sphere_simple
sphere_complex
```

The Zenodo record is `SELTO Dataset`, DOI `10.5281/zenodo.7781392`. It contains
the four 3D topology-optimization subsets above, each split into training and
validation/test archives. The paper reference is:

```text
Dittmer, S., Erzmann, D., Harms, H., Maass, P.
SELTO: Sample-Efficient Learned Topology Optimization.
arXiv:2209.05098.
```

## Training

Train the VAE/SDF modulation module:

```bash
python trainer.py \
  --dataset-name sphere_complex \
  --num-epochs 1000 \
  --learning-rate 1e-4 \
  --beta-kl 1e-5 \
  --prior-std 0.25
```

Outputs:

```text
checkpoints_mod/mod_last.pth
checkpoints_vae/vae_last.pth
modulation_module.pth
```

To create separate modulation/VAE checkpoints for all SELTO subsets and a
combined run, first request an interactive GPU allocation:

```bash
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpuhbm80g --gpus 1 --account=m5357
```

Then run:

```bash
./scripts/train_selto_checkpoints.sh
```

If `salloc` times out before Slurm grants a GPU, submit a persistent regular
GPU batch job:

```bash
mkdir -p logs
sbatch scripts/slurm_train_selto_checkpoints.sh
```

By default this runs one epoch and caps each SELTO subset to 256 samples so the
command produces initial checkpoint files quickly. Override settings with
environment variables:

```bash
EPOCHS=10 BATCH_SIZE=8 ./scripts/train_selto_checkpoints.sh
```

Use the full training split for each subset with:

```bash
MAX_SAMPLES_PER_DATASET=0 ./scripts/train_selto_checkpoints.sh
```

The script trains:

```text
disc_simple
disc_complex
sphere_simple
sphere_complex
combined_all
```

and writes:

```text
checkpoints_mod/<run-name>/mod_last.pth
checkpoints_mod/<run-name>/modulation_module.pth
checkpoints_vae/<run-name>/vae_last.pth
```

Train the latent VE score model:

```bash
python trainer_diffusion.py \
  --ckpt checkpoints_mod/mod_last.pth \
  --dataset-name sphere_complex \
  --epochs 100 \
  --batch-size 8 \
  --sigma-min 0.01 \
  --sigma-max 1.0
```

For SELTO-conditioned score training, add `--cond`. Conditional checkpoints use
an 8-channel pooled condition vector, so pass `--cond-dim 8` when sampling from
that checkpoint.

Outputs:

```text
checkpoints_diffusion/diffusion_epoch_<N>.pth
checkpoints_diffusion/diffusion_epoch_<N>_ema.pth
```

## Sampling

Sample from the VAE Gaussian prior:

```bash
python sample_sdf_obj.py \
  --ckpt checkpoints_mod/mod_last.pth \
  --grid 64 \
  --outfile sampled_shape.obj
```

Sample from the trained VE score model:

```bash
python sample_diffusion_model.py \
  --modulation-ckpt checkpoints_mod/mod_last.pth \
  --diffusion-ckpt checkpoints_diffusion/diffusion_epoch_100.pth \
  --grid 64 \
  --timesteps 1000 \
  --sigma-min 0.01 \
  --sigma-max 1.0 \
  --outfile sampled_diffusion_shape.obj
```

If the score checkpoint was trained with `trainer_diffusion.py --cond`, add:

```bash
--cond-dim 8
```

Useful sampling flags:

- `--pad-boundary`: force boundary SDF values from adjacent interior slices
  before marching cubes.
- `--repair-mesh`: run conservative mesh cleanup before export.
- `--true-stats`: load SELTO data and print ground-truth SDF statistics.
- `--save-sample`: export one dataset sample's ground-truth and predicted SDFs.

## Posterior Training

`trainer_posterior.py` trains an unconditional RealNVP flow for one fixed SELTO
problem. It combines:

- a non-differentiable dl4to PDE stress energy, applied with a score-function
  estimator;
- a differentiable VE probability-flow ODE prior from the score model;
- the flow entropy term `log q_phi(z)`.

Example:

```bash
python trainer_posterior.py \
  --modulation-ckpt checkpoints_mod/mod_last.pth \
  --diffusion-ckpt checkpoints_diffusion/diffusion_epoch_100.pth \
  --dataset-name sphere_complex \
  --epochs 20 \
  --batch-size 2 \
  --iters-per-epoch 50 \
  --pf-steps 10
```

If loading a conditional score checkpoint, add:

```bash
--score-cond-dim 8
```

This path is computationally expensive because each iteration decodes latents to
voxel densities and solves a PDE.

## Diagnostics

Run syntax/import checks:

```bash
PYTHONPATH=dl4to:. python -m py_compile \
  models.py trainer.py trainer_diffusion.py trainer_posterior.py \
  sample_sdf_obj.py sample_diffusion_model.py utils/preprocess_data.py
```

Check SDF coordinate conventions:

```bash
PYTHONPATH=dl4to:. python scripts/sanity_check_sdf_coords.py
```

Inspect prediction statistics for a trained modulation checkpoint:

```bash
PYTHONPATH=dl4to:. python scripts/diagnose_sdf_predictions.py \
  --ckpt checkpoints_mod/mod_last.pth \
  --encoding-dim 256 \
  --latent-dim 64
```

## Notes

- Main defaults use `encoding_dim=256` and `latent_dim=64`.
- The SDF decoder is conditioned on the VAE decoder output of size
  `encoding_dim`, not on the raw VAE latent `z`.
- The VE score model uses log-sigma time embeddings. Use the same
  `--sigma-min` and `--sigma-max` for training, sampling, and posterior prior
  evaluation.
- Run scripts from the repository root or set `PYTHONPATH=dl4to:.`.
