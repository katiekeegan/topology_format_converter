# Topology Format Converter

`topology_format_converter` is a small Python package for converting topology
optimization volume data into meshes and SDF training caches.

The package is intentionally explicit about lossy conversions. A SELTO/DL4TO
sample contains more than a surface: it can include density values, loads,
boundary conditions, design-space masks, voxel spacing, split name, and sample
index. Mesh exports such as OBJ, STL, and PLY cannot preserve all of that
information, so this package writes JSON metadata sidecars by default.

## Install

From this repository:

```bash
pip install -e .
```

On HPC systems where dependencies come from environment modules:

```bash
pip install -e . --no-deps
```

SELTO loading requires DL4TO. If you are using a vendored DL4TO package from
another project, include it on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/DL4TO-parent:.
```

## Supported Conversions

```text
SELTO/DL4TO sample -> OBJ/STL/PLY mesh
SELTO/DL4TO sample -> SDF training cache (.npz)
.npy/.npz/.pt volume -> OBJ/STL/PLY mesh
.npy/.npz/.pt volume -> SDF training cache (.npz)
mesh -> mesh format supported by trimesh
```

## Command Line

After installation, use:

```bash
topology-convert --help
```

Without installation:

```bash
python -m topology_format_converter.cli --help
```

Export one SELTO sample as a mesh:

```bash
topology-convert selto-to-mesh \
  --root . \
  --dataset sphere_complex \
  --split train \
  --index 0 \
  --coordinate-mode training \
  --out sphere_complex_0.obj
```

Build an SDF training cache from one SELTO sample:

```bash
topology-convert selto-to-cache \
  --root . \
  --dataset disc_simple \
  --index 0 \
  --num-surface-points 10000 \
  --num-query-points 10000 \
  --out disc_simple_0_cache.npz
```

Convert a volume file to a mesh:

```bash
topology-convert volume-to-mesh density.npy \
  --threshold 0.5 \
  --coordinate-mode unit-box \
  --out density.stl
```

Inspect a mesh or volume:

```bash
topology-convert inspect density.npy
topology-convert inspect density.obj
```

## Python API

```python
from topology_format_converter import (
    density_to_training_cache,
    export_mesh,
    load_selto_sample,
    save_training_cache,
)
from topology_format_converter.selto import selto_sample_to_mesh

sample = load_selto_sample(root=".", dataset="sphere_complex", split="train", index=0)
mesh = selto_sample_to_mesh(sample, threshold=0.5, coordinate_mode="training")
export_mesh(mesh, "sphere_complex_0.obj", metadata=sample.metadata)

cache = density_to_training_cache(sample.density, threshold=0.5)
save_training_cache(cache, "sphere_complex_0_cache.npz")
```

## Coordinate Modes

- `voxel`: preserve voxel-index units, optionally with `--spacing DX DY DZ`.
- `unit-box`: map each axis independently into `[-1, 1]`.
- `training`: use the normalization convention commonly used for SDF model
  training: the longest voxel axis maps to length 2.

## Metadata

Mesh and cache exports write a sidecar unless `--no-metadata` is passed.
Examples:

```text
sample.obj
sample.obj.json
sample_cache.npz
sample_cache.npz.json
```

The sidecar records source format, target format, dataset, split, sample index,
density threshold, voxel shape, spacing, coordinate mode, and SDF sign
convention.

## Development

Run tests:

```bash
python -m pytest tests -q
```
